"""Independent CPU token oracles for bounded fused diagonal readout regions."""
from __future__ import annotations

import pytest
import torch

from attn_engine._stateful_diagonal_kernels import prepare_state, read_state
from attn_engine._stateful_diagonal_block import diagonal_block_bonus_readout
from attn_engine._stateful_readout_regions import (
    fused_normalized_region,
    fused_readout_region,
    fused_shared_readouts,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _inputs(device, rows, columns, *, batch=2, heads=2, length=71, seed=43):
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(batch, length, heads, rows, generator=generator) * 0.2
    value = torch.randn(batch, length, heads, columns, generator=generator) * 0.2
    query = torch.randn(batch, length, heads, rows, generator=generator) * 0.2
    gr = torch.rand(batch, length, heads, rows, generator=generator) * 0.9
    gc = torch.rand(batch, length, heads, columns, generator=generator) * 0.9
    gr[:, 4::11] = 0
    gc[:, 9::13] = 0
    gr[:, 6::9] *= -1
    gc[:, 3::7] *= -1
    gr[:, 15:22] *= 0.001
    initial = torch.randn(batch, heads, rows, columns, generator=generator) * 0.1
    host = key, value, query, gr, gc, initial
    return host, tuple(tensor.to(device) for tensor in host)


def _prepare(data, *, row=True, column=True):
    key, value, _, gr, gc, initial = data
    batch, length, heads, rows = key.shape
    carry, final, factors = prepare_state(
        rows_gate=gr.reshape(batch * length, heads, rows) if row else None,
        columns_gate=gc.reshape(batch * length, heads, -1) if column else None,
        key=key.reshape(batch * length, heads, rows),
        value=value.reshape(batch * length, heads, -1),
        initial=initial, sequence_length=length, return_final_state=True,
        identity=not row and not column,
    )
    return (carry, factors, not row and not column), final


def _states(host, *, row=True, column=True):
    """No kernel helpers or optimized execution in this FP32 oracle."""
    key, value, _, gr, gc, initial = host
    state = initial.clone()
    prior, post = [], []
    for t in range(key.shape[1]):
        prior.append(state.clone())
        if row:
            state = state * gr[:, t].unsqueeze(-1)
        if column:
            state = state * gc[:, t].unsqueeze(-2)
        state = state + key[:, t].unsqueeze(-1) * value[:, t].unsqueeze(-2)
        post.append(state.clone())
    return prior, post, state


@pytest.mark.parametrize("prior_first,prior_second", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("axes", [(True, True), (True, False), (False, True), (False, False)])
def test_softmax_region_true_feature_reduction(gpu_device, prior_first, prior_second, axes):
    # Unequal rows/slots/output sizes expose transposed/wrong-axis reductions.
    first_host, first_data = _inputs(gpu_device, 13, 19)
    second_host, second_data = _inputs(gpu_device, 19, 11, seed=47)
    first, final1 = _prepare(first_data, row=axes[0], column=axes[1])
    second, final2 = _prepare(second_data, row=axes[1], column=axes[0])
    query = first_data[2].flatten(0, 1)
    actual = fused_readout_region(first, second, query, sequence_length=71,
                                 prior_first=prior_first, prior_second=prior_second)
    assert actual is not None
    before1, after1, expected_final1 = _states(first_host, row=axes[0], column=axes[1])
    before2, after2, expected_final2 = _states(second_host, row=axes[1], column=axes[0])
    states1 = before1 if prior_first else after1
    states2 = before2 if prior_second else after2
    expected = []
    for t, (state1, state2) in enumerate(zip(states1, states2)):
        logits = torch.einsum("bhr,bhrc->bhc", first_host[2][:, t], state1)
        weights = torch.softmax(logits, dim=-1)
        expected.append(torch.einsum("bhc,bhcd->bhd", weights, state2))
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(actual.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final1.cpu(), expected_final1, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final2.cpu(), expected_final2, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("prior", [False, True])
def test_normalized_region_and_independent_shared_outputs(gpu_device, prior):
    host, data = _inputs(gpu_device, 17, 9)
    numerator, _ = _prepare(data)
    # Positive initial, keys and values ensure denominator away from zero.
    den_host = tuple(t.clone() for t in host)
    den_host = (den_host[0].abs() + 0.1, torch.ones(2, 71, 2, 1), den_host[2],
                den_host[3].abs() + 0.05, torch.ones(2, 71, 2, 1),
                torch.ones(2, 2, 17, 1) * 0.1)
    den_data = tuple(t.to(gpu_device) for t in den_host)
    denominator, _ = _prepare(den_data)
    query_host = host[2].abs() + 0.1
    query = query_host.to(gpu_device).flatten(0, 1)
    normalized = fused_normalized_region(numerator, denominator, query,
                                         sequence_length=71, prior_first=prior, prior_second=prior)
    shared = fused_shared_readouts(numerator, (query, -query), sequence_length=71,
                                   priors=(prior, not prior))
    added = fused_shared_readouts(numerator, (query, -query), sequence_length=71,
                                  priors=(prior, not prior), epilogue="add")
    assert normalized is not None and shared is not None and added is not None
    pre, post, _ = _states(host)
    den_pre, den_post, _ = _states(den_host)
    selected = pre if prior else post
    opposite = post if prior else pre
    den_selected = den_pre if prior else den_post
    expected, expected_other, expected_normalized = [], [], []
    for t in range(71):
        n = torch.einsum("bhr,bhrc->bhc", query_host[:, t], selected[t])
        other = torch.einsum("bhr,bhrc->bhc", -query_host[:, t], opposite[t])
        d = torch.einsum("bhr,bhrc->bhc", query_host[:, t], den_selected[t])
        expected.append(n)
        expected_other.append(other)
        expected_normalized.append(n / d)
    expected, expected_other = torch.stack(expected, 1), torch.stack(expected_other, 1)
    expected_normalized = torch.stack(expected_normalized, 1)
    for result, oracle in ((normalized, expected_normalized), (shared[0], expected),
                           (shared[1], expected_other), (added[0], expected + expected_other)):
        torch.testing.assert_close(result.reshape_as(oracle).cpu(), oracle, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("prior_first,prior_second", [(False, False), (True, False), (False, True), (True, True)])
def test_identity_normalized_independent_keys_and_priors(gpu_device, prior_first, prior_second):
    first_host, first_data = _inputs(gpu_device, 17, 35)
    second_host, _ = _inputs(gpu_device, 17, 1, seed=71)
    second_host = (second_host[0].abs() + 0.1, second_host[1].abs() + 0.2,
                   second_host[2], second_host[3], second_host[4], second_host[5].abs() + 0.1)
    second_data = tuple(t.to(gpu_device) for t in second_host)
    first, _ = _prepare(first_data, row=False, column=False)
    second, _ = _prepare(second_data, row=False, column=False)
    query_host = first_host[2].abs() + 0.1
    epsilon = 1e-3
    actual = fused_normalized_region(
        first, second, query_host.to(gpu_device).flatten(0, 1), sequence_length=71,
        prior_first=prior_first, prior_second=prior_second, epsilon=epsilon,
    )
    pre1, post1, _ = _states(first_host, row=False, column=False)
    pre2, post2, _ = _states(second_host, row=False, column=False)
    selected1, selected2 = (pre1 if prior_first else post1), (pre2 if prior_second else post2)
    expected = torch.stack([
        torch.einsum("bhr,bhrc->bhc", query_host[:, t], selected1[t])
        / (torch.einsum("bhr,bhrc->bhc", query_host[:, t], selected2[t]) + epsilon)
        for t in range(71)
    ], 1)
    torch.testing.assert_close(actual.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)


def test_oversized_softmax_region_preserves_executor_fallback(gpu_device):
    _, first_data = _inputs(gpu_device, 96, 96, length=7)
    _, second_data = _inputs(gpu_device, 96, 33, length=7, seed=47)
    first, _ = _prepare(first_data)
    second, _ = _prepare(second_data)
    query = first_data[2].flatten(0, 1)
    assert fused_readout_region(first, second, query, sequence_length=7) is None
    # The previous executor remains valid for the rejected resource shape.
    logits = read_state(first[0], first[1], query, sequence_length=7, identity=first[2])
    result = read_state(second[0], second[1], logits.softmax(-1), sequence_length=7,
                        identity=second[2])
    _, post1, _ = _states(tuple(t.cpu() for t in first_data))
    _, post2, _ = _states(tuple(t.cpu() for t in second_data))
    expected = []
    for t in range(7):
        weights = torch.einsum("bhr,bhrc->bhc", first_data[2][:, t].cpu(), post1[t]).softmax(-1)
        expected.append(torch.einsum("bhr,bhrc->bhc", weights, post2[t]))
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(result.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("axes", ["scalar", "row", "column", "dual"])
@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 137])
def test_parallel_diagonal_block_fp32_oracle(gpu_device, axes, prior, length):
    host, _ = _inputs(gpu_device, 35, 23, length=length)
    key, value, query, gr, gc, initial = host
    row, column = axes != "column", axes in ("column", "dual")
    if axes == "scalar":
        gr = gr[..., :1].expand_as(gr).clone()
    host = key, value, query, gr, gc, initial
    data = tuple(t.to(gpu_device) for t in host)
    prepared, final = _prepare(data, row=row, column=column)
    actual = read_state(prepared[0], prepared[1], data[2].flatten(0, 1),
                        sequence_length=length, prior=prior, identity=False)
    before, after, expected_final = _states(host, row=row, column=column)
    selected = before if prior else after
    expected = torch.stack([
        torch.einsum("bhr,bhrc->bhc", query[:, t], state)
        for t, state in enumerate(selected)
    ], 1)
    torch.testing.assert_close(actual.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final.cpu(), expected_final, rtol=2e-4, atol=2e-5)
    if length > 64:
        # A nonaligned split changes both chunk and two-target tile phase.
        split = 17
        prefix_data = tuple(t[:, :split] for t in data[:-1]) + (data[-1],)
        prefix, prefix_final = _prepare(prefix_data, row=row, column=column)
        prefix_output = read_state(
            prefix[0], prefix[1], prefix_data[2].flatten(0, 1),
            sequence_length=split, prior=prior, identity=False,
        ).reshape(key.shape[0], split, key.shape[2], value.shape[-1])
        suffix_data = tuple(t[:, split:] for t in data[:-1]) + (prefix_final,)
        suffix, suffix_final = _prepare(suffix_data, row=row, column=column)
        suffix_output = read_state(
            suffix[0], suffix[1], suffix_data[2].flatten(0, 1),
            sequence_length=length - split, prior=prior, identity=False,
        ).reshape(key.shape[0], length - split, key.shape[2], value.shape[-1])
        continued = torch.cat((prefix_output, suffix_output), 1)
        torch.testing.assert_close(continued.cpu(), expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(suffix_final.cpu(), expected_final, rtol=2e-4, atol=2e-5)



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 3, 65, 137])
def test_diagonal_bonus_readout_matches_prior_recurrence(gpu_device, length, dtype):
    host, data = _inputs(gpu_device, 19, 23, length=length, seed=97)
    key, value, query, gr, _, initial = data
    if dtype == torch.bfloat16:
        key, value, query = key.to(dtype), value.to(dtype), query.to(dtype)
    bonus = torch.randn(
        key.shape, device=gpu_device, dtype=dtype,
        generator=torch.Generator(device=gpu_device).manual_seed(101),
    )
    carry, final, factors = prepare_state(
        rows_gate=gr.reshape(-1, key.shape[2], key.shape[3]),
        columns_gate=None,
        key=key.reshape(-1, key.shape[2], key.shape[3]),
        value=value.reshape(-1, value.shape[2], value.shape[3]),
        initial=initial,
        sequence_length=length,
        return_final_state=True,
        identity=False,
    )
    output = torch.empty(
        key.shape[0] * length, key.shape[2], value.shape[3],
        device=gpu_device, dtype=torch.float32,
    )
    scale = 19**-0.5
    diagonal_block_bonus_readout(
        carry, factors, query.reshape(-1, query.shape[2], query.shape[3]),
        bonus.reshape(-1, bonus.shape[2], bonus.shape[3]),
        key.reshape(-1, key.shape[2], key.shape[3]), output,
        sequence_length=length, query_scale=scale,
    )

    key_cpu, value_cpu, query_cpu, gr_cpu, initial_cpu = (
        key.cpu(), value.cpu(), query.cpu(), gr.cpu(), initial.cpu()
    )
    bonus_cpu = bonus.cpu()
    state = initial_cpu.clone()
    expected = []
    for token in range(length):
        q = query_cpu[:, token]
        b = bonus_cpu[:, token]
        k = key_cpu[:, token]
        v = value_cpu[:, token]
        if dtype == torch.bfloat16:
            scaled_q = (q * torch.tensor(scale, dtype=dtype)).to(dtype)
            bonus_score = (scaled_q * b * k).float().sum(-1, keepdim=True)
            state_term = torch.einsum("bhr,bhrc->bhc", scaled_q.float(), state)
        else:
            scaled_q = q * scale
            bonus_score = (scaled_q * b * k).sum(-1, keepdim=True)
            state_term = torch.einsum("bhr,bhrc->bhc", scaled_q, state)
        expected.append(state_term + bonus_score * v.float())
        state = state * gr_cpu[:, token].unsqueeze(-1) + k.float().unsqueeze(-1) * v.float().unsqueeze(-2)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(output.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final.cpu(), state, rtol=2e-4, atol=2e-5)

@pytest.mark.parametrize("prior", [False, True])
def test_diagonal_block_separated_product_features(gpu_device, prior):
    generator = torch.Generator().manual_seed(83)
    host, _ = _inputs(gpu_device, 35, 23, length=71)
    _, value, _, gr, gc, initial = host
    left = torch.randn(2, 71, 2, 5, generator=generator) * 0.2
    right = torch.randn(2, 71, 2, 7, generator=generator) * 0.2
    qleft = torch.randn(2, 71, 2, 5, generator=generator) * 0.2
    qright = torch.randn(2, 71, 2, 7, generator=generator) * 0.2
    # BF16 input factors are converted to FP32 before the separated product.
    key = (left.to(torch.bfloat16).float().unsqueeze(-1)
           * right.to(torch.bfloat16).float().unsqueeze(-2)).flatten(-2)
    query = (qleft.to(torch.bfloat16).float().unsqueeze(-1)
             * qright.to(torch.bfloat16).float().unsqueeze(-2)).flatten(-2)
    carry, final, factors = prepare_state(
        rows_gate=gr.to(gpu_device).flatten(0, 1),
        columns_gate=gc.to(gpu_device).flatten(0, 1),
        key=tuple(part.to(torch.bfloat16).to(gpu_device).flatten(0, 1) for part in (left, right)),
        value=value.to(gpu_device).flatten(0, 1), initial=initial.to(gpu_device),
        sequence_length=71, return_final_state=True, identity=False,
    )
    actual = read_state(carry, factors,
                        tuple(part.to(torch.bfloat16).to(gpu_device).flatten(0, 1)
                              for part in (qleft, qright)),
                        sequence_length=71, prior=prior, identity=False)
    before, after, expected_final = _states((key, value, query, gr, gc, initial))
    expected = torch.stack([
        torch.einsum("bhr,bhrc->bhc", query[:, t], state)
        for t, state in enumerate(before if prior else after)
    ], 1)
    torch.testing.assert_close(actual.reshape_as(expected).cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final.cpu(), expected_final, rtol=2e-4, atol=2e-5)
