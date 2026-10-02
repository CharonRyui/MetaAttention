"""FLA parity for the Stateful Operator Programs the coverage table lists.

Each case compiles one DSL Program, runs it on GPU tensors, and compares against
the Flash Linear Attention operator whose reference semantics the Program was
authored from. Tolerances follow the repository's BF16 convention.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import Target
from examples.based_attention import based_attention
from examples.gated_slot_attention import gated_slot_attention
from examples.hgrn import hgrn
from examples.normalized_linear_attention import EPS, normalized_linear_attention
from examples.rebased_attention import rebased_attention
from examples.retention_attention import retention_attention, retention_decay
from examples.rwkv6_attention import rwkv6_attention

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _compile(factory, named_roles, **kwargs):
    from attn_engine import FeatureRole, HeadRole

    operator = factory(**kwargs)
    roles = {
        role.name: role
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if isinstance(role, (HeadRole, FeatureRole))
    }
    return operator.compile(
        role_sizes={role: named_roles[name] for name, role in roles.items()},
        target=Target(),
        mode="dense",
        optimized=True,
    )


def test_normalized_linear_attention_matches_streaming_reference(gpu_device):
    """Two-State Program with a division Readout against a streaming FP32 recurrence.

    In FP32 the Program must reproduce the recurrence the library helper
    documents; in BF16 it must stay within the output-scale tolerance. The
    denominator can approach zero, so the BF16 bound is taken against the largest
    output magnitude rather than element by element.

    ``fla.ops.linear_attn.utils.normalize_output`` divides by
    ``(scale*q) . cumsum(k) + 1e-10`` and expects the ``[B, H, T, D]`` layout,
    while ``chunk_linear_attn`` passes it the operator's internal ``[B, T, H, D]``
    tensors. That layout mismatch makes the library operator's normalized output
    unusable as an oracle, so the streaming recurrence above is used instead; the
    mismatch itself is a FLA finding to report, not something to encode here.
    """
    batch, heads, length, dim, dim_value = 1, 4, 128, 64, 64
    for dtype, rtol, atol in (
        (torch.float32, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, None),
    ):
        generator = torch.Generator(device=gpu_device).manual_seed(0)
        query = torch.randn(
            batch,
            heads,
            length,
            dim,
            device=gpu_device,
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            batch,
            heads,
            length,
            dim,
            device=gpu_device,
            dtype=dtype,
            generator=generator,
        )
        value = torch.randn(
            batch,
            heads,
            length,
            dim_value,
            device=gpu_device,
            dtype=dtype,
            generator=generator,
        )

        compiled = _compile(
            normalized_linear_attention,
            {
                "query_heads": heads,
                "key_heads": heads,
                "value_heads": heads,
                "state_heads": heads,
                "key_dim": dim,
                "value_dim": dim_value,
            },
            batch=batch,
            heads=heads,
            seqlen=length,
            dim=dim,
            dim_value=dim_value,
            dtype=dtype,
        )
        actual = compiled(query=query, key=key, value=value).outputs["output"].float()
        assert torch.isfinite(actual).all()
        expected = _normalized_reference(query, key, value)
        difference = (actual - expected).abs()
        if atol is None:
            assert (difference / expected.abs().max()).max().item() < rtol
        else:
            assert (difference / (atol + rtol * expected.abs())).max().item() < 1.0


@pytest.mark.parametrize("layout,source_heads", [("bht", (2, 2, 2)), ("bth", (2, 1, 2))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [63, 64, 65, 130])
def test_normalized_final_states_continuation_and_authored_epsilon(
    gpu_device, dtype, length, layout, source_heads,
):
    from attn_engine import AlgorithmIR, Batch, Sequence, StatefulOperator, StateTuple, Target, TensorInput
    from examples.normalized_linear_attention import KEY, VALUE, QUERY_HEADS, KEY_HEADS, VALUE_HEADS, STATE_HEADS

    batch, heads, dim, dim_value = 2, (4 if layout == "bth" else 2), 17, 35
    qh, kh, vh = source_heads
    generator = torch.Generator().manual_seed(193)
    query = (torch.rand(batch, qh, length, dim, generator=generator) + 0.1).to(dtype)
    key = (torch.rand(batch, kh, length, dim, generator=generator) + 0.1).to(dtype)
    value = torch.randn(batch, vh, length, dim_value, generator=generator).to(dtype)
    memory = torch.rand(batch, heads, dim, dim_value, generator=generator) * 0.1 + 0.1
    key_sum = torch.zeros(batch, heads, dim)
    # Nonzero numerator and exactly zero denominator expose a dropped epsilon.
    key[:, :, 0] = 0
    initial = StateTuple(("memory", "key_sum"),
                         (memory.to(gpu_device), key_sum.to(gpu_device)))
    initial_before = {name: initial[name].clone() for name in initial.names}

    def assert_initial_unchanged(before, state, result):
        for name in state.names:
            tensor = state[name]
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
            if result.final_state is not None:
                for final in result.final_state:
                    assert final.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()

    expected_outputs = []
    q_oracle = query.repeat_interleave(heads // qh, dim=1)
    k_oracle = key.repeat_interleave(heads // kh, dim=1)
    v_oracle = value.repeat_interleave(heads // vh, dim=1)
    # CUDA TensorIterator converts the scalar to the input computation dtype.
    scale = torch.tensor(dim**-0.5, dtype=dtype)
    for token in range(length):
        memory = memory + k_oracle[:, :, token].float().unsqueeze(-1) * v_oracle[:, :, token].float().unsqueeze(-2)
        key_sum = key_sum + k_oracle[:, :, token].float()
        # Input arithmetic rounds in its authored dtype before the FP32 readout.
        scaled = (q_oracle[:, :, token] * scale).float()
        numerator = (scaled.unsqueeze(-1) * memory).sum(-2)
        denominator = (scaled * key_sum).sum(-1, keepdim=True)
        expected_outputs.append(numerator / (denominator + EPS))
    expected = torch.stack(expected_outputs, 2).to(dtype).float()
    algorithm = normalized_linear_attention(batch, heads, length, dim, dim_value, dtype=dtype).algorithm
    if layout == "bth":
        inputs_ir = tuple(TensorInput(spec.name, (Batch, Sequence, spec.roles[1], spec.roles[-1]), dtype)
                          for spec in algorithm.inputs)
        algorithm = AlgorithmIR(inputs=inputs_ir, states=algorithm.states,
                                transition=algorithm.transitions, readouts=algorithm.readouts,
                                head_mapping=algorithm.head_mapping)
    compiled = StatefulOperator(algorithm).compile(
        role_sizes={QUERY_HEADS: qh, KEY_HEADS: kh, VALUE_HEADS: vh, STATE_HEADS: heads,
                    KEY: dim, VALUE: dim_value}, target=Target(), mode="dense", optimized=True,
    )
    inputs = {name: (tensor.transpose(1, 2).contiguous() if layout == "bth" else tensor).to(gpu_device)
              for name, tensor in (("query", query), ("key", key), ("value", value))}
    full = compiled(**inputs, initial_state=initial, return_final_state=True)
    assert_initial_unchanged(initial_before, initial, full)
    output_only = compiled(**inputs, initial_state=initial)
    assert output_only.final_state is None
    assert_initial_unchanged(initial_before, initial, output_only)
    split = 64 if length > 64 else length // 2
    axis = 1 if layout == "bth" else 2
    prefix = compiled(**{name: tensor.narrow(axis, 0, split).contiguous() for name, tensor in inputs.items()},
                      initial_state=initial, return_final_state=True)
    assert_initial_unchanged(initial_before, initial, prefix)
    prefix_before = {name: prefix.final_state[name].clone() for name in prefix.final_state.names}
    suffix = compiled(**{name: tensor.narrow(axis, split, length - split).contiguous()
                        for name, tensor in inputs.items()},
                      initial_state=prefix.final_state, return_final_state=True)
    assert_initial_unchanged(initial_before, initial, suffix)
    assert_initial_unchanged(prefix_before, prefix.final_state, suffix)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-4
    for actual in (full.outputs["output"], output_only.outputs["output"],
                   torch.cat((prefix.outputs["output"], suffix.outputs["output"]), 2)):
        torch.testing.assert_close(actual.float().cpu(), expected, rtol=tolerance, atol=2e-5)
    for result in (full, suffix):
        assert result.final_state.names == ("memory", "key_sum")
        torch.testing.assert_close(result.final_state["memory"].cpu(), memory, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(result.final_state["key_sum"].cpu(), key_sum, rtol=2e-4, atol=2e-5)
    if dtype == torch.bfloat16:
        # Injections are FP32: lossless input promotion cannot change either
        # final State. Catch raw-load dot layouts reordering BF16 accumulation.
        promoted = AlgorithmIR(
            inputs=tuple(TensorInput(spec.name, spec.roles, torch.float32) for spec in algorithm.inputs),
            states=algorithm.states, transition=algorithm.transitions,
            readouts=algorithm.readouts, head_mapping=algorithm.head_mapping,
        )
        promoted_compiled = StatefulOperator(promoted).compile(
            role_sizes={QUERY_HEADS: qh, KEY_HEADS: kh, VALUE_HEADS: vh, STATE_HEADS: heads,
                        KEY: dim, VALUE: dim_value}, target=Target(), mode="dense", optimized=True,
        )
        promoted_result = promoted_compiled(
            **{name: tensor.float() for name, tensor in inputs.items()},
            initial_state=initial, return_final_state=True,
        )
        assert_initial_unchanged(initial_before, initial, promoted_result)
        for name in full.final_state.names:
            torch.testing.assert_close(full.final_state[name], promoted_result.final_state[name],
                                       rtol=0, atol=0)


def _normalized_reference(query, key, value):
    """Streaming numerator/denominator recurrence in FP32."""
    scale = key.shape[-1] ** -0.5
    state = torch.zeros(
        query.shape[0],
        query.shape[1],
        query.shape[-1],
        value.shape[-1],
        device=query.device,
    )
    key_sum = torch.zeros_like(state[..., 0])
    outputs = []
    for token in range(query.shape[2]):
        state = state + torch.einsum(
            "bhf,bhg->bhfg", key[:, :, token].float(), value[:, :, token].float()
        )
        key_sum = key_sum + key[:, :, token].float()
        scaled = query[:, :, token].float() * scale
        numerator = torch.einsum("bhf,bhfg->bhg", scaled, state)
        denominator = torch.einsum("bhf,bhf->bh", scaled, key_sum)
        outputs.append(numerator / (denominator.unsqueeze(-1) + EPS))
    return torch.stack(outputs, dim=2)


def _normalized_reference_head_axis(query, key, value):
    """The denominator ``normalize_output`` computes on an ``[B, T, H, D]`` argument."""
    scale = key.shape[-1] ** -0.5
    transposed_q = query.transpose(1, 2).float() * scale
    transposed_k = key.transpose(1, 2).float()
    state = torch.zeros(
        query.shape[0],
        query.shape[1],
        query.shape[-1],
        value.shape[-1],
        device=query.device,
    )
    outputs = []
    for token in range(query.shape[2]):
        state = state + torch.einsum(
            "bhf,bhg->bhfg", key[:, :, token].float(), value[:, :, token].float()
        )
        cumulative = transposed_k[:, token].cumsum(dim=-2)
        numerator = torch.einsum(
            "bhf,bhfg->bhg", query[:, :, token].float() * scale, state
        )
        denominator = torch.einsum("bhf,bhf->bh", transposed_q[:, token], cumulative)
        outputs.append(numerator / (denominator.unsqueeze(-1) + EPS))
    return torch.stack(outputs, dim=2)


def test_hgrn_matches_fla_chunk_hgrn(gpu_device):
    """Rank-1 State with an identity Readout vs ``chunk_hgrn``."""
    from fla.ops.hgrn import chunk_hgrn

    batch, length, dim = 1, 128, 64
    generator = torch.Generator(device=gpu_device).manual_seed(1)
    x = torch.randn(
        batch, length, dim, device=gpu_device, dtype=torch.bfloat16, generator=generator
    )
    g = torch.nn.functional.logsigmoid(
        torch.randn(batch, length, dim, device=gpu_device, generator=generator)
    )

    compiled = _compile(
        hgrn,
        {"value_heads": 1, "state_heads": 1, "value_dim": dim},
        batch=batch,
        heads=1,
        seqlen=length,
        dim_value=dim,
    )
    actual = compiled(value=x.unsqueeze(1), gate=g.unsqueeze(1)).outputs["output"]
    expected = chunk_hgrn(x, g, output_final_state=False)[0]
    torch.testing.assert_close(
        actual.float().squeeze(1), expected.float(), rtol=3e-2, atol=3e-2
    )


def test_retention_matches_fla_chunk_retention(gpu_device):
    """Per-head log decay supplied as a head-shaped Input vs ``chunk_retention``."""
    from fla.ops.retention import chunk_retention

    batch, heads, length, dim, dim_value = 1, 4, 128, 64, 64
    generator = torch.Generator(device=gpu_device).manual_seed(2)
    query = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        batch,
        heads,
        length,
        dim_value,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    compiled = _compile(
        retention_attention,
        {
            "query_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "state_heads": heads,
            "key_dim": dim,
            "value_dim": dim_value,
        },
        batch=batch,
        heads=heads,
        seqlen=length,
        dim=dim,
        dim_value=dim_value,
    )
    actual = compiled(
        query=query,
        key=key,
        value=value,
        decay=retention_decay(heads)
        .to(gpu_device)
        .expand(batch, heads, length)
        .contiguous(),
    ).outputs["output"]
    expected = chunk_retention(query, key, value, head_first=True)[0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_rwkv6_matches_fla_chunk_rwkv6(gpu_device):
    """Prior-State Readout with a bonus term vs ``chunk_rwkv6``."""
    from fla.ops.rwkv6 import chunk_rwkv6

    batch, heads, length, dim, dim_value = 1, 2, 128, 64, 64
    generator = torch.Generator(device=gpu_device).manual_seed(3)

    def randn(shape):
        return torch.randn(
            shape, device=gpu_device, dtype=torch.bfloat16, generator=generator
        )

    query, key, value = (
        randn((batch, heads, length, dim)),
        randn((batch, heads, length, dim)),
        randn((batch, heads, length, dim_value)),
    )
    decay = torch.nn.functional.logsigmoid(
        torch.randn(batch, heads, length, dim, device=gpu_device, generator=generator)
    )
    bonus = torch.randn(heads, dim, device=gpu_device, generator=generator).to(
        torch.bfloat16
    )

    compiled = _compile(
        rwkv6_attention,
        {
            "query_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "state_heads": heads,
            "key_dim": dim,
            "value_dim": dim_value,
        },
        batch=batch,
        heads=heads,
        seqlen=length,
        dim=dim,
        dim_value=dim_value,
    )
    actual = compiled(
        query=query,
        key=key,
        value=value,
        decay=decay,
        bonus=bonus[None, :, None, :].expand(batch, heads, length, dim).contiguous(),
    ).outputs["output"]
    expected = chunk_rwkv6(
        query,
        key,
        value,
        decay,
        bonus.float(),
        scale=dim**-0.5,
        head_first=True,
    )[0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_based_matches_fla_fused_chunk_based(gpu_device):
    """Second-order Taylor Program vs ``fused_chunk_based``."""
    from fla.ops.based import fused_chunk_based

    batch, heads, length, dim, dim_value = 1, 2, 128, 16, 64
    generator = torch.Generator(device=gpu_device).manual_seed(4)
    query = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        batch,
        heads,
        length,
        dim_value,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    compiled = _compile(
        based_attention,
        {
            "query_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "state_heads": heads,
            "unit_dim": 1,
            "key_dim": dim,
            "key_dim_x_key_dim": dim * dim,
            "value_dim": dim_value,
        },
        batch=batch,
        heads=heads,
        seqlen=length,
        dim=dim,
        dim_value=dim_value,
    )
    plan = compiled.operator._plans[compiled.specialization].plan
    assert plan.candidate.lowerer == "tilelang-product-feature-chunked"
    assert compiled.operator._plans[compiled.specialization].runtime_family == (
        "product-feature-chunked"
    )
    actual = compiled(query=query, key=key, value=value).outputs["output"]
    expected = fused_chunk_based(query, key, value, head_first=True)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_rebased_matches_fla_parallel_rebased(gpu_device):
    """Product-Feature Program vs ``parallel_rebased``."""
    from fla.ops.rebased import parallel_rebased

    batch, heads, length, dim, dim_value = 1, 2, 128, 16, 64
    generator = torch.Generator(device=gpu_device).manual_seed(5)
    query = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        batch,
        heads,
        length,
        dim,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        batch,
        heads,
        length,
        dim_value,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    compiled = _compile(
        rebased_attention,
        {
            "query_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "state_heads": heads,
            "key_dim": dim,
            "key_dim_x_key_dim": dim * dim,
            "value_dim": dim_value,
        },
        batch=batch,
        heads=heads,
        seqlen=length,
        dim=dim,
        dim_value=dim_value,
    )
    plan = compiled.operator._plans[compiled.specialization].plan
    assert plan.candidate.lowerer == "tilelang-product-feature-chunked"
    assert compiled.operator._plans[compiled.specialization].runtime_family == (
        "product-feature-chunked"
    )
    actual = compiled(query=query, key=key, value=value).outputs["output"]
    expected = parallel_rebased(query, key, value, head_first=True)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_gated_slot_matches_fla_chunk_gsa(gpu_device):
    """Two-State Program joined by a slot Softmax vs ``chunk_gsa``."""
    from fla.ops.gsa import chunk_gsa

    batch, heads, length, dim, dim_slot, dim_value = 1, 2, 128, 64, 16, 64
    generator = torch.Generator(device=gpu_device).manual_seed(6)

    def randn(shape):
        return torch.randn(
            shape, device=gpu_device, dtype=torch.bfloat16, generator=generator
        )

    query, key = randn((batch, heads, length, dim)), randn((batch, heads, length, dim))
    slot, value = (
        randn((batch, heads, length, dim_slot)),
        randn((batch, heads, length, dim_value)),
    )
    gate = torch.nn.functional.logsigmoid(
        torch.randn(
            batch, heads, length, dim_slot, device=gpu_device, generator=generator
        )
    )

    compiled = _compile(
        gated_slot_attention,
        {
            "query_heads": heads,
            "key_heads": heads,
            "value_heads": heads,
            "state_heads": heads,
            "key_dim": dim,
            "slot_dim": dim_slot,
            "value_dim": dim_value,
        },
        batch=batch,
        heads=heads,
        seqlen=length,
        dim=dim,
        dim_slot=dim_slot,
        dim_value=dim_value,
    )
    actual = compiled(
        query=query, key=key, slot=slot, value=value, gate=gate.to(query.dtype)
    ).outputs["output"]
    expected = chunk_gsa(
        query, key, value, slot, gate, scale=dim**-0.5, head_first=True
    )[0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
