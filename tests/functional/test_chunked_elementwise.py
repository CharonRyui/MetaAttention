"""The chunked diagonal schedule: boundary carry, chunk blocks, fused readout.

A chunk-aligned dense run of a diagonal Program takes the chunked path: one
kernel writes the running affine pair where each chunk ends, and one replays
each (sequence, head, chunk, column block) independently with the State column
in registers and the readout contracted in place. The per-token suite uses
unaligned lengths, so this file is what covers the path.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import StateTuple, Target
from examples.linear_attention import dual_gate_gla_attention, gla_attention

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

HEADS, LENGTH, DIM = 2, 128, 64


def _compile(operator, heads: int, dim: int, dim_value: int | None = None):
    state = operator.algorithm.states[0]
    return operator.compile(
        role_sizes={
            operator.algorithm.inputs[0].roles[1]: heads,
            operator.algorithm.inputs[1].roles[1]: heads,
            operator.algorithm.inputs[2].roles[1]: heads,
            state.head_role: heads,
            state.feature_roles[0]: dim,
            state.feature_roles[1]: dim if dim_value is None else dim_value,
        },
        target=Target(),
        mode="dense",
        optimized=True,
    )


def test_chunked_path_matches_the_token_recurrence(gpu_device):
    """A row-decay Program on a chunk-aligned length matches the FP32 oracle.

    Covers the boundary carry, the chunk blocks that read it, the fused readout,
    and the final State the last chunk writes.
    """
    generator = torch.Generator(device=gpu_device).manual_seed(3)
    query = torch.randn(1, HEADS, LENGTH, DIM, device=gpu_device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = -torch.rand(1, HEADS, LENGTH, DIM, device=gpu_device, generator=generator) * 0.05
    initial = 0.1 * torch.randn(1, HEADS, DIM, DIM, device=gpu_device)
    operator = gla_attention(1, HEADS, LENGTH, DIM, DIM)
    compiled = _compile(operator, HEADS, DIM)

    with torch.no_grad():
        result = compiled(
            query=query,
            key=key,
            value=value,
            gate=gate,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )

    state = initial[0].clone()
    outputs = []
    for token in range(LENGTH):
        state = state * gate[0, :, token].float().exp().unsqueeze(-1) + torch.einsum(
            "hk,hv->hkv", key[0, :, token].float(), value[0, :, token].float()
        )
        outputs.append(
            torch.einsum(
                "hk,hkv->hv",
                query[0, :, token].float() * DIM**-0.5,
                state,
            )
        )
    expected = torch.stack(outputs, dim=1)
    assert result.final_state is not None
    # the chunk boundaries are what the carry pass feeds forward: a wrong carry
    # would only move those tokens, and the sample covers first, interior, last
    for token in (63, 64, 127):
        torch.testing.assert_close(
            result.outputs["output"][0, :, token].float(),
            expected[:, token],
            rtol=0.03,
            atol=0.1,
        )
    torch.testing.assert_close(
        result.final_state["memory"][0].float(), state, rtol=0.01, atol=1e-4
    )


def test_chunked_path_handles_both_scaled_axes(gpu_device):
    """A pair of axis factors runs through the same chunked carry."""
    generator = torch.Generator(device=gpu_device).manual_seed(5)
    query = torch.randn(1, HEADS, LENGTH, DIM, device=gpu_device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    key_gate = -torch.rand(1, HEADS, LENGTH, DIM, device=gpu_device, generator=generator) * 0.05
    value_gate = -torch.rand(1, HEADS, LENGTH, DIM, device=gpu_device, generator=generator) * 0.05
    operator = dual_gate_gla_attention(1, HEADS, LENGTH, DIM, DIM)
    compiled = _compile(operator, HEADS, DIM)

    with torch.no_grad():
        result = compiled(query=query, key=key, value=value, key_gate=key_gate, value_gate=value_gate)

    state = torch.zeros(HEADS, DIM, DIM, device=gpu_device)
    outputs = []
    for token in range(LENGTH):
        state = state * (
            key_gate[0, :, token].float().exp().unsqueeze(-1)
            * value_gate[0, :, token].float().exp().unsqueeze(-2)
        ) + torch.einsum("hk,hv->hkv", key[0, :, token].float(), value[0, :, token].float())
        outputs.append(
            torch.einsum("hk,hkv->hv", query[0, :, token].float() * DIM**-0.5, state)
        )
    expected = torch.stack(outputs, dim=1)
    for token in (63, 64, 127):
        torch.testing.assert_close(
            result.outputs["output"][0, :, token].float(),
            expected[:, token],
            rtol=0.03,
            atol=0.1,
        )


def test_chunked_entry_runs_a_rank_one_state_beside_a_dense_one():
    """The entry carries several States, one of them rank-1, with rows != columns.

    A rank-1 State is one column per token: its single Feature lands on the rows,
    where a first-Feature readout contracts it, and its injection factor is the
    key while ``value=None`` means the constant one. Its final State returns at the
    declared rank. The two States differ in size, so each gets its own compiled
    kernels, and every other case in this file uses square States, where the row
    and column orientations coincide.
    """
    import torch

    from attention_engine.attn_engine._stateful_chunked import chunked_elementwise_dense

    torch.manual_seed(0)
    device = torch.device("cuda")
    sequences, heads, tokens, rows, columns = 1, 2, 128, 3, 5
    gate = torch.rand(tokens, heads, rows, device=device) * 0.5 + 0.25
    rank_gate = torch.rand(tokens, heads, columns, device=device) * 0.5 + 0.25
    key = torch.rand(tokens, heads, rows, device=device)
    value = torch.rand(tokens, heads, columns, device=device)
    rank_key = torch.rand(tokens, heads, columns, device=device)
    query = torch.rand(tokens, heads, rows, device=device)
    rank_query = torch.rand(tokens, heads, columns, device=device)
    initial = torch.rand(sequences, heads, rows, columns, device=device)
    rank_initial = torch.rand(sequences, heads, columns, device=device)

    outputs, finals = chunked_elementwise_dense(
        gate_rows=(gate, rank_gate),
        gate_columns=(None, None),
        key=(key, rank_key),
        value=(value, None),
        query=(query, rank_query),
        initial_state=(initial, rank_initial),
        output_dtype=torch.float32,
    )
    assert len(outputs) == 2 and len(finals) == 2
    assert outputs[0].shape == (sequences, heads, tokens, columns)
    assert outputs[1].shape == (sequences, heads, tokens, 1), "a rank-1 State is one column"
    assert finals[1].shape == (sequences, heads, columns)

    state = initial.clone()
    rank_state = rank_initial.clone()
    expected = torch.empty_like(outputs[0])
    expected_rank = torch.empty_like(outputs[1])
    for token in range(tokens):
        injection = key[token].unsqueeze(-1) * value[token].unsqueeze(-2)
        state = gate[token].unsqueeze(0).unsqueeze(-1) * state + injection.unsqueeze(0)
        rank_state = rank_gate[token].unsqueeze(0) * rank_state + rank_key[token].unsqueeze(0)
        # a first-Feature readout contracts the rows: out[g] = sum_f q[f] S[f, g]
        expected[:, :, token] = torch.einsum("shfg,hf->shg", state, query[token])
        expected_rank[:, :, token] = (
            rank_state * rank_query[token].unsqueeze(0)
        ).sum(-1, keepdim=True)
    torch.testing.assert_close(outputs[0].float(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(outputs[1].float(), expected_rank, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(finals[0], state, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(finals[1], rank_state, atol=1e-4, rtol=1e-4)
