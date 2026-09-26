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
from examples.linear_attention import dual_gate_gla_attention, gla_attention, simple_gla_attention

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


def test_scalar_decay_keeps_the_factorized_selection(gpu_device):
    """A whole-State decay still lowers to the fused scalar kernel."""
    operator = simple_gla_attention(1, HEADS, LENGTH, DIM, DIM)
    compiled = _compile(operator, HEADS, DIM)
    assert (
        compiled.operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-scalar-factorized"
    )
