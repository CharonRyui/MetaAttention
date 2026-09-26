"""FLA parity for Stateful Operator Programs authored in ``examples/fla_coverage.py``.

Each case compiles one DSL Program, runs it on GPU tensors, and compares against
the Flash Linear Attention operator whose reference semantics the Program was
authored from. Tolerances follow the repository's BF16 convention.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import Target
from examples.fla_coverage import (
    EPS,
    based_attention,
    gated_slot_attention,
    hgrn,
    normalized_linear_attention,
    rebased_attention,
    retention_attention,
    rwkv6_attention,
)
from examples.fla_coverage import retention_decay

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
        batch, heads, length, dim, device=gpu_device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        batch, heads, length, dim, device=gpu_device, dtype=torch.bfloat16, generator=generator
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
    actual = compiled(query=query, key=key, value=value).outputs["output"]
    expected = fused_chunk_based(query, key, value, head_first=True)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_rebased_matches_fla_parallel_rebased(gpu_device):
    """Product-Feature Program vs ``parallel_rebased``."""
    from fla.ops.rebased import parallel_rebased

    batch, heads, length, dim, dim_value = 1, 2, 128, 16, 64
    generator = torch.Generator(device=gpu_device).manual_seed(5)
    query = torch.randn(
        batch, heads, length, dim, device=gpu_device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        batch, heads, length, dim, device=gpu_device, dtype=torch.bfloat16, generator=generator
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

    query, key = randn((batch, heads, length, dim)), randn(
        (batch, heads, length, dim)
    )
    slot, value = randn((batch, heads, length, dim_slot)), randn(
        (batch, heads, length, dim_value)
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
