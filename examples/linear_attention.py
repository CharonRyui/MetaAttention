"""Flash Linear Attention families expressed with the StatefulOperator DSL.

All factories use the head-first ``[B, H, T, ...]`` runtime layout expected by
``StatefulOperator``. Their mathematical state is a ``[K, V]`` matrix:

``H_t = P_t(H_{t-1}) + k_t v_t^T`` and ``o_t = q_t H_t / sqrt(K)``.

The names match the corresponding FLA operators:

* :func:`linear_attention` matches ``chunk_linear_attn(normalize=False)``;
* :func:`simple_gla_attention` matches ``chunk_simple_gla``;
* :func:`gla_attention` matches ``chunk_gla`` with a feature-wise gate;
* :func:`lightning_attention` matches Lightning Attention's scalar gate schedule.

FLA's normalized linear attention keeps an additional scalar denominator state;
the first MetaAttention state schema models one matrix state, so that variant
is intentionally exposed only through ``normalize=False`` semantics. Forgetting
attention is softmax attention with decay, not this matrix-state recurrence.
"""
from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def _operator(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int,
    dtype: torch.dtype,
    gate: str | None,
) -> StatefulOperator:
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
    ]
    if gate == "scalar":
        inputs.append(
            TensorInput("gate", (Batch, VALUE_HEADS, Sequence), torch.float32)
        )
    elif gate == "feature":
        inputs.append(
            TensorInput(
                "gate", (Batch, VALUE_HEADS, Sequence, KEY), torch.float32
            )
        )
    elif gate is not None:
        raise ValueError(f"unknown gate kind: {gate!r}")

    propagation_roles = (KEY,) if gate == "feature" else ()
    propagations = (
        (AxisScale(Exp(Input("gate")), propagation_roles),)
        if gate in {"scalar", "feature"}
        else ()
    )
    algorithm = AlgorithmIR(
        inputs=tuple(inputs),
        states=(StateSpec("memory", (Batch, VALUE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=propagations,
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction(
                "output", "memory", Input("query") * (dim**-0.5), KEY, dtype
            ),
        ),
        head_mapping=HeadMapping(
            {
                QUERY_HEADS: VALUE_HEADS,
                KEY_HEADS: VALUE_HEADS,
                VALUE_HEADS: VALUE_HEADS,
            }
        ),
    )
    return StatefulOperator(algorithm)


def _validate_kv_heads(heads: int, kv_heads: int) -> None:
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")


def linear_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build FLA ``chunk_linear_attn(normalize=False)`` semantics."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype, gate=None)


def simple_gla_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build FLA ``chunk_simple_gla`` semantics with a scalar log gate."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype, gate="scalar")


def gla_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build FLA ``chunk_gla`` semantics with a feature-wise log gate."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype, gate="feature")


def lightning_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the FLA Lightning Attention state recurrence.

    The caller supplies the scalar log gate generated from ``layer_idx`` and
    ``num_layers`` by FLA's public API.
    """
    return simple_gla_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def forgetting_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build FLA forgetting-attention's scalar log-decay state recurrence."""
    return simple_gla_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def delta_rule_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build a matrix-state delta-rule-shaped recurrence.

    This keeps the rank-one matrix-state interface available for experiments;
    the current one-state schema represents the plain additive update.
    """
    return linear_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def recurrent_linear_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the explicit recurrent form of un-gated linear attention."""
    return linear_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )
