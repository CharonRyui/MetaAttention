"""FLA ``chunk_linear_attn(normalize=True)`` expressed with the Stateful Operator DSL.

The recurrence keeps FLA's extra scalar denominator State: the library divides
the linear-attention output by ``(scale*q) . cumsum(k) + 1e-10``."""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    Constant,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    Sequence,
    StatefulOperator,
    StateSpec,
    StateTerm,
    StateTransition,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")

# the epsilon FLA's ``normalize_output`` adds to the denominator
EPS = 1e-10


def normalized_linear_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_linear_attn(..., normalize=True)``.

    ``fla.ops.linear_attn.utils.normalize_output`` divides the recurrence output
    by ``(scale*q) . cumsum(k) + 1e-10``. The denominator is a rank-1 State that
    accumulates ``k``, so the normalized operator is one Program with two States.
    """
    scale = dim**-0.5
    scaled_query = Input("query") * scale
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
            StateSpec("key_sum", (Batch, STATE_HEADS, KEY)),
        ),
        transition=(
            StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "key_sum",
                injections=(ProductInjection((ProductFactor(Input("key"), (KEY,)),)),),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm("memory", scaled_query, KEY)
                / (StateTerm("key_sum", scaled_query, KEY) + Constant(EPS)),
                dtype,
            ),
        ),
        head_mapping=HeadMapping(
            {
                QUERY_HEADS: STATE_HEADS,
                KEY_HEADS: STATE_HEADS,
                VALUE_HEADS: STATE_HEADS,
                STATE_HEADS: STATE_HEADS,
            }
        ),
    )
    return StatefulOperator(algorithm)
