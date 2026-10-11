"""A fixed number of ordered Delta Rule writes per token, with scalar decay.

FLA interleaves propagation and injection at each inner step. Here the inner
steps are unrolled into one affine Transition: each earlier write is transformed
by all subsequent rank-one propagations before joining the final injection.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Sequence,
    Readout,
    StateTerm,
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


def gated_delta_product(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    num_householder: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Decay S, then apply S <- S + beta_j k_j (v_j - k_j^T S)^T in order.

    Readout is q^T S / sqrt(K), once after all inner steps. Inputs use head-first
    layout: query [B, Hq, T, K], scalar log gate [B, Hs, T], and key_j, value_j,
    beta_j for j in range(num_householder), each with T tokens. Split FLA's
    interleaved T*num_householder key/value/beta streams into these named inputs.

    The step count is fixed at factory construction. One step recovers GDN.
    """
    if (
        isinstance(num_householder, bool)
        or not isinstance(num_householder, int)
        or num_householder < 1
    ):
        raise ValueError("num_householder must be a positive integer")
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
    ]
    propagations = [AxisScale(Exp(Input("gate")))]
    injections = []
    for step in range(num_householder):
        inputs.extend(
            (
                TensorInput(f"key_{step}", (Batch, KEY_HEADS, Sequence, KEY), dtype),
                TensorInput(
                    f"value_{step}", (Batch, VALUE_HEADS, Sequence, VALUE), dtype
                ),
                TensorInput(
                    f"beta_{step}", (Batch, STATE_HEADS, Sequence), torch.float32
                ),
            )
        )
        key = Input(f"key_{step}")
        beta = Input(f"beta_{step}")
        propagations.append(RankOnePropagation(KEY, key, key, -beta))
        # If c_j = beta_j k_j v_j^T, its final contribution is
        # P_last ... P_(j+1) c_j. Expand the transformed KEY factor as
        # a sum of vectors with scalar coefficients. Each resulting outer
        # product is a separate injection, preserving every intermediate write.
        terms = [(key, ())]
        for later in range(step + 1, num_householder):
            later_key = Input(f"key_{later}")
            corrections = []
            for vector, coefficients in terms:
                corrections.append(
                    (
                        later_key,
                        (
                            *coefficients,
                            -Input(f"beta_{later}"),
                            Contract(later_key, vector, KEY),
                        ),
                    )
                )
            terms.extend(corrections)
        for vector, coefficients in terms:
            injections.append(
                ProductInjection(
                    (
                        ProductFactor(vector, (KEY,)),
                        ProductFactor(beta * Input(f"value_{step}"), (VALUE,)),
                        *(
                            ProductFactor(coefficient, ())
                            for coefficient in coefficients
                        ),
                    )
                )
            )
    return StatefulOperator(
        AlgorithmIR(
            inputs=inputs,
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory", propagations=propagations, injections=injections
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("memory", Input("query") * (dim**-0.5), KEY),
                    dtype,
                ),
            ),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )
