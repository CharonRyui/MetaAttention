"""Flash Linear Attention operators expressed with the StatefulOperator DSL.

Every factory returns a :class:`StatefulOperator` whose Algorithm IR is authored
from Flash Linear Attention's own reference semantics, so the compiled operator
can be compared against the library on the same tensors. Runtime layout is
head-first ``[B, H, T, ...]`` as required by ``StatefulOperator``.

:data:`FLA_COVERAGE` is the machine-readable map from every operator in
``fla.ops`` to the DSL construct that expresses it, or to the engine that owns it
when the operator is not a per-token linear recurrence.
"""

from __future__ import annotations

from typing import Callable

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Readout,
    Sequence,
    Shift,
    StateContraction,
    StateSpec,
    StateTerm,
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

EPS = 1e-10


def _head_mapping() -> HeadMapping:
    return HeadMapping(
        {
            QUERY_HEADS: STATE_HEADS,
            KEY_HEADS: STATE_HEADS,
            VALUE_HEADS: STATE_HEADS,
            STATE_HEADS: STATE_HEADS,
        }
    )


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
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


def hgrn(
    batch: int,
    heads: int,
    seqlen: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_hgrn``: a per-Feature vector State whose readout is itself.

    ``fla.ops.hgrn.naive`` updates ``h = exp(g) * h + x`` and writes ``o = h``.
    HGRN has no head axis, so callers pass ``heads=1`` and supply the headless
    ``[B, T, D]`` tensors as ``[B, 1, T, D]``; ``gate`` is the per-Feature log
    decay.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, VALUE), torch.float32),
        ),
        states=(StateSpec("hidden", (Batch, STATE_HEADS, VALUE)),),
        transition=StateTransition(
            "hidden",
            propagations=(AxisScale(Exp(Input("gate")), (VALUE,)),),
            injections=(ProductInjection((ProductFactor(Input("value"), (VALUE,)),)),),
        ),
        readouts=(Readout("output", StateTerm("hidden", Constant(1.0), None), dtype),),
        head_mapping=HeadMapping({VALUE_HEADS: STATE_HEADS, STATE_HEADS: STATE_HEADS}),
    )
    return StatefulOperator(algorithm)


def rwkv6_recurrence(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """RWKV-6 style recurrence: token shift, per-Feature decay, removal term.

    ``fla.ops.rwkv6.chunk_rwkv6`` advances the State through
    ``S_t = S_{t-1} exp(-w_t) (I - k_t k_t^T) + k_t v_t^T`` with the shifted key
    ``k_t = shift(k)_t * mix_t + k_t * (1 - mix_t)``. The token shift is authored
    with :class:`Shift`, which zero-fills the tokens before a sequence start.
    """
    shifted_key = Shift(Input("key"), 1) * Input("mix") + Input("key") * (
        1 - Input("mix")
    )
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("mix", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(-Input("decay")), (KEY,)),
                RankOnePropagation(KEY, shifted_key, shifted_key, -1.0),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(shifted_key, (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), KEY, dtype),),
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


def retention_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_retention`` with the standard per-head decay schedule.

    ``fla.ops.retention`` builds ``g = log(1 - 2 ** (-5 - h))`` per head and
    delegates to simple GLA, so the decay is a head-shaped constant. The caller
    supplies it as the ``decay`` input of shape ``[B, H, T]``.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("decay"))),),
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
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


def retention_decay(heads: int) -> torch.Tensor:
    """The per-head log decay FLA's retention op constructs."""
    return torch.log1p(
        -torch.tensor(2.0).pow(-5.0 - torch.arange(heads).float())
    ).reshape(1, heads, 1)


def rwkv7_state(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """RWKV-7 / DPLR state algebra from ``chunk_dplr_delta_rule``.

    The transition is ``S <- diag(exp(w)) S - beta k (a^T S) + beta k v^T``.
    RWKV-7's channel mixing and token shift belong to its layer rather than to
    this recurrence, so both are supplied as inputs.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("removal", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("beta", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("decay")), (KEY,)),
                RankOnePropagation(KEY, Input("key"), Input("removal"), -Input("beta")),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value") * Input("beta"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), KEY, dtype),),
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


FLA_COVERAGE: tuple[tuple[str, str, str], ...] = (
    (
        "fla.ops.linear_attn.chunk_linear_attn(normalize=False)",
        "examples/linear_attention.py:linear_attention",
        "one matrix State",
    ),
    (
        "fla.ops.linear_attn.chunk_linear_attn(normalize=True)",
        "normalized_linear_attention",
        "rank-1 key-sum State + division Readout",
    ),
    (
        "fla.ops.simple_gla.chunk_simple_gla",
        "examples/linear_attention.py:simple_gla_attention",
        "scalar AxisScale",
    ),
    (
        "fla.ops.gla.chunk_gla",
        "examples/linear_attention.py:gla_attention",
        "key-axis AxisScale",
    ),
    (
        "fla.ops.gla.fused_recurrent_gla(gk, gv)",
        "gla_attention with two AxisScales",
        "both Feature roles",
    ),
    (
        "fla.ops.lightning_attn.chunk_lightning_attn",
        "examples/linear_attention.py:lightning_attention",
        "per-head constant decay",
    ),
    (
        "fla.ops.retention.chunk_retention",
        "retention_attention",
        "per-head log decay Input",
    ),
    (
        "fla.ops.delta_rule.chunk_delta_rule",
        "examples/linear_attention.py:delta_rule_attention",
        "RankOne + ProductInjection",
    ),
    (
        "fla.ops.gated_delta_rule.chunk_gated_delta_rule",
        "examples/gated_delta_rule.py:gated_delta_rule",
        "gate + RankOne + beta Injection",
    ),
    (
        "fla.ops.generalized_delta_rule.chunk_dplr_delta_rule",
        "examples/gated_delta_rule.py:gated_delta_rule",
        "AxisScale + RankOne (left != right)",
    ),
    (
        "fla.ops.generalized_delta_rule.chunk_iplr_delta_rule",
        "examples/gated_delta_rule.py:gated_delta_rule",
        "RankOne with coefficient",
    ),
    ("fla.ops.rwkv7.chunk_rwkv7", "rwkv7_state", "DPLR State algebra"),
    ("fla.ops.hgrn.chunk_hgrn", "hgrn", "rank-1 State + identity Readout"),
    (
        "fla.ops.rwkv6.chunk_rwkv6",
        "rwkv6_recurrence",
        "token Shift + decay + removal term",
    ),
    (
        "fla.ops.based.fused_chunk_based",
        "UNSUPPORTED",
        "needs higher-order States (K*K Feature) and 3-factor Injections",
    ),
    (
        "fla.ops.rebased.parallel_rebased",
        "UNSUPPORTED",
        "needs higher-order States (K*K Feature) and 3-factor Injections",
    ),
    ("fla.ops.abc.chunk_abc", "UNSUPPORTED", "two States with a Feature-map product"),
    ("fla.ops.gsa.chunk_gsa", "UNSUPPORTED", "slot State plus a Feature-map product"),
    (
        "fla.ops.ttt.chunk_ttt_linear",
        "UNSUPPORTED",
        "chunk-local inner loop reinitialises State",
    ),
    (
        "fla.ops.titans.chunk_titans_linear",
        "UNSUPPORTED",
        "inner loop plus momentum and bias States",
    ),
    (
        "fla.ops.attn.parallel_attn",
        "AttentionEngine",
        "softmax attention, not a linear recurrence",
    ),
    (
        "fla.ops.forgetting_attn.parallel_forgetting_attn",
        "AttentionEngine",
        "softmax attention with decay",
    ),
    ("fla.ops.nsa.parallel_nsa", "AttentionEngine", "sparse softmax attention"),
    (
        "fla.ops.rwkv4.fused_recurrent_rwkv4",
        "AttentionEngine",
        "growing key/value history, not a fixed State",
    ),
)


FACTORIES: dict[str, Callable[..., StatefulOperator]] = {
    "normalized_linear_attention": normalized_linear_attention,
    "hgrn": hgrn,
    "rwkv6_recurrence": rwkv6_recurrence,
    "retention_attention": retention_attention,
    "rwkv7_state": rwkv7_state,
}
