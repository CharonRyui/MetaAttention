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
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    Outer,
    ProductFactor,
    ProductInjection,
    ProductFeatureRole,
    RankOnePropagation,
    Readout,
    Sequence,
    Softmax,
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


def rwkv6_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_rwkv6``: decay, bonus skip, and a pre-transition readout.

    ``fla.ops.rwkv6.recurrent_naive`` advances ``h = h * exp(w) + k (x) v`` and
    reads the *previous* State with a current-token bonus:

    ``o_i = ((h + u (x) k_i v_i) . (scale * q_i))``

    so the readout needs a prior-State term and a Feature-axis contraction of
    ``scale * q * u * k``. ``w`` is the log decay, as in the library.
    """
    scaled_query = Input("query") * (dim**-0.5)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("bonus", (Batch, STATE_HEADS, Sequence, KEY), dtype),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("decay")), (KEY,)),),
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
            Readout(
                "output",
                StateTerm("memory", scaled_query, KEY, prior=True)
                + Contract(
                    scaled_query * Input("bonus") * Input("key"),
                    Constant(1.0),
                    KEY,
                )
                * Input("value"),
                dtype,
            ),
        ),
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


def based_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``fused_chunk_based``: second-order Taylor feature map with normalizer.

    The recurrence keeps six States: the zero, first, and second order numerator
    terms ``h0 [V]``, ``h1 [K, V]``, ``h2 [K*K, V]`` and their denominator
    counterparts ``z0 [1]``, ``z1 [K]``, ``z2 [K*K]``. The readout evaluates
    ``1 + (q.k) + 0.5 (q.k)^2`` through those States and divides by the matching
    denominator with the library's ``1e-6`` epsilon.
    """
    scale = dim**-0.5
    scaled_query = Input("query") * scale
    second_order = ProductFeatureRole(KEY, KEY)
    unit = FeatureRole("unit_dim")
    query_pair = Outer(scaled_query, scaled_query, second_order)
    key_pair = Outer(Input("key"), Input("key"), second_order)
    zero_order_injection = ProductInjection((ProductFactor(Constant(1.0), (unit,)),))
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("numerator_0", (Batch, STATE_HEADS, VALUE)),
            StateSpec("numerator_1", (Batch, STATE_HEADS, KEY, VALUE)),
            StateSpec("numerator_2", (Batch, STATE_HEADS, second_order, VALUE)),
            StateSpec("denominator_0", (Batch, STATE_HEADS, unit)),
            StateSpec("denominator_1", (Batch, STATE_HEADS, KEY)),
            StateSpec("denominator_2", (Batch, STATE_HEADS, second_order)),
        ),
        transition=(
            StateTransition(
                "numerator_0",
                injections=(
                    ProductInjection((ProductFactor(Input("value"), (VALUE,)),)),
                ),
            ),
            StateTransition(
                "numerator_1",
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
                "numerator_2",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(key_pair, (second_order,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition("denominator_0", injections=(zero_order_injection,)),
            StateTransition(
                "denominator_1",
                injections=(ProductInjection((ProductFactor(Input("key"), (KEY,)),)),),
            ),
            StateTransition(
                "denominator_2",
                injections=(
                    ProductInjection((ProductFactor(key_pair, (second_order,)),)),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                (
                    StateTerm("numerator_0", Constant(1.0), None)
                    + StateTerm("numerator_1", scaled_query, KEY)
                    + StateTerm("numerator_2", query_pair, second_order) * 0.5
                )
                / (
                    StateTerm("denominator_0", Constant(1.0), None)
                    + StateTerm("denominator_1", scaled_query, KEY)
                    + StateTerm("denominator_2", query_pair, second_order) * 0.5
                    + Constant(1e-6)
                ),
                dtype,
            ),
        ),
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


def rebased_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
) -> StatefulOperator:
    """FLA ``parallel_rebased``: squared attention with a normalizer.

    ``attn = (scale*q . k) ** 2`` keeps only the second-order feature map, so the
    Program declares a ``[K*K, V]`` numerator State and a ``[K*K]`` denominator
    State and divides by the library's configurable epsilon.
    """
    scale = dim**-0.5
    scaled_query = Input("query") * scale
    second_order = ProductFeatureRole(KEY, KEY)
    query_pair = Outer(scaled_query, scaled_query, second_order)
    key_pair = Outer(Input("key"), Input("key"), second_order)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("numerator", (Batch, STATE_HEADS, second_order, VALUE)),
            StateSpec("denominator", (Batch, STATE_HEADS, second_order)),
        ),
        transition=(
            StateTransition(
                "numerator",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(key_pair, (second_order,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "denominator",
                injections=(
                    ProductInjection((ProductFactor(key_pair, (second_order,)),)),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm("numerator", query_pair, second_order)
                / (StateTerm("denominator", query_pair, second_order) + Constant(eps)),
                dtype,
            ),
        ),
        head_mapping=_head_mapping(),
    )
    return StatefulOperator(algorithm)


def gated_slot_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_slot: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_gsa``: two gated States joined by a slot softmax.

    ``hk [K, M]`` accumulates ``k (x) s`` and ``hv [M, V]`` accumulates ``s (x) v``,
    each decayed by the supplied per-slot gate. The readout contracts ``hk``,
    softmaxes the slot axis, and contracts ``hv`` with the result, which is how
    the library composes its two States.
    """
    slot = FeatureRole("slot_dim")
    slot_decay = Exp(Input("gate"))
    scaled_query = Input("query") * (dim**-0.5)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("slot", (Batch, KEY_HEADS, Sequence, slot), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, slot), dtype),
        ),
        states=(
            StateSpec("key_slot", (Batch, STATE_HEADS, KEY, slot)),
            StateSpec("slot_value", (Batch, STATE_HEADS, slot, VALUE)),
        ),
        transition=(
            StateTransition(
                "key_slot",
                propagations=(AxisScale(slot_decay, (slot,)),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("slot"), (slot,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "slot_value",
                propagations=(AxisScale(slot_decay, (slot,)),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("slot"), (slot,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm(
                    "slot_value",
                    Softmax(StateTerm("key_slot", scaled_query, KEY), slot),
                    slot,
                ),
                dtype,
            ),
        ),
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
        "rwkv6_attention",
        "prior-State Readout + Contract Feature reduction",
    ),
    (
        "fla.ops.based.fused_chunk_based",
        "based_attention",
        "three States incl. a K*K product Feature + normalized Readout",
    ),
    (
        "fla.ops.rebased.parallel_rebased",
        "rebased_attention",
        "product Feature State + normalized Readout",
    ),
    (
        "fla.ops.abc.chunk_abc",
        "gated_slot_attention",
        "two States + slot Softmax (ABC's beta Injection is out of scope)",
    ),
    (
        "fla.ops.gsa.chunk_gsa",
        "gated_slot_attention",
        "slot State + Softmax Node joining two States",
    ),
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
    "rwkv6_attention": rwkv6_attention,
    "retention_attention": retention_attention,
    "rwkv7_state": rwkv7_state,
    "based_attention": based_attention,
    "rebased_attention": rebased_attention,
    "gated_slot_attention": gated_slot_attention,
}
