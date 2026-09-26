"""Coverage map from ``fla.ops`` operators to the Stateful Operator DSL.

Every Program lives in its own ``examples/<family>.py`` module and is imported from
there; this file records which construct expresses each library operator, and which
engine owns the operators that are not per-token linear recurrences.

Each row is ``(fla_operator, program, note)``, where ``program`` is
``examples/<file>.py:<factory>`` for an authored Program, an engine name for an
operator that belongs to another engine, or ``UNSUPPORTED`` with the reason in
``note``.

``tests/unit/test_fla_coverage.py`` resolves every Program pointer and checks that
every installed ``fla.ops`` module appears here.
"""

FLA_COVERAGE: tuple[tuple[str, str, str], ...] = (
    (
        "fla.ops.linear_attn.chunk_linear_attn(normalize=False)",
        "examples/linear_attention.py:linear_attention",
        "one matrix State",
    ),
    (
        "fla.ops.linear_attn.chunk_linear_attn(normalize=True)",
        "examples/normalized_linear_attention.py:normalized_linear_attention",
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
        "examples/linear_attention.py:gla_attention",
        "two AxisScales (gk and gv)",
    ),
    (
        "fla.ops.lightning_attn.chunk_lightning_attn",
        "examples/linear_attention.py:lightning_attention",
        "per-head constant decay",
    ),
    (
        "fla.ops.retention.chunk_retention",
        "examples/retention_attention.py:retention_attention",
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
    (
        "fla.ops.rwkv7.chunk_rwkv7",
        "examples/rwkv7_state.py:rwkv7_state",
        "DPLR State algebra",
    ),
    (
        "fla.ops.hgrn.chunk_hgrn",
        "examples/hgrn.py:hgrn",
        "rank-1 State + identity Readout",
    ),
    (
        "fla.ops.rwkv6.chunk_rwkv6",
        "examples/rwkv6_attention.py:rwkv6_attention",
        "prior-State Readout + Contract Feature reduction",
    ),
    (
        "fla.ops.based.fused_chunk_based",
        "examples/based_attention.py:based_attention",
        "three States incl. a K*K product Feature + normalized Readout",
    ),
    (
        "fla.ops.rebased.parallel_rebased",
        "examples/rebased_attention.py:rebased_attention",
        "product Feature State + normalized Readout",
    ),
    (
        "fla.ops.abc.chunk_abc",
        "examples/gated_slot_attention.py:gated_slot_attention",
        "two States + slot Softmax (ABC's beta Injection is out of scope)",
    ),
    (
        "fla.ops.gsa.chunk_gsa",
        "examples/gated_slot_attention.py:gated_slot_attention",
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
    (
        "fla.ops.nsa.parallel_nsa",
        "AttentionEngine",
        "sparse softmax attention",
    ),
    (
        "fla.ops.rwkv4.fused_recurrent_rwkv4",
        "AttentionEngine",
        "growing key/value history, not a fixed State",
    ),
)
