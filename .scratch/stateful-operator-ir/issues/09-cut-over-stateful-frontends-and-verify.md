# 09 — Cut over stateful frontends and verify the release

**What to build:** Complete the first-phase rollout by making `StatefulOperator` the sole IR-first public interface for Gated Linear Attention, RetNet recurrent, Mamba2, and GDN; retain `LinearAttentionEngine` and `GDNEngine` only as deprecated adapters; document the state-transition vocabulary and migration paths; and verify CPU-safe and supported-GPU behavior.

**Blocked by:** 03 — Support initial state, final state, and continuation; 04 — Migrate RetNet recurrent to Algorithm IR; 05 — Run Mamba2 through State Transition IR; 06 — Enforce Algorithm IR validation and structural cache identity; 07 — Convert legacy modifiers to Algorithm IR; 08 — Run GDN through StatefulOperator.

**Status:** ready-for-agent

- [ ] Official stateful examples construct Algorithm IR explicitly and invoke `StatefulOperator`; imports do not compile kernels.
- [ ] Public stateful documentation uses Algorithm IR, Stateful Operator, state transition, propagation composition, rank-one delta propagation, state injection, state readout, state tuple, Tensor Input, Head Mapping, Backward IR, frontend sugar, and lowering strategy consistently.
- [ ] Documentation describes keyword-only input binding, initial/final state usage, concrete specialization metadata, supported first-phase algebra, explicit normalization exclusion, both deprecated adapters, and migration from `LinearAttentionEngine` and `GDNEngine`.
- [ ] The CPU-safe unit suite and official example import coverage pass.
- [ ] Gated Linear Attention, RetNet recurrent, Mamba2, and GDN GPU forward/backward/reference/continuation cases pass on supported hardware.
- [ ] The full functional GPU suite passes on supported hardware, including unchanged ordinary AttentionEngine behavior.
- [ ] Verification confirms generated operators use a lowering selected from IR structure and backend capability, with no eager fallback or model-name dispatch.
- [ ] The deprecated `LinearAttentionEngine` and `GDNEngine` adapters remain available only for this migration phase; their later removal is documented but not performed here.
