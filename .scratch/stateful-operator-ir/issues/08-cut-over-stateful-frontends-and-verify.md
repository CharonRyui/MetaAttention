# 08 — Cut over stateful frontends and verify the release

**What to build:** Complete the first-phase rollout by making the official Gated Linear Attention, RetNet recurrent, and Mamba2 frontends IR-first, documenting the state-transition vocabulary and modifier migration path, and verifying the repository across CPU-safe and supported-GPU behavior.

**Blocked by:** 03 — Support initial state, final state, and continuation; 04 — Migrate RetNet recurrent to Algorithm IR; 05 — Run Mamba2 through State Transition IR; 06 — Enforce Algorithm IR validation and structural cache identity; 07 — Convert legacy modifiers to Algorithm IR.

**Status:** ready-for-agent

- [ ] All three official stateful frontends construct Algorithm IR by default and remain importable without compiling a kernel.
- [ ] Public stateful documentation uses Algorithm IR, state transition, state injection, state readout, state tuple, Tensor Input, Head Mapping, IR-derived backward, frontend sugar, and lowering strategy consistently.
- [ ] Documentation describes initial/final state usage, supported first-phase algebra, explicit normalization exclusion, deprecation behavior, and migration from modifier construction.
- [ ] The CPU-safe unit suite and official example import coverage pass.
- [ ] Gated Linear Attention, RetNet recurrent, and Mamba2 GPU forward/backward/reference/continuation cases pass on supported hardware.
- [ ] The full functional GPU suite passes on supported hardware, including unchanged ordinary AttentionEngine behavior.
- [ ] Verification confirms generated operators use the optimized lowering and do not silently select an eager fallback.
- [ ] The deprecated modifier constructor remains available only for this migration phase; its later removal is documented but not performed here.
