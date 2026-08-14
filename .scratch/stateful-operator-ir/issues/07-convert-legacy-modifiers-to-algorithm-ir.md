# 07 — Convert legacy modifiers to Algorithm IR

**What to build:** Turn the existing modifier-based `LinearAttentionEngine` constructor into a temporary deprecated frontend adapter. It must validate and convert every modifier pattern used by Gated Linear Attention, RetNet recurrent, and Mamba2 into the same Algorithm IR path, preserving behavior without retaining a second lowering implementation.

**Blocked by:** 04 — Migrate RetNet recurrent to Algorithm IR; 05 — Run Mamba2 through State Transition IR; 06 — Enforce Algorithm IR validation and structural cache identity.

**Status:** ready-for-agent

- [ ] Existing modifier-based construction emits a clear deprecation warning with an IR-first migration direction.
- [ ] Supported `q_mod`, `k_mod`, `v_mod`, and `decay_mod` expression patterns convert to validated input transforms, propagation, injection, and readout semantics.
- [ ] Stateful-path `CustomIO` declarations convert to typed Tensor Inputs while ordinary AttentionEngine `CustomIO` behavior remains unchanged.
- [ ] Deprecated and IR-first entrypoints for Gated Linear Attention, RetNet recurrent, and Mamba2 match for forward and all differentiable gradients within established tolerances.
- [ ] An unsupported modifier expression fails during conversion, identifies the modifier and unsupported expression, and never reaches kernel compilation.
- [ ] There is one StatefulOperator compilation path: no legacy kernel path, eager fallback, frontend-type dispatch, or model-name escape hatch remains.
- [ ] New production code and new tests do not depend on modifier construction except explicit compatibility coverage.
