# 05 — Run Mamba2 through State Transition IR

**What to build:** Express and execute Mamba2 through the same Algorithm IR and lowering strategy as other stateful operators, using structured diagonal or elementwise propagation, elementwise input injection, a typed post-transition readout, Tensor Inputs, and explicit grouped Head Mapping. The completed operator must not depend on a model-name special case.

**Blocked by:** 02 — Run Gated Linear Attention through Algorithm IR; 03 — Support initial state, final state, and continuation.

**Status:** ready-for-agent

- [ ] The Mamba2 frontend defines selective state evolution and readout directly instead of presenting them as value and decay modifiers.
- [ ] Tensor Inputs declare selective parameters, logical shape roles, dtypes, and gradient requirements without using stateful-path `CustomIO`.
- [ ] Supported grouped Head Mapping validates expansion and backward reduction relationships before lowering.
- [ ] Forward output matches the independent PyTorch Mamba2 reference within established dtype-specific tolerances.
- [ ] Gradients for primary inputs and selective Tensor Inputs, including propagation and injection parameters, match the independent reference.
- [ ] Initial/final-state continuation works for Mamba2 and matches a full-sequence invocation.
- [ ] No `Mamba2SpecialCase`, equivalent model-name dispatch, arbitrary Python transition, or eager fallback exists.
