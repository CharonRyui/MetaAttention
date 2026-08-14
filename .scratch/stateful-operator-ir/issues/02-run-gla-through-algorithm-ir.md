# 02 — Run Gated Linear Attention through Algorithm IR

**What to build:** Make an IR-first Gated Linear Attention operator execute end to end through the existing optimized chunked backend. Deliver the smallest complete typed Algorithm IR needed for this behavior: a named state tuple, typed tensor inputs, equal-head Head Mapping, elementwise-scale propagation, outer-product state injection, post-transition matrix readout, and compiler-derived backward behavior.

**Blocked by:** 01 — Establish the Stateful Operator compile seam.

**Status:** ready-for-agent

- [ ] A model author can construct Gated Linear Attention with typed explicit Algorithm IR constructors rather than modifier callbacks.
- [ ] The state transition is represented as structured affine propagation plus outer-product injection, and readout observes the updated state.
- [ ] Forward output matches the independent PyTorch Gated Linear Attention reference within established dtype-specific tolerances.
- [ ] Gradients for query, key, value, and gate match the independent reference within established tolerances.
- [ ] Supported IR nodes derive their backward semantics from node VJPs; the frontend does not provide an independent backward.
- [ ] The operator uses the existing optimized chunked lowering and does not select an eager recurrence.
- [ ] Unsupported transition structures fail before backend code generation.
