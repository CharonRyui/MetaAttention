# 04 — Migrate RetNet recurrent to Algorithm IR

**What to build:** Make the RetNet recurrent frontend define its model through structured affine state propagation, outer-product injection, and post-transition readout, then execute through the shared stateful-operator compilation seam and existing optimized lowering.

**Blocked by:** 02 — Run Gated Linear Attention through Algorithm IR.

**Status:** ready-for-agent

- [ ] The RetNet recurrent frontend uses typed Algorithm IR construction rather than modifier slots as its semantic definition.
- [ ] RetNet decay and query scaling are represented by supported structured propagation and input-transform semantics, not arbitrary transition functions.
- [ ] Forward output matches the independent PyTorch RetNet reference within established dtype-specific tolerances.
- [ ] Gradients for every differentiable primary input match the independent reference within established tolerances.
- [ ] The IR-first operator matches the pre-migration generated operator for forward and gradients on identical seeded inputs.
- [ ] RetNet uses the shared lowering strategy without model-name dispatch or a RetNet-specific compiler path.
- [ ] Existing tuning controls remain available and outside Algorithm IR semantics.
