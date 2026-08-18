# 04 — Lower Backward IR and State continuation

**What to build:** Lower canonical Backward IR through the same parallel compiler and complete initial/final State, continuation, and active-gradient specialization without a separate backward implementation.

**Blocked by:** 03 — Build generic dense-affine H20 lowering.

**Status:** ready-for-agent

- [ ] Generated backward consumes any subset of named output and final-State cotangents.
- [ ] Every active differentiable binding and provided initial State receives the correct gradient dtype and value.
- [ ] Identity/contiguous-group Head Mapping reverse reductions are generated from Backward IR.
- [ ] Identical Tensor-object aliases accumulate contributions; distinct overlapping views are rejected.
- [ ] Training/inference specialization follows grad mode and active differentiable bindings.
- [ ] Omitted State creates FP32 zeros; provided State is read-only; unrequested final State is not materialized.
- [ ] Output-only, State-only, joint-loss, missing-cotangent, and initial-State-gradient cases match references.
- [ ] Prefix/suffix continuation matches full execution within profile tolerance.
- [ ] No hand-written profile backward, eager State path, or legacy kernel backward remains.
- [ ] CPU VJP tests and focused H20 backward tests cover both Feature roles and unequal Feature dimensions.
