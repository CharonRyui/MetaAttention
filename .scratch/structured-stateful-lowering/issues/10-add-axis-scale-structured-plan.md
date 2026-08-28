# 10: Add the Axis Scale structured plan

**What to build:** Recognize reusable Axis Scale and elementwise affine structure in Simplified IR, enumerate a specialized Mathematical Plan beside the generic baseline, and compile mathematically equivalent forward and backward execution.

**Blocked by:** 06/Migrate existing structured forward paths; 09/Enforce named forward-backward context contracts.

**Status:** ready-for-agent

- [ ] Matching depends only on typed Algorithm IR structure and State Feature roles, never model, profile, frontend, or input names.
- [ ] The planner rule declares its pattern, preconditions, replacement identity, VJP equivalence, and conflicts.
- [ ] Scalar, first-Feature, second-Feature, and both-Feature Axis Scale semantics preserve role-directed action.
- [ ] Authored noncommutative Propagation order is preserved when Axis Scale appears with other Propagations.
- [ ] Specialized forward outputs, final State, and generated gradients match the generic dense baseline and independent references.
- [ ] Optimization disabled selects only generic execution; a failed Axis Scale candidate leaves the Scanable Program valid.
