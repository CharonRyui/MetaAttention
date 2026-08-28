# 12: Add the Rank-One factorized plan

**What to build:** Recognize eligible ordered Rank-One Propagation sequences and enumerate a factorized Mathematical Plan, such as a WY/KKT-style representation, beside generic dense affine execution.

**Blocked by:** 06/Migrate existing structured forward paths; 09/Enforce named forward-backward context contracts.

**Status:** ready-for-agent

- [ ] Matching supports distinct left and right vectors, positive and negative coefficients, both State Feature directions, and non-square State.
- [ ] Ordered noncommuting Rank-One Propagations are never reordered or reassociated outside the declared numerical contract.
- [ ] The factorized plan declares complete preconditions, conflicts, forward equivalence, and VJP equivalence.
- [ ] Unsupported or resource-infeasible factorization leaves the generic dense plan available.
- [ ] Factorized forward outputs, final State, input gradients, and initial-State gradient match generic and independent references.
- [ ] No model, GDN, or complete known-profile identity participates in matching or lowering.
