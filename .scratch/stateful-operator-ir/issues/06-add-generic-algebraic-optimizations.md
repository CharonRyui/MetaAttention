# 06 — Add generic algebraic optimizations

**What to build:** Optimize common typed subgraphs without changing semantic identity or introducing complete-profile dispatch.

**Blocked by:** 05 — Add packed variable-length execution.

**Status:** ready-for-agent

- [ ] Adjacent compatible Axis Scales fuse only under proven equivalence.
- [ ] Product Injection role structure selects GEMM/outer-product schedules without model or input-name checks.
- [ ] Rank-One sequences may use equivalent WY/KKT-style factorization while retaining dense affine semantics.
- [ ] Multiple State Contractions sharing State/input data may fuse.
- [ ] Schedule selection uses typed structure, Feature sizes, dtype, sequence representation, and Target resources only.
- [ ] Every optimization has differential dense/packed forward and backward tests against generic fallback.
- [ ] Failure or inapplicability of an optimization always falls back to generic dense affine lowering.
- [ ] Structural checks reject model names, frontend identities, exact known profile identities, and dedicated FlashQLA execution.
- [ ] Numeric differences remain within predeclared profile tolerances.
- [ ] Optimization metadata remains separate from Algorithm IR semantics.
