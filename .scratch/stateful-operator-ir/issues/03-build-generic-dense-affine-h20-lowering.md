# 03 — Build generic dense-affine H20 lowering

**What to build:** Assemble one parallel chunked TileLang forward compiler from node semantics, using dense affine summaries as the semantic fallback for every supported Scanable Program.

**Blocked by:** 02 — Derive affine summaries and Backward IR.

**Status:** ready-for-agent

- [ ] Generated kernels construct token summaries, compose chunk summaries associatively, propagate State, and evaluate post-transition State Contractions.
- [ ] No model/profile dispatch, eager recurrence, per-token framework launch, or token-serial TileLang fallback exists.
- [ ] Optimization misses use generic dense affine lowering rather than rejection.
- [ ] H20 capability supports Feature pairs `(64,64)`, `(64,128)`, `(128,64)`, and `(128,128)`.
- [ ] BF16/FP32 inputs and outputs use normative promotion; State algebra remains FP32.
- [ ] Canonical role-order contiguous layout is required and no implicit copy/transpose occurs.
- [ ] Positive unaligned dense lengths use compiler-defined inactive identity lanes.
- [ ] Concrete workspace/register/memory limits produce unsupported-specialization errors without changing execution class.
- [ ] Structural inspection proves parallel chunk summary construction and composition.
- [ ] Focused H20 tests compare generic forward output and final State with independent references.
