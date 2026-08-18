# 03 — Build compositional TileLang lowering

**What to build:** Replace whole-algorithm pattern selection with one H20 stateful kernel generator assembled from node semantics. The generated execution must construct and compose compact state summaries in parallel chunks and support masked tail chunks.

**Blocked by:** 02 — Derive scanability and Backward IR.

**Status:** ready-for-agent

- [ ] Each supported node lowers token-local evaluation, summary construction, associative composition, state propagation, and readout fragments.
- [ ] Previously unseen scanable compositions of supported algebra lower without new whole-algorithm Python patterns.
- [ ] Generated execution is parallel chunked; no Python recurrence, per-token framework launch, or token-serial TileLang fallback exists.
- [ ] Positive unaligned sequence lengths use a masked tail chunk; zero length is rejected before lowering.
- [ ] Concrete shape, dtype, layout, and stride constraints come from H20 Target capability.
- [ ] All concrete sizes specialize initially; contiguous-only constraints, if retained, fail before cache/codegen without hidden copies.
- [ ] IR-generic canonicalization, fusion, layout, tiling, and structural schedule selection remain in the single lowering pipeline.
- [ ] Deterministic structural inspection proves chunk-summary construction and parallel composition.
