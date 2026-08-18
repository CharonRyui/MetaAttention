# 02 — Derive affine summaries and Backward IR

**What to build:** Implement internal compiler analysis for role/dtype inference, canonical affine State summaries, target-independent Scanable/Recurrent classification, compiler-owned VJPs, and canonical Backward IR.

**Blocked by:** 01 — Define the typed Algorithm language.

**Status:** ready-for-agent

- [ ] Canonical summary semantics use a dense affine State transform and additive State term with storage independent of sequence length.
- [ ] Identity, Axis Scale, Rank-One propagation, ordered propagation, and Product Injection derive closed associative summaries.
- [ ] Rank-One left/right vectors may differ and may act on either Feature role.
- [ ] Scanable and Recurrent classification is independent of Target and concrete resources.
- [ ] Every expression, propagation, injection, contraction, State operation, and Head Mapping operation owns a VJP.
- [ ] Backward IR represents output/final-State cotangents, reverse State propagation, initial-State gradient, Head Mapping reductions, and aliased-input accumulation.
- [ ] Missing cotangents are zero and multiple output/input paths accumulate.
- [ ] State and State cotangents remain FP32 within one invocation.
- [ ] CPU FP32 references compare every node, noncommuting sequence, summary composition, and VJP against PyTorch autograd.
- [ ] Recurrent and malformed programs produce stable earliest-root-cause errors without code generation.
