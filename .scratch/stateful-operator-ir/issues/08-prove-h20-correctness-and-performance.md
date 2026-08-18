# 08 — Prove H20 correctness and performance

**What to build:** Run the complete semantic, functional, structural, and performance release proof on an otherwise idle H20. No GPU test or benchmark may share a card with another process.

**Blocked by:** 06 — Cut over GDN and public callers; 07 — Make cache and compilation deterministic.

**Status:** ready-for-agent

- [ ] CPU-safe suite proves validation, analysis JSON, scanability, VJPs, cache identity, deterministic source, and migration contracts.
- [ ] GLA, RetNet recurrent, corrected Mamba2, and GDN pass independent-reference H20 forward and gradient cases.
- [ ] Each profile passes output-only, final-state-only, joint-loss, initial-state-gradient, continuation, grouped-head where applicable, and masked-tail cases.
- [ ] Generated execution inspection proves parallel chunk summaries and no token-serial or specialized bypass.
- [ ] Before cutover comparison, current profile baselines are recorded with repository warmup/repetition methodology, fixed shapes, and observed variance.
- [ ] Generic compiler median latency and variance are compared against baselines.
- [ ] Any statistically significant regression blocks completion pending explicit review; no implicit percentage is accepted.
- [ ] Full documentation and example import tests pass.
- [ ] ADR 0001 becomes accepted after CPU analysis/VJP proof and the first generic parallel H20 profile clears correctness and baseline review.
- [ ] No merge occurs until all four profiles and every specification acceptance criterion pass.
