# 09 — Prove H20 correctness and performance

**What to build:** Run the complete public-seam correctness, structural, cache, compatibility, and predeclared performance proof on an exclusive H20 before accepting the compiler.

**Blocked by:** 07 — Migrate profiles and remove GDNEngine; 08 — Harden Target, cache, and error contracts.

**Status:** ready-for-agent

- [ ] CPU-safe tests prove constructors, canonicalization, VJPs, classification, errors, cache identity, and failure cleanup.
- [ ] GLA, RetNet recurrent, selective State evolution, GDN, and the pressure profile pass dense and packed H20 forward/backward references.
- [ ] Feature pairs, grouped heads, aliases, empty sequences, inactive tails, all cotangent paths, and continuation pass.
- [ ] Generated execution inspection proves parallel summary composition, generic Backward IR, and absence of profile/token-serial bypass.
- [ ] Frozen old/new cases run interleaved for at least ten independent batches per side on an otherwise idle H20.
- [ ] Paired median latency ratio uses a 95% bootstrap interval; upper bound ≤1.10 passes, lower bound >1.10 fails, overlap requires more batches.
- [ ] Peak allocated memory uses the same 1.10× upper-bound gate.
- [ ] First compilation time is recorded but not gated.
- [ ] Pressure profile meets explicit absolute timeout and memory limits.
- [ ] Full documentation, example imports, CPU suite, focused H20 suite, and existing functional GPU suite pass before merge.
- [ ] ADR 0001 becomes accepted only after generic lowering, derived backward, one real profile, and pressure profile clear correctness and performance review.
