# 02 — Derive scanability and Backward IR

**What to build:** Implement compiler-owned typed algebra analysis and node VJPs. Validated Algorithm IR becomes a Stateful Program classified as scanable or recurrent; Backward IR is derived rather than supplied by a frontend or inherited from a selected kernel.

**Blocked by:** 01 — Define Target, analysis, and strict results.

**Status:** ready-for-agent

- [ ] Every expression, propagation, injection, readout, state operation, and Head Mapping operation used by the four profiles has a VJP.
- [ ] Typed algebra derives compact associative summaries and canonical reductions without model names or exact-sequence whitelists.
- [ ] Mathematically valid non-closed programs classify as recurrent and never enter backend lowering.
- [ ] Recurrent diagnostics identify the minimal non-closed structural path, operand domains, and supported reformulations.
- [ ] Backward IR covers any subset of named output cotangents, optional final-state cotangent, and provided initial-state gradients.
- [ ] Head Mapping owns grouped-head inverse reductions.
- [ ] Aliased Tensor Input bindings accumulate semantic gradient contributions correctly.
- [ ] State and cotangent operations obey StateSpec accumulation dtype and declared storage casts.
- [ ] ProgramAnalysis exposes node-level backward summaries without exposing constructible Backward IR.
- [ ] CPU FP64/FP32 tests compare node and composition VJPs with independent PyTorch autograd recurrences.
