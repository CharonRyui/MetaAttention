# 06 — Enforce Algorithm IR validation and structural cache identity

**What to build:** Make compilation of every first-phase stateful operator safe and deterministic: validate the complete Algorithm IR before backend generation and derive generated-kernel cache identity from canonical operator semantics and relevant compile metadata.

**Blocked by:** 04 — Migrate RetNet recurrent to Algorithm IR; 05 — Run Mamba2 through State Transition IR.

**Status:** ready-for-agent

- [ ] Compilation validates state names and references, logical shape roles, storage and accumulation dtypes, supported propagation/injection/readout nodes, Head Mapping, differentiability, initial-state compatibility, and lowering support.
- [ ] Invalid head divisibility, duplicate or unknown states, incompatible shapes/dtypes, unsupported state tuples, and unsupported nodes fail before TileLang code generation with actionable diagnostics.
- [ ] Normalization behavior is rejected explicitly in this phase rather than ignored or partially lowered.
- [ ] Structurally equivalent Algorithm IR definitions produce the same cache identity regardless of wrapper or Python object identity.
- [ ] Semantically different transitions, tensor metadata, backend identities, and relevant compile options cannot collide.
- [ ] CPU-safe behavioral tests cover validation, canonicalization, Head Mapping, cache identity, and VJP construction without asserting internal dataclass or hash implementation details.
- [ ] Gated Linear Attention, RetNet recurrent, and Mamba2 continue passing their functional reference checks.
