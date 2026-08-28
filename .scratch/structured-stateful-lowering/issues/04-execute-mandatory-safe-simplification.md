# 04: Execute mandatory safe simplification

**What to build:** Route every validated Algorithm IR program through deterministic canonicalization and mandatory safe simplification before generic execution, without changing its observable floating-point, State, or gradient semantics.

**Blocked by:** 03/Complete the metadata-only compile cutover.

**Status:** ready-for-agent

- [ ] Explicit identity and empty Propagation canonicalize to the same structure, and nested Propagation is flattened without reordering nodes.
- [ ] Registered zero, one, and other mandatory rules apply only when valid under declared dtype and VJP semantics.
- [ ] Add and Multiply use canonical operand order while authored association remains observable and unchanged.
- [ ] Noncommutative Propagation order and ordered State Feature roles remain distinct in canonical and simplified identities.
- [ ] The Simplified IR identity is deterministic across equivalent constructions and participates in compilation identity.
- [ ] Public compile-and-invoke tests prove outputs, final State, and gradients match independent references before and after simplification.
