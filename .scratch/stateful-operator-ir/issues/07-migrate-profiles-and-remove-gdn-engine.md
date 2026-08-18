# 07 — Migrate profiles and remove GDNEngine

**What to build:** Express the four real acceptance profiles and one model-independent pressure profile exclusively through generic IR, remove the dedicated GDN adapter, and leave LinearAttentionEngine unchanged.

**Blocked by:** 06 — Add generic algebraic optimizations.

**Status:** ready-for-agent

- [ ] GLA and RetNet recurrent use Axis Scale, Product Injection, and State Contraction without model-specific nodes.
- [ ] Selective State evolution uses role-directed Axis Scale and generic contraction semantics.
- [ ] GDN uses Axis Scale, general Rank-One propagation, Product Injection, and State Contraction with no dedicated backend.
- [ ] No profile/model name or input name participates in lowering, backward, or cache identity.
- [ ] The compositional pressure profile combines AxisScale → RankOne(left != right) → AxisScale, two injections, two readouts, `(64,128)` State, grouped heads, packed empty sequences, and inactive tails.
- [ ] Pressure-profile support adds no whole-program pattern.
- [ ] Every profile consumes immutable ExecutionResult and StateTuple.
- [ ] GDNEngine implementation, export, tests, docs, and repository callers are removed.
- [ ] LinearAttentionEngine source and behavior remain unchanged outside the new compiler guarantee.
- [ ] Independent references validate all outputs, State, continuation, and gradients on H20.
