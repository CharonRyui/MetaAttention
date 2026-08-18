# 06 — Cut over GDN and public callers

**What to build:** Complete the strict public migration. Remove GDNEngine and every adapter artifact, migrate all repository StatefulOperator callers, and leave LinearAttentionEngine source and external behavior unchanged outside the new compiler guarantee.

**Blocked by:** 05 — Migrate four algorithm profiles.

**Status:** ready-for-agent

- [ ] `GDNEngine` implementation, export, documentation, adapter tests, and repository callers are removed.
- [ ] GDN remains demonstrated only through explicit Algorithm IR in `examples/gated_delta_rule.py`.
- [ ] Every StatefulOperator caller supplies explicit H20 Target and typed constants.
- [ ] Every caller consumes ExecutionResult and NamedOutputTuple; no bare-tensor or dictionary-state shim remains.
- [ ] StateTuple is the sole initial-state container.
- [ ] Non-H20 StatefulOperator use fails explicitly through Target capability/mismatch diagnostics.
- [ ] LinearAttentionEngine implementation and bare-tensor interface remain unchanged.
- [ ] Ordinary AttentionEngine and non-stateful examples remain unchanged.
- [ ] API documentation publishes the H20 capability matrix and temporary architecture restriction.
