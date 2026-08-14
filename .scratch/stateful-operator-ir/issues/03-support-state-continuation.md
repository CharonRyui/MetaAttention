# 03 — Support initial state, final state, and continuation

**What to build:** Let an IR-first stateful operator accept an optional named initial state tuple, default to a zero state when omitted, and return a named final state tuple when requested. Callers can split a sequence across invocations without the engine retaining mutable state internally.

**Blocked by:** 02 — Run Gated Linear Attention through Algorithm IR.

**Status:** ready-for-agent

- [ ] Omitting initial state produces the same result as supplying an explicit zero state.
- [ ] A caller can request final state without changing the default output-only return contract.
- [ ] Running a prefix, continuing its suffix from the returned final state, and concatenating outputs matches a single full-sequence invocation within dtype-specific tolerances.
- [ ] Final state names, shapes, storage dtypes, and accumulation semantics match the declared state tuple.
- [ ] A deterministic behavioral test proves that state readout occurs after the current token transition.
- [ ] Reusing an engine for independent calls does not leak or mutate state between calls.
- [ ] Invalid initial state names, shapes, or dtypes fail before kernel compilation.
