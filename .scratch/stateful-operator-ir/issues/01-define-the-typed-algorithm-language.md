# 01 — Define the typed Algorithm language

**What to build:** Replace the prototype node catalog with the exact first compiler schema: typed logical roles, one two-Feature matrix State, canonical Input Expressions and Constants, ordered Axis Scale and Rank-One propagation, Product Injections, State Contractions, generic Head Mapping, StateTuple, NamedOutputTuple, ExecutionResult, optional Target, and structured errors.

**Blocked by:** None.

**Status:** ready-for-agent

- [ ] Roles are typed Batch, Sequence, named Head, and named Feature values rather than free strings.
- [ ] Algorithm IR accepts exactly one State with two distinct ordered Feature roles.
- [ ] Input Expressions support only Input, BF16/FP32 Typed Constant, Add, Multiply, Negate, and Exp.
- [ ] Constants canonicalize to exact dtype bits, preserve signed zero, and reject NaN/Inf.
- [ ] Add/Multiply canonicalize operand order without flattening association.
- [ ] Transition has ordered propagation and Product Injection tuples; empty tuples mean identity and additive zero.
- [ ] Axis Scale, Rank-One propagation, Product Injection, State Contraction, and Head Mapping enforce the specification's role rules.
- [ ] Multiple named State Contractions produce immutable NamedOutputTuple inside invariant ExecutionResult.
- [ ] Target is optional; no public ProgramAnalysis interface remains.
- [ ] Structured StatefulCompilationError exposes stable code, public-IR path, and JSON-safe details.
- [ ] CPU tests cover all constructors, canonical identity, typing failures, deterministic validation order, and immutable containers.
