# 01 — Define Target, analysis, and strict results

**What to build:** Replace implicit StatefulOperator construction and conditional tensor returns with the strict compiler interface from `docs/stateful-operator-ir-spec.md`: explicit H20 Target, code-generation-free ProgramAnalysis, typed constants, named readouts, StateTuple-only state input, NamedOutputTuple, and ExecutionResult.

**Blocked by:** None.

**Status:** ready-for-agent

- [ ] `Target` is immutable, architecture-only, and required by StatefulOperator.
- [ ] `analyze_program` always returns versioned, deterministic, JSON-serializable ProgramAnalysis without generating code.
- [ ] ProgramAnalysis carries validity, execution class, Target capability, specialization constraints, structural diagnostics, and node-level backward summary.
- [ ] `Constant` requires explicit dtype and canonical value.
- [ ] Algorithm IR has ordered uniquely named readouts.
- [ ] Invocation binds Tensor Inputs by keyword and initial state by StateTuple only.
- [ ] Invocation always returns ExecutionResult containing NamedOutputTuple and optional StateTuple.
- [ ] Tuple containers support deterministic name, index, string lookup, and iteration.
- [ ] Runtime Target mismatch and invalid gradient declarations produce stable StatefulCompilationError codes before cache lookup or code generation.
- [ ] CPU tests cover deterministic serialization, strict cutover behavior, and error details.
