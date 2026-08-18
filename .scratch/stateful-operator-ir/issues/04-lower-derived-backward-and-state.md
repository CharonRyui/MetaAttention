# 04 — Lower derived backward and state continuation

**What to build:** Generate optimized H20 backward and continuation behavior from the Stateful Program. Initial state, final state, output cotangents, final-state cotangents, and initial-state gradients must stay on the same compositional parallel pipeline as stateless forward execution.

**Blocked by:** 03 — Build compositional TileLang lowering.

**Status:** ready-for-agent

- [ ] Omitted initial state and explicit zero state are tolerance-equivalent; distinct optimized specializations are allowed.
- [ ] Caller-provided initial state is never mutated or reused as final-state storage.
- [ ] Final state uses StateSpec storage dtype; internal state/cotangents use accumulation dtype.
- [ ] `return_final_state` creates a distinct specialization and avoids final-state work when false.
- [ ] Backward combines any subset of named output and final-state cotangents, treating missing cotangents as zero.
- [ ] Gradients reach every active Tensor Input and provided initial state, including grouped-head reductions and aliased bindings.
- [ ] Inference and training gradient modes specialize separately.
- [ ] Prefix/suffix continuation matches full-sequence execution.
- [ ] No separately authored specialized backward or eager state path remains.
- [ ] CPU and H20 tests cover output-only, state-only, joint losses, post-transition readout, stateless reuse, and gradient paths.
