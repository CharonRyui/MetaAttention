---
status: accepted
---

# Express Several States and Readout Expressions in the Algorithm IR

## Decision

Algorithm IR schema v2 keeps every v1 form and adds the vocabulary the remaining
Flash Linear Attention families need, without giving up the Scanable guarantee:

- **Several independent States.** `AlgorithmIR` accepts one `StateTransition`
  per declared `StateSpec`. States share one Head role and use at most two
  distinct Feature roles. A State Transition is driven by Inputs only, so every
  State keeps an associative affine summary whose storage is bounded by Feature
  dimensions.
- **Rank-1 States.** `StateSpec(name, (Batch, Head, Feature))` declares a vector
  State, which is how a linear-attention denominator or a HGRN hidden vector is
  represented. Internally it is promoted to a matrix with a singleton second
  Feature, so the existing per-State summary and scan kernels apply unchanged.
- **Expression readouts.** `StateContraction` stays the canonical single-role
  readout. `Readout(name, expression, output_dtype)` adds the general form:
  `StateTerm` nodes reference post-transition States, `Divide` expresses a
  denominator, `Clamp` and the transcendental nodes express normalizers, and an
  Input term expresses a non-State additive contribution.
- **Richer Input Expressions.** `Divide`, `Log`, `Sigmoid`, `Sqrt`,
  `Reciprocal`, `Clamp`, and `Shift(expr, lag)` join `Input`, `Constant`, `Add`,
  `Multiply`, `Negate`, and `Exp`. Transcendental and reciprocal forms compute in
  FP32, exactly as `Exp` already did.
- **Token shift.** `Shift` reads the value of an expression `lag` tokens earlier
  inside the same sequence and zero-fills the tokens before a sequence start, in
  both dense and packed execution.

Schema-v2 Programs are lowered by one reference lowering that composes the same
parallel segmented scans the v1 specialized plans use: per-State affine
summaries are built from the typed IR, scanned, and the readouts evaluate the
authored expression against the post-transition States. Gradients flow through
the same lowering. V1-shaped Programs keep their existing specialized TileLang
plans unchanged, so schema v2 cannot regress a v1 workload.

## Considered Options

A dedicated lowering per new FLA family was rejected: it reintroduces the model
catalog this compiler exists to avoid, and each new family would need its own
analysis, backward, and cache identity. Extending the language and lowering each
Program through the shared summary/scan machinery keeps one semantic source.

State-dependent Transition operands (TTT, Titans) were rejected for this schema:
letting a Transition read its own previous State makes the per-token map
state-dependent, which breaks the affine summary that makes Scanable Programs
parallel. Those operators need an inner-loop construct and a separate decision.

Higher-order States with a product Feature (Based, Rebased: a `[K*K, V]`
numerator) were rejected for this schema: they need more than two Feature roles
and Product Injections with three or more factors. They remain expressible only
after a later schema revision. (ADR 0003 makes that revision: Feature roles of
any arity, `Outer`, `Contract`, `Softmax`, and prior-State readouts.)

## Consequences

- Every per-token linear recurrence in `fla.ops` is expressed by a Program in
  `examples/`, and `examples/fla_coverage.py` records the coverage table that
  points at those factories; `tests/functional/test_fla_coverage.py` checks
  library parity for the operators whose reference semantics were confirmed from
  source.
- Softmax and sparse attention (`attn`, `forgetting_attn`, `nsa`) stay with
  `AttentionEngine`; a stateful linear-recurrence language is the wrong home for
  them, and that engine already implements their online forms.
- The v1 fast paths are unchanged, so the workspace envelope is now charged per
  State of the Program rather than from the first State alone.
- Programs that use v2 vocabulary select the reference plan, which is a
  correctness-first lowering: it is not yet a competitive kernel for long
  sequences, and no autotuned kernel configuration exists for it. Making the
  reference lowering fast (chunked kernels for rank-1 States and diagonal decay,
  chunk-parallel output passes) is the next performance decision, not a
  semantics change.
