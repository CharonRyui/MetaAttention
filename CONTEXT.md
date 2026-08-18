# MetaAttention

## Language

**Gated Delta Rule (GDN)**:
The QLA operator targeted by this repository. It is distinct from scalar-decay gated retention: GDN takes query, key, value, log-space gate, and beta, and uses a delta-corrected recurrent state update.
_Avoid_: Gated Attention, GLA

**QLA**:
The upstream FlashQLA implementation used as a mathematical and performance reference for Gated Delta Rule. It is not a Stateful Operator execution path.

**GDN Grouped Value Attention (GVA)**:
A head mapping where `Hv % Hk == 0`; each query/key head serves `Hv/Hk` value and recurrent-state heads.

# Stateful Operator Domain

## Glossary

**Algorithm IR** — A backend-independent description of a stateful sequence operator: input transforms, a structured state transition, and a state readout. It defines model mathematics, not a GPU schedule.

**Stateful Operator** — The sole IR-first public compiler interface for state-transition algorithms such as GDN and Mamba2. It accepts validated Algorithm IR and typed input specifications, derives Backward IR, selects a compatible lowering strategy, and returns a stateless callable operator.

**State transition** — The per-token transformation from a prior state to a next state. It combines an ordered structured propagation with an input-derived injection.

**Structured affine transition** — A state transition whose propagation is a compiler-recognized node or an ordered composition of such nodes, plus an input-derived injection. Its composition and backward rules are compiler-derived.

**Rank-one delta propagation** — A structured matrix-state propagation `S -> S - beta * k * (k^T * S)` for an arbitrary unnormalized key and raw differentiable beta. GDN composes scalar `exp(log_gate)` decay before this propagation, then adds a beta-weighted key-value outer-product injection.

**State injection** — The input-derived additive term in a structured affine transition. The first phase recognizes outer product, elementwise product, direct input, and zero injection.

**State tuple** — An ordered collection of named tensor states. Each state declares storage and accumulation dtypes; transitions read the prior tuple, commit next states simultaneously, and never mutate caller-owned storage.

**Initial state** — An optional caller-provided state tuple used before the first token. Missing initial state means a zero-initialized state tuple.

**Final state** — The state tuple after the final token. Callers may request it for streaming or continuation through the stable Execution Result interface.

**Frontend sugar** — Model-facing conveniences outside Algorithm IR. Stateful Operator examples construct IR explicitly; legacy `LinearAttentionEngine` modifier inputs remain outside the new compiler guarantee.

**Backward IR** — The internal compiler-derived vector-Jacobian-product program for an Algorithm IR. Frontends cannot construct or override it, and generated backward execution implements it compositionally.

**Stateful program** — Validated Algorithm IR paired with compiler-derived Backward IR, execution class, and Target capability analysis for lowering. It is an internal compiler artifact, not a model-authoring interface.

**IR-derived backward** — Backward semantics derived from Algorithm IR node VJPs, including named output and final-state cotangents, initial-state gradients, and Head Mapping reductions.

**Head mapping** — The explicit relationship between input heads and state heads, including valid group broadcasting and reduction rules. It is Algorithm IR semantics, not a kernel layout inference.

**Tensor input** — A named, typed external tensor available to input transforms, transitions, or readouts. Its shape roles, dtype, and gradient requirement are declared by an input specification; it replaces `CustomIO` only for stateful operators.

**Lowering strategy** — The backend-specific schedule, tiling, layout, fusion, state materialization, and kernel generation used to execute a Scanable Program. It does not replace node semantics or select model-specific implementations.

**Backend capability** — The concrete dtype, shape, layout, and architecture combinations a Target can lower for forward, continuation, and backward. Public IR constructors define mathematical language; capability determines whether a Scanable Program can execute on that Target.

**Execution class** — A compiler-derived classification of validated Algorithm IR by its state-summary algebra. A scanable program has a compact associative summary and may enter Stateful Operator lowering; a recurrent program does not and is rejected before backend lowering in this phase.

**Scanable program** — A stateful program whose typed propagation composition has a compiler-proven compact associative summary suitable for parallel chunked execution.

**Recurrent program** — A mathematically valid stateful program without a compiler-proven compact associative summary. It has no Stateful Operator execution path in the current phase.

**Target** — An immutable description of the intended stateful backend and hardware architecture used for capability analysis and specialization. It is explicit rather than inferred from process-global device state.

**Program analysis** — The compiler-owned, code-generation-free derivation of execution class, Backward IR, target capability, specialization constraints, and actionable diagnostics from Algorithm IR and a Target.

**Named output tuple** — An ordered collection of uniquely named token-output tensors produced by post-transition readouts. Output order and names are Algorithm IR semantics.

**Execution result** — The immutable result of a Stateful Operator invocation, containing a Named Output Tuple and an optional final State Tuple with stable return shape.
