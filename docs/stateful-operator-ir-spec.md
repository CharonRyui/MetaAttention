# Compositional Stateful Operator Compiler Specification

Status: `proposed`

## Problem

The current `StatefulOperator` validates a backend-independent Algorithm IR but lowers it by matching whole algorithms to existing implementations. Linear state continuation falls back to a Python token loop, GDN bypasses generic lowering through a dedicated implementation, and gradients come from whichever legacy kernel was selected rather than a compiler-derived Backward IR. The result is an interface over a catalog of implementations, not a compiler for new compositions of known state-transition nodes.

The refactor must preserve `LinearAttentionEngine` unchanged while replacing the IR-first path with one H20-targeted compiler. The compiler must generate parallel chunked TileLang forward and backward kernels compositionally from finite, compiler-known node semantics. It must not dispatch on model names, use eager recurrence, or bypass semantic lowering through specialized whole-algorithm kernels.

## Goals

1. Compile previously unseen scanable algorithms composed from existing IR nodes without adding whole-algorithm Python patterns.
2. Derive Backward IR from node-owned VJPs and use it as the semantic contract for generated backward kernels.
3. Support Gated Linear Attention, RetNet recurrent, Mamba2, and GDN end to end on H20 through the same lowering pipeline.
4. Support optional initial state, optional final state, losses through any named output or final state, and gradients to provided initial state.
5. Generate only parallel chunked execution. Mathematically valid programs without compact associative summaries are classified as recurrent and rejected before backend lowering.
6. Preserve mathematical IR independently from target, shape, layout, schedule, and tuning details.
7. Make capability, diagnostics, specialization, generated source, and cache behavior deterministic and inspectable.

## Non-goals

- Modifying the `LinearAttentionEngine` interface or routing it through `StatefulOperator`.
- Supporting architectures other than NVIDIA H20 in the first compiler cutover.
- Accepting arbitrary Python transition, readout, composition, or backward functions.
- Providing a sequential CPU, eager PyTorch, or token-serial GPU execution fallback.
- Preserving `GDNEngine`; it is removed in the clean cutover.
- Adding public model-specific factories. Model construction remains explicit in `examples/`.
- Implementing arbitrary coupled multi-state transitions in this phase.
- Guaranteeing bitwise equivalence under floating-point reassociation.

## Domain Model

### Algorithm IR

Algorithm IR is the backend-independent mathematical program. It contains:

- uniquely named typed Tensor Inputs;
- an ordered tuple of named State Specs;
- state transitions;
- ordered, uniquely named post-transition readouts;
- explicit Head Mapping;
- typed input-expression nodes.

It contains no device index, concrete tensor size, stride, chunk size, TileLang source, schedule, tuning result, or backend identity.

The future multi-state form has one transition per named state. All transitions read the same prior state tuple and commit simultaneously. Cross-state transition dependencies require future compiler-known nodes and are not added implicitly. Dedicated compiler-known readout nodes may consume multiple updated states when a concrete use case requires them.

The first cutover supports the current single-state profiles while keeping internal analysis and result interfaces compatible with ordered transitions and named readouts.

### Typed constants

`Constant` has an explicit dtype and exact canonical value. Constant type and value participate in structural identity. Python numeric inference is not part of Algorithm IR semantics.

### Pure state semantics

Algorithm IR is purely functional. A state transition defines a next state and never mutates a caller-owned state or Tensor Input. Code generation may reuse internal storage only when alias analysis proves the reuse unobservable.

Readouts consume the fully updated state tuple at the current token. Pre-transition readout is unsupported.

### Execution classes

Compiler analysis classifies validated Algorithm IR as:

- **Scanable Program**: every state transition has a compiler-proven compact associative summary suitable for parallel chunked execution.
- **Recurrent Program**: the mathematics is valid, but at least one transition lacks such a summary.

Stateful Operator executes only Scanable Programs. A Recurrent Program produces a deterministic analysis report identifying the minimal non-closed structural subexpression and is rejected before backend lowering. It never selects a slower implementation.

Scanability is derived from typed algebra rules owned by the compiler. Authors cannot assert associativity or supply composition rules. The compiler canonicalizes and proves reductions based on operand domains rather than whitelisting model names or exact node sequences.

### Stateful Program and Backward IR

A Stateful Program is the validated forward Algorithm IR, compiler-derived Backward IR, execution classification, and Target capability result used by lowering.

Every differentiable expression, propagation, injection, readout, state operation, and Head Mapping operation owns a VJP rule. Backward derivation covers:

- all differentiable Tensor Inputs;
- all named output cotangents, with missing cotangents treated as zero;
- optional final-state cotangents;
- gradients to a provided initial State Tuple;
- grouped-head broadcast reversal and reduction through Head Mapping;
- gradient accumulation when multiple input names alias the same tensor storage.

Backward IR remains internal. Program analysis exposes a read-only node-level summary of differentiated inputs and states, cotangent paths, reductions, and any unsupported VJP location. Frontends cannot construct or override Backward IR.

## Public Interface

### Target

`Target` is an immutable architecture descriptor. The first public target is H20. It identifies backend and architecture, not a CUDA device index. Identical H20 cards share analysis and compiled artifacts.

`StatefulOperator` requires an explicit Target. There is no inferred-target compatibility mode:

```python
operator = StatefulOperator(algorithm, target=Target.h20(), compile_options=options)
```

Runtime tensor architecture is checked against Target before cache lookup or code generation.

### Program analysis

```python
report = analyze_program(algorithm, target, compile_options)
```

`analyze_program` performs no code generation and always returns an immutable `ProgramAnalysis`. It has a versioned stable JSON representation containing:

- validity and structural diagnostics;
- execution class;
- minimal non-closed structural path, operand domains, and supported reformulations;
- Target capability and concrete specialization constraints;
- node-level backward summary;
- canonical semantic identity.

`StatefulOperator` consumes the same internal analysis. A failed report becomes one `StatefulCompilationError` with a stable machine-readable code and structured details.

### Invocation

Runtime Tensor Inputs bind by keyword name only. Runtime flags must not request gradients for a Tensor Input declared `requires_grad=False`; this mismatch is rejected. An input declared differentiable may bind a tensor with gradients disabled, producing an inference specialization.

Initial state accepts `StateTuple` only. A missing initial state has zero-state semantics. The compiler may specialize missing and explicit-zero state separately, but results and gradients must agree within declared tolerance. Caller-provided initial state is read-only.

```python
result = operator(
    query=query,
    key=key,
    value=value,
    initial_state=state_tuple,
    return_final_state=True,
)
```

Invocation always returns immutable `ExecutionResult`:

- `outputs`: `NamedOutputTuple`;
- `final_state`: `StateTuple | None`.

`NamedOutputTuple` and `StateTuple` provide deterministic names, integer indexing, string lookup, and iteration. Dynamic attribute access is not required. Even a single output remains wrapped. `return_final_state` controls whether final state is materialized and participates in specialization identity.

Algorithm IR carries an ordered tuple of uniquely named readouts and may therefore produce multiple named token outputs. The first profiles each use one output, but every repository caller migrates to `ExecutionResult` without a bare-tensor compatibility path.

## Numeric Semantics

State propagation, injection, chunk summaries, and backward state cotangents compute in each `StateSpec.accumulation_dtype`. Casts occur only at declared input, state-storage, output, or explicitly typed expression seams. Optimizations may not silently reduce precision.

Final states use each `StateSpec.dtype`; accumulation dtype does not redefine storage dtype. Different states may use different storage and accumulation dtypes when Target capability permits.

GDN is an Algorithm IR construction, not a fixed dtype contract. The H20 Target publishes a capability matrix for supported input, output, state-storage, and accumulation dtype combinations. Unsupported concrete dtype combinations fail analysis before code generation.

Compiler optimizations may reassociate floating-point operations while preserving the mathematical graph and declared dtype boundaries. Correctness uses profile-specific tolerance against independent PyTorch references, not bitwise identity.

## Head and Shape Semantics

Head Mapping owns forward broadcast semantics and inverse grouped-head gradient reductions. Individual nodes do not duplicate head-indexing policy.

Logical roles belong to Algorithm IR. Concrete sizes, strides, and layouts belong to specialization metadata. The first compiler specializes all batch, sequence, head, and dimension sizes. Sequence length must be positive. Unaligned positive lengths use a masked tail chunk; the operator does not truncate, expose padding, or reject solely for chunk alignment.

Target capability declares accepted layouts. Initial H20 lowering may require contiguous tensors; it must reject layout mismatch before cache lookup or code generation rather than copying implicitly.

Two named Tensor Inputs may alias the same tensor or storage. Generated backward must accumulate all semantic gradient contributions correctly.

## Compositional TileLang Lowering

Each supported IR node contributes typed lowering behavior for:

1. token-local forward evaluation;
2. compact state-summary construction;
3. associative summary composition;
4. chunk-to-chunk state propagation;
5. post-transition readout;
6. VJP and cotangent accumulation.

One stateful kernel generator assembles these fragments. Whole-algorithm implementation selection is prohibited. Model names and frontend identities never participate in lowering.

IR-generic optimizations are allowed, including canonicalization, identity elimination, adjacent-scale fusion, algebraic simplification, layout selection, tiling, fusion, and schedule selection from structural properties. These optimizations remain inside the one semantic lowering pipeline.

The cutover does not use FlashQLA/GDN kernels as an execution path. QLA may remain a mathematical and performance reference. GLA, RetNet recurrent, corrected selective-diagonal Mamba2, and rank-one-delta GDN all execute generated compositional TileLang kernels.

No supported profile may use a Python recurrence, per-token framework launch, token-serial TileLang fallback, or a separately authored specialized backward.

## Cache and Compilation

Cache identity includes:

- explicit compiler semantic-schema version;
- canonical forward Algorithm IR;
- derived backward semantic identity;
- execution class;
- concrete tensor metadata, including shapes, dtypes, layouts, and active gradient mode;
- Target backend and architecture;
- final-state materialization mode;
- relevant compile and schedule options.

Frontend object identity, diagnostic labels, device index, source locations, and formatting do not participate.

Identical identity inputs generate byte-identical semantic source across processes. Traversal order, temporary names, and serialization are deterministic.

Autotuning records are versioned auxiliary artifacts keyed by semantic kernel family, Target, and concrete shape metadata. They are separate from semantic program identity.

Shared compiled artifacts use process-safe locking and atomic publication. Concurrent callers may perform duplicate private work, but no caller observes a partial artifact. Transient compilation, driver, or resource failures never poison the cache; temporary artifacts are removed. Deterministic analysis failures live only in ProgramAnalysis.

## Capability and Error Contract

Public IR constructors define mathematical language, not an unconditional execution promise. Target capability defines executable dtype, layout, shape, and hardware combinations. Within the compiler-known scanable algebra, no semantic whole-combination rejection is permitted: compositional lowering must handle previously unseen valid compositions.

Valid Recurrent Programs are rejected because Stateful Operator's interface promises parallel chunked execution. Invalid IR, recurrent classification, unsupported Target capability, target mismatch, and code-generation failure use distinct stable codes under `StatefulCompilationError`.

Diagnostics identify deterministic structural paths such as `transitions[memory].propagation.nodes[1]`; source-stack capture and wrapper identity are excluded.

## Migration

This is a strict, end-to-end cutover. No compatibility aliases or partially switched execution paths remain.

- `LinearAttentionEngine` source and external behavior remain unchanged and outside the new compiler guarantee. Shared low-level lowering logic may evolve.
- `GDNEngine` is removed, including exports, adapter tests, documentation, and repository callers.
- GDN construction remains an explicit example, not a library factory or execution wrapper.
- GLA, RetNet recurrent, Mamba2, and GDN examples keep callable operator factory functions but construct Algorithm IR, Target, and StatefulOperator explicitly.
- Mamba2 is corrected to use selective diagonal propagation and its intended compiler-known readout semantics while retaining the example factory's arguments.
- Every StatefulOperator caller supplies Target, typed constants, StateTuple, and consumes ExecutionResult/NamedOutputTuple.
- Documentation publishes the H20 capability matrix and explicit temporary rejection of other architectures.

## Verification

### CPU-safe semantic proof

- Validate and canonicalize every node and expression.
- Compare each VJP and supported composition against a small PyTorch autograd recurrence in FP64 or FP32.
- Cover output-only, final-state-only, and joint losses.
- Cover provided initial-state gradients, missing cotangents, grouped-head reductions, and aliased Tensor Inputs.
- Prove post-transition readout with a deterministic short sequence.
- Verify omitted initial state against explicit zero state.
- Verify stable ProgramAnalysis JSON, deterministic source generation, schema-version invalidation, and cache keys.
- Verify minimal recurrent structural diagnostics and Target capability errors without code generation.

### H20 functional proof

On an otherwise idle H20, each of GLA, RetNet recurrent, Mamba2, and GDN must:

- compile through the generic compositional path;
- match an independent PyTorch reference for outputs and every declared gradient;
- support omitted and explicit initial state;
- return named final state with declared storage dtype;
- combine output and final-state losses;
- match full-sequence execution with prefix/suffix continuation;
- handle a positive unaligned sequence through a masked tail chunk;
- expose no token-serial recurrence or specialized implementation bypass.

### Performance proof

Before cutover, record current profile baselines using the repository benchmark warmup/repetition methodology on an idle H20 with fixed shapes and configuration. Compare median latency and observed variance against the generic compiler. Any statistically significant regression blocks merge pending explicit review; no unstated percentage is automatically accepted.

Structural inspection must also demonstrate parallel chunked state-summary construction and composition. Correctness without this structure is insufficient.

## Acceptance

The change lands only when the strict cutover, all four H20 profiles, compiler-derived backward, state continuation, named results, migration, CPU semantic proof, H20 functional proof, and performance review are complete. No scaffold, eager fallback, specialized execution path, temporary adapter, or partial interface migration is acceptable.

ADR 0001 remains proposed until CPU analysis/VJP proof and at least one generic parallel H20 profile satisfy correctness and baseline review. It becomes accepted at that evidence gate; full merge still requires every acceptance criterion above.
