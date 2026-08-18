# Compositional Stateful Operator Compiler Specification

Status: `proposed`

## Problem Statement

The current branch is not yet an implementation of this contract: its prior
authoring path selected legacy whole algorithms underneath Algorithm IR.
This specification explicitly forbids eager token loops, dedicated GDN
execution, and backend-specific backward as the implementation target.

Model authors need one small mathematical interface for state evolution. A supported state transition must compile because its typed algebra is Scanable, not because the repository recognizes a model name or exact graph. The same mathematical program must drive forward, backward, dense execution, packed variable-length execution, initial/final state, capability diagnostics, and cache identity. Kernel schedules may optimize common algebraic structures, but they must not replace their semantics.

## Solution

Build one H20-first compositional Stateful Operator compiler. A model author declares typed Tensor Inputs, one matrix State, an ordered structured affine transition, explicit Head Mapping, and one or more named post-transition State Contractions. The first compiler schema provides a finite language:

- typed elementwise Input Expressions: Input, Typed Constant, Add, Multiply, Negate, and Exp;
- role-directed Axis Scale propagation;
- general Rank-One propagation;
- ordered propagation sequences;
- additive tuples of Product Injections;
- single-Feature-role State Contractions;
- identity and contiguous-group Head Mapping.

Compiler-owned typing and affine algebra derive an associative state summary. Summary storage may depend on State Feature dimensions but never sequence length; the canonical summary may be a dense affine state map. Programs with such a summary are Scanable Programs. Mathematically valid programs without one are Recurrent Programs and are rejected without a slower fallback.

Every forward node owns a VJP. The compiler composes those VJPs into an internal canonical Backward IR, then lowers forward and backward through the same parallel chunked TileLang compiler. A generic dense affine lowering is the semantic fallback for supported Scanable Programs. Proven local algebraic optimizations such as scale fusion, product/GEMM recognition, WY/KKT-style factorization, shared readout fusion, and shape-specific scheduling are permitted. Model-name dispatch, whole-profile matching, eager recurrence, token-serial execution, dedicated GDN execution, and hand-written profile backward are prohibited.

The first production lowering supports CUDA H20, dense uniform sequences, token-major packed variable-length sequences, BF16/FP32 inputs and outputs, FP32 state, contiguous canonical layouts, and matrix Feature dimension pairs drawn from `{64, 128} × {64, 128}`. Algorithm IR remains independent of Target, concrete size, layout, and schedule.

The primary and only public behavioral test seam is constructing a `StatefulOperator` from Algorithm IR and invoking it with real keyword-bound tensors. Every invocation returns immutable `ExecutionResult`. Compiler analysis remains internal; public failures are stable structured `StatefulCompilationError` values.

## User Stories

1. As a stateful-attention author, I want to describe state evolution directly, so that I do not encode a recurrence as unrelated Q/K/V modifiers.
2. As a stateful-attention author, I want previously unseen combinations of supported nodes to compile, so that adding an algorithm does not require adding its name to a dispatcher.
3. As a compiler user, I want Algorithm IR to contain mathematical semantics only, so that target, layout, chunking, and scheduling remain implementation choices.
4. As a compiler user, I want one public invocation seam, so that outputs, state continuation, gradients, errors, and specialization behave consistently.
5. As an operator author, I want typed Batch, Sequence, Head, and Feature roles, so that shape meaning is not encoded by free strings or concrete sizes.
6. As an operator author, I want exactly one matrix State in the first schema, so that the public language matches the compiler's delivered capability.
7. As an operator author, I want the two State Feature roles to be ordered and distinct, so that left and right state actions are unambiguous.
8. As an operator author, I want an empty propagation tuple to mean identity, so that a pure accumulator needs no placeholder node.
9. As an operator author, I want propagation nodes to execute left to right, so that noncommuting state updates preserve authored order.
10. As an operator author, I want Axis Scale to identify logical roles, so that scalar, diagonal, and elementwise scaling use one model-independent node.
11. As an operator author, I want Axis Scale direction to follow State Feature-role position, so that left and right diagonal actions do not depend on physical layout.
12. As an operator author, I want a general Rank-One propagation, so that left and right vectors, coefficient, and transformed role are not tied to GDN terminology.
13. As a GDN author, I want to compose gate scaling, rank-one propagation, Product Injection, and State Contraction, so that GDN is an ordinary IR program rather than a dedicated backend.
14. As an operator author, I want multiple additive Product Injections, so that a transition may add several input-derived state terms.
15. As an operator author, I want an empty injection tuple to mean additive zero, so that zero injection needs no special node.
16. As an operator author, I want every Product Injection to contain at least one factor, so that multiplicative identity is never confused with additive zero.
17. As an operator author, I want Product Injection factors aligned by Feature roles, so that outer, direct, and elementwise construction are consequences of typing rather than separate algorithm nodes.
18. As an operator author, I want shared Feature-role multiplication to be explicit in Input Expressions, so that Product Injection never guesses broadcast semantics.
19. As an operator author, I want multiple named State Contractions, so that one updated State can produce main and auxiliary token outputs.
20. As an operator author, I want each State Contraction to name one Feature role, so that readout is not an unrestricted einsum language.
21. As an operator author, I want every readout to consume post-transition State, so that the current token's injection contributes to its current output.
22. As an operator author, I want readouts to be independent, so that output ordering does not create a hidden computation DAG.
23. As an operator author, I want Input, Typed Constant, Add, Multiply, Negate, and Exp expressions, so that common gates and scales are composable without arbitrary callbacks.
24. As a compiler maintainer, I want finite Input Expression semantics, so that typing, canonicalization, VJPs, and lowering remain complete.
25. As a compiler maintainer, I want Typed Constants quantized at construction, so that semantic identity is deterministic across processes and frontends.
26. As a compiler maintainer, I want NaN and infinity rejected in Constants, so that equality and canonicalization remain unambiguous.
27. As a compiler maintainer, I want Add and Multiply to canonicalize operand order but retain association, so that cache reuse does not erase floating-point rounding structure.
28. As a training user, I want every differentiable node to own a compiler VJP, so that backward cannot drift from forward semantics.
29. As a training user, I want output-only, final-state-only, and joint losses, so that segmented recurrent training remains complete.
30. As a training user, I want gradients to provided initial State, so that sequence continuation can participate in optimization.
31. As a training user, I want missing output or final-state cotangents treated as zero, so that unused result fields need no synthetic loss.
32. As a training user, I want grouped-head broadcasts reversed by compiler-owned reductions, so that shared input heads receive complete gradients.
33. As a training user, I want two input names to bind the identical Tensor object, so that parameter sharing accumulates all semantic gradient contributions.
34. As a training user, I want overlapping distinct views rejected, so that the first schema does not pretend to provide undefined storage-level gradient merging.
35. As an inference user, I want differentiable inputs usable without gradients, so that one Algorithm IR serves training and inference.
36. As a compiler user, I want training specialization selected from grad mode and active bindings, so that no second gradient-control interface conflicts with PyTorch.
37. As a dense-batch user, I want any uniform positive sequence length, so that kernel chunk size does not leak into model inputs.
38. As a packed-batch user, I want token-major variable-length values and shared sequence offsets, so that different sequence lengths require no padding.
39. As a packed-batch user, I want one independent State entry per logical sequence, so that recurrence never crosses offsets.
40. As a packed-batch user, I want empty sequences, so that callers need not filter and remap State entries.
41. As a packed-batch user, I want an all-empty invocation to be valid, so that identity State behavior is total rather than conditional.
42. As a packed-batch user, I want offset values to remain runtime data, so that changing a length distribution does not force recompilation.
43. As a CUDA Graph user, I want device-local fixed-shape offsets, so that packed execution does not require host synchronization for specialization.
44. As a streaming user, I want an omitted initial State to mean per-sequence zeros, so that full-sequence execution is concise.
45. As a streaming user, I want requested final State in logical sequence order, so that continuation mapping is deterministic.
46. As a streaming user, I want empty sequences to preserve initial State, so that identity continuation has exact semantics.
47. As a streaming user, I want prefix/suffix continuation to match full execution within tolerance, so that invocation boundaries do not change the model materially.
48. As a numerical-kernel author, I want State to remain FP32 during an invocation, so that chunk and schedule choices do not introduce storage-rounding seams.
49. As a numerical-kernel author, I want State storage casts only at invocation entry and final materialization, so that continuation precision is explicit.
50. As a model author, I want BF16 and FP32 Tensor Inputs and outputs, so that common mixed-precision recurrent models are representable.
51. As a model author, I want FP32 Exp and FP32 State algebra, so that gates and recurrent accumulation remain stable.
52. As a compiler user, I want unsupported dtype combinations rejected before code generation, so that no kernel silently changes precision.
53. As a compiler user, I want inactive tail lanes to be affine identity, so that unaligned dense and packed chunks cannot alter State or gradients.
54. As a compiler user, I want inactive lanes not to read Tensor Inputs, so that out-of-range values have no accidental semantics.
55. As a grouped-head author, I want explicit identity and contiguous-group Head Mapping, so that equal and grouped heads are deliberate.
56. As a grouped-head author, I want arbitrary role names, so that Head Mapping does not assume query, key, or value vocabulary.
57. As a framework user, I want Algorithm IR to permit arbitrary positive Feature sizes, so that mathematical semantics are not fixed at 128.
58. As an H20 user, I want `{64,128} × {64,128}` matrix State capability, so that square and unequal Feature dimensions are supported deliberately.
59. As an H20 user, I want unsupported concrete resources rejected as specialization errors, so that hardware limits are not mislabeled as mathematical recurrence.
60. As an H20 user, I want canonical contiguous layouts, so that hidden layout copies do not distort performance or memory.
61. As an H20 user, I want no implicit transpose or `.contiguous()` call, so that input preparation cost stays visible.
62. As a deployment author, I want Target optional, so that ordinary invocation can derive hardware from bound tensors.
63. As a deployment author, I want an optional explicit Target constraint, so that deployment to unintended hardware fails deterministically.
64. As a deployment author, I want H20 identified by product family and capability, so that H100 is not accepted merely because capability numbers overlap.
65. As a deployment author, I want unknown products and unidentifiable MIG devices rejected, so that tuning and correctness evidence are not extrapolated silently.
66. As a performance engineer, I want tuning records isolated by exact product and environment, so that incompatible measurements are never reused by architecture alone.
67. As a compiler maintainer, I want Scanable and Recurrent to be target-independent classifications, so that hardware resource limits do not alter mathematics.
68. As a compiler maintainer, I want a dense affine canonical summary, so that rank-one composition remains closed even when factors become dense.
69. As a compiler maintainer, I want summary storage independent of sequence length, so that Scanable has a precise compactness criterion.
70. As a compiler maintainer, I want generic dense affine lowering as semantic fallback, so that optimization misses cannot reject valid Scanable programs.
71. As a performance engineer, I want proven local algebraic optimizations, so that common structures reach competitive schedules without whole-profile dispatch.
72. As a reviewer, I want a model-independent compositional pressure profile, so that four recognized examples cannot hide an algorithm catalog.
73. As a reviewer, I want unequal Feature dimensions and left-not-equal-right rank-one vectors in the pressure profile, so that the compiler cannot assume GDN's square symmetric form.
74. As a reviewer, I want multiple injections and readouts in the pressure profile, so that composition is exercised rather than merely declared.
75. As a reviewer, I want packed empty sequences and unaligned tails in the pressure profile, so that generic boundary semantics are exercised together.
76. As a compiler user, I want stable structured errors, so that frontends and tests can act on failures without parsing prose.
77. As a compiler user, I want errors to identify canonical public-IR paths, so that unsupported combinations are actionable.
78. As a compiler maintainer, I want internal analysis reports free to evolve, so that public diagnostics do not freeze compiler implementation.
79. As a compiler maintainer, I want transient compile and resource failures excluded from failure caches, so that temporary conditions do not poison future calls.
80. As a caller, I want every invocation to return immutable `ExecutionResult`, so that return shape never depends on output or State count.
81. As a caller, I want named and indexed output access, so that multiple outputs remain deterministic and ergonomic.
82. As a caller, I want final State absent and unmaterialized unless requested, so that output-only execution avoids unnecessary work.
83. As an existing MetaAttention user, I want `LinearAttentionEngine` behavior unchanged, so that this compiler does not create an unrelated migration.
84. As a project maintainer, I want the dedicated `GDNEngine` removed, so that GDN cannot retain a second semantic execution path.
85. As a project maintainer, I want GLA, RetNet recurrent, Mamba2-style selective State, and GDN expressed only through generic nodes, so that examples teach the compiler language rather than model factories.
86. As a correctness reviewer, I want independent PyTorch references, so that generated forward and backward cannot agree with the same compiler defect.
87. As a performance reviewer, I want a predeclared statistical gate, so that merge decisions cannot be chosen after observing benchmark results.
88. As a performance reviewer, I want first compilation time recorded but not gated, so that runtime regression and compiler-development cost remain separate evidence.
89. As a maintainer, I want the compiler schema version in semantic identity, so that language changes invalidate incompatible artifacts.
90. As a concurrent caller, I want atomic cache publication and process-safe locking, so that no caller observes partial generated artifacts.

## Implementation Decisions

- Build one deep Stateful Operator module with one public behavioral seam: construct from Algorithm IR and invoke with keyword-bound tensors. The module owns validation, canonicalization, Backward IR derivation, Target resolution, specialization, lowering, caching, compilation, and execution.
- Algorithm IR contains no concrete size, stride, device, chunk size, TileLang construct, schedule, tuning result, model name, or frontend identity.
- The first compiler schema contains exactly one named matrix State with exactly two distinct ordered Feature roles. Multi-State programs and other State ranks require a later schema version.
- Tensor Inputs declare a unique name, full logical role tuple, BF16 or FP32 dtype, and `differentiable` capability. Logical roles are typed singleton Batch and Sequence roles, named Head roles, and named Feature roles.
- The State's logical shape is sequence count, one named State Head role, then its two Feature roles. Dense sequence count is batch size; packed sequence count is `len(sequence_offsets) - 1`.
- Input Expressions are immutable trees over Input, Typed Constant, Add, Multiply, Negate, and Exp. Arbitrary Python functions, tracing, control flow, comparisons, `where`, division, Log, clamp, and general tensor contractions are not accepted.
- Typed Constants support BF16 and FP32 only. Construction deterministically rounds to the declared dtype and stores canonical bits. Dtype plus bits define identity; signed zeros remain distinct; NaN and infinity are invalid.
- Add and Multiply canonicalize binary operand order by semantic identity but retain the authored association tree. Canonicalization does not use associativity, distributivity, or transcendental rewrites. Local constant folding observes the normative dtype rules.
- The normative dtype table keeps equal floating dtypes unchanged, promotes BF16 with FP32 to FP32, and makes Exp compute and return FP32. State-affecting operations and State Contractions compute in FP32. Output casts occur once at each readout's declared output dtype. Input gradients use their input dtype.
- A State Transition contains an ordered propagation tuple and an ordered tuple of Product Injections. The transition semantics are `next_state = Pn(...P2(P1(prior_state))) + sum(injections)`.
- The propagation tuple executes left to right. An empty tuple and explicit identity are semantically identical. Canonicalization flattens nesting and removes identity but never reorders unproven noncommuting propagations.
- Axis Scale accepts a typed factor and an explicit tuple of State Feature roles. Empty roles apply scalar scaling. For matrix State, the first Feature role defines left diagonal action and the second defines right diagonal action; both roles define per-axis elementwise action. Logical role position, not storage layout, defines direction.
- Rank-One propagation accepts one State Feature role, typed left and right expressions, and a typed coefficient. Its semantics are `S -> S + coefficient * left * contract(right, S, role)`. Left and right may differ; sign and magnitude live in coefficient.
- Each Product Injection contains at least one typed factor. Factor Feature-role union must cover both State Feature roles. A factor broadcasts over missing Feature roles. Shared Feature-role multiplication must already be explicit in one Input Expression. No implicit transpose, reduction, or contraction occurs. Batch/head scalars may broadcast through Head Mapping.
- An empty injection tuple is additive zero. There is no public Zero Injection node.
- Algorithm IR carries one or more ordered, uniquely named State Contractions. Every contraction reads post-transition State, contracts exactly one Feature role with an operand expression, introduces no additional Feature role, retains the other State Feature role, and declares output dtype. Readouts cannot depend on other outputs.
- Head Mapping is a set of explicit source-Head-role mappings to the State Head role. The first schema supports identity and contiguous group broadcast only. Each mapping defines its inverse gradient reduction. No node assumes query, key, or value names.
- A Scanable Program has an associative affine summary whose storage depends only on State Feature sizes, never sequence length. The canonical semantic summary may contain a dense State transform and additive State term. Dense closure makes general Rank-One composition Scanable.
For the selective diagonal/Mamba2-style acceptance profile, the intended
composition is explicit: `a_t = exp(dt_t * A)`, `u_t = dt_t * key_t`,
`S_t = AxisScale(a_t, S_{t-1}) + ProductInjection(u_t, value_t)`, and
`y_t = StateContraction(S_t, query_t)`. `A` is a typed constant or broadcast
Input with one State Head role; `dt` and all products obey the declared
BF16/FP32 promotion and FP32 State rules. This equation is a profile example,
not a compiler primitive or dispatch key.

- A summary is represented semantically as an affine map `(M, b)` over the
  matrix State. Summary composition is `(M2, b2) ∘ (M1, b1) = (M2 M1,
  M2 b1 + b2)`, and applying a token summary to State `S` yields `M S + b`.
  Factorized, WY/KKT, low-rank, or tiled forms are permitted only as equivalent
  storage and schedule optimizations.
- A Recurrent Program is mathematically valid but lacks a compiler-proven compact associative summary. It receives a stable rejection before backend lowering. Target or resource limitations never change execution class.
- Every forward expression, propagation, Product Injection, State Contraction, State operation, and Head Mapping operation owns a compiler VJP. The compiler composes an internal canonical Backward IR. Frontends cannot construct or override it.
- Backward IR explicitly represents named output cotangents, optional final-State cotangent, reverse State propagation, initial-State gradient, Head Mapping reductions, and aliased-input accumulation. Generated backward has no hand-written profile path.
- Compiler analysis has no public entry point or stable report type. It remains an internal code-generation-free phase.
- Public failures use `StatefulCompilationError` with a stable category code, canonical public-IR path, and JSON-safe details. Human message text and internal reports are not compatibility contracts.
- Deterministic validation order is IR structure, role/dtype typing, scanability and VJP completeness, runtime binding/sequence/State validity, Target matching, concrete resource capability, then code generation.
- Runtime Tensor Inputs bind by keyword name only. All required and no unknown names must be present.
- `differentiable=False` rejects a grad-enabled binding requiring gradients. `differentiable=True` permits training or inference. Training specialization is selected when grad mode is enabled and any differentiable binding or provided initial State requires gradients; otherwise inference is selected.
- Distinct Tensor Input names may bind the identical Tensor object. Compiler backward accumulates every contribution to that object. Distinct Tensor objects with overlapping storage are unsupported; nonoverlapping views are independent.
- Dense execution uses canonical contiguous role order `[batch, heads, sequence, features...]` and requires positive uniform sequence length.
- Packed execution uses canonical contiguous token-major values `[total_tokens, heads, features...]` and one shared device-local contiguous `int32` sequence-offset tensor.
- Packed offsets have shape `[num_sequences + 1]`, start at zero, end at `total_tokens`, and are nondecreasing. Shape and dtype enter specialization identity; values remain runtime data and do not enter compiled-artifact identity.
- Packed empty sequences are valid, including an all-empty invocation. Empty sequences emit no output, preserve initial or zero State, and pass final-State cotangent directly to initial State.
- Compiler-created inactive tail lanes do not read Tensor Inputs, apply affine identity with zero injection, emit no output, produce no input gradient, preserve State, and pass State cotangent unchanged. Inactive semantics override Input Expressions rather than manufacturing padding values.
- Initial State accepts immutable `StateTuple` only. Omitted State creates one FP32 zero State per logical sequence. Provided FP32 State is read-only and converted once to accumulation representation at invocation entry.
- Final State is materialized only when requested, one entry per logical sequence in batch or offset order. Continuation requires matching sequence count and role shape.
- Every invocation returns immutable `ExecutionResult` containing immutable `NamedOutputTuple` and `StateTuple | None`. Single outputs and State are never unwrapped. Containers support deterministic name lookup, integer indexing, and iteration.
- Target is an optional deployment constraint. If absent, it resolves solely from all bound runtime tensors; process-global current-device state is forbidden. If supplied, every binding must match.
- The first production Target is CUDA H20. Resolution requires both approved product-family identity and approved CUDA capability. H100, other CUDA products, CPU, ROCm, unknown same-capability products, and unidentifiable MIG devices are rejected.
- The first H20 capability accepts only canonical role-order contiguous tensors, BF16/FP32 inputs and outputs, FP32 State, and two-State-Feature dimension pairs in `{64,128} × {64,128}`. Unequal Feature sizes are required capability.
- Algorithm IR permits arbitrary positive logical Feature sizes. Unsupported concrete sizes, head counts, sequence metadata, gradient modes, workspace, register pressure, or memory receive stable specialization rejection before cache lookup or code generation.
- The compiler performs no implicit transpose, `.contiguous()`, device move, dtype conversion outside declared seams, padding, or fallback execution.
- One generic dense affine TileLang lowering is the semantic fallback for every H20-supported Scanable Program. Failure to match an optimization cannot reject such a program.
- Proven generic subgraph optimizations may fuse adjacent scales, recognize Product Injection as GEMM, factor affine Rank-One sequences in WY/KKT-style forms, fuse shared State Contractions, and choose schedules from structural and concrete metadata. They cannot inspect model names, frontend identity, input names, or complete known profile identity.
- Semantic source identity includes compiler schema version, canonical Algorithm IR, canonical Backward IR, execution class, concrete tensor metadata, sequence representation, active differentiable bindings, final-State materialization mode, resolved Target architecture, and relevant compilation options.
- Semantic source identity excludes Python object identity, source locations, diagnostic messages, device index, runtime offset values, and tuning measurements.
- Binary artifact identity adds compiler, TileLang, CUDA ABI, and compatible driver/toolchain fingerprint.
- Performance-tuning identity is isolated by exact H20 product variant, resource fingerprint, software stack, and clock policy. Tuning results are never shared merely by compute capability or architecture.
- Shared artifacts use process-safe locking and atomic publication. Transient compilation, driver, and resource failures remove temporary artifacts and never create persistent negative cache entries.
- `LinearAttentionEngine` remains unchanged and outside the new compiler guarantee. It is not routed through Stateful Operator in this scope.
- `GDNEngine` is removed, including export, implementation, tests, documentation, and repository callers. GDN remains an explicit Algorithm IR example and independent-reference profile.
- GLA, RetNet recurrent, selective Axis-Scale state evolution, and GDN are acceptance profiles, never lowering identities or library factories.

## Testing Decisions

- Test through the highest seam: construct explicit Algorithm IR, create `StatefulOperator`, invoke with real keyword-bound tensors, consume `ExecutionResult`, and observe outputs, optional State, gradients, continuation, errors, cache behavior, and performance. Tests must not call lowering helpers as substitutes for this seam.
- Use independent PyTorch recurrences as numerical references. Reference execution is test infrastructure, not a Stateful Operator backend or production fallback.
- Test every Input Expression node and VJP against PyTorch autograd in FP32; test BF16/FP32 promotion and output casts at observable tensor boundaries.
- Test Typed Constant canonical bits, signed zero distinction, NaN/Inf rejection, deterministic identity, local constant folding, commutative operand ordering, and preserved association.
- Test Axis Scale on scalar, first Feature role, second Feature role, and both roles. Use asymmetric non-square State values so left/right mistakes are observable.
- Test general Rank-One propagation with `left != right`, positive and negative coefficients, both Feature roles, and gradients to every operand.
- Test Product Injection with disjoint Feature-role factors, full-State direct factors, explicit shared-role elementwise expressions, Batch/head scalars, multiple injections, empty injection tuple, and invalid role coverage.
- Test State Contraction along each Feature role, BF16/FP32 outputs, multiple independent outputs, output-only subsets, and rejection of extra Feature roles or multi-axis contraction.
- Test propagation order with deterministic noncommuting nodes. Verify empty and explicit identity canonicalize equally while reordered noncommuting nodes remain distinct.
- Test Head Mapping identity and contiguous group broadcast in forward and inverse backward reduction. Reject nondivisible and undeclared mappings before code generation.
- Test the same Tensor object bound to multiple names and verify accumulated gradient. Reject distinct overlapping views; accept nonoverlapping views.
- Test training/inference specialization using grad mode, differentiable bindings, initial-State gradients, and non-differentiable rejection. Verify active binding set changes specialization identity.
- Test dense positive lengths including `1`, `63`, `64`, `65`, `127`, and `129`. Verify inactive tail identity in output, final State, every input gradient, and initial-State gradient.
- Test packed offsets with unequal lengths, empty interior sequences, leading/trailing empty sequences, and all-empty invocations. Verify sequence isolation, output token order, per-sequence State order, and final-to-initial State cotangent identity for empty sequences.
- Reject invalid packed offset dtype, device, shape, start, end, monotonicity, and mismatched token counts with stable error codes and paths.
- Reuse one compiled packed specialization with different offset contents but identical metadata. Verify offset values do not change semantic or binary cache identity.
- Test omitted initial State against explicit zero State, requested/unrequested final State, output-only/final-only/joint losses, and prefix/suffix continuation against full execution within declared profile tolerance.
- Test FP32 State remains the visible State dtype and that no chunk-boundary storage cast changes results. Test output dtype per State Contraction and input-gradient dtype per Tensor Input.
- Test Algorithm IR and execution class without a GPU through internal unit seams only where required; public tests must remain centered on `StatefulOperator` and structured errors.
- Test stable `StatefulCompilationError` categories, canonical public-IR paths, JSON-safe details, deterministic first-root-cause ordering, wrapped compiler causes, and absence of negative caching for transient failures.
- Test H20 product-family and capability resolution. Reject H100, CPU, ROCm, unknown same-capability products, and unidentifiable MIG without fallback.
- Test all H20 Feature pairs `(64,64)`, `(64,128)`, `(128,64)`, and `(128,128)` in forward and backward. Reject unsupported logical sizes as specialization failures, not Recurrent classification.
- Test canonical contiguous dense and packed layouts. Reject noncontiguous tensors before cache lookup/codegen and verify no hidden copy occurs.
- Test generated execution structurally for parallel chunk-summary construction and composition. Correct numerical output from token-serial or eager execution is a failure.
- Test GLA, RetNet recurrent, selective Axis-Scale state evolution, and GDN as explicit generic-node profiles. Each profile covers dense and packed execution, initial/final State, output/final/joint cotangents, all declared gradients, grouped heads where applicable, unaligned tails, and independent references.
- Add one model-independent compositional pressure profile using ordered `AxisScale → RankOnePropagation(left != right) → AxisScale`, at least two Product Injections, at least two State Contractions, `(64,128)` State, grouped Head Mapping, packed variable lengths, empty sequences, and inactive tails. It must compile without adding any whole-program pattern.
- Inspect compiler decisions to prove the pressure profile uses generic dense affine semantics and compiler-derived Backward IR. Tests may assert the absence of model/profile dispatch and token-serial paths, but must not lock template source text or internal dataclass layout.
- Freeze benchmark profile, shape, mode, software, and clock configuration before implementation comparison. Run on an otherwise idle H20 with no competing GPU process.
- Benchmark old and new implementations in interleaved order with at least ten independent measurement batches per side, fixed warmup/repetitions, and paired batch medians.
- Compute a 95% bootstrap confidence interval for the paired latency ratio. Pass when the interval upper bound is at most `1.10`; fail when the interval lower bound exceeds `1.10`; otherwise add batches and rerun.
- Apply the same `1.10×` upper-bound rule to peak allocated memory. Record first compilation time without using it as a merge gate.
- Give the compositional pressure profile explicit absolute timeout and peak-memory limits because it has no legacy baseline.
- Run the CPU-safe semantic/cache/error suite, focused H20 profile suite, packed and tail suite, performance gate, and then the existing full functional GPU suite before acceptance.

## Out of Scope

- Changing or routing `LinearAttentionEngine` through Stateful Operator.
- Changing ordinary `AttentionEngine`, decode attention, MLA, score modifiers, mask modifiers, Online Functions, or their `CustomIO` behavior.
- Preserving or replacing `GDNEngine` with another compatibility wrapper.
- Public model-specific GLA, RetNet, Mamba2, or GDN factories.
- Model-name, frontend-name, input-name, or whole-profile lowering dispatch.
- FlashQLA/GDN kernels as a Stateful Operator execution backend.
- Arbitrary Python transition, injection, readout, composition, or backward callbacks.
- Public construction or override of Backward IR.
- A public compiler-analysis entry point or stable diagnostic-report JSON.
- More than one State or State with a Feature rank other than two.
- Coupled multi-State transitions or normalized numerator/denominator State.
- General einsum, arbitrary tensor contraction, multi-axis State Contraction, or readout dependency graphs.
- Input Expression division, Log, clamp, comparisons, conditional selection, arbitrary activations, or control flow.
- General low-rank propagation with a public rank dimension; the first schema exposes single Rank-One propagation only.
- Noncontiguous or arbitrary-stride layouts, implicit copies, implicit transposes, padding-based packed execution, or automatic device moves.
- Distinct overlapping-storage Tensor views.
- Stateful Operator production execution on CPU, ROCm, H100, non-H20 CUDA products, or unidentified MIG devices.
- FP16 or FP64 production execution, BF16 State storage, integer State algebra, NaN/Inf Constants, or mixed FP16/BF16 inputs.
- H20 Feature dimensions outside `{64,128} × {64,128}` in the first backend capability.
- Persistent mutable State stored inside `StatefulOperator` instances.
- Bitwise equivalence across floating-point reassociation, factorized optimization, or continuation invocation boundaries.
- Runtime offset values as compilation or tuning cache identity.
- Sharing tuning records across product variants, software stacks, resource fingerprints, or clock policies.
- Making first compilation latency a merge gate.

## Further Notes

- The normative language is the project glossary. Algorithm names are acceptance profiles only; they are not IR terms or lowering identities.
- The clean public seam is `StatefulOperator.__call__`. Internal node and Backward IR unit tests support diagnosis, but acceptance evidence crosses the public invocation seam.
- Dense affine summary is the canonical mathematical fallback. WY, KKT, low-rank factorization, GEMM recognition, fusion, and schedule selection are equivalent compiler optimizations.
- Scanability and Target feasibility are deliberately separate. A Scanable Program may be rejected for an unsupported concrete H20 specialization without becoming a Recurrent Program.
- Packed offsets are invocation metadata rather than Tensor Inputs because they define execution segmentation, are not differentiable model data, and must remain dynamic across compiled calls.
- The first schema is intentionally narrow in State rank and expression vocabulary but deep in behavior: every legal composition receives typing, scan analysis, forward lowering, Backward IR, dense/packed execution, State continuation, diagnostics, and H20 verification.
- ADR 0001 remains `proposed` until the generic dense affine path, compiler-derived Backward IR, one real profile, and the compositional pressure profile pass H20 correctness and performance review. Full merge still requires every acceptance criterion in this specification.
