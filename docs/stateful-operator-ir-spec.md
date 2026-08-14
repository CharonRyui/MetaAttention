# Stateful Operator IR Refactor Specification

Status: `ready-for-agent`

## Problem Statement

MetaAttention currently exposes stateful sequence operators through `LinearAttentionEngine` modifier slots such as `q_mod`, `k_mod`, `v_mod`, `decay_mod`, and `CustomIO`. These slots are useful frontend conveniences, but they make tensor modification—not state evolution—the compiler’s central abstraction.

This works naturally for Linear Attention, Gated Linear Attention, and RetNet because their implementations resemble modified Q/K/V operations. It is less natural for Mamba2, whose defining behavior is a selective state transition and state readout. The current Mamba2 frontend therefore expresses its behavior through decay and value modifiers even though those modifiers do not describe the model’s mathematical structure.

The existing lowering also combines several responsibilities: modifier graph conversion, algorithm assumptions, chunk-local recurrence, chunk-boundary state propagation, output computation, backward generation, scheduling, and TileLang code generation. As more stateful algorithms are added, extending modifier slots risks producing a parameterized template with model-specific escape hatches instead of a compiler abstraction.

Users need a stable way to define supported stateful sequence operators according to their mathematical state transition and readout while retaining the current optimized chunked backend, training support, and existing model behavior. Unsupported algebra must fail before kernel compilation rather than silently falling back to a slow recurrence or a separate legacy lowering path.

## Solution

Introduce a backend-independent Algorithm IR for stateful sequence operators. The Algorithm IR describes typed tensor inputs, a named state tuple, a structured affine state transition, and a post-transition state readout. It expresses model mathematics and remains separate from chunking, scan selection, tiling, fusion, state materialization, and backend code generation.

The first phase supports structured affine transitions of the form:

`next_state = propagation(inputs, prior_state) + injection(inputs)`

Propagation is restricted to compiler-recognized identity, elementwise-scale, and diagonal-scale forms. State injection is restricted to outer product, elementwise product, direct input, and zero. The compiler derives composition and backward behavior from these IR nodes; users cannot supply arbitrary transition functions or independent backward definitions.

Provide a typed, explicit IR construction interface for new stateful operators. Model-specific frontends wrap this interface so ordinary users do not need to construct every node manually. Migrate Gated Linear Attention, RetNet recurrent, and Mamba2 to this shared Algorithm IR while retaining the existing optimized lowering strategy.

Keep the existing `LinearAttentionEngine` modifier constructor during this phase as a deprecated compatibility adapter. The adapter captures and validates supported modifier expressions, converts them into the same Algorithm IR, and then uses the same lowering path as IR-first operators. It must reject expressions that cannot be represented by the supported IR. It must not retain a second legacy lowering, silently execute an eager recurrence, or create model-specific fallback behavior.

The core operator accepts an optional named initial state tuple, defaulting to zero, and may return the named final state tuple for streaming continuation. Compatibility calls continue to return token outputs only unless the new state interface is used.

Normalization is explicitly unsupported in this phase. The design reserves named state tuples for future stateful numerator/denominator formulations, but this specification does not define denominator accumulation, epsilon, clamp, or detach semantics.

## User Stories

1. As a model author, I want to describe a stateful sequence operator using a state transition and readout, so that the compiler interface matches the model’s mathematics.
2. As a Linear Attention author, I want to express state injection as an outer product, so that `KᵀV` accumulation is represented directly.
3. As a Gated Linear Attention author, I want to express elementwise state decay and outer-product injection, so that gating is not hidden in a generic modifier slot.
4. As a RetNet author, I want to express recurrent decay as structured state propagation, so that the compiler can recognize its scan algebra.
5. As a Mamba2 author, I want to express selective diagonal propagation and input-derived injection, so that Mamba2 does not masquerade as modified Linear Attention.
6. As a model author, I want readout to consume the updated state, so that the current token’s state injection contributes to the current token’s causal output.
7. As a model author, I want typed explicit IR constructors, so that unsupported structures are rejected clearly instead of being discovered during generated-kernel execution.
8. As a model frontend maintainer, I want to wrap explicit IR constructors in concise model-specific functions, so that common operators remain easy to instantiate.
9. As a compiler maintainer, I want a finite set of transition nodes, so that associativity, composition, scanability, and backward rules are compiler-known properties.
10. As a compiler maintainer, I want arbitrary Python transition functions to be rejected, so that hidden control flow and unsupported state mutation cannot compromise parallel lowering.
11. As a compiler maintainer, I want propagation forms to be explicit, so that identity, elementwise scale, and diagonal scale can select valid lowering behavior.
12. As a compiler maintainer, I want injection forms to be explicit, so that outer-product and elementwise state construction can be analyzed and lowered independently.
13. As a compiler maintainer, I want Algorithm IR separated from lowering strategy, so that mathematical meaning does not depend on chunk size, tile layout, warp count, or backend.
14. As a backend maintainer, I want to reuse the existing chunked forward and backward kernel skeletons, so that this semantic refactor does not simultaneously rewrite GPU scheduling.
15. As a backend maintainer, I want unsupported IR combinations to fail before kernel compilation, so that no slow or numerically different fallback is selected silently.
16. As a training user, I want backward graphs derived from the same IR as forward graphs, so that forward and gradient semantics cannot drift apart.
17. As a training user, I want gradients for Q, K, V, decay inputs, and additional tensor inputs to remain correct, so that existing models continue training after migration.
18. As a compiler maintainer, I want each supported IR node to define a vector-Jacobian product, so that backward generation is compositional and testable.
19. As a frontend author, I want named typed tensor inputs instead of stateful-path `CustomIO`, so that shape roles, dtype, and gradient requirements are explicit.
20. As an AttentionEngine user, I want the non-stateful AttentionEngine `CustomIO` behavior to remain unchanged, so that this refactor does not affect score-, mask-, or online-function compilation.
21. As a grouped-head model author, I want head relationships represented by an explicit Head Mapping, so that legal broadcasts and reductions are checked rather than inferred accidentally.
22. As a compiler user, I want invalid head divisibility or group relationships rejected with a compile-time diagnostic, so that indexing errors do not surface inside a GPU kernel.
23. As a streaming inference user, I want to provide an initial state tuple, so that a sequence can continue from a previous invocation.
24. As a streaming inference user, I want to request the final state tuple, so that I can persist state for the next invocation.
25. As a batch training user, I want omitted initial state to mean a zero state, so that ordinary full-sequence calls remain concise.
26. As a concurrent caller, I want the engine itself to remain stateless, so that state is not shared accidentally across requests or autograd graphs.
27. As an operator author, I want state entries to be named and ordered, so that transitions and readouts reference states deterministically.
28. As a future normalized Linear Attention author, I want the IR state model not to assume exactly one tensor forever, so that a later numerator/denominator extension does not require replacing the public state model.
29. As a current user, I want existing `LinearAttentionEngine` calls to keep working during this migration phase, so that adopting the new IR is not an immediate breaking change.
30. As a current user, I want deprecated modifier calls to produce a clear migration warning, so that I know the compatibility interface will be removed.
31. As a compiler maintainer, I want compatibility modifiers converted into Algorithm IR, so that there is only one semantic source and one lowering path.
32. As a compiler maintainer, I want an unsupported legacy modifier to produce a precise conversion error, so that it cannot become a permanent escape hatch.
33. As a project maintainer, I want new code and new tests to use only the IR-first interface, so that dependency on the deprecated constructor decreases rather than grows.
34. As a Gated Linear Attention user, I want migrated output and gradients to match the previous generated operator within dtype-specific tolerances, so that the refactor preserves behavior.
35. As a RetNet user, I want migrated output and gradients to match the previous generated operator within dtype-specific tolerances, so that recurrent retention remains compatible.
36. As a Mamba2 user, I want migrated output and gradients—including gradients of selective parameters—to match the previous generated operator within dtype-specific tolerances, so that the abstraction pressure test is meaningful.
37. As a correctness reviewer, I want migrated operators also compared with independent PyTorch references, so that new and legacy implementations cannot agree on the same bug unnoticed.
38. As a numerical-kernel maintainer, I want tolerance-based rather than bitwise compatibility, so that legal floating-point reassociation and backend variation remain possible.
39. As a performance maintainer, I want the existing schedule and kernel skeleton retained in this phase, so that semantic regressions can be distinguished from scheduling regressions.
40. As a documentation reader, I want Algorithm IR and lowering strategy described as separate concepts, so that future changes are placed at the correct seam.
41. As a documentation reader, I want Mamba2 described through its state transition rather than Q/K/V modifier terminology, so that examples teach the intended abstraction.
42. As a maintainer, I want generated-code cache identity to include the structural Algorithm IR and relevant input metadata, so that semantically different operators cannot collide.
43. As a maintainer, I want structurally equivalent IR to produce stable cache identity, so that frontend wrapper details do not cause needless recompilation.
44. As a user, I want malformed shapes, dtypes, state references, and unsupported nodes diagnosed before backend code generation, so that failures are actionable.
45. As a user, I want diagnostics to identify the unsupported IR node or relationship, so that I can reformulate the operator without reading generated TileLang.
46. As a project maintainer, I want the compatibility interface removed in a subsequent migration phase, so that the project does not maintain two permanent authoring interfaces.
47. As a project maintainer, I want ordinary attention, decoding, mask-based retention, and MLA behavior unchanged, so that this refactor remains scoped to stateful operators.
48. As a future backend author, I want Algorithm IR independent of TileLang, so that another backend can consume the same stateful operator semantics.

## Implementation Decisions

- Build one deep stateful-operator module whose external interface accepts a validated Algorithm IR plus compile options and returns a callable operator. The same interface is the primary behavioral test seam.
- Algorithm IR is backend-independent. It describes input transforms, a named state tuple, structured affine transitions, post-transition state readout, typed tensor inputs, and Head Mapping. It does not contain chunk sizes, tile shapes, scan implementation, warp configuration, memory placement, or backend source code.
- The first-phase state transition is affine: propagation of prior state plus input-derived injection.
- Supported propagation nodes are identity, elementwise scale, and diagonal scale. The compiler determines their composition behavior; callers cannot assert associativity manually.
- Supported state-injection nodes are outer product, elementwise product, direct input, and zero.
- Readout occurs after transition. The updated state is visible to the output at the same sequence position.
- The initial readout set must cover the existing stateful examples, including matrix-style readout for Linear Attention/retention and elementwise or contraction-based Mamba2 readout. Readout forms remain typed and enumerated rather than arbitrary Python callbacks.
- State is represented as an ordered tuple of named tensor states. Each state declares logical shape roles, storage dtype, and accumulation dtype. References are by state name.
- The initial lowering strategy may accept only the state tuple configurations required by the three migration targets. Other valid Algorithm IR configurations receive an explicit unsupported-lowering diagnostic.
- Initial state is optional and defaults to zeros. Final state is returned only when requested. State is caller-owned; the engine does not mutate persistent internal state between calls.
- Tensor inputs replace `CustomIO` only within stateful operators. Each input declares a name, logical shape roles, dtype, and whether gradients are required.
- Head Mapping explicitly relates query heads, state heads, key/input groups, and value heads. Valid divisibility, broadcasting, and gradient-reduction relationships are checked before lowering.
- Input transforms may reuse the existing symbolic expression machinery for supported elementwise leaf transformations, but symbolic expression DAGs are not themselves state-transition IR.
- The IR-first authoring interface uses typed explicit constructors and factory functions. It does not trace arbitrary Python transition/readout functions and does not parse Python source or AST.
- Every differentiable IR node owns a vector-Jacobian-product rule. The compiler derives backward semantics from the forward Algorithm IR. Frontends cannot provide an independent custom backward.
- Existing chunked forward, state propagation, output, and backward kernel skeletons remain the first lowering strategy. Refactoring them into a separate Schedule IR is deferred.
- Existing tuning controls and generated-kernel compilation remain available through the stateful operator module. Schedule configuration is not embedded into Algorithm IR.
- Generated-code caching uses a canonical structural representation of Algorithm IR plus relevant tensor metadata, backend identity, and compile/tuning options. Python object identity and wrapper function identity are not cache keys.
- Validation runs before backend code generation. It checks state references, shape-role compatibility, dtypes, propagation/injection/readout node support, Head Mapping, differentiability requirements, initial-state compatibility, and lowering support.
- Unsupported behavior is a compile-time error. There is no eager recurrence fallback, no separate legacy kernel path, and no model-name special case.
- The existing modifier-based `LinearAttentionEngine` constructor remains temporarily as a compatibility adapter. It emits a deprecation warning, converts supported modifier DAGs and `CustomIO` declarations into Algorithm IR, and invokes the same stateful operator module.
- Compatibility conversion accepts only modifier expressions that map to supported input transforms, propagation, injection, or readout behavior. Failed conversion identifies the modifier and unsupported expression.
- New production code, examples added after this refactor, and new tests use the IR-first interface. The compatibility adapter is tested only for migration parity and warning behavior.
- Gated Linear Attention, RetNet recurrent, and Mamba2 frontends migrate to Algorithm IR in this phase. They must not select a model-specific lowering branch.
- Mamba2 is the abstraction pressure test: its selective state transition and readout must lower without a `Mamba2SpecialCase` or equivalent model-name dispatch.
- Normalization is unsupported in this phase. No denominator state convention, epsilon, clamp, detach, or normalized readout contract is introduced.
- The compatibility modifier interface is scheduled for removal in the next explicit migration phase after repository callsites and published examples have moved to IR-first construction.
- Documentation uses the domain terms Algorithm IR, state transition, structured affine transition, state injection, state readout, state tuple, Tensor Input, Head Mapping, frontend sugar, IR-derived backward, and lowering strategy consistently.

## Testing Decisions

- Test primarily through the highest seam: construct or obtain a callable stateful operator, invoke it with real tensors, and observe outputs, optional final state, gradients, warnings, and validation errors. Tests should not assert generated source text, internal dataclass layout, helper call counts, or template string fragments.
- Reuse the existing functional GPU-test style and independent PyTorch references. Existing tests already compare official example factories and direct engine calls with dtype- and operator-specific tolerances, including backward gradients where supported.
- Add IR-first functional coverage for Gated Linear Attention, RetNet recurrent, and Mamba2. Each test compares output with the independent PyTorch reference.
- Add differential migration coverage that runs equivalent IR-first and deprecated modifier-based operators on identical seeded inputs and compares outputs within existing dtype-specific tolerances.
- Compare gradients for every differentiable primary input and every differentiable Tensor Input. Mamba2 coverage includes selective parameters used in propagation and injection.
- Do not require bitwise equality. Use established project tolerances for float16 and bfloat16, tightening only where current reference behavior supports it.
- Test post-transition readout with a short deterministic sequence where the current token’s injection produces an observable current-position output. This prevents accidental pre-transition readout.
- Test omitted initial state against an explicit zero state.
- Test sequence continuation: running a prefix while requesting final state, then running the suffix with that state, must match a single full-sequence invocation within dtype-specific tolerances.
- Test requested final-state shape, dtype, names, and numerical value against a simple reference recurrence.
- Test that callers which do not request final state receive the existing output-only contract.
- Test that engine instances do not retain state across independent calls.
- Test valid and invalid Head Mapping relationships, including equal heads, supported grouped heads, non-divisible groups, and required gradient reductions.
- Test validation failures for missing state references, duplicate state names, incompatible initial-state shapes/dtypes, unsupported propagation/injection/readout nodes, and unsupported state-tuple lowering combinations.
- Test that normalization constructs are rejected explicitly rather than miscompiled or ignored.
- Test compatibility-adapter deprecation warnings and successful conversion for every modifier pattern used by the three migration targets.
- Test that an unsupported modifier DAG fails during compatibility conversion and never reaches kernel compilation.
- Add focused CPU-safe tests for Algorithm IR validation, canonicalization, structural cache identity, shape-role inference, Head Mapping validation, and VJP graph construction where these behaviors do not require executing a GPU kernel.
- VJP unit tests verify mathematical behavior of supported nodes at observable tensor boundaries, preferably against PyTorch autograd on small tensors. They do not inspect the internal graph representation.
- Cache tests verify that structurally equivalent IR definitions share identity and semantically different transitions do not. They do not assert a particular hash algorithm or filename.
- Retain the existing official example import test so all migrated factories remain importable without compiling kernels.
- Run the existing CPU-safe unit suite after implementation.
- Run the specific GPU functional cases for Gated Linear Attention, RetNet recurrent, and Mamba2 on supported hardware. Then run the full functional GPU suite to detect accidental changes to shared compilation infrastructure.
- Performance benchmarking is not an acceptance gate for the semantic refactor, but the generated operators must continue using the existing optimized lowering rather than an eager fallback. Any observed material regression is a blocker for release and must be diagnosed separately from numerical correctness.

## Out of Scope

- Rewriting chunking, scan strategy, tiling, warp scheduling, memory layout, or TileLang kernel skeletons into a new Schedule IR.
- Adding a general arbitrary-function state machine DSL.
- Allowing users to declare associativity, composition, or custom backward rules.
- Supporting arbitrary tensor contractions or a general-purpose einsum IR.
- Supporting normalized Linear Attention, denominator states, epsilon/clamp policies, or detach semantics.
- Migrating `retention_parallel`, which uses the score/mask-based AttentionEngine rather than the state-transition seam.
- Refactoring ordinary AttentionEngine, softmax attention, decode attention, MLA, score modifiers, mask modifiers, online functions, or their `CustomIO` behavior.
- Supporting every theoretically valid named state tuple in the first backend lowering.
- Introducing persistent mutable state inside engine instances.
- Requiring bitwise numerical identity across old and new generated kernels.
- Replacing tuning infrastructure or changing benchmark configurations.
- Removing the deprecated modifier constructor in this phase; removal belongs to the subsequent migration phase.

## Further Notes

- The first phase is intentionally a semantic refactor over the existing backend. Changing Algorithm IR and GPU scheduling simultaneously would make correctness, performance, and abstraction regressions difficult to isolate.
- Mamba2 is a required acceptance case because it checks that the common abstraction is state transition plus readout rather than modified K/V accumulation. Passing Mamba2 through a model-name special case does not satisfy this specification.
- The compatibility adapter is temporary frontend sugar, not a second compiler seam. Its deletion should remove adapter complexity without changing Algorithm IR or lowering behavior.
- The project glossary defines the canonical vocabulary for this work. Implementation and documentation should avoid using `q_mod`, `k_mod`, `v_mod`, or `decay_mod` as names for core IR concepts.
- Recommended implementation sequence:
  1. Define and validate typed Algorithm IR nodes, state tuples, Tensor Inputs, Head Mapping, canonicalization, and VJPs.
  2. Add the stateful-operator compilation seam and adapt the existing lowering strategy to consume validated IR.
  3. Implement initial/final-state behavior and IR-derived backward through the existing kernel skeletons.
  4. Migrate Gated Linear Attention and RetNet recurrent.
  5. Migrate Mamba2 without model-specific lowering.
  6. Add the deprecated modifier compatibility adapter and differential parity coverage.
  7. Update public stateful-operator documentation and migration guidance.
