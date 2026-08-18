---
status: proposed
---

# Compile Stateful Programs Compositionally

## Decision

Stateful Operator is a compositional compiler, not an algorithm catalog. It lowers every H20-supported Scanable Program from a finite typed algebra through one parallel chunked forward and backward pipeline. Scanability is target-independent: a program is Scanable when its affine State summary has storage bounded only by State Feature dimensions, never sequence length. The canonical summary may be a dense affine State map.

The first compiler schema contains one matrix State with two ordered Feature roles. Model authors compose that State from typed Input Expressions, role-directed Axis Scale propagation, general Rank-One propagation, Product Injections, explicit Head Mapping, and named single-role State Contractions. GLA, RetNet recurrent, selective State evolution, and GDN are acceptance profiles rather than compiler identities.

Every forward node owns a compiler VJP. Their composition forms an internal canonical Backward IR, which is the sole semantic source for generated backward. The production path has no eager recurrence, token-serial fallback, dedicated GDN execution, hand-written profile backward, model-name dispatch, or whole-profile pattern selection.

The compiler may apply proven local algebraic optimizations such as scale fusion, product/GEMM recognition, affine factorization, shared-readout fusion, and shape-specific schedules. Generic dense affine lowering remains the semantic fallback; failure to match an optimization cannot reject a Target-supported Scanable Program.

The first production Target is CUDA H20. It supports canonical contiguous dense and token-major packed layouts, BF16/FP32 inputs and outputs, FP32 State, and Feature pairs in `{64,128} × {64,128}`. Concrete resource limits may reject a specialization without changing its Scanable classification. H20 identity uses both approved product family and CUDA capability, and tuning artifacts are isolated by exact product and environment fingerprint.

## Considered Options

A catalog of whole-program templates was rejected because every unseen algorithm would require a new dispatcher case and semantic implementation. A mixed generic/specialized architecture was rejected because dedicated GDN or model-specific paths would preserve multiple semantic sources. A sequential generic recurrence was rejected because parallel chunked execution is part of the Stateful Operator promise. An unrestricted tensor/state DSL was rejected because arbitrary contractions, callbacks, and State ranks cannot receive complete scan, VJP, lowering, and performance guarantees in the first schema.

`LinearAttentionEngine` remains unchanged outside this compiler guarantee. Routing all legacy modifier behavior through the new finite language would either enlarge this change substantially or retain a fallback path. The dedicated `GDNEngine` is removed because GDN must be an ordinary composition of generic nodes.

## Consequences

Algorithm IR remains independent of concrete size, layout, Target, schedule, and tuning. The public interface stays small: construct `StatefulOperator`, bind runtime tensors by keyword, optionally pass `StateTuple` and packed sequence offsets, and always receive immutable `ExecutionResult`. Compiler analysis is internal; structured Stateful Compilation Errors expose stable codes and public-IR paths without freezing internal reports.

The first schema deliberately supports exactly one matrix State, a finite elementwise expression language, identity/contiguous-group Head Mapping, Product Injection, Axis Scale, Rank-One propagation, and single-role State Contraction. Adding a node or State shape later requires a schema change plus typing, canonical summary, VJP, dense and packed lowering, errors, cache identity, and H20 evidence.

The implementation burden is substantial: generic dense affine forward and backward, packed segmentation including empty sequences, inactive-tail identity, State continuation, canonical caching, resource validation, and a compositional pressure profile are all required before merge. Correctness alone is insufficient; four real profiles must remain within the predeclared 10% latency and memory confidence-interval gate on an exclusive H20.

This ADR remains proposed until the generic dense affine path, compiler-derived Backward IR, one real profile, and the model-independent compositional pressure profile pass H20 correctness and performance review.
