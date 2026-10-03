# AttentionEngine API


## Custom Attention Level API
This level API is designed for users to define their own attention mechanism. Users needs to define the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `score_mod`: elementwise modification on attention scores
- `mask_mod`: mask modification on attention scores
- `online_func`: online function for attention scores
- `custom_fwd_inputs`: custom inputs for attention mechanism
```py
mod = AttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    score_mod: Callable[[Tensor, CustomIO, int, int, int, int], Tensor],
    custom_fwd_inputs: CustomIO,
    online_func: OnlineFunc,
    mask_mod: Callable[[int, int, int, int], Bool]
)
```

For compiled Attention module, users can use pytorch-compatible API to interact with the module.
```py
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_inputs: Optional[List[torch.Tensor]]
)
output.backward(do)
```

### OnlineFunc

OnlineFunc is a class that defines the online function for attention scores, such as online softmax and retention.
```py
class OnlineFunc:
    def __init__(self):
        pass
    def online_fwd(scores, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        pass
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        pass
    def forward(scores, online_rowscales, b, h, q_idx, kv_idx) -> Tensor:
        pass
    def backward(dp, scores, final_rowscales, doosum, b, h, q_idx, kv_idx) -> Tensor:
        pass
```
Examples can be found in the [Getting-started Example](./getting_started_example.md).

### score_mod
`score_mod` takes the following inputs:
- `score`: attention scores
- `custom_fwd_inputs`: custom input tensors
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`score_mod` returns the modified attention scores.
```
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx) -> Tensor:
    return new_score
```

Examples can be found in the [Getting-started Example](./getting_started_example.md).

### mask_mod
`mask_mod` takes the following inputs:
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`mask_mod` returns Bool value to indicate whether the attention score should be masked.
```
def mask_mod(b, h, q_idx, kv_idx) -> Bool:
    return True
```


## Stateful Operator API

Stateful sequence operators are authored as model-independent Algorithm IR and
compiled explicitly from logical metadata. Compilation accepts fixed Head and
Feature role sizes, an explicit Target, one physical mode, and independent
optimization, autotuning, and backward choices. It never accepts example
Tensors. Batch size, dense Sequence length, packed logical-sequence count,
packed token count, and offsets remain dynamic at invocation.

```py
from attn_engine import StatefulOperator, Target

operator = StatefulOperator(algorithm)
compiled = operator.compile(
    role_sizes={
        query_heads: 2,
        key_heads: 2,
        value_heads: 2,
        state_heads: 2,
        key_dim: 64,
        value_dim: 128,
    },
    target=Target(
        # Optional compiled envelope; larger runtime workloads fail explicitly.
        workspace_bytes=1 << 30,
    ),
    mode="dense",
    optimized=True,
    autotune=False,
    backward=True,
)
result = compiled(
    query=query,
    key=key,
    value=value,
    gate=gate,
    return_final_state=True,
)
output = result.outputs["output"]
state = result.final_state["memory"]
```

One dense compiled callable accepts different positive Batch and Sequence
sizes. A packed callable uses `mode="packed"`; invocation supplies contiguous
token-major inputs and a device-local contiguous int32 `sequence_offsets`
vector. Different offsets shapes and values, including empty sequences and
all-empty workloads, reuse the same artifact. Runtime tensors must preserve the
compiled role sizes, Algorithm IR dtypes, Target, mode, and canonical contiguous
layout. No implicit copy, transpose, dtype conversion, device move, replanning,
or recompilation occurs.

Every invocation returns immutable `ExecutionResult`, containing immutable
named outputs and optional named final State. Missing initial State means one
FP32 zero State per logical sequence. Final State is materialized only when
requested. Calling an uncompiled `StatefulOperator` raises; only the callable
returned by `compile()` executes.

With `backward=False`, grad-enabled invocation is rejected. With
`backward=True`, ordinary PyTorch autograd transports cotangents and saved
context through the compiler-selected forward/backward composition. The
compiler owns Backward IR and context semantics; no eager or framework-derived
fallback is used.

The first schema has one named matrix State with two ordered Feature roles. It
supports typed Input Expressions (`Input`, BF16/FP32 `Constant`, `Add`,
`Multiply`, `Negate`, `Exp`), role-directed Axis Scale, general Rank-One
propagation, Product Injection tuples, explicit Head Mapping, and single-role
State Contractions. Compiler analysis, Mathematical Plans, Lowered Candidates,
KernelConfig, context contracts, and caches are internal.

Schema v2 keeps every v1 form and adds what the remaining Flash Linear
Attention families need:

- **Several independent States** with one Transition each; States share one Head
  role and use at most two distinct Feature roles.
- **Rank-1 States** (`Batch, Head, Feature`) for scalar accumulators such as a
  linear-attention denominator.
- **Expression readouts**: `Readout(name, expression, output_dtype)` where the
  expression may combine `StateTerm` contractions, inputs, and arithmetic,
  including `Divide` by a denominator State, `Clamp`, and a non-State additive
  term. A `StateTerm` with `role=None` reads a rank-1 State without contraction.
- **More Input Expressions**: `Divide`, `Log`, `Sigmoid`, `Sqrt`, `Reciprocal`,
  `Clamp`, and `Shift(expr, lag)` for token shift. Transcendental and reciprocal
  forms compute in FP32, as `Exp` already did.
- **Independent States stay Scanable**: a Transition is driven by Inputs only,
  so the per-token summary remains affine and the parallel scan still applies.

Schema v3 keeps every v2 form and adds the vocabulary the product-Feature
families need (`docs/adr/0003-product-features-and-prior-state-readouts.md`):

- **Product Feature roles.** `ProductFeatureRole(left, right)` declares a Feature
  whose axis is the flattened product of two Features (named `left_x_right`); a
  State may use it as an axis, and `Outer(left, right, role)` builds that axis
  from two expressions. Based and Rebased keep their `[K*K, V]` numerator as an
  ordinary matrix State.
- **Feature reductions.** `Contract(operand, coefficient, role)` reduces one
  Feature axis and keeps a singleton axis, so a Program can write a coefficient
  term such as RWKV6's `(q * u * k)` bonus. Its `coefficient` may be feature-free.
- **Axis softmax.** `Softmax(operand, role)` normalizes one Feature axis, which is
  how a Program joins two States through a slot distribution (Gated Slot
  Attention) rather than through a fixed contraction.
- **Prior-State readouts.** `StateTerm(..., prior=True)` reads the State before
  the current token's transition, which pre-transition readouts such as RWKV6
  require.
- **Feature-free Injection factors.** An Injection factor may declare a Feature
  role its expression does not carry; the value broadcasts along that axis, so a
  constant `+1` Injection expresses a token count.

Schema-v2 Programs enter the reference lowering in
`attention_engine/attn_engine/_stateful_reference.py`. Dense inference whose
typed transitions are diagonal, with disjoint injection factors and supported
expression readouts, uses FP32 chunk summaries, boundary carries and independent
64-token replay. Identity propagation uses tiled GEMMs; Product Feature operands
are synthesized inside tiles rather than stored as token-sized outer products.
Prior-State terms and readout dependencies retain their authored semantics.
Packed and gradient execution retain the segmented affine scan. Programs within
the v1 language retain typed specialized TileLang plans.

Each Flash Linear Attention operator that is a per-token linear recurrence has
its Program in its own `examples/<family>.py` module, and
`examples/fla_coverage.py` records the coverage table that points at those
factories; `tests/functional/test_fla_coverage.py` checks parity with the library
operators.

The first production Target is CUDA H20. It accepts BF16/FP32 inputs and
outputs, FP32 State, canonical contiguous layouts, and Feature pairs in
`{64,128} × {64,128}`. Invalid IR, unsupported composition, Target mismatch,
invalid packed offsets, unsupported specialization, resource failure, and
codegen failure surface as structured `StatefulCompilationError` values. The
complete contract is in `docs/stateful-operator-ir-spec.md`.

### Structural inference schedules

Identity and scalar propagation use the existing fused scalar-factorized dense
kernel. A previously measured aligned scalar chunk schedule improved CUDA Graph
replay but regressed public-call latency; scalar inference now consistently uses
the raw dense kernel rather than that multi-launch schedule.

Diagonal multi-State inference supports batches, nonaligned tails, nonzero
initial States, FP32 final States and continuation. Readout expressions consume
contracted FP32 terms without materializing per-token matrix States. Unsupported
typed structures retain the generic lowering; selection does not inspect model
or example names.

Ordered first-axis Rank-One propagations with row, column or whole-State scales
use affine chunk summaries `S_out = P @ S_in @ diag(d) + b`, an FP32 boundary
carry pass and independent chunk replay. Operation order is preserved inside
each chunk; replay reads the incoming carry, not the outgoing summary. Multiple
injections and independent first-axis contractions share the same replay.
No inverse gate or BF16 State approximation is used in inference. Existing
single-readout backward execution retains its recorded-State reverse walk;
unsupported gradient compositions retain the generic lowering.

When every ordered propagation is a whole-State scalar scale or first-axis
Rank-One update, at most four rank/injection events and dimensions at most 128
use the compact WY triangular solve and block contractions. This path reads
each FP32 factor through its binding strides; low-precision factors retain the
existing FP32 promotion before the dot operands. It does not pack/copy operation
buffers or stack multiple injection/readout factors.
Zero/signed scalar gates, distinct left/right factors, nonzero initial States
and continuation retain the same recurrence. Noncommuting row/column scales
keep the existing affine-summary runtime; packed and backward paths are unchanged.

The workspace envelope charges token factors and expression outputs separately
from chunk carries: `B*T*bytes_per_token + B*(ceil(T/64)+1)*bytes_per_chunk`.
The extra chunk allowance covers final States. The runtime checks this bound
before launching; it no longer allocates an unused duplicate reservation buffer.

`testing/benchmark_stateful_plans.py --families based,rebased,rwkv6,gsa,hgrn,gdn,rwkv7,pressure --seqlen 512 2048 --fla --json /tmp/plans.json`
reports the selected lowerer, public-call timing, CUDA Graph replay, launches and
workspace. Stock FLA is timed as a performance baseline only; its outputs are
not a correctness oracle.

Measured on an idle H20, BF16, `B=1,H=8,T=2048,K=V=64`, five warmups and
median of twenty CUDA-event samples around the public callable with final State:

| Structure/example | Before end-to-end ms | After end-to-end ms | After Graph ms | FLA Graph ms |
| --- | ---: | ---: | ---: | ---: |
| Based (`K=16`) | 11.384 | 1.931 | 0.816 | 0.468 |
| ReBased (`K=16`) | 2.814 | 1.084 | 0.609 | 0.078 |
| Normalized linear | 0.766 | 0.874 | 0.203 | 0.068 |
| RWKV6 | 2.796 | 0.908 | 0.365 | 0.131 |
| GSA (`slots=16`) | 2.119 | 0.988 | 0.312 | 0.142 |
| HGRN (`H=1`) | 0.570 | 0.519 | 0.060 | 0.065 |
| GDN | 4.661 | 1.074 | 0.604 | 0.190 |
| RWKV7 State | 4.968 | 1.031 | 0.601 | 0.351 |
| Ordered composition pressure | 82.628 | 1.372 | 0.924 | unavailable |

Before Based and pressure needed an explicit 8 GiB workspace envelope; every
after case runs with the default 1 GiB envelope. Before pressure's auxiliary
output failed the independent FP32 tolerance; after output and auxiliary pass.
Normalized linear end-to-end latency regresses despite faster Graph replay.
FLA remains faster for most cases; its timing is not a correctness claim.
After results pass an independent FP32 token recurrence for outputs and every
final State at lengths 65, 512 and 2048 with nonzero initial States, plus batched
continuation split across nonaligned chunks.

The follow-up packing pass reuses single injection/readout bindings and leaves
unused Triton operation buffers at one element. On the same H20 workload, the
`T=2048` graph timings changed from `0.792` to `0.712 ms` for Based and from
`0.600` to `0.566 ms` for ReBased; profiler-attributed auxiliary launches fell

from `71` to `47` and `33` to `21`, respectively. GDN changed from `0.598` to
`0.592 ms` graph time. Output, final State, and continuation correctness remain
covered by the focused GPU suite.

The compiler-only compositional comparison was measured on an otherwise idle H20
with BF16 inputs, FP32 gates and State, `B=1`, `(row,column)=(64,128)`, `T=129`,
one public call with final State, three warmups, five CUDA-event samples per batch,
and ten alternating batches per side. The generic side used the same Algorithm IR
and `optimized=False`; compilation was excluded. Paired batch medians were:

| Composition | Optimized lowerer | Generic CUDA-event ms | Optimized CUDA-event ms | Generic / optimized | Optimized peak / generic peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Axis scales | `tilelang-chunked-stateful` | 5.106 | 0.233 | 21.95x | 0.28% |
| Axis + Rank-One | `tilelang-factored-recurrence` | 5.319 | 0.778 | 6.83x | 1.88% |
| Multiple injections | `tilelang-product-gemm` | 0.744 | 0.761 | 0.98x | 100.42% |
| Rank-One | `tilelang-factored-recurrence` | 5.131 | 0.543 | 9.46x | 1.41% |

The multiple-injection candidate is semantically valid and stays within the
predeclared 1.10x latency and memory upper-bound gate: its optimized/generic
CUDA-event ratio is 1.023 with a bootstrap 95% interval of [1.006, 1.050], and
its optimized/generic incremental-allocation ratio is 1.004. It is 2.3% slower
than generic at this workload, so the measurement supports retaining the
candidate for semantic coverage, not claiming a speedup. Full raw records are
in `benchmark_results/composition_profiles_final.json`; this artifact is local
measurement evidence, not a tracked API contract.

The follow-up dense-State pass also reuses identical broadcast value bindings
across States; it changes no numerical path and is covered by the same focused
GPU regression suite.

Dense inference selected as `chunked-stateful` or `product-feature-chunked`
now enters a direct executor with compile-time State/readout/layout recipes,
without constructing reference sequence segments or repeating eligibility
analysis. Reference execution remains the semantic path for unsupported
structures, packed programs, gradients, and unoptimized v2 programs.

The rank-1 diagonal/identity-readout structure (including HGRN) uses raw BF16/FP32
bindings and native vector affine scans. A sequence of at most 64 tokens uses
one output kernel; longer sequences use summary, carry, and output kernels.
Three FP32 chunk vectors are scratch; beyond 64 chunks, recursive group summaries
bound prefix depth and add smaller vector buffers. No promoted matrix, dense query
of ones, FP32 input copy, or token-sized State scratch is created. Output casting
and optional FP32 final-State writing occur in the output kernel. HeadMapping,
nonzero initial State, tails, zero/signed gates and continuation are preserved.

On H20, the HGRN matrix `B={1,2}`, `H={1,8,16}`,
`T={63,64,65,130,2048,8192}` and all three contracts produced 108 runnable
records. Of 106 before/after pairs, public latency geometric means decreased
by 82.5% (output-only), 82.2% (final-state), and 80.2% (continuation), with no
observed paired regression. Two B=2/H=16/T=8192 calls previously rejected by
the old resource estimate now run with the native workspace bound. At B=H=1,
T=64 output-only changed from 0.485 to 0.122 ms and CUDA activity count from
11 to 1. Measurements used three warmups and five repetitions; FLA is only a
performance baseline, and continuation includes two calls plus output concat.
The headless FLA HGRN adapter rejects H>1; those rows compare Meta before/after
only and do not establish a cross-framework speedup.

Matrix diagonal inference now binds authored pointwise expressions directly to
raw BF16/FP32 inputs. Row and column summary scales occupy separate vectors,
not a broadcast matrix. Prefix composition uses multiplication/addition only,
including zero or signed gates, and recursively groups long chunk sequences.
One-chunk calls bypass summary/carry allocation; output-only calls omit the last
summary because no following chunk or final State consumes it.

Readout schedules are compiler-selected structural candidates: bounded chunk
replay, shared score generation followed by block contractions when the column
action is identity, and a two-State contraction/slot-softmax/contraction region
with an explicit materialization-cut alternative. Scores are reused across equal
State/query/prior/bonus expressions. Softmax reduces the typed Feature axis, not
Sequence. Fusion eligibility uses the existing padded-axis/live-FP32 resource
bound. Candidate feasibility, tuning allocation checks, and runtime workspace
policy consume the same recipe-derived output and scratch estimates, including
every output dtype and final-State allowance. Dynamic B/T do not select a new
plan. Successful device constraints are cached per process/device/constraint;
tensor binding, layout, alias, and shape checks still run on each invocation.

Shared scores were slower than replay on the measured RWKV6 shape and are not
enabled by default; fusion likewise remains a candidate, not a universal rule.
The native replay/fused defaults passed independent CPU FP32 recurrences at
`B=2,H=8,T={1,63,64,65,130,2048,8192}` for RWKV6, GSA, and HGRN, including nonzero
initial States, output-only, all final States, and nonaligned continuation.
Focused public-call regressions cover unequal Feature widths, shared heads,
BF16 authored expression rounding, zero/signed gates, prior-State reads, shared
scores, and the fused/cut softmax alternatives.
Verification: 187 unit tests passed (3 GPU-marked tests skipped), 113 focused
GPU tests passed, and the additional rank-1 autotune regression passed. RWKV6,
GSA, and HGRN autotune invocations also passed output and
final-State recurrence checks. At `B=2,H=8,T=8192`, allocator peaks for replay,
shared-score, fused/cut, and vector paths all stayed below their compiled bounds.

Final H20 measurements use BF16, `B=1`, output-only, five warmups and twenty
samples. Public-call medians below include CPU dispatch; CUDA Graph replay does
not. Full records: `benchmark_results/diagonal_native_final.json`.

| Structure | H | T | Meta public ms | Meta Graph ms | FLA public ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RWKV6 | 1 | 64 | 0.263 | 0.064 | 0.303 |
| RWKV6 | 1 | 2048 | 0.355 | 0.084 | 0.312 |
| RWKV6 | 8 | 64 | 0.266 | 0.065 | 0.362 |
| RWKV6 | 8 | 2048 | 0.397 | 0.159 | 0.378 |
| GSA | 1 | 64 | 0.332 | 0.123 | 0.562 |
| GSA | 1 | 2048 | 0.501 | 0.137 | 0.505 |
| GSA | 8 | 64 | 0.333 | 0.130 | 0.568 |
| GSA | 8 | 2048 | 0.637 | 0.325 | 0.576 |
| HGRN | 1 | 64 | 0.119 | 0.014 | 0.113 |
| HGRN | 1 | 2048 | 0.175 | 0.020 | 0.115 |

FLA remains faster on the longer RWKV6/H=8 GSA/HGRN calls. HGRN/H>1 still has no
runnable FLA comparison. Profiler activity was unavailable in this final run;
the records leave kernel time and launch counts null rather than infer them.

The subsequent host-overhead pass compiles each native dense input's
`(Batch, Head, Sequence, Feature)` axis map once. Vector, normalized-pair, and
matrix-diagonal loaders reuse that immutable recipe rather than constructing a
full `BindingDescriptor` or searching logical roles on every call. The general
descriptor/reference interface is unchanged. Recipes retain only input names
and axis positions, never input tensors, pointers, strides, dynamic B/T, or
cross-call validation results. Current tensor strides and head extents are read
at each invocation, after the existing public runtime checks.

Same-process H20 measurements, BF16, `B=H=1`, output-only, ten warmups and median
of one hundred public-call samples; kernels and workspace policy are unchanged:

| Structure | T | Before public ms | After public ms | After Graph ms |
| --- | ---: | ---: | ---: | ---: |
| HGRN | 64 | 0.118 | 0.088 | 0.013 |
| HGRN | 2048 | 0.169 | 0.138 | 0.019 |
| RWKV6 | 64 | 0.261 | 0.169 | 0.064 |
| RWKV6 | 2048 | 0.355 | 0.248 | 0.083 |
| GSA | 64 | 0.326 | 0.241 | 0.122 |
| GSA | 2048 | 0.474 | 0.369 | 0.136 |
| Normalized linear | 64 | 0.158 | 0.120 | 0.015 |
| Normalized linear | 2048 | 0.227 | 0.175 | 0.025 |

Graph replay times remain effectively unchanged. A 500-call HGRN host profile
dropped from 227501 to 166501 function calls; full descriptor construction no
longer appears on its raw-input path. Records:
`benchmark_results/dense_raw_binding_final.json`. This removes repeated host
structure work, not GPU arithmetic, and does not establish cross-device speedup.
Verification: 330 unit/focused GPU tests passed. A fresh-process smoke reused one
compiled handle per structure across `B/T={(1,1),(2,65),(1,130),(2,8192)}`, `H=8`,
checking independent FP32 outputs, all final States, output-only, nonaligned
continuation, and dtype rejection after warm calls. The permanent vector test
also changes input storage offsets, uses a headless gate, and retains earlier
outputs to catch stale pointers/strides or accidental output reuse.

The next host-validation pass compiles immutable structural validation metadata
with the specialization. It stores input names, declaration roles, dtype and
differentiability, dense/packed Sequence positions, State role suffixes,
HeadMapping, and the reference-lowering classification. Runtime validation still
checks the current tensors, dynamic B/T or packed offsets, device, layout, dtype,
gradient contract, aliases, fixed role sizes, state tensors, and workspace.
No tensor, pointer, stride, offset value, or successful runtime validation result
is retained.

On the same H20 setup, BF16 `B=H=1,T=2048`, ten warmups and one hundred public
samples, the current public medians were HGRN `0.116 ms`, RWKV6 `0.217 ms`, GSA
`0.331 ms`, and normalized-linear `0.148 ms`; Graph medians were respectively
`0.018`, `0.081`, `0.134`, and `0.024 ms`. A 500-call HGRN cProfile decreased
from `175001` to `156001` function calls, with `_validate_runtime` cumulative
time decreasing from `0.025 s` to `0.018 s`. These are host-structure measurements
and do not imply a GPU arithmetic change.

## Legacy Customized Linear Attention API
The modifier-based `LinearAttentionEngine` remains available with its existing
interface and behavior. It is outside the Stateful Operator compiler guarantee
and is not routed through this compiler.


```py
mod = LinearAttentionEngine(qkv_meta, q_mod=..., k_mod=..., v_mod=..., decay_mod=..., custom_io=...)
output = mod(q, k, v, decay, custom_input)
```
GDN is an explicit generic Algorithm IR composition. A dedicated GDN engine is
not part of the public interface; FlashQLA is only an independent mathematical
or performance reference, never a Stateful Operator execution backend.

See `docs/stateful-operator-ir-spec.md` for the complete contract.

# Upcoming Features

## Attention Library Level API
This level API is designed for users to use the existing attention mechanism in the library.
```py
mod = AttentionLibrary(
    attn_type: str="SoftmaxAttention",
    mask_type: str="Causal",
    use_types: str="Train",
)
```

## Custom Attention Level API
- Support for varlen, block-sparse mask and block-sparse indices
```py
mod = AttentionEngine(
    ...
)
mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_len_q: torch.Tensor=None,
    cu_len_kv: torch.Tensor=None,
    block_sparse_mask: torch.Tensor=None,
    block_sparse_indices: torch.Tensor=None,
    block_num: torch.Tensor=None,
)
```
- Support OnlineFunc for decoding
```py
class OnlineFunc:
    ...
    def combine(final_rowscales)-> Tensor:
        """Compute logic for the combine kernel"""
        return o_scale

```
- Support mask_mod for decoding

```py
def mask_mod(b, h, q_idx, kv_idx, custom_fwd_inputs) -> Bool:
    """The offset of q need to be passed by custom_fwd_inputs"""
    return True
```
