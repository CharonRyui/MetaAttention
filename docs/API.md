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
Only three FP32 chunk vectors are scratch; no promoted matrix, dense query of
ones, FP32 input copy, or token-sized State scratch is created. Output casting
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
