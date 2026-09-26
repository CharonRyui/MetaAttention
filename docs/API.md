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

Schema-v2 Programs execute through the reference lowering in
`attention_engine/attn_engine/_stateful_reference.py`: the same parallel
segmented scans the specialized lowerings use, with per-State affine summaries
and expression readouts. Gradients flow through the same lowering, so no eager
token loop and no framework-derived backward is involved. Programs that stay
within the v1 language keep their existing specialized TileLang plans.

`examples/fla_coverage.py` authors every Flash Linear Attention operator that is
a per-token linear recurrence in this language and records the coverage table;
`tests/functional/test_fla_coverage.py` checks parity with the library operators.

The first production Target is CUDA H20. It accepts BF16/FP32 inputs and
outputs, FP32 State, canonical contiguous layouts, and Feature pairs in
`{64,128} × {64,128}`. Invalid IR, unsupported composition, Target mismatch,
invalid packed offsets, unsupported specialization, resource failure, and
codegen failure surface as structured `StatefulCompilationError` values. The
complete contract is in `docs/stateful-operator-ir-spec.md`.

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
