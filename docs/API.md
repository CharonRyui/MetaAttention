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

Stateful sequence operators are authored by constructing explicit Algorithm IR
and compiling it through `StatefulOperator`. The IR is model-independent: GLA,
RetNet recurrent state evolution, selective state evolution, GDN, and custom
compositions use the same typed nodes. Runtime tensors bind by keyword to named
Tensor Inputs. `LinearAttentionEngine` remains a separate legacy generator and
is not routed through this compiler.

The first compiler schema has one named matrix State with two ordered Feature
roles. It supports typed Input Expressions (`Input`, BF16/FP32 `Constant`,
`Add`, `Multiply`, `Negate`, `Exp`), role-directed Axis Scale, general
Rank-One propagation, Product Injection tuples, explicit Head Mapping, and
single-role State Contractions. Arbitrary Python callbacks, generic einsum,
multi-state programs, and other State ranks are not supported.

```py
from attn_engine import StatefulOperator

operator = StatefulOperator(algorithm)  # Target may be supplied explicitly
result = operator(
    query=query,
    key=key,
    value=value,
    initial_state=state_tuple,       # optional StateTuple
    sequence_offsets=offsets,        # packed mode only; int32 CUDA tensor
    return_final_state=True,
)

output = result.outputs["output"]
state = result.final_state["memory"]
```

Every invocation returns immutable `ExecutionResult`, containing a named output
tuple and optional named final State. Single outputs and States are never
unwrapped. Missing initial State means one FP32 zero State per logical sequence.
Final State is materialized only when requested. Packed values use contiguous
token-major `[total_tokens, heads, features...]`; dense values use canonical
role-order contiguous layout. Empty sequences and unaligned tail chunks have
identity-transition semantics and do not read padding.

The first production Target is CUDA H20. It accepts BF16/FP32 inputs and
outputs, FP32 State, canonical contiguous layouts, and Feature dimension pairs
`(64,64)`, `(64,128)`, `(128,64)`, and `(128,128)`. H100, CPU, ROCm, unknown
same-capability products, and unidentifiable MIG devices are rejected. No
implicit copies, transposes, padding, eager recurrence, or dedicated GDN
backend are used.

Compiler analysis is internal. Invalid IR, unsupported composition, target
mismatch, invalid packed offsets, unsupported concrete specialization, and
code-generation failures surface as structured `StatefulCompilationError`
values with stable category codes, public-IR paths, and JSON-safe details.

## Legacy Customized Linear Attention API

The modifier-based `LinearAttentionEngine` remains available with its existing
interface and behavior. It is outside the Stateful Operator compiler guarantee
and is not changed by this specification.

```py
mod = LinearAttentionEngine(qkv_meta, q_mod=..., k_mod=..., v_mod=..., decay_mod=..., custom_io=...)
output = mod(q, k, v, decay, custom_input)
```

## Gated Delta Rule

GDN is an explicit Algorithm IR composition of generic Axis Scale, Rank-One
propagation, Product Injection, and State Contraction nodes. `GDNEngine` is not
part of the public interface and is removed; FlashQLA is only a mathematical
or performance reference, never a Stateful Operator execution backend.

See `docs/stateful-operator-ir-spec.md` for the complete language, numeric,
packed-sequence, cache, Target, error, and verification contracts.

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
