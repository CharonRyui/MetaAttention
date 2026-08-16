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

Stateful sequence operators are authored with explicit Algorithm IR and compiled
through `StatefulOperator`. Runtime tensors bind by keyword to declared Tensor
Input names. The operator is stateless; an omitted initial state means zero, and
`return_final_state=True` returns a named `StateTuple`.

```py
from attn_engine import StatefulOperator

operator = StatefulOperator(algorithm)
output, final_state = operator(
    query=query,
    key=key,
    value=value,
    gate=gate,
    beta=beta,
    initial_state={"memory": state},
    return_final_state=True,
)
```

The first-phase Algorithm IR supports identity, elementwise or diagonal scale,
ordered propagation composition, rank-one delta propagation, outer-product and
elementwise state injection, direct input, zero injection, and typed state
readouts. Head Mapping declares grouped-head relationships and validates
divisibility before lowering. Normalized numerator/denominator state is not
supported in this phase.

`LinearAttentionEngine` and `GDNEngine` remain deprecated compatibility adapters
for migration. New production code should construct Algorithm IR directly;
arbitrary Python transition and readout callbacks are not accepted.

## Deprecated Customized Linear Attention API

The modifier-based constructor remains available only for migration and emits a
`DeprecationWarning`. Its existing positional call contract is unchanged.

```py
mod = LinearAttentionEngine(qkv_meta, q_mod=..., k_mod=..., v_mod=..., decay_mod=..., custom_io=...)
output = mod(q, k, v, decay, custom_input)
```

## Gated Delta Rule API

Gated Delta Rule is authored by constructing explicit rank-one-delta
Algorithm IR and invoking `StatefulOperator`. `GDNEngine` remains the
deprecated positional compatibility adapter and unwraps the named `memory`
state for compatibility.

The IR retains log-space `gate` as a Tensor Input and expresses `Exp(gate)`
explicitly. `beta` is raw and differentiable; keys are not normalized.

`Hv` must be a positive multiple of `Hk`; each query/key head serves
`Hv/Hk` value and state heads. `T` must be positive and divisible by the fixed
chunk size 64. Inputs are contiguous and head-first. Output is BF16; requested
final state and a provided initial-state gradient are FP32.

The first release supports only the repository's validated NVIDIA H20 target, fixed key/value dimensions 128, BF16 query/key/value, and FP32 gate/beta/state. It rejects other devices, dtypes, layouts, dimensions, and unaligned lengths before kernel execution.
Q/K normalization, variable lengths, padding, and decode caches are outside this API.

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
