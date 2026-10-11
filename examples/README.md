# Stateful examples

These factories describe recurrences with the existing Stateful Operator IR.
They return a `StatefulOperator`; compilation binds the head and feature roles,
then invocation binds tensors by name. The factory's batch, head and sequence
arguments follow the existing example convention; runtime tensors and
`role_sizes` determine those extents. `dim` sets the query scale `1/sqrt(K)`.

Dense inputs use head-first `[B, H, T, ...]` layout. Query, key and value inputs
use the factory's `dtype` (BF16 by default); the gates and scalar learning rates
in the examples below use FP32. Keys and values arrive as supplied: input
projection, key normalization, gate activation and layer wrappers belong to the
caller. All outputs read the state after the current token's update.

| File / factory | Recurrence on state `S` with shape `[K, V]` | Additional inputs |
| --- | --- | --- |
| [`delta_rule.py`](delta_rule.py) / `delta_rule_attention` | `S += beta k (v - k^T S)^T` | Scalar `beta` |
| [`kda.py`](kda.py) / `kda` | `S' = diag(exp(gate)) S`; `S = S' + beta k (v - k^T S')^T` | KEY-channel `gate`, scalar `beta` |
| [`gdn2.py`](gdn2.py) / `gdn2` | `S' = diag(exp(gate)) S`; `S = S' + k (write * v - (erase * k)^T S')^T` | KEY-channel `gate` and `erase`, VALUE-channel `write` |
| [`comba.py`](comba.py) / `comba` | `S' = exp(gate) S`; `S = S' + beta k (v - p^T S')^T` | `prediction_key` (`p`), scalar `gate` and `beta` |
| [`generalized_delta_rule.py`](generalized_delta_rule.py) / `iplr_delta_rule` | `S += beta (alpha^T S) + k v^T` | Independent KEY vectors `alpha`, `beta` |
| [`generalized_delta_rule.py`](generalized_delta_rule.py) / `dplr_delta_rule` | `S = diag(exp(gate)) S + beta (alpha^T S) + k v^T` | KEY-channel `gate`, KEY vectors `alpha`, `beta` |
| [`gated_oja_rule.py`](gated_oja_rule.py) / `gated_oja_rule` | `S' = S diag(exp(gate))`; `S = S' + beta (k - S' v) v^T` | VALUE-channel `gate`, scalar `beta` |
| [`gated_delta_product.py`](gated_delta_product.py) / `gated_delta_product` | Scalar decay, then ordered delta writes within each token | Scalar `gate`; named `key_j`, `value_j`, `beta_j` |
| [`abc_attention.py`](abc_attention.py) / `abc_attention` | Weighted key/value slot sums plus a weight sum; normalized slot softmax readout | `slot_logits` with shape `[B, Hk, T, M]` |

Every gate is in natural-log space. Scalar inputs have shape `[B, Hs, T]`;
channel gates add their indicated feature axis. Generalized Delta's `beta` is a
KEY vector, unlike the scalar learning rate in ordinary Delta Rule. The legacy
`examples.linear_attention.delta_rule_attention` now delegates to the real
Delta Rule factory and requires the scalar `beta` tensor when invoked.

DPLR's low-rank term reads the **prior** state. The example factors its map as
`(I + beta (exp(-gate) * alpha)^T) diag(exp(gate))`; both exponentials must remain
representable in FP32. This differs from applying a low-rank term directly to
an already decayed state.

Delta Product fixes `num_householder` when constructing the operator (default:
2). Split FLA's interleaved `[B, T*num_householder, H, ...]` streams into
head-first `key_0`, `value_0`, `beta_0`, etc. Each earlier injection includes the
effect of later corrections. The explicit outer-product expansion grows exponentially with
the step count and is intended to illustrate small fixed counts.

ABC maintains `key_slot [K, M]`, `slot_value [M, V]`, and `normalizer [M]`.
It implements vanilla ABC using `exp(slot_logits)` weights, normalization and
slot softmax in the readout. All three raw sums are needed for continuation;
they are different from FLA's normalized two-state cache. Supply logits whose
exponentials and cumulative sums remain finite and positive in FP32. This
example does not include the logcumsumexp stabilization used by FLA's recurrent
ABC implementation.

## Compile and run KDA

From the repository root on a supported NVIDIA H20:

```python
import torch
from attn_engine import StateTuple, Target
from examples.kda import (
    KEY, VALUE, QUERY_HEADS, KEY_HEADS, VALUE_HEADS, STATE_HEADS, kda,
)

batch, heads, length, dim, dim_value = 1, 2, 17, 64, 128
operator = kda(batch, heads, length, dim, dim_value)
compiled = operator.compile(
    role_sizes={
        QUERY_HEADS: heads,
        KEY_HEADS: heads,
        VALUE_HEADS: heads,
        STATE_HEADS: heads,
        KEY: dim,
        VALUE: dim_value,
    },
    target=Target(),
    mode="dense",
)
query = torch.randn(batch, heads, length, dim, device="cuda", dtype=torch.bfloat16)
key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1).contiguous()
value = torch.randn(batch, heads, length, dim_value, device="cuda", dtype=torch.bfloat16)
gate = -torch.rand(batch, heads, length, dim, device="cuda", dtype=torch.float32)
beta = torch.rand(batch, heads, length, device="cuda", dtype=torch.float32)
initial = torch.zeros(batch, heads, dim, dim_value, device="cuda", dtype=torch.float32)
result = compiled(
    query=query, key=key, value=value, gate=gate, beta=beta,
    initial_state=StateTuple(("memory",), (initial,)),
    return_final_state=True,
)
print(result.outputs["output"].shape)
# Supply result.final_state as initial_state to continue with another segment.
```

Omit `initial_state` to start from zero. Each source head count must divide the
State head count; `HeadMapping` expands contiguous groups. Compile with
`backward=True` for differentiable invocation. A different key or value
feature size requires matching `role_sizes` and a matching query scale in the
factory.

## Validate

The example tests compare public operator calls with independent FP32 token
recurrences, including nonzero initial states, grouped heads, unequal key/value
dimensions, gradients, and state continuation:

```bash
uv run --extra test pytest -q tests/unit/test_delta_family_examples.py tests/unit/test_example_imports.py
CUDA_VISIBLE_DEVICES=0 uv run --extra test pytest -q --run-gpu \
  tests/unit/test_delta_family_examples.py -k execute_on_gpu
```

The GPU cases use BF16 inputs and `[64, 128]` matrix states. These checks establish
correctness at the tested shapes; they do not establish performance parity with
FLA or coverage of complete model layers.

## Benchmark against FLA

Both `testing/benchmark_stateful_plans.py` (Meta only) and
`testing/benchmark_fla_stateful.py` now register all nine factories above. For a
recent FLA checkout and an otherwise idle NVIDIA GPU:

```bash
uv run --group fla-benchmark python testing/benchmark_fla_stateful.py \
  --families delta-rule,kda,gdn2,comba,iplr,dplr,oja,delta-product,abc \
  --batches 1 --heads 8 --seqlens 512 2048 \
  --contract output-only --timing both --warmup 5 --repetitions 20 \
  --workspace-gib 8 --fla-source ../flash-linear-attention \
  --output-dir benchmark_results/new_examples_h20_8gib
```

Use `--contract all` to additionally check final states and two-call continuation.
ABC is comparable only for output-only from zero: FLA's two normalized caches
are different from Meta's three raw accumulators. The other eight families
compare all three contracts. FLA uses its public fused recurrent API, except
Delta Product and ABC, which use their public chunk APIs. This is a forward
benchmark; it does not measure training backward or the fastest FLA chunk kernel
for each recurrent family.

The default workloads use BF16 query/key/value, FP32 gates/state, `K=V=64`,
16 ABC slots, and two Delta Product writes. Inputs normalize correction vectors,
use small independent low-rank vectors for IPLR/DPLR, and use exactly
BF16-representable ABC logits so both sides receive the same values. Gate/logit
preparation happens before timing; layout conversion, ABC's FLA logit cast,
ComBa's prediction-key adjustment and Delta Product interleaving remain inside
the public-call interval. This FLA checkout's ComBa fused kernel predicts from
the prior state; multiplying its prediction key by `exp(gate)` matches the
example's prediction from the decayed state. The adjustment uses FP32 to avoid
introducing another BF16 rounding step.

The 8 GiB workspace budget permits the full measured workload on H20. With the
default 1 GiB budget, some `T=2048` workloads are rejected before execution;
those rejections are recorded rather than assigned latency values.

Each comparable result must pass the independent FP32 oracle and pairwise output
checks before its ratio is reported. Records include the selected lowerer,
public-call latency, synchronized wall time, CUDA Graph replay when available,
peak allocator usage, and source/environment provenance. `reference-stateful`
is a possible selected path; its measurements include that implementation.
Graph-capture failures and missing FLA APIs are recorded explicitly. The runner
writes `records.json`, `manifest.json`, `summary.csv`, and comparison plots.
See [the measured H20 results](../docs/stateful-example-benchmarks.md) for the
workload, selected lowerers and performance limits.
