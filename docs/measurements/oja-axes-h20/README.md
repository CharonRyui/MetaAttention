# Independent low-rank update/readout axes on H20

The structural low-rank recipe now records the internal State orientation and
readout reduction axis. Rank-one propagations must all use one Feature axis;
readouts must all contract one Feature axis. These axes can differ. Updates on
the public second axis run over a transposed internal State; initial/final State
and their gradients retain the public layout. Mixed update or readout axes keep
the existing affine fallback.

Oja selects `tilelang-factored-recurrence` instead of `tilelang-product-gemm`.
Ordered summaries, carry, replay and inverse-free checkpoint VJPs consume the
axis recipe. Dense and packed calls support grouped heads, multiple readouts,
continuation and empty packed segments. Opposite-axis forward readouts hold
both State dimensions in a replay block at the supported 64/128 feature sizes.
Training still tiles columns and deterministically reduces factor gradients.

## Measurement

Workspace: `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl`, H20 GPU 3,
B=1, H=8, K=V=64, BF16 inputs/output, FP32 gates and State, output-only calls.
Five warmups and twenty repetitions exclude compilation. The default 1 GiB
workspace limit is sufficient. Measurements used the Oja working-tree changes
on `be05d11`; the manifest also records the measured source digest.

| Tokens | Meta public ms | FLA public ms | Meta Graph ms | FLA Graph ms | Meta incremental MiB | FLA incremental MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.668 | 0.360 | 0.143 | 0.269 | 6.78 | 0.75 |
| 2048 | 1.227 | 4.119 | 0.749 | 4.018 | 68.52 | 12.06 |

Both points pass the benchmark's independent FP32 recurrence oracle and FLA
output checks. FLA uses its public recurrent Oja implementation. At 2048 tokens
Meta's public call is 3.36 times faster than that baseline. The previous merged
revision measured about 101.57 ms public, 101.14 ms Graph and 4100.1 MiB
incremental allocation at this shape. Those historical numbers came from a
separate run, so their ratios are indicative rather than paired statistics.
Short public calls remain slower than FLA despite faster Graph execution;
factor binding and submission still contribute significant overhead. This is a
single-device measurement without confidence intervals.

Reproduce on an idle H20 from the repository root:

```bash
.venv/bin/python testing/benchmark_fla_stateful.py --families oja \
  --seqlens 128 2048 --batches 1 --heads 8 --timing both \
  --warmup 5 --repetitions 20 --gpu 3 \
  --output-dir docs/measurements/oja-axes-h20
```

## Validation

33 new GPU tests and 10 compiler tests pass. Coverage includes both 64x128 and
128x64 States, BF16/FP32, nonzero initial State, shared heads, two readouts with
different output dtypes, chunk tails, continuation, empty packed segments and
all-empty packed execution. Input/initial-State VJPs cover output-only,
final-only and joint losses, including a zero channel decay.

The independent gradient oracle retains authored expression dtype boundaries.
In particular, DSL `Constant` values are FP32 tensors; replacing the query
scale with a Python scalar changes BF16 rounding and even FP32 State gradients.
It also keeps separately authored use-site casts separate.

Shared regression: 175 tests pass, 9 parameter combinations skip; 179 CPU tests
pass, 59 GPU-dependent cases skip in the CPU runs. Ruff and diff checks pass.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_lowrank_axes.py -q --run-gpu -x
.venv/bin/pytest tests/unit/test_lowrank_axes.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_weighted_lowrank.py tests/functional/test_stateful_checkpoint.py tests/functional/test_stateful_packed_checkpoint.py tests/functional/test_lowrank_wy.py tests/unit/test_delta_family_examples.py tests/unit/test_stateful_factored_recipes.py -q --run-gpu -x
.venv/bin/pytest tests/unit/test_stateful_factored_recipes.py tests/unit/test_weighted_lowrank.py tests/unit/test_delta_family_examples.py tests/unit/test_stateful_readout_normalization.py -q
.venv/bin/pytest tests/unit/test_stateful_operator.py tests/unit/test_stateful_workspace.py tests/unit/test_stateful_bindings.py tests/unit/test_stateful_selection.py -q
```
