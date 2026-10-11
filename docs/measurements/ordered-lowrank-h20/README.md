# Ordered weighted low-rank execution on H20

Weighted recurrences with multiple rank-one updates now reuse each row reduction
across a 64-column replay fragment during long dense inference. Selection uses
operation codes, authored scalar write factors and feature sizes, with at most
128 rows. Propagation and injection order, every FP32 multiplication boundary,
and public output/final-State layouts are retained. Recorded and packed calls
keep the 16-token checkpoint ABI and existing inverse-free VJP.

The compiler also admits weighted low-rank recipes into CUDA packed execution.
They previously used the reference executor despite the shared segmented
checkpoint pipeline already supporting their factors and gradients. Unoptimized
programs retain their previous selection.

## Paired measurement

Workspace: `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl`, H20 GPU 3,
B=1, H=8, BF16 inputs/output, FP32 gates/State, output plus final State. Five
warmups and twenty repetitions exclude compilation. Before uses the checkpoint
module from `43de339`; after uses the implementation in this directory's
measurement revision. Both use the same operator, input tensors and seed.

| State K x V | Tokens | Public before / after ms | Graph before / after ms | Incremental MiB |
| --- | ---: | ---: | ---: | ---: |
| 64 x 64 | 128 | 0.857 / 0.989 | 0.193 / 0.193 | 9.43 |
| 64 x 64 | 2048 | 2.281 / 2.009 | 1.618 / 1.387 | 112.83 |
| 64 x 128 | 128 | 1.056 / 1.003 | 0.267 / 0.267 | 13.32 |
| 64 x 128 | 2048 | 4.040 / 3.133 | 2.992 / 2.482 | 149.02 |
| 128 x 64 | 128 | 1.143 / 1.050 | 0.330 / 0.327 | 19.06 |
| 128 x 64 | 2048 | 3.931 / 3.748 | 3.371 / 3.214 | 204.96 |

Long-sequence Graph improvement is 4.7--17.1%. Allocation does not change. The
short schedule is unchanged; public timing variation there is submission noise.
All paired output/final-State checks pass; BF16 output differences are at most
0.001953125, and final-State differences are at most 1.1921e-7. These are
single-device measurements without confidence intervals.

The separate output-only FLA comparison on GPU 0 passes the independent FP32
oracle and FLA output checks at 128 and 2048 tokens. At 2048, Meta public/Graph
is 2.020/1.373 ms versus FLA 0.712/0.212 ms, with 112.83 versus 38.50 MiB
incremental allocation. A substantial gap remains; this change does not claim
FLA parity or compact-WY execution for weighted writes.

Reproduce the paired comparison from the repository root:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 .venv/bin/python docs/measurements/ordered-lowrank-h20/paired_probe.py
.venv/bin/python testing/benchmark_fla_stateful.py --families delta-product \
  --seqlens 128 2048 --batches 1 --heads 8 --timing both \
  --warmup 5 --repetitions 20 --gpu 0 --output-dir /tmp/meta_ordered_fla
```

## Rejected candidates

A propagation-only compact-WY solve combined with authored bias summary and
replay preserved the tested outputs but increased launches and allocation.
At 2048 tokens its direct executor took 2.160 ms and 68.44 MiB, versus 1.487 ms
and 16.19 MiB for the existing checkpoint executor. Its factor indexing also
prevented Graph capture. The candidate was removed before dispatch integration;
`rejected-compact-records.json` preserves the negative evidence. These timings
exclude public factor binding and are not directly comparable to the table.

Chunk sweeps (16/32/64/128 tokens) did not improve long execution over 64.
Wider summary fragments also failed to improve it. The recorded sweeps preserve
these alternatives and the replay-width result. No weighted write was collapsed
into a vector: doing so changes the authored multiplication association and can
change overflow/gradient behavior.

## Validation

34 new GPU tests pass: both 64x128 and 128x64, BF16/FP32, shared heads, two
readouts, nonzero initial States, zero coefficients/decay, chunk tails,
continuation, empty packed segments, every input/initial-State gradient and
output-only/final-only/joint losses. A third ordered update is also covered.
The independent oracle retains authored dtype boundaries and separate casts.

Shared GPU regression: 127 passed, 9 skipped. Compiler/workspace CPU checks:
42 passed, 33 GPU tests skipped. Ruff and `git diff --check` pass.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_ordered_lowrank.py -q --run-gpu -x
CUDA_VISIBLE_DEVICES=3 .venv/bin/pytest tests/functional/test_weighted_lowrank.py tests/functional/test_stateful_checkpoint.py tests/functional/test_stateful_packed_checkpoint.py tests/functional/test_lowrank_axes.py tests/unit/test_stateful_factored_recipes.py -q --run-gpu -x
.venv/bin/pytest tests/unit/test_stateful_factored_recipes.py tests/unit/test_lowrank_axes.py tests/unit/test_stateful_workspace.py -q
```
