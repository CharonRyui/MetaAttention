# Readout normalization: H20 inference benchmark

All four programs selected `tilelang-factored-recurrence` at both tested lengths. A diagnostic invocation intercepted the factored and reference executor entry points and verified that both optimized frontend spellings actually entered the factored executor. Timing runs used the uninstrumented public call. The authored `Readout(StateTerm(...))` and equivalent `StateContraction` had identical binary identities and exactly equal outputs and final States.

NVIDIA H20, physical GPU 0; B=1, H=8, K=V=64; BF16 inputs/output and FP32 State. Dense inference returns output and final State, starting from zero State. Five warmups and 20 measured repetitions; median latency, compilation excluded. CUDA event timing surrounds the public call and includes launch gaps; CUDA Graph timing measures replay. Incremental memory is the warmed single-call PyTorch allocator peak, including returned output and State.

| Program | Length | Optimized public ms | Optimized Graph ms | Contraction Graph ms | Incremental MiB | Unoptimized public ms | Public speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gdn2 | 512 | 0.652 | 0.242 | 0.242 | 17.2 | 16.818 | 25.78x |
| iplr | 512 | 0.639 | 0.333 | 0.333 | 24.2 | INVALID | — |
| dplr | 512 | 0.666 | 0.245 | 0.245 | 17.2 | 16.801 | 25.23x |
| delta-product | 512 | 1.659 | 1.109 | 1.106 | 28.4 | 11.217 | 6.76x |
| gdn2 | 2048 | 1.099 | 0.736 | 0.737 | 68.3 | 81.383 | 74.05x |
| iplr | 2048 | 1.519 | 1.248 | 1.249 | 96.3 | INVALID | — |
| dplr | 2048 | 1.130 | 0.748 | 0.748 | 68.3 | 81.382 | 72.01x |
| delta-product | 2048 | 3.941 | 3.501 | 3.504 | 113.1 | 49.459 | 12.55x |

All 16 optimized runs passed the independent authored-order FP32 token recurrence oracle. Acceptance tolerances: output rtol=0.02/atol=0.05; final State rtol=0.01/atol=0.02. Exact equality between the two frontend spellings was checked separately with zero tolerance.

The unoptimized IPLR affine scan failed correctness at both lengths (maximum final-State absolute error 1.520 at length 512 and 9.279 at length 2048). These two runs have no timing or speedup claim. A subsequent diagnosis traced these failures to single-TF32 GEMM in the generic affine scan, which rounds away small corrections in near-identity transition matrices. The precision fix and new measurements below supersede the IPLR INVALID verdicts; the original raw records remain unchanged. GDN2 and DPLR use the generic affine scan when optimization is disabled; Delta Product uses the reference executor. The reference executor does not support CUDA Graph capture in this workload, so its Graph latency is omitted.

This is a small inference benchmark of the current worktree. It does not establish training performance or performance across other shapes. Concurrent work was active on other GPUs.

Raw data: [records.json](records.json), [summary.csv](summary.csv), [manifest.json](manifest.json). Reproduction:

```bash
cd /dev/shm/metaattention-math-structured
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/attention_engine:$PWD" \
  /mnt/nvme/linyanxuan/MetaAttentionImpl/.venv/bin/python \
  docs/measurements/readout-normalization-h20/benchmark.py
```

The script resumes completed entries from records.json; use a fresh output directory or move the existing records file to collect fresh timings.

## IPLR affine-scan precision fix

The generic affine scan used `T.gemm` on FP32 operands, but generated CUDA used a single TF32 tensor-core product. In near-identity IPLR transitions, rounding the operands loses small rank-one corrections at each scan composition. The fix reuses `fp32_gemm_macro`: split both operands into TF32 high and residual terms, then accumulate high×high, low×high, and high×low. Both the forward prefix and reverse scan for backward share this implementation. This is a three-product approximation, not exact FP32 multiplication.

A differential probe with one head and length 64 held affine summaries and composition order fixed. Final-State maximum absolute error was 0.077161 for the original GPU scan, 0.000105 with CPU FP32 scan, and 0.000111 with the residual-preserving GPU GEMM. This isolates the main loss to GPU scan multiplication.

The original benchmark inputs (seed 42), oracle, and tolerances were reused at lengths 512 and 2048. All four new runs passed. Final-State maximum absolute errors:

| Length | Generic before fix | Generic after fix | Factored after fix |
|---|---:|---:|---:|
| 512 | 1.519603 | 0.001106 | 1.144e-05 |
| 2048 | 9.278517 | 0.006536 | 4.292e-05 |

BF16 output maximum absolute errors after the fix were 0.031648/0.066135 for generic and 0.031235/0.062181 for factored at lengths 512/2048. Acceptance uses elementwise absolute and relative tolerance together; maximum absolute error alone is not the verdict. The generic path retains a length-dependent numerical gap versus the residual recurrence oracle. Factored output and State error metrics match the original measurements.

Fresh timings on physical H20 GPU 0, otherwise using the configuration above:

| Length | Generic public ms | Generic Graph ms | Factored public ms | Factored Graph ms | Public speedup |
|---|---:|---:|---:|---:|---:|
| 512 | 20.788 | 20.348 | 0.647 | 0.337 | 32.14x |
| 2048 | 101.023 | 100.446 | 1.547 | 1.243 | 65.30x |

There is no valid latency comparison against the original unoptimized IPLR scan: it failed correctness and was not timed. These speedups compare the factored path against the repaired generic path.

The minimal public-call regression uses two tokens, a nonzero initial State, and ±1e-4 rank-one updates. All four dense/packed cases failed before the fix (output errors 0.000200/0.000776) and passed afterward, including output, final State, input gradients, and initial-State gradients. Regression tolerance is rtol=2e-5/atol=2e-6.

New raw data: [iplr-precision-fix.json](iplr-precision-fix.json), including the affine implementation SHA256. Reproduction:

```bash
cd /dev/shm/metaattention-math-structured
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/attention_engine:$PWD" \
  /mnt/nvme/linyanxuan/MetaAttentionImpl/.venv/bin/python \
  docs/measurements/readout-normalization-h20/iplr_precision_probe.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/attention_engine:$PWD" \
  /mnt/nvme/linyanxuan/MetaAttentionImpl/.venv/bin/python -m pytest -q --run-gpu \
  tests/functional/test_stateful_affine_precision.py
```

## Integration validation

The merge with the weighted outer-product implementation preserves its ordered scalar factors and zero-safe VJP. Normalized Delta Product readouts now select compiled factored training as well as inference. Validation after resolving overlapping analysis, planning, and low-rank execution changes: the full unit suite passed (315 passed, 69 GPU tests skipped); the selected GPU suite passed (100 passed, 9 inapplicable cases skipped). GPU coverage includes equivalent readouts in dense/packed execution with gradients, recovered forward/continuation cases, FP32/BF16 Delta Product backward, weighted-write factor order and zero gradients, and affine precision regressions.
