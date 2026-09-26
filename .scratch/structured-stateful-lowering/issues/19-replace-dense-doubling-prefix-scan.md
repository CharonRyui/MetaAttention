# 19: Replace both prefix scans with one chunked recurrent schedule

**What to build:** Compose per-token affine summaries through a chunked recurrent kernel instead of the two host-driven prefix implementations, for every State whose summary is dense (rank-one propagations) and every State whose summary is a per-axis diagonal (feature-wise decay).

**Blocked by:** 12/Add Rank One Factorized Plan; 17/Compose Optimized Forward and Backward.

**Status:** ready-for-agent

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64, after 20):

| family | plan | T=512 | T=2048 | launches @2048 |
| --- | --- | --- | --- | --- |
| `gdn` | rank-one | 17.0 ms | 81.6 ms | 44 |
| `rwkv7` | rank-one | 16.9 ms | 81.6 ms | 44 |
| `gla` | elementwise | 1.34 ms | 4.59 ms | 1 |
| `mamba2` | elementwise | 1.37 ms | 4.42 ms | 1 |
| `linear` / `simple-gla` (scalar-factorized) | scalar | 0.14 ms | 0.30 ms | 1 |

Two implementations spend this time, and both must go:

1. `_tilelang_dense_affine_scan` walks `offset = 1, 2, 4, ... token_count` and launches four fused-matmul kernels per step with six full-size temporaries per step (left `[T,H,K,K]`, right `[T,H,V,V]`, four `[T,H,K,V]`). That is 4·log2(T) launches and ~1.6 GiB of temporaries per step at T=2048.
2. `_compile_elementwise_prefix` is a single kernel, but it parallelizes over `(heads, K*V/4)` blocks with four matrix lanes each and scans tokens serially inside the block with two thread syncs per doubling stage. It materializes the per-token State and runs ~15x the scalar-factorized schedule for the same State shape.

The scalar-factorized kernel already shows the schedule to copy: grid over `(sequences x heads, column blocks)`, one shared-memory State `[rows, block_columns]` carried across a serial chunk loop, chunk-local GEMMs for the intra-chunk part, and no per-token State written to global memory. Feature-wise decay fits the same shape by pre-scaling the chunk's query and key factors with their per-feature prefix sums.

- [ ] One chunk-summary path serves dense and per-axis diagonal summaries: chunk summaries computed by a kernel parallel over chunks and heads, a scan over chunk summaries (and segment carries), and an epilogue that emits per-token States and readouts.
- [ ] Feature-wise decay is applied as a per-feature prefix rescale of the chunk factors, not as a per-token elementwise scale.
- [ ] The doubling scan is deleted or retained only as a CPU reference; no CUDA path reaches it.
- [ ] Authored propagation order, segmentation, `initial_state`/final-State semantics, packed mode, and gradient parity are unchanged against the current implementation, checked by the existing H20 suites.
- [ ] `gdn`, `rwkv7`, `gla`, and `mamba2` land within 2x of the scalar-factorized group at T=2048 with one launch per pass.
