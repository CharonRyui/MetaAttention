# 19: Replace the dense doubling prefix scan

**What to build:** Compose per-token affine summaries through a chunk-summary scan instead of the current host-side doubling scan, for every State whose summary is dense (rank-one propagations and single-role Axis Scale).

**Blocked by:** 12/Add Rank One Factorized Plan; 17/Compose Optimized Forward and Backward.

**Status:** ready-for-agent

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64):

| family | T=512 | T=2048 | launches @2048 |
| --- | --- | --- | --- |
| `gla` (axis-scale) | 16.5 ms | 80.0 ms | 44 |
| `gdn` (rank-one) | 17.0 ms | 81.6 ms | 44 |
| `rwkv7` (rank-one) | 16.9 ms | 81.6 ms | 44 |
| `mamba2` (axis-scale) | 16.5 ms | 80.0 ms | 44 |
| `linear` (scalar-factorized) | 0.14 ms | 0.30 ms | 1 |

`_tilelang_dense_affine_scan` walks `offset = 1, 2, 4, ... token_count` and launches four fused-matmul kernels per step, allocating six full-size temporaries per step (left `[T,H,K,K]`, right `[T,H,V,V]`, four `[T,H,K,V]`). That is 4·log2(T) launches and ~1.6 GiB of temporaries per step at T=2048.

- [ ] Chunk summaries are computed by one parallel kernel over chunks and heads, not by a host doubling loop.
- [ ] A scan over the chunk summaries (and, where needed, over segment carries) replaces the token-axis doubling.
- [ ] An epilogue kernel produces per-token States and readouts from the chunk entry State and the chunk-local factors.
- [ ] The doubling scan is deleted or retained only as a CPU reference; no CUDA path reaches it.
- [ ] Authored propagation order, segmentation, `initial_state`/final-State semantics, and gradient parity are unchanged against the current implementation.
- [ ] The four families above land within 2x of the scalar-factorized group at T=2048 with one launch per pass.
