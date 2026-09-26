# 19: Replace both prefix scans with one streaming schedule

**What to build:** Compose per-token affine summaries without the two host-driven prefix implementations, so dense (rank-one) and diagonal (feature-wise decay) States both run in one launch whose cost is bounded by memory traffic.

**Blocked by:** 12/Add Rank One Factorized Plan; 17/Compose Optimized Forward and Backward.

**Status:** in progress (19a delivered, 19b open)

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64):

| family | plan | T=2048 forward before | after 19a | forward+backward before | after 19a |
| --- | --- | --- | --- | --- | --- |
| `gla` | elementwise | 4.59 ms | 1.87 ms | 16.5 ms | 7.9 ms |
| `mamba2` | elementwise | 4.42 ms | 1.92 ms | 16.7 ms | 8.0 ms |
| `simple-gla` | scalar-factorized | 0.30 ms | 0.30 ms | 15.2 ms | 7.3 ms |
| `linear` | scalar-factorized | 0.30 ms | 0.30 ms | 15.5 ms | 7.6 ms |
| `gdn` | rank-one | 81.6 ms | 81.6 ms | 169.5 ms | 169.5 ms |
| `rwkv7` | rank-one | 81.6 ms | 81.6 ms | — | — |

19a (delivered): `_compile_elementwise_prefix` is now a streaming kernel. Every thread owns one State element and walks its segment accumulating `(A_t, B_t)` with `S_t = A_t S_0 + B_t`, so the cross-thread scan, its barriers, and the shared-memory chunk loop are gone; per-axis, per-token-scalar, and full elementwise factors all lower to the same accumulation. The kernel also composes in reverse (`reverse=True`), which removed the seven flipped token-sized copies the elementwise VJP used to make.

Measured effect of 19a: the three backward scans fall from 7.5-8.5 ms to 1.0 ms, forward falls 2.4x, and the whole training step falls ~2x.

What remains for 19b, in the measured order of impact:

1. **Rank-one States (gdn, rwkv7)**: `_tilelang_dense_affine_scan` still doubles over the token axis with four fused-matmul launches and six full-size temporaries per step (4*log2(T) launches, ~1.6 GiB per step at T=2048). These need a chunked factored kernel: per-token factors (`left [T,H,R,1]`, `right [T,H,1,C]`, coefficient) applied to a shared-memory State, with the readout contracted in the kernel.
2. **Readout fusion (see 21)**: the elementwise family still materializes per-token States and then reads them back, which is why the VJP spends ~2 ms in token-sized torch passes and why GLA forward is 6x the scalar-factorized schedule. Fusing the contraction into the streaming kernel removes those passes entirely.
3. **Chunk-level parallelism**: the streaming kernel parallelizes over State elements only (heads x R*C/threads blocks), which is enough at K=V=64..128 but not at small State sizes.

- [x] The elementwise prefix composes in one launch with no cross-thread scan, for per-axis and scalar factors, in dense and packed modes.
- [ ] One chunked path serves dense and diagonal summaries with chunk-level parallelism.
- [ ] The doubling scan is deleted or retained only as a CPU reference; no CUDA path reaches it.
- [ ] Authored propagation order, segmentation, `initial_state`/final-State semantics, packed mode, and gradient parity are unchanged against the current implementation, checked by the existing H20 suites.
- [ ] `gdn`, `rwkv7`, `gla`, and `mamba2` land within 2x of the scalar-factorized group at T=2048 with one launch per pass.
