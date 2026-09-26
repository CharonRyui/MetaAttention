# 19: Replace both prefix scans with one streaming schedule

**What to build:** Compose per-token affine summaries without the two host-driven prefix implementations, so dense (rank-one) and diagonal (feature-wise decay) States both run in one launch whose cost is bounded by memory traffic.

**Blocked by:** 12/Add Rank One Factorized Plan; 17/Compose Optimized Forward and Backward.

**Status:** in progress (19a and 19b-i delivered, 19b-ii open)

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64):

| family | plan | T=2048 forward before | after 19a | forward+backward before | after 19a |
| --- | --- | --- | --- | --- | --- |
| `gla` | elementwise | 4.59 ms | 1.87 ms | 16.5 ms | 7.9 ms |
| `mamba2` | elementwise | 4.42 ms | 1.92 ms | 16.7 ms | 8.0 ms |
| `simple-gla` | scalar-factorized | 0.30 ms | 0.30 ms | 15.2 ms | 7.3 ms |
| `linear` | scalar-factorized | 0.30 ms | 0.30 ms | 15.5 ms | 7.6 ms |
| `gdn` | rank-one | 81.6 ms | 81.6 ms | 169.5 ms | 169.5 ms |
| `rwkv7` | rank-one | 81.6 ms | 81.6 ms | — | — |

19b-i (delivered): a `factored-recurrence` Plan Representation and `tilelang-factored-recurrence` lowerer add `_stateful_factored.py`, one kernel that keeps every State column in registers, replays the authored whole-State or first-feature Axis Scale and first-feature Rank One operations per token, adds the injection outer products, and contracts the readout. Eligibility (`rank_one_factored_eligible`) requires one Transition, one contracting first-feature readout, disjoint single-feature injection factors, and at least one rank-one contraction, so the diagonal-only Programs keep the scalar and elementwise kernels that already fuse the same work into fewer, shorter kernels. The family is forward-only: with a compiled backward the plan stays on the previous path, exactly as `scalar-factorized` does.

| family | T=2048 forward before | after 19b-i | launches | workspace |
| --- | --- | --- | --- | --- |
| `gdn` | 81.6 ms | 3.73 ms | 1 + 20 | 0 (was 384 KiB/token) |
| `rwkv7` | 81.6 ms | 4.00 ms | 1 + 18 | 0 |
| `gdn` T=8192 | rejected | 17.7 ms | 1 + 19 | 0 |

Evidence: `test_gdn_factored_recurrence_matches_independent_reference` and `test_gdn_factored_recurrence_continues_from_a_returned_state` compare against `gated_delta_rule_reference` at T=130 with an initial State and a prefix/suffix continuation.

19a (delivered): `_compile_elementwise_prefix` is now a streaming kernel. Every thread owns one State element and walks its segment accumulating `(A_t, B_t)` with `S_t = A_t S_0 + B_t`, so the cross-thread scan, its barriers, and the shared-memory chunk loop are gone; per-axis, per-token-scalar, and full elementwise factors all lower to the same accumulation. The kernel also composes in reverse (`reverse=True`), which removed the seven flipped token-sized copies the elementwise VJP used to make.

Measured effect of 19a: the three backward scans fall from 7.5-8.5 ms to 1.0 ms, forward falls 2.4x, and the whole training step falls ~2x. Against Flash Linear Attention at T=2048 (`testing/benchmark_linear_attention.py`, both sides VALID against the FP32 oracle) `gla` moves 250x -> 11.9x -> 5.84x end-to-end and 366x -> 16.4x on graph replay across 20 and 19a. The remaining 5.84x is the per-token State materialization and the token-serial accumulation, which 21 and 19b remove.

What remains for 19b, in the measured order of impact:

1. **Rank-one backward (19b-ii)**: the factored family has no VJP yet, so a Program compiled with `backward=True` keeps the doubling scan and the dense per-token factor matrices (`gdn` training is still 169 ms at T=2048). The backward needs the State adjoint recurrence in reverse over `(l, r, c)`, the per-token gradients `dL/dl`, `dL/dr`, `dL/dc`, the injection and readout operand gradients, and the initial-State gradient; the `grad_r`/`grad_readout` terms need the per-token State, so the kernel must either recompute it per chunk or carry the WY matrix described below.

   Two findings from 19b-i that shape 19b-ii: every per-token parameter gradient (`dL/dr`, `dL/dq`, `dL/dc`, `dL/d(scale)`) is a reduction across State columns, so the backward has to be tile based rather than thread-per-column like the forward; and the State must come from storage or a checkpointed replay, not from inverting the forward ops.

   A rejected design, recorded because it cost a full implementation: reconstructing `s_{t-1}` from `s_t` by inverting each operation (scales divide, injections subtract, a rank-one update inverts through Sherman-Morrison) needs no extra storage, but it is numerically unusable. A decaying State shrinks like `exp(-sum |g|)`, so the values the walk recovers are the *forward* States, which round to zero in FP32 well before the sequence ends; dividing the surviving low bits back up amplifies the rounding error instead of the signal. Measured against the FP64 oracle on the GDN profile: 1e-2 relative error at T=8 and 1e13 at T=130, with the kernel's arithmetic otherwise matching. 19b-ii must therefore store State checkpoints and replay the intra-token operation chain forward for the intermediates, or carry the WY matrix described below so intra-chunk prefixes have a closed form.

   Design note for the implementation: within a chunk, a chain of rank-one updates `prod_s (I + c_s l_s r_s^T)` is `I + A W B^T` with `A = [l_1..l_n]`, `B = [r_1..r_n]` and an `n x n` coefficient matrix `W` from a triangular solve (the WY represention). That makes the chunk summary `O(n^2 R)` instead of `O(n R^2)`, the chunk scan cheap (`T/n` summaries), and the per-token epilogue a prefix of the same form. The same factors serve the VJP: the state adjoint recurrences in reverse over `(l, r, c)` plus the per-token gradients `dL/dl = ..., dL/dr = ..., dL/dc = l^T S r`, none of which need a `[T,H,R,R]` tensor.
2. **Readout fusion (see 21)**: the elementwise family still materializes per-token States and then reads them back, which is why the VJP spends ~2 ms in token-sized torch passes and why GLA forward is 6x the scalar-factorized schedule. Fusing the contraction into the streaming kernel removes those passes entirely.
3. **Chunk-level parallelism**: the streaming kernel parallelizes over State elements only (heads x R*C/threads blocks), which is enough at K=V=64..128 but not at small State sizes.

- [x] The elementwise prefix composes in one launch with no cross-thread scan, for per-axis and scalar factors, in dense and packed modes.
- [ ] One chunked path serves dense and diagonal summaries with chunk-level parallelism.
- [ ] The doubling scan is deleted or retained only as a CPU reference; no CUDA path reaches it.
- [ ] Authored propagation order, segmentation, `initial_state`/final-State semantics, packed mode, and gradient parity are unchanged against the current implementation, checked by the existing H20 suites.
- [ ] `gdn`, `rwkv7`, `gla`, and `mamba2` land within 2x of the scalar-factorized group at T=2048 with one launch per pass.
