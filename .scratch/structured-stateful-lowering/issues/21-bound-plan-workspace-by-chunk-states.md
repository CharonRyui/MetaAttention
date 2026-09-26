# 21: Bound plan workspace by chunk States, not by tokens

**What to build:** Every specialized plan holds only chunk-level States and the output, so workspace stops growing with the token count past the chunk boundary.

**Blocked by:** 19/Replace the dense doubling prefix scan.

**Status:** ready-for-agent

Evidence: every plan reports the same 384 KiB/token (`WorkspacePolicy.bytes_per_token`), i.e. the per-token State `[T,H,K,V]` plus its scale/bias copies are materialized for the whole sequence. At T=8192 that is ~3 GiB, so 10 of 11 families are rejected with `RUNTIME_RESOURCE` before running; `linear` at T=2048 reads and writes 268 MB per pass (0.21 ms ≈ HBM roofline) for a 4 MB output.

- [x] The `tilelang-factored-recurrence` family keeps only the output and the final State: `bytes_per_token == 0` in inference, so it is accepted at T=32768. A compiled backward records the per-token State it replays from and charges `state_heads * rows * columns * 2` per token, which puts the 1 GiB envelope at T≈4096 for 8 heads of 128x128.
- [x] A regression test (`tests/unit/test_stateful_workspace.py`) pins the per-token charge of the factored and the summary plans and the token count where each stops fitting the envelope.
- [ ] Chunk States, chunk summaries, and the output are the only sequence-scaled buffers for the summary families too; per-token summaries are consumed inside the kernel that produces them.
- [ ] Every coverage family compiles and runs at T=32768 in dense and packed modes within the H20 envelope.
- [ ] Output and final State are unchanged against the pre-change implementation for every family the functional suite covers.

**Design note (measured).** The factored family's structure -- one block per (sequence, head), one State column in registers per thread, all operations replayed in the thread -- removes the per-token State but is *not* the schedule the summary families should adopt: for the same 8x128x128 State shape at T=2048 it runs in 4.65 ms while `tilelang-scalar-factorized` runs the same State in 0.30 ms. One block per sequence and head with 128 threads walks the whole sequence serially, which is latency bound, whereas the scalar-factorized kernel tiles the sequence and uses tensor-core GEMMs. The remaining families therefore need a *tile+GEMM chunked* kernel that generalizes the scalar-factorized one to a per-feature decay: inside a chunk the constant decay can be folded into the operands, but a per-feature decay makes the causal weight `exp(P_t[i] - P_s[i])` depend on the summed axis, so it does not reduce to rescaling one [T, T] score matrix. The form that keeps the tensor cores is a per-chunk state build (`M_t = diag(exp(-P_t)) * cumsum(diag(exp(P_s)) k_s v_s^T)`, cheap and tile-parallel) followed by a GEMM with the pre-scaled queries, with the chunk-end state carried by the existing block-level scan.
