# 21: Bound plan workspace by chunk States, not by tokens

**What to build:** Every specialized plan holds only chunk-level States and the output, so workspace stops growing with the token count past the chunk boundary.

**Blocked by:** 19/Replace the dense doubling prefix scan.

**Status:** in progress (the factored-recurrence family carries no per-token State in inference, the resource contract is tested, and the per-axis diagnosis below is measured; the summary families still materialize one per token)

Evidence: every plan reports the same 384 KiB/token (`WorkspacePolicy.bytes_per_token`), i.e. the per-token State `[T,H,K,V]` plus its scale/bias copies are materialized for the whole sequence. At T=8192 that is ~3 GiB, so 10 of 11 families are rejected with `RUNTIME_RESOURCE` before running; `linear` at T=2048 reads and writes 268 MB per pass (0.21 ms ≈ HBM roofline) for a 4 MB output.

- [x] The `tilelang-factored-recurrence` family keeps only the output and the final State: `bytes_per_token == 0` in inference, so it is accepted at T=32768. A compiled backward records the per-token State it replays from and charges `state_heads * rows * columns * 2` per token, which puts the 1 GiB envelope at T≈4096 for 8 heads of 128x128.
- [x] A regression test (`tests/unit/test_stateful_workspace.py`) pins the per-token charge of the factored and the summary plans and the token count where each stops fitting the envelope.
- [ ] Chunk States, chunk summaries, and the output are the only sequence-scaled buffers for the summary families too; per-token summaries are consumed inside the kernel that produces them.
- [ ] Every coverage family compiles and runs at T=32768 in dense and packed modes within the H20 envelope.
- [ ] Output and final State are unchanged against the pre-change implementation for every family the functional suite covers.

**Measured gap (T=512, H=8, K=V=64, `testing/benchmark_linear_attention.py`).** Kernel-only graph replay separates the schedule from launch and readout overhead:

| Program | MetaAttention kernel | stock FLA kernel | MetaAttention end to end | stock FLA end to end |
| --- | --- | --- | --- | --- |
| `simple-gla` (scalar decay, scalar-factorized) | 0.062 ms | 0.024 ms | 0.135 ms | 0.253 ms |
| `gla` (per-feature decay, elementwise) | 0.450 ms | 0.044 ms | 0.690 ms | 0.330 ms |

Two separate gaps. The `gla` kernel is 10.2x off `chunk_gla`, which is the schedule difference the design note below explains, and its end-to-end time carries a further 0.24 ms because the elementwise family materializes the per-token State and contracts the readout in a separate torch pass -- exactly the buffer this ticket removes. The scalar-decay kernel is 2.6x off `chunk_simple_gla` but wins end to end, and stock FLA is reported INVALID against the oracle there (its toggle semantics differ), so only the `gla` row is a like-for-like baseline.

**Where the per-axis cost actually is (measured, T=512, H=8, K=V=64, idle H20).** The `gla` forward is three passes over one `[T, H, 64, 64]` fp32 tensor, each materializing it in global memory:

| stage | time |
| --- | --- |
| dense injection outer product (torch, builds `[T,H,R,C]`) | 94 us |
| elementwise prefix kernel (writes the per-token State) | 102 us |
| readout einsum (reads the per-token State) | 77 us |

No stage is pathological; each runs near its own traffic, and their sum is what one fused pass should replace. Two fused candidates measured against them:

| schedule | time | note |
| --- | --- | --- |
| `tilelang-factored-recurrence` kernel, same chain and shape | 588 us | one State column per thread in registers, `sequence_count * heads` = 8 blocks with 64 of 128 threads active |
| elementwise prefix kernel, same chain and shape | 102 us | thread per State element, `heads * cdiv(R*C, 128)` = 256 blocks |

The register-per-column shape is therefore *not* the schedule to copy as it stands: it gives up 64x of the parallel width (512 active lanes against 32768) and loses 5.8x for identical per-token work. Widening the factored eligibility to per-axis chains was implemented, measured at 0.690 ms -> 0.990 ms end to end, and reverted.

**Design note.** The lever is block count, not per-element arithmetic (~192 flops per thread per token). The chunked form is what buys blocks: take the per-chunk State carry from a cheap first pass over `[chunks, heads, R, C]`, then give each `(chunk, column block)` pair its own block with the State in registers and a thread-local readout row sum. A per-feature decay is compatible with the tensor-core form after all: inside a chunk, pre-scale the operands (`q' = diag(exp(P_t)) q`, `k' = diag(exp(-P_s)) k`) so that `q'_t . k'_s == q_t . diag(exp(P_t - P_s)) . k_s`, and the chunk-end summary becomes `diag(exp(-P_end)) * sum_s k'_s v_s^T`. That is the shape a per-axis chunked kernel should take.
