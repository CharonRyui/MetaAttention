# 21: Bound plan workspace by chunk States, not by tokens

**What to build:** Every specialized plan holds only chunk-level States and the output, so workspace stops growing with the token count past the chunk boundary.

**Blocked by:** 19/Replace the dense doubling prefix scan.

**Status:** ready-for-agent

Evidence: every plan reports the same 384 KiB/token (`WorkspacePolicy.bytes_per_token`), i.e. the per-token State `[T,H,K,V]` plus its scale/bias copies are materialized for the whole sequence. At T=8192 that is ~3 GiB, so 10 of 11 families are rejected with `RUNTIME_RESOURCE` before running; `linear` at T=2048 reads and writes 268 MB per pass (0.21 ms ≈ HBM roofline) for a 4 MB output.

- [ ] Chunk States, chunk summaries, and the output are the only sequence-scaled buffers; per-token summaries are consumed inside the kernel that produces them.
- [ ] `WorkspacePolicy`/`required_bytes` encode the chunk-State bound, and `Target.workspace_bytes` is re-derived from it.
- [ ] Every coverage family compiles and runs at T=32768 in dense and packed modes within the H20 envelope.
- [ ] A regression test asserts that no plan's `bytes_per_token` exceeds the output bytes per token plus the chunk-State bytes per token.
- [ ] Output and final State are unchanged against the pre-change implementation for every family the functional suite covers.
