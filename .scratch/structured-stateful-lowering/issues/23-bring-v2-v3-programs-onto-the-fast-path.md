# 23: Bring v2/v3 vocabulary Programs onto the fast path

**What to build:** Give multi-State, rank-1, product-Feature, and expression-readout Programs the same chunk-summary execution the specialized v1 plans get, so the reference lowering stops being their only executor.

**Blocked by:** 19/Replace the dense doubling prefix scan; 21/Bound plan workspace by chunk States; 02/Establish metadata-only packed compilation.

**Status:** ready-for-agent

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64) against 0.14–0.32 ms for the specialized plans:

| family | T=512 | T=2048 | note |
| --- | --- | --- | --- |
| `normalized-linear` | 4.8 ms | rejected | `RUNTIME_RESOURCE` |
| `gsa` | 9.6 ms | 40.0 ms | 0/323 launches |
| `rwkv6` | 9.9 ms | 44.7 ms | 0/173 launches |
| `hgrn` | 2.2 ms | 2.8 ms | elementwise scan |
| `based`, `rebased` | rejected | rejected | `RUNTIME_RESOURCE` |

`feature_layout` builds one union layout over every Input and State Feature role and the reference lowering materializes per-token States in that layout, which is both the workspace cliff and the reason a product-Feature Program cannot run at all.

- [ ] Each State carries its own layout and shape; no union `[T,H,*roles]` tensor spans the Program.
- [ ] Rank-1 and per-axis diagonal States reuse the elementwise chunk scan; dense States reuse the chunk-summary scan.
- [ ] `StateTerm`, `Contract`, `Outer`, `Softmax`, and prior-State readouts are evaluated against chunk States in the epilogue, so the readout adds no full-sequence pass.
- [ ] Every v2/v3 coverage family compiles and runs at T=4096 within the H20 envelope, and each is at most 3x the specialized plan for the same State shape.
- [ ] CPU parity tests and the H20 coverage parity tests keep passing unchanged, and per-family H20 numbers are recorded as evidence.
