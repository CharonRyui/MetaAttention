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

**Implementation map (surveyed, not yet built).** The union layout has exactly two consumers, both in `_stateful_reference.py`:

- `feature_layout` (line 105) builds the ordered union of every Feature role over inputs and States and returns the axis map.
- line 155 lays out *inputs* as token-major `[tokens, heads, *union features]`; line 260 passes the same axis map into `_scan_state`, which is where the per-token State is materialized in that union shape.

So the first box is narrower than it looks: the input layout can stay union-based (inputs really are shared per Program), and what has to change is that `_scan_state` allocates and carries each State with its *own* `state.feature_roles` order and shape. The expression evaluator already keys everything by role (`axes[role]`), so the change is a substitution at the state-term sites -- `StateTerm`, `Contract`, `Outer`, `Softmax` -- from the union axis map to a per-State one, with both maps available where a State term and an Input term meet.

Payoff, in the order it unblocks the rest of this ticket: a State no longer pays for Feature roles it does not use, the per-token State buffer shrinks to `[tokens, heads, *own roles]`, and that per-State shape is exactly what the chunk scan needs to take over from the reference lowering.

Scope correction on this ticket's second box. The chunked schedule from ticket 19 reached the *v1* diagonal families only: both the compile-time eligibility and the runtime gate require one Transition, one injection, and one readout, so no multi-State or multi-readout Program can reach it. "Rank-1 and diagonal States on the elementwise chunk scan" is therefore untouched for the v2/v3 vocabulary -- `hgrn` is the one coverage family the elementwise family serves today, and only because it is single-Transition. This box follows the per-State layout work rather than preceding it. Then the readout epilogue can be evaluated against chunk States, and the `_stateful_chunked.py` schedule already built for ticket 19 extends to multi-State Programs once each State carries its own carry buffer.

