# 23: Bring v2/v3 vocabulary Programs onto the fast path

**What to build:** Give multi-State, rank-1, product-Feature, and expression-readout Programs the same chunk-summary execution the specialized v1 plans get, so the reference lowering stops being their only executor.

**Blocked by:** 19/Replace the dense doubling prefix scan; 21/Bound plan workspace by chunk States; 02/Establish metadata-only packed compilation.

**Status:** in progress (the per-State layout box is verified; per-axis scales elementwise, the streaming State scan, and the per-State charge are delivered; the multi-State chunk scan, the readout epilogue, and the remaining envelope case are open)

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

**First box verified satisfied; the ticket's premise was stale.** Reading the code rather than trusting the note: `_scan_state` allocates each per-token State at `[tokens, heads, left_size, right_size]` taken from that State's own `view.feature_roles`, `_injection_terms` builds the per-token injection at the same sizes, and `_contract_state_term` contracts by `state_spec.feature_roles` (`thf,thfv->thv` for a first-feature contraction). The union `feature_axes` map is used only to locate a role inside the *input-derived operand*, which genuinely carries several roles. So "each State carries its own layout and shape; no union `[T,H,*roles]` tensor spans the Program" already holds; the union layout spans inputs, by design, not States.

That means the ticket's stated evidence ("the reference lowering materializes per-token States in that layout, which is ... the reason a product-Feature Program cannot run at all") is wrong about the mechanism. The real blockers, in order:

1. `_affine_summary` sets `dense = any(rank-one or single-role AxisScale over the propagations)` and then builds dense `[T,H,R,R]` and `[T,H,C,C]` factors for any State whose Transition carries a per-axis scale. That is exactly the problem tickets 20 and 24 removed for the v1 families, and it is why `hgrn`, `gsa` and `rwkv6` are slow. The v1 answer -- one elementwise factor per scaled axis, with the pair composed per element -- applies unchanged here, and this is the highest-value step in the ticket.
2. `based` and `rebased` are rejected at T=512 by the workspace envelope, not by a layout rule: a Product Feature State makes `dense = rows^2 + columns^2 + rows*columns` large in the estimate, and the reference branch charges `heads * (2 * dense + 4 * states) * 4` per token. Fixing (1) shrinks that charge directly for the diagonal cases; the product-feature cases then need the estimate priced against what the kernel actually materializes.
3. Per-token State materialization itself (the workspace cliff and the readout pass) is what the chunk scan removes, and that is the second half of this ticket, as the earlier note says.

So the first box closes by inspection; the work starts at `_affine_summary`.

**Delivered: per-axis scales stay elementwise in the reference lowering.** `_affine_summary` no longer forces dense `[T,H,R,R]`/`[T,H,C,C]` factors for a single-feature Axis Scale; it composes one factor per axis exactly as tickets 20 and 24 do for the v1 families, and `_scan_state` only asks for dense matrices when a rank-one contraction is present. The reference estimate was updated with it: without a rank-one contraction it now charges the factor, the injection, and the per-token State instead of the dense pair (previously it over-charged because the model assumed the matrices were still there, which briefly pushed gsa and rwkv6 past the envelope at T=2048).

Measured with `testing/benchmark_stateful_plans.py` on an idle H20 (card 4):

| family | T=512 before | T=512 after | T=2048 before | T=2048 after | workspace/token (T=2048) |
| --- | --- | --- | --- | --- | --- |
| `gsa` | 9.6 ms | 2.66 ms | 40.0 ms | 8.16 ms | 192 KiB |
| `rwkv6` | 9.9 ms | 3.61 ms | 44.7 ms | 14.80 ms | 384 KiB |
| `hgrn` | 2.2 ms | 1.24 ms | 2.8 ms | 1.45 ms | 0.8 KiB |
| `normalized-linear` | 4.8 ms | 4.40 ms | rejected | rejected | 768 KiB |
| `based` | rejected | 17.2 ms | rejected | 66.2 ms | 36 KiB |
| `rebased` | rejected | rejected | rejected | rejected | - |

`based` and `rebased` are the product-Feature cases: `based` now runs where it was rejected, `rebased` still exceeds the envelope. Verified by the H20 coverage suites (`test_fla_coverage.py`, `test_gdn_h20.py`, `test_linear_attention_examples.py`, `test_chunked_elementwise.py`: 25 pass) and 153 unit tests.

What is left in this ticket: the readout still costs one torch op per token (90-654 launches per family per the table above), so the readout epilogue against chunk States is the next lever, and with it the chunk scan for multi-State Programs. `rebased` needs the product-Feature charge revisited once its readout stops dominating.

**Delivered: the reference State scan is the streaming kernel.** `_scan_state` no longer runs the host-driven doubling scan for an elementwise summary on CUDA. It calls the same streaming prefix kernel the v1 elementwise family uses, gated on no bound value requiring grad so a differentiable call keeps the torch scan and autograd is untouched. That removes the per-halving passes and their token-sized intermediates, which is where the launch counts in the table below came from.

Measured with `testing/benchmark_stateful_plans.py --seqlen 2048` on an idle H20 (card 4), cumulative across this ticket's two commits:

| family | baseline | after elementwise summary | after streaming scan | launches (baseline -> now) |
| --- | --- | --- | --- | --- |
| `gsa` | 40.0 ms | 8.16 ms | **2.13 ms** | 220 -> 48 |
| `rwkv6` | 44.7 ms | 14.80 ms | **2.81 ms** | 122 -> 36 |
| `hgrn` | 2.8 ms | 1.45 ms | **0.58 ms** | 106 -> 20 |
| `based` | rejected | 66.2 ms | **11.36 ms** | 654 -> 139 |
| `normalized-linear` | rejected | rejected | rejected | - |

Against 0.14-0.32 ms for the specialized plans, that leaves `hgrn` inside this ticket's 3x bar and `gsa` (6.7x) and `rwkv6` (8.8x) still above it, where they started at 125x and 140x. `normalized-linear` is now the outlier: it still exceeds the envelope at T=2048 even though its siblings with the same shape do not, so its per-State charge needs the same treatment the others just got. The remaining structural levers are unchanged: the readout is still evaluated over the materialized per-token State rather than against chunk States, and multi-State Programs cannot reach the chunked schedule.

**Delivered: the reference charge is per State.** The estimate priced the reference lowering as `first_state_size * state_count`, so a Program with a rank-2 numerator and a rank-1 denominator paid twice for the larger one and `normalized-linear` was rejected at T=2048 while its siblings with the same structure ran. `EstimationShape` now carries the sum of each State's own element count and the reference branch charges that. Measured at T=2048 on an idle H20: `normalized-linear` compiles and runs in 3.02 ms (was `RUNTIME_RESOURCE`), with `gsa` 2.10, `rwkv6` 2.83 and `hgrn` 0.63 ms unchanged.

All five non-product coverage families now run at T=2048 (`rebased` was still rejected in the last full sweep and is the remaining envelope case). Against the 0.14-0.32 ms specialized band that is `hgrn` 2.0x, `gsa` 6.6x, `rwkv6` 8.8x and `normalized-linear` 9.4x, so three families are still above this ticket's 3x bar. Their remaining cost is the readout, which is still evaluated over the materialized per-token State rather than against chunk States -- the next item.

**`rebased` rejection is honest, not a mis-charge.** Measured at T=512 it runs in 2.87 ms but declares 1560 KiB/token, so T=2048 is 3.2 GB and correctly exceeds the 1 GiB envelope. That number is the per-State sum the reference lowering genuinely materializes (elementwise factor, injection, and per-token State for each of its States), not the `first_state * state_count` artefact `normalized-linear` had. So there is nothing to re-price here: `rebased` becomes reachable at T=2048 exactly when multi-State Programs reach the chunked schedule, whose chunk States cost `rows * columns / 64` elements per token -- about 1 KiB/token for this shape, against 1560.

That makes the three remaining boxes one piece of work: extend `_stateful_chunked.py` from one State to many, give each State its own carry pair, and evaluate the readout epilogue for contractions against chunk States (expression readouts over several States need the per-token State and would keep the reference path until a chunk-level form exists for them).

**Note for pinning the estimate per family.** A first attempt at a regression test that pins each family's declared charge was reverted mid-edit rather than committed half-applied, but it produced one useful datum: `hgrn` at `heads=1, dim_value=64` declares 6144 bytes per token, which is `1 * 3 * 512 * 4` -- so its State carries 512 elements, not the 64 a single per-Feature vector would suggest (its roles include the input feature axis alongside the state axis). A per-family pin must therefore read the roles from the operator and derive the expected charge from them, rather than assuming one element per role; the two estimate bugs this session were both silent capability changes, so such a pin is worth having.

**Implementation checklist for the last piece (multi-State chunk scan + readout epilogue).** Everything below was read or measured in this session, so it is the shortest path to closing the three open boxes.

1. `_stateful_chunked.py`: generalize the entry to per-State inputs. `chunked_elementwise_dense` takes `gate_rows`, `key`, `value`, `query`, `initial_state`, `output_dtype` for one State today; make the per-State tensors tuples and loop the two kernels once per State, giving each State its own `carry_scale`/`carry_bias` pair sized `[chunks, heads, rows_i, columns_i]`. The kernels themselves need no change: they are already parameterized by `rows`/`columns`, so each State gets its own compiled instance (the `lru_cache` keys already include the sizes).
2. `stateful_operator._lower_chunked_elementwise`: widen the gate and gather per State. Today it requires `len(transitions) == 1`, one injection and one readout. Drop the Transition count to `>= 1`, and per State take that State's Transition injection factors (evaluate both factor expressions as `_lower_factored_recurrence` does) plus the readout operand. Keep the requirement that every readout is a `StateContraction` on its own State's first Feature: an expression readout over several States needs the per-token State and must stay on the reference path until a chunk-level form exists for it, which is exactly this ticket's remaining third box.
3. `_stateful_plan.py`: extend `chunked_eligible` with the same per-State conditions (dense, no compiled backward, diagonal Transition, CUDA target, one injection per State, contraction readouts only) and charge the aligned bound per State -- `sum(heads * rows_i * columns_i * 4 // CHUNK_TOKENS)` per token instead of the current single-State figure.
4. Verification order, cheapest gate first: (a) the CPU coverage tests (`tests/unit/test_fla_coverage.py`) still compile and run every family; (b) add a two-State case to `tests/functional/test_chunked_elementwise.py` comparing against an FP32 recurrence at a chunk-aligned length, boundary tokens included; (c) the H20 suites (`test_fla_coverage.py`, `test_gdn_h20.py`, `test_linear_attention_examples.py`, `test_two_axis_elementwise.py`, `test_chunked_elementwise.py`) as the oracle gate; (d) `testing/benchmark_stateful_plans.py --seqlen 512 2048 --families gsa,rwkv6,normalized-linear,rebased` and compare against the 0.14-0.32 ms specialized band -- the bar is 3x, and `rebased` should stop being rejected because its chunk States cost `rows*columns/64` elements per token against the 1560 KiB/token its reference path declares.

Do not start with the estimate or the policy: the charge change is the last step, after the kernels and the gate agree with what they actually materialize. That ordering is what kept the two estimate bugs this session from becoming capability regressions in both directions.

