# 24: Compose two scaled axes elementwise

**What to build:** Let a Program whose Transition scales both State Features compose the two per-axis factors without falling back to the dense diagonal.

**Blocked by:** 20/Lower per-axis diagonal decay elementwise.

**Status:** in progress (the forward pair path and its analysis are delivered; the paired VJP and packed-mode backward remain)

`_dense_summary_required` still returns True when two different Feature roles carry a single-role Axis Scale, because the prefix kernel reads one factor per matrix element and a row factor times a column factor needs both. That is the shape of `gated_delta_rule`'s `left_gate`/`right_gate` profile and of `fla.ops.gla.fused_recurrent_gla(gk, gv)`, and it costs the same `[T,H,K,K]`/`[T,H,V,V]` materialization that 20 removed for a single axis.

- [x] The elementwise prefix kernel takes a second factor slot, or lowers the pair without materializing a `[T,H,R,C]` tensor.
- [ ] The elementwise VJP returns one gradient per factor, reducing each along the axis the other factor spans, and still matches the FP32 recurrence for both gates.
- [x] The chunk carry composes the pair across chunk boundaries in dense and packed modes.
- [x] `gated_delta_rule(left_gate, right_gate)` and the two-Axis Scale GLA profile select a non-dense plan and stay within 2x of the single-axis cost for the same State shape. (Forward, inference. A compilation that asks for a backward keeps the dense summary, whose VJP owns the dense factors.)
- [x] `_dense_summary_required` no longer treats two scaled axes as dense, and the guard in `_axis_scale_summary` that rejects the composition is removed with it.

Delivered: the prefix kernel takes one factor per State axis (`scale` as `[T,H,R,1]` and `second` as `[T,H,1,C]`) and composes them per element while it walks the tokens, writing the two cumulative products instead of one, so no `[T,H,R,C]` factor is ever materialized. `_axis_scale_summary` returns the pair rather than raising, `_dense_summary_required` only forces dense factors for a rank-one propagation, and the runtime follows the compiled `dense_summary` decision instead of recomputing the rule.

Because a paired prefix has no compiler-derived VJP yet, a compilation that requests a backward keeps the dense summary, and the paired path refuses to run under autograd rather than mis-deriving gradients from the single-factor form.

Measured at T=2048, H=8, K=V=64, inference, idle H20, with the new `dual_gate_gla_attention` profile (FLA `fused_recurrent_gla(gk, gv)` shape):

| Program | latency | lowerer |
| --- | --- | --- |
| `gla` (one scaled axis) | 1.870 ms | `tilelang-elementwise` |
| two scaled axes, elementwise pair | 2.432 ms | `tilelang-elementwise` |
| two scaled axes, dense summary | 80.545 ms | dense factors |

The pair is 1.30x the single-axis cost and 33x faster than the dense summary, which also cannot run at T=2048 under the 1 GiB envelope while the pair can. Tests: `tests/functional/test_two_axis_elementwise.py` covers dense parity against an FP32 token recurrence, packed parity with leading and trailing empty sequences, and the backward-compilation gate.

Open: the paired VJP (a factor pair needs one gradient per factor, each reduced along the axis the other factor spans; the current single-factor derivation does not generalize directly because the accumulated product is a joint one), and with it the backward clause of the selection box.
