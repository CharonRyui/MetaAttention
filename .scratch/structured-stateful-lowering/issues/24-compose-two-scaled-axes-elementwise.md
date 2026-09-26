# 24: Compose two scaled axes elementwise

**What to build:** Let a Program whose Transition scales both State Features compose the two per-axis factors without falling back to the dense diagonal.

**Blocked by:** 20/Lower per-axis diagonal decay elementwise.

**Status:** ready-for-agent

`_dense_summary_required` still returns True when two different Feature roles carry a single-role Axis Scale, because the prefix kernel reads one factor per matrix element and a row factor times a column factor needs both. That is the shape of `gated_delta_rule`'s `left_gate`/`right_gate` profile and of `fla.ops.gla.fused_recurrent_gla(gk, gv)`, and it costs the same `[T,H,K,K]`/`[T,H,V,V]` materialization that 20 removed for a single axis.

- [ ] The elementwise prefix kernel takes a second factor slot, or lowers the pair without materializing a `[T,H,R,C]` tensor.
- [ ] The elementwise VJP returns one gradient per factor, reducing each along the axis the other factor spans, and still matches the FP32 recurrence for both gates.
- [ ] The chunk carry composes the pair across chunk boundaries in dense and packed modes.
- [ ] `gated_delta_rule(left_gate, right_gate)` and the two-Axis Scale GLA profile select a non-dense plan and stay within 2x of the single-axis cost for the same State shape.
- [ ] `_dense_summary_required` no longer treats two scaled axes as dense, and the guard in `_axis_scale_summary` that rejects the composition is removed with it.
