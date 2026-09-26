# 20: Lower per-axis diagonal decay elementwise

**What to build:** Represent a single-role Axis Scale as a per-axis elementwise factor instead of a dense diagonal matrix, so a feature-wise decay never materializes `[T,H,K,K]`.

**Blocked by:** 10/Add Axis Scale Structured Plan.

**Status:** ready-for-agent

Evidence: `_lower_parallel_scan` sets `needs_dense_transform` whenever a propagation is a rank-one or a single-role `AxisScale`, and `_axis_scale_summary` then builds `torch.diag_embed(factor) @ identity`, i.e. a `[T,H,K,K]` tensor per token — 384 KiB/token of workspace for a Program whose only reason to be dense is a feature-wise gate. The elementwise prefix kernel already resolves `scale[token, head, scale_row, scale_column]` with `scale_row = 0 if scale_rows == 1 else row`, so per-axis factors are supported by the kernel; only `tilelang_elementwise_prefix`'s shape guard (`scale.shape in (bias.shape, (*bias.shape[:2], 1, 1))`) and the summary builder reject them.

- [ ] A first-feature Axis Scale produces a `[T,H,K,1]` factor and a second-feature Axis Scale a `[T,H,1,V]` factor; neither builds a diagonal matrix.
- [ ] Two single-role Axis Scales compose as a row factor times a column factor: the prefix kernel accepts both, or the summary folds them without densifying.
- [ ] The elementwise shape guard admits per-axis factors and still rejects malformed ones.
- [ ] `gla`, `fused_recurrent_gla(gk, gv)`, `rwkv6`, and `mamba2` select the elementwise plan without a dense transform, and `retention_attention`/`simple_gla_attention` keep their scalar path.
- [ ] Workspace for axis-scale Programs is `O(T·H·(K+V))` plus the State output, verified through `WorkspacePolicy.bytes_per_token`.
- [ ] Forward outputs, final State, and gradients match the dense diagonal baseline elementwise.
