# 22: Rank candidates by a shape-aware estimate

**What to build:** Rank Lowered Candidates by their estimated cost for the concrete Feature sizes and workload instead of by the hard-coded per-lowerer constants, so a Program whose summary is diagonal cannot select a dense-scan candidate.

**Blocked by:** 14/Select feasible lowered candidates; 19/Replace the dense doubling prefix scan.

**Status:** ready-for-agent

Evidence: `select_candidate` orders by `LOWERERS[...].static_cost` (`tilelang-elementwise` 30 beats `tilelang-generic` 100 and `reference-stateful` 200 regardless of shape), while `CandidateEstimate` already computes `workspace_bytes_per_token` and is used only for feasibility filtering. `gla` therefore selects `tilelang-elementwise` and pays the dense doubling scan.

- [ ] The estimate covers bytes moved per token, per-token arithmetic, and expected launches, derived only from typed IR structure and concrete Feature sizes.
- [ ] Ranking uses the estimate; the deterministic lowerer/identity tie-break stays.
- [ ] `optimized=False` still selects generic execution only, and an infeasible Top candidate still leaves the Program valid with a lower-ranked plan.
- [ ] A test proves that a Program with a feature-wise decay or a rank-one propagation never selects a dense-doubling candidate.
- [ ] Selection decisions and their estimates are inspectable from the compiled operator for evidence collection.
- [ ] The scalar-factorized and rank-one families keep their current selections and measured latency.
