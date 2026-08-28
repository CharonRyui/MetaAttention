# 14: Select feasible Lowered Candidates

**What to build:** Choose deterministically between generic and structured Lowered Candidates using correctness, capability, resource feasibility, and static estimates while preserving successful compilation when specialized attempts fail.

**Blocked by:** 10/Add the Axis Scale structured plan.

**Status:** ready-for-agent

- [ ] The selector accepts multiple lowerers for one Mathematical Plan and multiple Mathematical Plans for one Simplified IR.
- [ ] Semantic, Target-capability, context, workspace, register, shared-memory, and other resource failures remove a candidate before publication.
- [ ] The generic dense affine baseline remains available throughout selection.
- [ ] With autotuning disabled and no cached measurements, selection uses deterministic static estimates and a stable tie-break.
- [ ] All failed specialized attempts produce one aggregate warning naming candidate, stage, exception type, concise reason, and selected result.
- [ ] Failure of the generic baseline raises StatefulCompilationError rather than returning an invalid callable.
- [ ] Failed attempts create no persistent negative cache entries.
