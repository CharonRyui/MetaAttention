# 07: Support dynamic workspace and resource validation

**What to build:** Let one fixed compiled implementation derive launch bounds and reusable workspace from runtime dense and packed workload dimensions without selecting another plan, lowerer, or KernelConfig.

**Blocked by:** 05/Execute generic dense forward through a Mathematical Plan.

**Status:** ready-for-agent

- [ ] Runtime Batch, dense Sequence length, packed logical-sequence count, and packed total-token count determine launch and workspace sizes without code generation.
- [ ] Workspace is obtained and reused through a compiler-controlled allocation path; generated kernels perform no device-side dynamic allocation.
- [ ] Runtime workload dimensions do not alter Mathematical Plan, Lowered Candidate, KernelConfig, or binary identity.
- [ ] A workload outside the selected implementation's resource envelope raises a structured resource error without fallback, replanning, or recompilation.
- [ ] All-empty packed execution preserves State identity without unnecessary token workspace.
- [ ] Public tests cover workspace reuse and resource failure across dynamic dense and packed workloads.
