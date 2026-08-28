# 05: Execute generic dense forward through a Mathematical Plan

**What to build:** Compile every Target-supported Scanable Program through an explicit generic dense affine Mathematical Plan, Lowered Candidate, and separate KernelConfig while preserving the public forward behavior.

**Blocked by:** 04/Execute mandatory safe simplification.

**Status:** ready-for-agent

- [ ] Mathematical Plan represents only algorithm structure and equivalent mathematical representation, not kernel parameters, schedules, or measurements.
- [ ] Lowered Candidate identifies the backend implementation separately from Mathematical Plan and KernelConfig.
- [ ] A generic dense affine Mathematical Plan and lowerer are available for every supported Scanable Program.
- [ ] A previously unseen legal Algorithm IR composition compiles without a model, profile, frontend, or whole-program dispatch case.
- [ ] A lowerer mismatch does not change Scanable/Recurrent classification and falls through to the generic candidate.
- [ ] Generic forward outputs and optional final State match independent dense and packed references through the public seam.
