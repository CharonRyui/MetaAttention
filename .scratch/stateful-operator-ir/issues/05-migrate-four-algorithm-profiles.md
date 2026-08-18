# 05 — Migrate four algorithm profiles

**What to build:** Express Gated Linear Attention, RetNet recurrent, Mamba2, and GDN through explicit Algorithm IR and the generic H20 compiler. Preserve example factory arguments, but migrate every StatefulOperator call to Target, typed constants, named outputs, StateTuple, and ExecutionResult.

**Blocked by:** 04 — Lower derived backward and state continuation.

**Status:** ready-for-agent

- [ ] GLA uses compiler-known scale propagation, injection, and post-transition readout nodes.
- [ ] RetNet recurrent uses its declared structured decay rather than frontend modifier semantics.
- [ ] Mamba2 is corrected to selective diagonal propagation and its intended compiler-known readout semantics.
- [ ] GDN uses scale plus rank-one delta propagation and raw differentiable beta without a dedicated execution implementation.
- [ ] Each example keeps a callable operator factory but exposes no library-level model factory.
- [ ] Independent PyTorch references validate outputs, all declared Tensor Input gradients, and initial/final-state paths on H20.
- [ ] Each profile covers omitted/explicit initial state, continuation, named results, and positive unaligned lengths.
- [ ] No model name, frontend identity, or whole-profile pattern participates in lowering or cache identity.
