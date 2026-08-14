# 08 — Run GDN through StatefulOperator

**What to build:** Extend Algorithm IR with ordered propagation composition and rank-one delta propagation, then run Gated Delta Rule through the sole IR-first `StatefulOperator` interface and the existing optimized H20 implementation. Retain `GDNEngine(device)` only as a deprecated adapter that constructs equivalent IR and uses the same compilation seam.

**Blocked by:** 03 — Support initial state, final state, and continuation; 06 — Enforce Algorithm IR validation and structural cache identity.

**Status:** ready-for-agent

- [ ] Algorithm IR expresses GDN exactly as `D_t = exp(g_t) * S_{t-1}`, `S_t = D_t - beta_t * k_t * (k_t^T * D_t) + k_t * (beta_t * v_t)^T`, and `o_t = (scale * q_t)^T * S_t`; readout observes the post-transition state.
- [ ] `gate` remains a log-space FP32 Tensor Input and `Exp(gate)` is explicit in IR. `RankOneDelta` accepts arbitrary unnormalized keys and raw differentiable beta without normalization, clamping, or value-range validation.
- [ ] `RankOneDelta` and ordered propagation composition have validated shape, dtype, canonicalization, and compiler-derived VJP semantics; frontends cannot supply or override Backward IR.
- [ ] GDN uses the existing grouped Head Mapping to broadcast query/key heads to value/state heads and reduce query/key gradients when `Hv % Hk == 0`.
- [ ] New GDN code constructs the general IR nodes explicitly and invokes `StatefulOperator` with keyword-only named tensors and `return_final_state`; no public GDN-specific IR factory or model-name dispatch is introduced.
- [ ] Lowering selection uniquely matches IR structure and backend capability. The H20 GDN lowering rejects unsupported hardware, dimensions, dtypes, layouts, and per-invocation sequence alignment before code generation.
- [ ] The existing staged gate-cumsum, KKT, state/output, and backward kernels implement the selected lowering without eager fallback or a second compiler/cache path.
- [ ] IR-first GDN preserves the existing BF16/FP32 precision contract, default and custom readout-query scale semantics, GVA, optional named initial state, named final state, aligned continuation, and gradients from output and final state.
- [ ] `GDNEngine(device)` preserves its existing positional call, `output_final_state` flag, and naked state-tensor return contract; emits `DeprecationWarning`; maps state to the named `memory` entry; and caches equivalent `StatefulOperator` instances by resolved scale.
- [ ] CPU-safe tests cover IR validation, canonical identity, VJP behavior, lowering capability diagnostics, keyword binding, and legacy warning/conversion without asserting internal representation.
- [ ] H20 functional tests compare IR-first and legacy-adapter output, final state, and all supported gradients against the FP64 reference for equal heads, GVA, one/multiple chunks, zero/nonzero state, and default/custom scale within the existing 2% relative-L2 bound.
