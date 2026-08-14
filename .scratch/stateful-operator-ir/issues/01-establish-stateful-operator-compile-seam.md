# 01 — Establish the Stateful Operator compile seam

**What to build:** Introduce one stateful-operator compilation interface that owns runtime compilation, tuning inputs, generated-module loading, and kernel cache behavior while preserving every existing `LinearAttentionEngine` call. This prefactor makes the existing optimized linear lowering callable from the future Algorithm IR without creating a second compilation path.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Existing Gated Linear Attention, RetNet recurrent, and Mamba2 frontends retain their current output and gradient behavior through the new compilation seam.
- [ ] Runtime compilation, generated-module loading, tuning options, and cache behavior have one implementation shared by current and future stateful frontends.
- [ ] The seam accepts a backend-neutral operator description without exposing TileLang template internals to callers.
- [ ] Existing CPU-safe unit and official example import tests pass.
- [ ] Relevant existing GPU functional cases pass on supported hardware.
- [ ] No eager fallback, model-name dispatch, or duplicate legacy compiler is introduced.
