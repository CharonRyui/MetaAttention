# 05 — Add packed variable-length execution

**What to build:** Extend the same generic forward/backward compiler to token-major packed values, dynamic device-local sequence offsets, independent per-sequence State, empty sequences, and per-sequence inactive tails.

**Blocked by:** 04 — Lower Backward IR and State continuation.

**Status:** ready-for-agent

- [ ] Packed values use contiguous `[total_tokens, heads, features...]` layout and shared CUDA `int32` offsets.
- [ ] Offsets validate shape, device, start, end, monotonicity, and token count without entering compiled-artifact identity by value.
- [ ] Offset shape/dtype, total tokens, and sequence count participate in specialization identity.
- [ ] Recurrence resets at every offset and State order matches logical sequence order.
- [ ] Interior, leading, trailing, and all-empty sequences preserve identity semantics.
- [ ] Empty-sequence final-State cotangent passes directly to initial State.
- [ ] Every per-sequence tail lane is affine identity and contributes no output or Tensor-input gradient.
- [ ] Different offset contents reuse one compatible compiled specialization.
- [ ] Dense and packed encodings of equivalent sequences match outputs, final State, and gradients.
- [ ] Packed output-only, State-only, joint-loss, continuation, grouped-head, and alias cases pass H20 reference comparison.
