# MetaAttention

## Language

**Gated Delta Rule (GDN)**:
The QLA operator targeted by this repository. It is distinct from scalar-decay gated retention: GDN takes query, key, value, log-space gate, and beta, and uses a delta-corrected recurrent state update.
_Avoid_: Gated Attention, GLA

**QLA**:
The upstream FlashQLA implementation of Gated Delta Rule chunked prefill. In this repository, “QLA adaptation” means adding a dedicated GDN path, not changing the existing scalar-decay linear-attention path.

**First Release**:
The initial GDN contract: H20/sm89 BF16 fixed-length training with forward and backward execution, GVA, and initial/final recurrent state. Variable-length and automatic intra-card context parallelism are excluded.

**GDN Engine**:
The dedicated module that owns GDN lowering and kernel execution. It is separate from the scalar-decay LinearAttentionEngine.

**Head-first Layout**:
The GDN Engine public tensor layout: query and key are `[B,Hk,T,K]`, value is `[B,Hv,T,V]`, and gate and beta are `[B,Hv,T]`. This is the native MetaAttention layout.
_Avoid_: Token-first Layout

**GDN State**:
The optional fp32 recurrent state with layout `[B,Hv,128,128]`. A missing initial state means zero state; final state is returned only when explicitly requested.

**GDN Grouped Value Attention (GVA)**:
The first-release head mapping where `Hv % Hk == 0`. Each query/key head serves `Hv/Hk` value and recurrent-state heads.

**GDN Chunk Alignment**:
The first release requires `T % 64 == 0`. Unaligned input is rejected explicitly; the engine never truncates or pads sequences implicitly.

**H20 GDN Adapter**:
The sm89 TileLang implementation of GDN. It follows FlashQLA's mathematical reference and test behavior but does not reuse Hopper-only warp-specialized kernels.

**GDN Precision Contract**:
Query, key, value, and output use bfloat16. Log-space gate, beta, initial state, final state, and state accumulation use float32.
# Stateful Operator Domain

## Glossary

**Algorithm IR** — A backend-independent description of a stateful sequence operator: input transforms, a structured state transition, and a state readout. It defines model mathematics, not a GPU schedule.

**State transition** — The per-token transformation from a prior state to a next state. In the first refactor phase it is a structured affine transition, `s_next = L(inputs) * s + b(inputs)`.

**Structured affine transition** — A state transition whose propagation is one of the compiler-recognized forms: identity, elementwise scale, or diagonal scale, plus an input-derived injection. Its composition and backward rules are compiler-derived.

**State readout** — The per-token computation performed after the state transition. It produces output from the updated state and transformed inputs, so the current token's state injection is visible to its output.

**State injection** — The input-derived additive term in a structured affine transition. The first phase recognizes outer product, elementwise product, direct input, and zero injection.

**State tuple** — An ordered collection of named tensor states. Each state has its own shape and dtype, and transitions and readouts reference states by name. A lowering strategy may reject state combinations it does not support.

**Initial state** — An optional caller-provided state tuple used before the first token. Missing initial state means a zero-initialized state tuple.

**Final state** — The state tuple after the final token. Callers may request it for streaming or continuation; compatibility frontends return only token outputs by default.

**Frontend sugar** — Model-facing conveniences such as `q_mod`, `k_mod`, `v_mod`, and `decay_mod`. These are lowered into Algorithm IR and are not core compiler semantics.

**IR-derived backward** — The backward graph is derived from Algorithm IR node VJPs. Frontends do not provide independent backward functions; lowering strategies implement the derived graph.

**Head mapping** — The explicit relationship between input heads and state heads, including valid group broadcasting and reduction rules. It is Algorithm IR semantics, not a kernel layout inference.

**Tensor input** — A named, typed external tensor available to input transforms, transitions, or readouts. Its shape roles, dtype, and gradient requirement are declared by an input specification; it replaces `CustomIO` only for stateful operators.

**Lowering strategy** — A backend-specific execution method, including chunking, scans, tiling, fusion, state materialization, and backward kernels. It is separate from Algorithm IR.
