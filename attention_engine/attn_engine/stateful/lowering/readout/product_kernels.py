"""Identity ProductFeature readouts with chunk-local base-Feature GEMMs.

Only the inter-chunk State contraction uses the expanded product axis. Local
scores use (q_left @ k_left.T) * (q_right @ k_right.T), with an optional reusable
64-token score tile for readouts sharing the same typed Outer operands. All
bindings are contiguous FP32 token-major tensors; the product axis is flattened
left-major/right-minor, exactly as the authored Outer expression.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = tl.constexpr(64)


@triton.jit
def _local_scores(
    KL,
    KR,
    QL,
    QR,
    seq,
    head,
    chunk,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    PRIOR: tl.constexpr,
):
    token = chunk * CHUNK + tl.arange(0, CHUNK)
    feature = tl.arange(0, 32)
    left = tl.full((CHUNK, CHUNK), 0.0, tl.float32)
    right = tl.full((CHUNK, CHUNK), 0.0, tl.float32)
    for block in range(tl.cdiv(LEFT, 32)):
        f = block * 32 + feature
        q = tl.load(
            QL + ((seq * LENGTH + token[:, None]) * HEADS + head) * LEFT + f[None, :],
            (token[:, None] < LENGTH) & (f[None, :] < LEFT),
            0.0,
        )
        k = tl.load(
            KL + ((seq * LENGTH + token[None, :]) * HEADS + head) * LEFT + f[:, None],
            (token[None, :] < LENGTH) & (f[:, None] < LEFT),
            0.0,
        )
        left += tl.dot(q, k, input_precision="tf32x3")
    for block in range(tl.cdiv(RIGHT, 32)):
        f = block * 32 + feature
        q = tl.load(
            QR + ((seq * LENGTH + token[:, None]) * HEADS + head) * RIGHT + f[None, :],
            (token[:, None] < LENGTH) & (f[None, :] < RIGHT),
            0.0,
        )
        k = tl.load(
            KR + ((seq * LENGTH + token[None, :]) * HEADS + head) * RIGHT + f[:, None],
            (token[None, :] < LENGTH) & (f[:, None] < RIGHT),
            0.0,
        )
        right += tl.dot(q, k, input_precision="tf32x3")
    causal = token[None, :] < token[:, None] if PRIOR else token[None, :] <= token[:, None]
    return tl.where(causal, left * right, 0.0)


@triton.jit
def _scores(
    KL,
    KR,
    QL,
    QR,
    Scores,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    CHUNKS: tl.constexpr,
    PRIOR: tl.constexpr,
):
    sh, chunk = tl.program_id(0), tl.program_id(1)
    scores = _local_scores(
        KL, KR, QL, QR, sh // HEADS, sh % HEADS, chunk,
        LENGTH, HEADS, LEFT, RIGHT, PRIOR,
    )
    row = tl.arange(0, CHUNK)
    col = tl.arange(0, CHUNK)
    tl.store(
        Scores + (sh * CHUNKS + chunk) * CHUNK * CHUNK + row[:, None] * CHUNK + col[None, :],
        scores,
    )


@triton.jit
def _readout(
    KL,
    KR,
    QL,
    QR,
    V,
    Carry,
    Scores,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    PRIOR: tl.constexpr,
    REUSE_SCORES: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    token = chunk * CHUNK + tl.arange(0, CHUNK)
    col = block * 32 + tl.arange(0, 32)
    if REUSE_SCORES:
        index = tl.arange(0, CHUNK)
        scores = tl.load(
            Scores + (sh * CHUNKS + chunk) * CHUNK * CHUNK
            + index[:, None] * CHUNK + index[None, :]
        )
    else:
        scores = _local_scores(
            KL, KR, QL, QR, seq, head, chunk,
            LENGTH, HEADS, LEFT, RIGHT, PRIOR,
        )
    out = tl.full((CHUNK, 32), 0.0, tl.float32)
    feature = tl.arange(0, 32)
    for rb in range(tl.cdiv(LEFT * RIGHT, 32)):
        row = rb * 32 + feature
        q = tl.load(
            QL + ((seq * LENGTH + token[:, None]) * HEADS + head) * LEFT
            + row[None, :] // RIGHT,
            (token[:, None] < LENGTH) & (row[None, :] < LEFT * RIGHT),
            0.0,
        )
        q *= tl.load(
            QR + ((seq * LENGTH + token[:, None]) * HEADS + head) * RIGHT
            + row[None, :] % RIGHT,
            (token[:, None] < LENGTH) & (row[None, :] < LEFT * RIGHT),
            0.0,
        )
        state = tl.load(
            Carry + ((sh * CHUNKS + chunk) * LEFT * RIGHT + row[:, None]) * COLS
            + col[None, :],
            (row[:, None] < LEFT * RIGHT) & (col[None, :] < COLS),
            0.0,
        )
        out += tl.dot(q, state, input_precision="tf32x3")
    value = tl.load(
        V + ((seq * LENGTH + token[:, None]) * HEADS + head) * COLS + col[None, :],
        (token[:, None] < LENGTH) & (col[None, :] < COLS),
        0.0,
    )
    out += tl.dot(scores, value, input_precision="tf32x3")
    tl.store(
        Output + ((seq * LENGTH + token[:, None]) * HEADS + head) * COLS + col[None, :],
        out,
        (token[:, None] < LENGTH) & (col[None, :] < COLS),
    )


def product_chunk_scores(key, query, *, sequence_length, prior=False):
    """Build reusable local scores, never a full-sequence attention matrix.

    ``key`` and ``query`` are normalized ``(left, right)`` FP32 tensors of shape
    ``[batch * sequence_length, heads, feature]``. Their respective left and
    right extents must agree. The return shape is
    ``[batch * heads, ceil(sequence_length / 64), 64, 64]``. Reuse is valid only
    for the same factors, sequence length and prior/post-State read convention.
    """
    left, right = key
    query_left, query_right = query
    heads = left.shape[1]
    batch = left.shape[0] // sequence_length
    chunks = triton.cdiv(sequence_length, CHUNK)
    scores = torch.empty(
        batch * heads, chunks, CHUNK, CHUNK, device=left.device, dtype=torch.float32
    )
    _scores[(batch * heads, chunks)](
        left, right, query_left, query_right, scores,
        sequence_length, heads, left.shape[-1], right.shape[-1], chunks, prior,
    )
    return scores


def read_product_state(
    carry, key, query, value, *, sequence_length, prior=False, scores=None
):
    """Contract an identity product State without expanded chunk-local Q/K.

    Factors use the normalized token-major layout of ``product_chunk_scores``;
    ``value`` is ``[batch * sequence_length, heads, columns]``. ``carry`` contains
    each chunk's incoming FP32 State in shape
    ``[batch * heads, chunks, left * right, columns]``. Returns FP32 token-major
    ``[batch * sequence_length, heads, columns]``. If ``scores`` is omitted,
    base-Feature GEMMs and their product are fused into the readout kernel; a
    shared score tile can instead serve numerator and denominator readouts.
    Work and optional score storage are linear in sequence length at fixed chunk
    size. No State reconstruction, recurrence inversion or final-State mutation
    occurs here; prepare_state remains responsible for explicit chunk carries.
    """
    left, right = key
    query_left, query_right = query
    batch_heads, chunks, _, columns = carry.shape
    heads = left.shape[1]
    output = torch.empty(
        left.shape[0], heads, columns, device=left.device, dtype=torch.float32
    )
    _readout[(batch_heads, chunks, triton.cdiv(columns, 32))](
        left, right, query_left, query_right, value, carry, scores, output,
        sequence_length, heads, left.shape[-1], right.shape[-1], columns,
        chunks, prior, scores is not None,
    )
    return output
