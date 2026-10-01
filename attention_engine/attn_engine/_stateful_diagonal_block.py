"""Inverse-free diagonal block algebra with separated row/column decays.

Direct interval products use parallel associative scans, including signed/zero
factors. No reciprocal, log difference or reconstructed prefix quotient occurs.
Each target contracts weighted factors with TF32x3 GEMMs; no token States are
stored. Chunk summaries likewise use weighted key/value block GEMMs.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _multiply(a, b):
    return a * b


@triton.jit
def _factor(P, P2, flat, r, ROWS: tl.constexpr, KL: tl.constexpr, KR: tl.constexpr, VALID):
    if KR:
        k = tl.load(P + flat * KL + r // KR, VALID & (r < ROWS), 0.0)
        k *= tl.load(P2 + flat * KR + r % KR, VALID & (r < ROWS), 0.0)
    else:
        k = tl.load(P + flat * ROWS + r, VALID & (r < ROWS), 0.0)
    return k


@triton.jit
def _block_summary(GR, GC, K, K2, V, Scale, Bias,
                   LENGTH: tl.constexpr, HEADS: tl.constexpr,
                   ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
                   UNIT_ROWS: tl.constexpr, UNIT_COLS: tl.constexpr,
                   KL: tl.constexpr, KR: tl.constexpr):
    sh, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tile // tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    c = tile % tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    local = tl.arange(0, 64)
    t = chunk * 64 + local
    flat = (seq * LENGTH + t) * HEADS + head
    # Shift gates by one: an injection at s propagates with gates s+1..end.
    next_flat = flat + HEADS
    rg = tl.full((32, 64), 1.0, tl.float32)
    cg = tl.full((64, 32), 1.0, tl.float32)
    if not UNIT_ROWS:
        rg = tl.load(GR + next_flat[None, :] * ROWS + r[:, None],
                     (local[None, :] < 63) & (t[None, :] + 1 < LENGTH) & (r[:, None] < ROWS), 1.0)
    if not UNIT_COLS:
        cg = tl.load(GC + next_flat[:, None] * COLS + c[None, :],
                     (local[:, None] < 63) & (t[:, None] + 1 < LENGTH) & (c[None, :] < COLS), 1.0)
    row_suffix = tl.associative_scan(rg, 1, _multiply, reverse=True)
    column_suffix = tl.associative_scan(cg, 0, _multiply, reverse=True)
    k = _factor(K, K2, flat[None, :], r[:, None], ROWS, KL, KR, t[None, :] < LENGTH)
    v = tl.load(V + flat[:, None] * COLS + c[None, :],
                (t[:, None] < LENGTH) & (c[None, :] < COLS), 0.0)
    bias = tl.dot(k * row_suffix, v * column_suffix, input_precision="tf32x3")
    # Suffix at index zero excludes the first gate; multiply it explicitly.
    row_end = tl.sum(tl.where(local[None, :] == 0, row_suffix, 0.0), 1)
    column_end = tl.sum(tl.where(local[:, None] == 0, column_suffix, 0.0), 0)
    first_flat = (seq * LENGTH + chunk * 64) * HEADS + head
    if not UNIT_ROWS:
        row_end *= tl.load(GR + first_flat * ROWS + r, r < ROWS, 1.0)
    if not UNIT_COLS:
        column_end *= tl.load(GC + first_flat * COLS + c, c < COLS, 1.0)
    addr = ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :]
    valid = (r[:, None] < ROWS) & (c[None, :] < COLS)
    tl.store(Scale + addr, row_end[:, None] * column_end[None, :], valid)
    tl.store(Bias + addr, bias, valid)


@triton.jit
def _block_readout(GR, GC, K, K2, V, Q, Q2, Carry, Output,
                   LENGTH: tl.constexpr, HEADS: tl.constexpr,
                   ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
                   UNIT_ROWS: tl.constexpr, UNIT_COLS: tl.constexpr, PRIOR: tl.constexpr,
                   KL: tl.constexpr, KR: tl.constexpr, QL: tl.constexpr, QR: tl.constexpr):
    sh, target_index, cb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    chunk, target = target_index // 64, target_index % 64
    # Prior contraction targets S_(t-1), not a pre-decayed S_t.
    end = target - 1 if PRIOR else target
    local = tl.arange(0, 64)
    t = chunk * 64 + local
    flat = (seq * LENGTH + t) * HEADS + head
    qflat = (seq * LENGTH + target_index) * HEADS + head
    c = cb * 32 + tl.arange(0, 32)
    cg = tl.full((64, 32), 1.0, tl.float32)
    if not UNIT_COLS:
        cg = tl.load(GC + (flat[:, None] + HEADS) * COLS + c[None, :],
                     (local[:, None] < end) & (t[:, None] + 1 < LENGTH) & (c[None, :] < COLS), 1.0)
    column_suffix = tl.associative_scan(cg, 0, _multiply, reverse=True)
    column_prefix = tl.sum(tl.where(local[:, None] == 0, column_suffix, 0.0), 0)
    if not UNIT_COLS:
        column_prefix *= tl.load(GC + ((seq * LENGTH + chunk * 64) * HEADS + head) * COLS + c,
                                 (end >= 0) & (c < COLS), 1.0)
    scores = tl.full((16, 64), 0.0, tl.float32)
    boundary = tl.full((16, 32), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + tl.arange(0, 32)
        rg = tl.full((32, 64), 1.0, tl.float32)
        if not UNIT_ROWS:
            rg = tl.load(GR + (flat[None, :] + HEADS) * ROWS + r[:, None],
                         (local[None, :] < end) & (t[None, :] + 1 < LENGTH) & (r[:, None] < ROWS), 1.0)
        row_suffix = tl.associative_scan(rg, 1, _multiply, reverse=True)
        row_prefix = tl.sum(tl.where(local[None, :] == 0, row_suffix, 0.0), 1)
        if not UNIT_ROWS:
            row_prefix *= tl.load(GR + ((seq * LENGTH + chunk * 64) * HEADS + head) * ROWS + r,
                                  (end >= 0) & (r < ROWS), 1.0)
        q = _factor(Q, Q2, qflat, r, ROWS, QL, QR, target_index < LENGTH)
        # Matrix dot has a hardware minimum of 16 rows; only row zero is stored.
        qq = tl.broadcast_to(q[None, :], (16, 32))
        key = _factor(K, K2, flat[None, :], r[:, None], ROWS, KL, KR,
                      (local[None, :] <= end) & (t[None, :] < LENGTH))
        scores += tl.dot(qq, key * row_suffix, input_precision="tf32x3")
        state = tl.load(Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
                        (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
        boundary += tl.dot(qq * row_prefix[None, :], state, input_precision="tf32x3")
    value = tl.load(V + flat[:, None] * COLS + c[None, :],
                    (local[:, None] <= end) & (t[:, None] < LENGTH) & (c[None, :] < COLS), 0.0)
    out = boundary * column_prefix[None, :] + tl.dot(scores, value * column_suffix, input_precision="tf32x3")
    result = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, out, 0.0), 0)
    tl.store(Output + qflat * COLS + c, result, (target_index < LENGTH) & (c < COLS))


def diagonal_block_summary(rows_gate, columns_gate, key, key2, value, scale, bias,
                           *, sequence_length, heads, rows, columns, chunks, kl, kr):
    bh = bias.shape[0]
    _block_summary[(bh, chunks, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))](
        rows_gate, columns_gate, key, key2, value, scale, bias,
        sequence_length, heads, rows, columns, chunks,
        rows_gate is None, columns_gate is None, kl, kr,
        num_warps=4, enable_fp_fusion=False,
    )


def diagonal_block_readout(carry, factors, query, query2, output, *, sequence_length,
                           prior, ql, qr):
    bh, chunks, rows, columns = carry.shape
    gr, gc, key, key2, value, kl, kr = factors
    heads = key.shape[1]
    _block_readout[(bh, sequence_length, triton.cdiv(columns, 32))](
        gr, gc, key, key2, value, query, query2, carry, output,
        sequence_length, heads, rows, columns, chunks, gr is None, gc is None,
        prior, kl, kr, ql, qr, num_warps=4, enable_fp_fusion=False,
    )
