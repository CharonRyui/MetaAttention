"""Inverse-free diagonal block algebra with separated row/column decays.

Direct interval products use parallel associative scans, including signed/zero
factors. No reciprocal, log difference or reconstructed prefix quotient occurs.
Bounded target tiles contract weighted factors with FP32 reductions; no token
States are stored. Chunk summaries use TF32x3 weighted key/value block GEMMs.
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
                   KL: tl.constexpr, KR: tl.constexpr, QL: tl.constexpr, QR: tl.constexpr,
                   TARGETS: tl.constexpr):
    sh, target_block, cb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    chunk = target_block // (64 // TARGETS)
    target = (target_block % (64 // TARGETS)) * TARGETS + tl.arange(0, TARGETS)
    target_index = chunk * 64 + target
    # Prior contraction targets S_(t-1), not a pre-decayed S_t.
    end = target - 1 if PRIOR else target
    local = tl.arange(0, 64)
    t = chunk * 64 + local
    flat = (seq * LENGTH + t) * HEADS + head
    qflat = (seq * LENGTH + target_index) * HEADS + head
    c = cb * 32 + tl.arange(0, 32)
    scores = tl.full((TARGETS, 64), 0.0, tl.float32)
    boundary = tl.full((TARGETS, 32), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + tl.arange(0, 32)
        rg = tl.full((32, 64), 1.0, tl.float32)
        if not UNIT_ROWS:
            rg = tl.load(GR + (flat[None, :] + HEADS) * ROWS + r[:, None],
                         (local[None, :] < 63) & (t[None, :] + 1 < LENGTH)
                         & (r[:, None] < ROWS), 1.0)
        rg = tl.where(local[None, None, :] < end[:, None, None], rg[None, :, :], 1.0)
        row_suffix = tl.associative_scan(rg, 2, _multiply, reverse=True)
        row_prefix = tl.sum(tl.where(local[None, None, :] == 0, row_suffix, 0.0), 2)
        if not UNIT_ROWS:
            first_row = tl.load(GR + ((seq * LENGTH + chunk * 64) * HEADS + head) * ROWS + r,
                                r < ROWS, 1.0)
            row_prefix *= tl.where(end[:, None] >= 0, first_row[None, :], 1.0)
        q = _factor(Q, Q2, qflat[:, None], r[None, :], ROWS, QL, QR,
                    target_index[:, None] < LENGTH)
        key = _factor(K, K2, flat[None, :], r[:, None], ROWS, KL, KR,
                      t[None, :] < LENGTH)
        weighted_key = key[None, :, :] * row_suffix
        scores += tl.sum(q[:, :, None] * weighted_key, 1)
        state = tl.load(Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
                        (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
        boundary += tl.sum((q * row_prefix)[:, :, None] * state[None, :, :], 1)
    scores = tl.where(local[None, :] <= end[:, None], scores, 0.0)
    cg = tl.full((64, 32), 1.0, tl.float32)
    if not UNIT_COLS:
        cg = tl.load(GC + (flat[:, None] + HEADS) * COLS + c[None, :],
                     (local[:, None] < 63) & (t[:, None] + 1 < LENGTH)
                     & (c[None, :] < COLS), 1.0)
    # Share source factors across a bounded target tile. Products remain direct
    # source-to-target scans, so signed and exactly-zero gates need no inverse.
    cg = tl.where(local[None, :, None] < end[:, None, None], cg[None, :, :], 1.0)
    column_suffix = tl.associative_scan(cg, 1, _multiply, reverse=True)
    column_prefix = tl.sum(tl.where(local[None, :, None] == 0, column_suffix, 0.0), 1)
    if not UNIT_COLS:
        first_column = tl.load(GC + ((seq * LENGTH + chunk * 64) * HEADS + head) * COLS + c,
                               c < COLS, 1.0)
        column_prefix *= tl.where(end[:, None] >= 0, first_column[None, :], 1.0)
    value = tl.load(V + flat[:, None] * COLS + c[None, :],
                    (t[:, None] < LENGTH) & (c[None, :] < COLS), 0.0)
    weighted_value = tl.where(local[None, :, None] <= end[:, None, None],
                              value[None, :, :], 0.0) * column_suffix
    result = boundary * column_prefix + tl.sum(scores[:, :, None] * weighted_value, 1)
    tl.store(Output + qflat[:, None] * COLS + c[None, :], result,
             (target_index[:, None] < LENGTH) & (c[None, :] < COLS))


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
    targets = 2
    _block_readout[(bh, triton.cdiv(sequence_length, targets), triton.cdiv(columns, 32))](
        gr, gc, key, key2, value, query, query2, carry, output,
        sequence_length, heads, rows, columns, chunks, gr is None, gc is None,
        prior, kl, kr, ql, qr, targets, num_warps=4, enable_fp_fusion=False,
    )
