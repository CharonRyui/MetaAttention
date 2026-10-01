"""Dense diagonal recurrence kernels; only chunk-boundary States reach memory.

Identity propagation uses chunk GEMMs. General diagonal propagation replays a
64-token chunk with a distributed FP32 State tile and a parallel row reduction.
All tensor layouts are token-major; sequence length is a static dense contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = tl.constexpr(64)


@triton.jit
def _identity_pairs(
    K,
    K2,
    V,
    Bias,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    KR: tl.constexpr,
    KL: tl.constexpr,
):
    sh = tl.program_id(0)
    chunk = tl.program_id(1)
    tile = tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tile // tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    c = tile % tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    t = chunk * CHUNK + tl.arange(0, CHUNK)
    if KR:
        k = tl.load(
            K + ((seq * LENGTH + t[None, :]) * HEADS + head) * KL + r[:, None] // KR,
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
        k *= tl.load(
            K2 + ((seq * LENGTH + t[None, :]) * HEADS + head) * KR + r[:, None] % KR,
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
    else:
        k = tl.load(
            K + ((seq * LENGTH + t[None, :]) * HEADS + head) * ROWS + r[:, None],
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
    v = tl.load(
        V + ((seq * LENGTH + t[:, None]) * HEADS + head) * COLS + c[None, :],
        (t[:, None] < LENGTH) & (c[None, :] < COLS),
        0,
    )
    bias = tl.dot(k, v, input_precision="tf32x3")
    tl.store(
        Bias + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
        bias,
        (r[:, None] < ROWS) & (c[None, :] < COLS),
    )


@triton.jit
def _diagonal_pairs(
    GR,
    GC,
    K,
    V,
    Scale,
    Bias,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    UNIT_ROWS: tl.constexpr,
    UNIT_COLS: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    x = block * 256 + tl.arange(0, 256)
    r, c = x // COLS, x % COLS
    scale = tl.full((256,), 1.0, tl.float32)
    bias = tl.full((256,), 0.0, tl.float32)
    for step in range(CHUNK):
        t = chunk * CHUNK + step
        flat = (seq * LENGTH + t) * HEADS + head
        rg = tl.full((256,), 1.0, tl.float32)
        cg = tl.full((256,), 1.0, tl.float32)
        if not UNIT_ROWS:
            rg = tl.load(GR + flat * ROWS + r, (t < LENGTH) & (r < ROWS), 1)
        if not UNIT_COLS:
            cg = tl.load(GC + flat * COLS + c, (t < LENGTH) & (r < ROWS), 1)
        k = tl.load(K + flat * ROWS + r, (t < LENGTH) & (r < ROWS), 0)
        v = tl.load(V + flat * COLS + c, (t < LENGTH) & (r < ROWS), 0)
        bias = rg * cg * bias + k * v
        scale *= rg * cg
    base = (sh * CHUNKS + chunk) * ROWS * COLS + x
    tl.store(Scale + base, scale, x < ROWS * COLS)
    tl.store(Bias + base, bias, x < ROWS * COLS)


@triton.jit
def _carry(
    Scale,
    Bias,
    Initial,
    Carry,
    Final,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
):
    sh, block = tl.program_id(0), tl.program_id(1)
    x = block * 256 + tl.arange(0, 256)
    state = tl.load(Initial + sh * ROWS * COLS + x, x < ROWS * COLS, 0)
    for chunk in range(CHUNKS):
        base = (sh * CHUNKS + chunk) * ROWS * COLS + x
        tl.store(Carry + base, state, x < ROWS * COLS)
        bias = tl.load(Bias + base, x < ROWS * COLS, 0)
        if IDENTITY:
            state += bias
        else:
            scale = tl.load(Scale + base, x < ROWS * COLS, 1)
            state = state * scale + bias
    if WRITE_FINAL:
        tl.store(Final + sh * ROWS * COLS + x, state, x < ROWS * COLS)


@triton.jit
def _identity_readout(
    K,
    K2,
    V,
    Q,
    Q2,
    Carry,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    PRIOR: tl.constexpr,
    KR: tl.constexpr,
    KL: tl.constexpr,
    QR: tl.constexpr,
    QL: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    t = chunk * CHUNK + tl.arange(0, CHUNK)
    src = chunk * CHUNK + tl.arange(0, CHUNK)
    c = block * 32 + tl.arange(0, 32)
    rr = tl.arange(0, 32)
    scores = tl.full((CHUNK, CHUNK), 0.0, tl.float32)
    out = tl.full((CHUNK, 32), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + rr
        if QR:
            q = tl.load(
                Q
                + ((seq * LENGTH + t[:, None]) * HEADS + head) * QL
                + r[None, :] // QR,
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
            q *= tl.load(
                Q2
                + ((seq * LENGTH + t[:, None]) * HEADS + head) * QR
                + r[None, :] % QR,
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
        else:
            q = tl.load(
                Q + ((seq * LENGTH + t[:, None]) * HEADS + head) * ROWS + r[None, :],
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
        if KR:
            k = tl.load(
                K
                + ((seq * LENGTH + src[None, :]) * HEADS + head) * KL
                + r[:, None] // KR,
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
            k *= tl.load(
                K2
                + ((seq * LENGTH + src[None, :]) * HEADS + head) * KR
                + r[:, None] % KR,
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
        else:
            k = tl.load(
                K + ((seq * LENGTH + src[None, :]) * HEADS + head) * ROWS + r[:, None],
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
        state = tl.load(
            Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
            (r[:, None] < ROWS) & (c[None, :] < COLS),
            0,
        )
        scores += tl.dot(q, k, input_precision="tf32x3")
        out += tl.dot(q, state, input_precision="tf32x3")
    causal = src[None, :] < t[:, None] if PRIOR else src[None, :] <= t[:, None]
    scores = tl.where(causal, scores, 0.0)
    v = tl.load(
        V + ((seq * LENGTH + src[:, None]) * HEADS + head) * COLS + c[None, :],
        (src[:, None] < LENGTH) & (c[None, :] < COLS),
        0,
    )
    out += tl.dot(scores, v, input_precision="tf32x3")
    tl.store(
        Output + ((seq * LENGTH + t[:, None]) * HEADS + head) * COLS + c[None, :],
        out,
        (t[:, None] < LENGTH) & (c[None, :] < COLS),
    )


@triton.jit
def _diagonal_readout(
    GR,
    GC,
    K,
    V,
    Q,
    Carry,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    UNIT_ROWS: tl.constexpr,
    UNIT_COLS: tl.constexpr,
    PRIOR: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tl.arange(0, BR)
    c = block * 16 + tl.arange(0, 16)
    state = tl.load(
        Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
        (r[:, None] < ROWS) & (c[None, :] < COLS),
        0,
    )
    for step in range(CHUNK):
        t = chunk * CHUNK + step
        flat = (seq * LENGTH + t) * HEADS + head
        q = tl.load(Q + flat * ROWS + r, (t < LENGTH) & (r < ROWS), 0)
        if PRIOR:
            out = tl.sum(state * q[:, None], 0)
        rg = tl.full((BR,), 1.0, tl.float32)
        cg = tl.full((16,), 1.0, tl.float32)
        if not UNIT_ROWS:
            rg = tl.load(GR + flat * ROWS + r, (t < LENGTH) & (r < ROWS), 1)
        if not UNIT_COLS:
            cg = tl.load(GC + flat * COLS + c, (t < LENGTH) & (c < COLS), 1)
        k = tl.load(K + flat * ROWS + r, (t < LENGTH) & (r < ROWS), 0)
        v = tl.load(V + flat * COLS + c, (t < LENGTH) & (c < COLS), 0)
        state = state * rg[:, None] * cg[None, :] + k[:, None] * v[None, :]
        if not PRIOR:
            out = tl.sum(state * q[:, None], 0)
        tl.store(Output + flat * COLS + c, out, (t < LENGTH) & (c < COLS))


def prepare_state(
    *,
    rows_gate,
    columns_gate,
    key,
    value,
    initial,
    sequence_length,
    return_final_state,
    identity=False,
):
    """Return reusable FP32 carries, final State, and normalized factor bindings."""
    batch, heads, rows, columns = initial.shape
    initial = initial.contiguous()
    chunks = triton.cdiv(sequence_length, CHUNK)
    if isinstance(key, tuple):
        key, key2 = (part.float().contiguous() for part in key)
        kl, kr = key.shape[-1], key2.shape[-1]
    else:
        key = key.reshape(batch * sequence_length, heads, rows).float().contiguous()
        key2, kl, kr = key, rows, 0
    value = value.reshape(batch * sequence_length, heads, columns).float().contiguous()
    rows_gate = None if rows_gate is None else rows_gate.float().contiguous()
    columns_gate = None if columns_gate is None else columns_gate.float().contiguous()
    bias = torch.empty(
        batch * heads, chunks, rows, columns, device=initial.device, dtype=torch.float32
    )
    scale = None if identity else torch.empty_like(bias)
    carry = torch.empty_like(bias)
    final = torch.empty_like(initial) if return_final_state else initial
    if identity:
        _identity_pairs[
            (batch * heads, chunks, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))
        ](key, key2, value, bias, sequence_length, heads, rows, columns, chunks, kr, kl)
    else:
        _diagonal_pairs[(batch * heads, chunks, triton.cdiv(rows * columns, 256))](
            rows_gate,
            columns_gate,
            key,
            value,
            scale,
            bias,
            sequence_length,
            heads,
            rows,
            columns,
            chunks,
            rows_gate is None,
            columns_gate is None,
        )
    _carry[(batch * heads, triton.cdiv(rows * columns, 256))](
        scale,
        bias,
        initial,
        carry,
        final,
        rows,
        columns,
        chunks,
        identity,
        return_final_state,
    )
    return carry, final, (rows_gate, columns_gate, key, key2, value, kl, kr)


def read_state(carry, factors, query, *, sequence_length, prior=False, identity=False):
    batch_heads, chunks, rows, columns = carry.shape
    rows_gate, columns_gate, key, key2, value, kl, kr = factors
    heads = key.shape[1]
    if isinstance(query, tuple):
        query, query2 = (part.float().contiguous() for part in query)
        ql, qr = query.shape[-1], query2.shape[-1]
    else:
        query = query.reshape(key.shape[0], heads, rows).float().contiguous()
        query2, ql, qr = query, rows, 0
    output = torch.empty(
        key.shape[0], heads, columns, device=key.device, dtype=torch.float32
    )
    if identity:
        _identity_readout[(batch_heads, chunks, triton.cdiv(columns, 32))](
            key,
            key2,
            value,
            query,
            query2,
            carry,
            output,
            sequence_length,
            heads,
            rows,
            columns,
            chunks,
            prior,
            kr,
            kl,
            qr,
            ql,
        )
    else:
        _diagonal_readout[(batch_heads, chunks, triton.cdiv(columns, 16))](
            rows_gate,
            columns_gate,
            key,
            value,
            query,
            carry,
            output,
            sequence_length,
            heads,
            rows,
            columns,
            chunks,
            triton.next_power_of_2(rows),
            rows_gate is None,
            columns_gate is None,
            prior,
            num_warps=4 if rows <= 128 else 8,
        )
    return output
