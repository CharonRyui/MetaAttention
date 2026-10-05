"""Ordered compact WY execution for scalar-scaled rank-one recurrences.

For ordered events e, U_e is coefficient*left (or an injection key), R_e
is the arbitrary rank-one right (or zero), and J_e is an injection value
(or zero). With forward scalar interval products d(e,j), event responses obey

    Z = W S_in + V,  W = (I-A)^-1 diag(gamma) R,
    V = (I-A)^-1 J,  A[e,j] = d(e,j) R_e U_j  (j < e).

A is strictly lower triangular, so its finite geometric series is exact in
real arithmetic. This is a triangular solve, NOT an inverse State recurrence.
No scalar gate is divided or inverted: zero and negative gates are supported.
Scalar scales commute with every rank-one propagation, so they may be collected
at each token's start; row/column scales do not have that proof and are rejected.
All event order, injection placement, carries and GEMM accumulators are FP32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def eligible(operation_codes, injection_count, rows, columns):
    """Prove scalar commutation and bound the compiler's matrix tile resources."""
    events = operation_codes.count("rank-one") + injection_count
    return (
        "rank-one" in operation_codes
        and all(code in ("scale", "rank-one") for code in operation_codes)
        and 1 <= events <= 4
        and rows <= 128
        and columns <= 128
    )


def chunk_tokens(event_count):
    return 32 if event_count <= 2 else 16


@triton.jit
def _load_factor(P, flat, feature, S0: tl.constexpr, S1: tl.constexpr,
                 S2: tl.constexpr, HEADS: tl.constexpr, VALID):
    return tl.load(
        P + (flat // HEADS) * S0 + (flat % HEADS) * S1
        + feature * S2, VALID, 0.0,
    ).to(tl.float32)


@triton.jit
def _solve(
    L, R, C, IK, IV, U, W, V, Intervals, Prefix,
    LS: tl.constexpr, RS: tl.constexpr, CS: tl.constexpr,
    IKS: tl.constexpr, IVS: tl.constexpr,
    LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
    COLS: tl.constexpr, CHUNKS: tl.constexpr,
    RANK_OPS: tl.constexpr, SCALE_OPS: tl.constexpr,
    INJECTIONS: tl.constexpr, EVENTS: tl.constexpr,
    CHUNK: tl.constexpr, N: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    SOLVE_STEPS: tl.constexpr,
):
    sh, chunk = tl.program_id(0), tl.program_id(1)
    seq, head = sh // HEADS, sh % HEADS
    e = tl.arange(0, N)
    row = tl.arange(0, BR)
    col = tl.arange(0, BC)
    local_token = e // EVENTS
    token = chunk * CHUNK + local_token
    flat = (seq * LENGTH + token) * HEADS + head
    valid = (local_token < CHUNK) & (token < LENGTH)
    u = tl.full((N, BR), 0.0, tl.float32)
    r = tl.full((N, BR), 0.0, tl.float32)
    j = tl.full((N, BC), 0.0, tl.float32)
    for index in tl.static_range(len(RANK_OPS)):
        mask = valid & (e % EVENTS == index)
        coefficient = _load_factor(
            C[RANK_OPS[index]], flat, 0,
            CS[RANK_OPS[index]][0], CS[RANK_OPS[index]][1], CS[RANK_OPS[index]][2],
            HEADS, mask,
        )
        left = _load_factor(
            L[index], flat[:, None], row[None, :],
            LS[index][0], LS[index][1], LS[index][2], HEADS,
            mask[:, None] & (row[None, :] < ROWS),
        )
        right = _load_factor(
            R[index], flat[:, None], row[None, :],
            RS[index][0], RS[index][1], RS[index][2], HEADS,
            mask[:, None] & (row[None, :] < ROWS),
        )
        u += coefficient[:, None] * left
        r += right
    for index in tl.static_range(INJECTIONS):
        mask = valid & (e % EVENTS == len(RANK_OPS) + index)
        u += _load_factor(
            IK[index], flat[:, None], row[None, :],
            IKS[index][0], IKS[index][1], IKS[index][2], HEADS,
            mask[:, None] & (row[None, :] < ROWS),
        )
        j += _load_factor(
            IV[index], flat[:, None], col[None, :],
            IVS[index][0], IVS[index][1], IVS[index][2], HEADS,
            mask[:, None] & (col[None, :] < COLS),
        )
    t = tl.arange(0, CHUNK)
    intervals = tl.full((CHUNK, CHUNK), 1.0, tl.float32)
    prefix = tl.full((CHUNK,), 1.0, tl.float32)
    event_intervals = tl.full((N, N), 1.0, tl.float32)
    event_prefix = tl.full((N,), 1.0, tl.float32)
    for step in range(CHUNK):
        absolute = chunk * CHUNK + step
        scalar_flat = (seq * LENGTH + absolute) * HEADS + head
        gate = tl.full((), 1.0, tl.float32)
        for index in tl.static_range(len(SCALE_OPS)):
            gate *= _load_factor(
                C[SCALE_OPS[index]], scalar_flat, 0,
                CS[SCALE_OPS[index]][0], CS[SCALE_OPS[index]][1], CS[SCALE_OPS[index]][2],
                HEADS, absolute < LENGTH,
            )
        gate = tl.where(absolute < LENGTH, gate, 1.0)
        prefix *= tl.where(step <= t, gate, 1.0)
        event_prefix *= tl.where(step <= local_token, gate, 1.0)
        intervals *= tl.where((step > t[None, :]) & (step <= t[:, None]), gate, 1.0)
        event_intervals *= tl.where(
            (step > local_token[None, :]) & (step <= local_token[:, None]), gate, 1.0,
        )
    gram = tl.dot(r, tl.trans(u), input_precision="tf32x3")
    a = tl.where(e[:, None] > e[None, :], gram * event_intervals, 0.0)
    identity = (e[:, None] == e[None, :]).to(tl.float32)
    # (I+A)(I+A^2)... terminates after log2(N) factors because A^N=0.
    triangular = identity + a
    power = a
    for stage in tl.static_range(1, SOLVE_STEPS):
        power = tl.dot(power, power, input_precision="tf32x3")
        triangular += tl.dot(power, triangular, input_precision="tf32x3")
    w = tl.dot(triangular, event_prefix[:, None] * r, input_precision="tf32x3")
    v = tl.dot(triangular, j, input_precision="tf32x3")
    base = (sh * CHUNKS + chunk) * N
    tl.store(U + (base + e[:, None]) * ROWS + row[None, :], u, row[None, :] < ROWS)
    tl.store(W + (base + e[:, None]) * ROWS + row[None, :], w, row[None, :] < ROWS)
    tl.store(V + (base + e[:, None]) * COLS + col[None, :], v, col[None, :] < COLS)
    tl.store(
        Intervals + (sh * CHUNKS + chunk) * CHUNK * CHUNK + t[:, None] * CHUNK + t[None, :],
        intervals,
    )
    tl.store(Prefix + (sh * CHUNKS + chunk) * CHUNK + t, prefix)


@triton.jit
def _carry(
    U, W, V, Intervals, Prefix, Initial, Carry, Final,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    EVENTS: tl.constexpr, CHUNK: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
):
    sh, block = tl.program_id(0), tl.program_id(1)
    row = tl.arange(0, BR)
    col = block * 32 + tl.arange(0, 32)
    e = tl.arange(0, N)
    event_valid = e < CHUNK * EVENTS
    state = tl.load(
        Initial + (sh * ROWS + row[:, None]) * COLS + col[None, :],
        (row[:, None] < ROWS) & (col[None, :] < COLS), 0.0,
    )
    for chunk in range(CHUNKS):
        base = (sh * CHUNKS + chunk) * N
        tl.store(
            Carry + ((sh * CHUNKS + chunk) * ROWS + row[:, None]) * COLS + col[None, :],
            state, (row[:, None] < ROWS) & (col[None, :] < COLS),
        )
        u = tl.load(
            U + (base + e[None, :]) * ROWS + row[:, None], row[:, None] < ROWS, 0.0,
        )
        w = tl.load(
            W + (base + e[:, None]) * ROWS + row[None, :], row[None, :] < ROWS, 0.0,
        )
        v = tl.load(
            V + (base + e[:, None]) * COLS + col[None, :], col[None, :] < COLS, 0.0,
        )
        decay = tl.load(
            Intervals + (sh * CHUNKS + chunk) * CHUNK * CHUNK + (CHUNK - 1) * CHUNK + e // EVENTS,
            event_valid, 0.0,
        )
        gamma = tl.load(Prefix + (sh * CHUNKS + chunk) * CHUNK + CHUNK - 1)
        response = tl.dot(w, state, input_precision="tf32x3") + v
        state = gamma * state + tl.dot(u * decay[None, :], response, input_precision="tf32x3")
    tl.store(
        Final + (sh * ROWS + row[:, None]) * COLS + col[None, :],
        state, (row[:, None] < ROWS) & (col[None, :] < COLS),
    )


@triton.jit
def _readouts(
    Q, U, W, V, Intervals, Prefix, Carry, Output, QS: tl.constexpr,
    LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
    COLS: tl.constexpr, CHUNKS: tl.constexpr, BATCH_HEADS: tl.constexpr,
    EVENTS: tl.constexpr, READOUTS: tl.constexpr,
    CHUNK: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    t = tl.arange(0, CHUNK)
    e = tl.arange(0, N)
    row = tl.arange(0, BR)
    col = block * 32 + tl.arange(0, 32)
    token = chunk * CHUNK + t
    flat = (seq * LENGTH + token) * HEADS + head
    base = (sh * CHUNKS + chunk) * N
    u = tl.load(U + (base + e[None, :]) * ROWS + row[:, None], row[:, None] < ROWS, 0.0)
    w = tl.load(W + (base + e[:, None]) * ROWS + row[None, :], row[None, :] < ROWS, 0.0)
    v = tl.load(V + (base + e[:, None]) * COLS + col[None, :], col[None, :] < COLS, 0.0)
    incoming = tl.load(
        Carry + ((sh * CHUNKS + chunk) * ROWS + row[:, None]) * COLS + col[None, :],
        (row[:, None] < ROWS) & (col[None, :] < COLS), 0.0,
    )
    decay = tl.load(
        Intervals + (sh * CHUNKS + chunk) * CHUNK * CHUNK + t[:, None] * CHUNK + e[None, :] // EVENTS,
        (e[None, :] < CHUNK * EVENTS) & (e[None, :] // EVENTS <= t[:, None]), 0.0,
    )
    gamma = tl.load(Prefix + (sh * CHUNKS + chunk) * CHUNK + t)
    # The compact response is shared by every authored readout.
    response = tl.dot(w, incoming, input_precision="tf32x3") + v
    for readout in tl.static_range(READOUTS):
        q = _load_factor(
            Q[readout], flat[:, None], row[None, :],
            QS[readout][0], QS[readout][1], QS[readout][2], HEADS,
            (token[:, None] < LENGTH) & (row[None, :] < ROWS),
        )
        h = tl.dot(q, u, input_precision="tf32x3") * decay
        output = gamma[:, None] * tl.dot(q, incoming, input_precision="tf32x3")
        output += tl.dot(h, response, input_precision="tf32x3")
        tl.store(
            Output + (readout * LENGTH * BATCH_HEADS + flat[:, None]) * COLS + col[None, :],
            output, (token[:, None] < LENGTH) & (col[None, :] < COLS),
        )


def execute(*, codes, left, right, coefficient, injection_left, injection_right,
            query, initial_state, length, output_dtypes):
    batch, heads, rows, columns = initial_state.shape
    rank_ops = tuple(i for i, code in enumerate(codes) if code == 2)
    scale_ops = tuple(i for i, code in enumerate(codes) if code == 0)
    injection_count, readout_count = len(injection_left), len(query)
    events = len(rank_ops) + injection_count
    chunk = chunk_tokens(events)
    n = triton.next_power_of_2(chunk * events)
    chunks = triton.cdiv(length, chunk)
    sh = batch * heads
    device = initial_state.device
    u = torch.empty((sh, chunks, n, rows), device=device, dtype=torch.float32)
    w = torch.empty_like(u)
    v = torch.empty((sh, chunks, n, columns), device=device, dtype=torch.float32)
    intervals = torch.empty((sh, chunks, chunk, chunk), device=device, dtype=torch.float32)
    prefix = torch.empty((sh, chunks, chunk), device=device, dtype=torch.float32)
    carry = torch.empty((sh, chunks, rows, columns), device=device, dtype=torch.float32)
    final = torch.empty_like(initial_state, dtype=torch.float32)
    output = torch.empty((readout_count, batch * length, heads, columns), device=device, dtype=torch.float32)
    _solve[(sh, chunks)](
        left, right, coefficient, injection_left, injection_right, u, w, v, intervals, prefix,
        tuple(part.stride() for part in left), tuple(part.stride() for part in right),
        tuple(part.stride() for part in coefficient),
        tuple(part.stride() for part in injection_left),
        tuple(part.stride() for part in injection_right),
        length, heads, rows, columns, chunks, rank_ops, scale_ops,
        injection_count, events, chunk, n,
        max(16, triton.next_power_of_2(rows)), max(16, triton.next_power_of_2(columns)),
        n.bit_length() - 1,
        num_warps=8,
    )
    _carry[(sh, triton.cdiv(columns, 32))](
        u, w, v, intervals, prefix, initial_state, carry, final,
        rows, columns, chunks, events, chunk, n, max(16, triton.next_power_of_2(rows)),
    )
    _readouts[(sh, chunks, triton.cdiv(columns, 32))](
        query, u, w, v, intervals, prefix, carry, output,
        tuple(part.stride() for part in query),
        length, heads, rows, columns, chunks, sh, events, readout_count,
        chunk, n, max(16, triton.next_power_of_2(rows)),
    )
    return tuple(
        output[i].reshape(batch, length, heads, columns).permute(0, 2, 1, 3).to(dtype)
        for i, dtype in enumerate(output_dtypes)
    ), final
