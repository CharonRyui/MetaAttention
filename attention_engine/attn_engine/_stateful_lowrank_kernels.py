"""Ordered low-rank affine chunk summaries, FP32 carries and parallel replay."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = tl.constexpr(64)


@triton.jit
def _pairs(
    L,
    R,
    C,
    G,
    IK,
    IV,
    P,
    D,
    Bias,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    OPS: tl.constexpr,
    INJECTIONS: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tl.arange(0, BR)
    c = block * 32 + tl.arange(0, 32)
    # P and additive bias are replayed together; only bias sees column decay.
    is_transform = c < ROWS
    vcol = c - ROWS
    matrix = tl.where((r[:, None] == c[None, :]) & is_transform[None, :], 1.0, 0.0)
    column_scale = tl.full((32,), 1.0, tl.float32)
    for step in range(CHUNK):
        token = chunk * CHUNK + step
        flat = (seq * LENGTH + token) * HEADS + head
        for op in tl.static_range(len(OPS)):
            if OPS[op] == 0:
                factor = tl.load(
                    C + op * LENGTH * tl.num_programs(0) + flat, token < LENGTH, 1.0
                )
                matrix *= factor
            elif OPS[op] == 1:
                factor = tl.load(
                    L + (op * LENGTH * tl.num_programs(0) + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    1.0,
                )
                matrix *= factor[:, None]
            elif OPS[op] == 3:
                factor = tl.load(
                    G + (op * LENGTH * tl.num_programs(0) + flat) * COLS + vcol,
                    (token < LENGTH) & (~is_transform) & (vcol < COLS),
                    1.0,
                )
                matrix *= factor[None, :]
                column_scale *= factor
            else:
                left = tl.load(
                    L + (op * LENGTH * tl.num_programs(0) + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    0.0,
                )
                right = tl.load(
                    R + (op * LENGTH * tl.num_programs(0) + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    0.0,
                )
                coefficient = tl.load(
                    C + op * LENGTH * tl.num_programs(0) + flat, token < LENGTH, 0.0
                )
                dot = tl.sum(matrix * right[:, None], 0)
                matrix += coefficient * left[:, None] * dot[None, :]
        for injection in tl.static_range(INJECTIONS):
            key = tl.load(
                IK + (injection * LENGTH * tl.num_programs(0) + flat) * ROWS + r,
                (token < LENGTH) & (r < ROWS),
                0.0,
            )
            value = tl.load(
                IV + (injection * LENGTH * tl.num_programs(0) + flat) * COLS + vcol,
                (token < LENGTH) & (~is_transform) & (vcol < COLS),
                0.0,
            )
            matrix += key[:, None] * value[None, :]
    tl.store(
        P + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * ROWS + c[None, :],
        matrix,
        (r[:, None] < ROWS) & is_transform[None, :],
    )
    tl.store(
        Bias + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + vcol[None, :],
        matrix,
        (r[:, None] < ROWS) & (~is_transform[None, :]) & (vcol[None, :] < COLS),
    )
    tl.store(
        D + (sh * CHUNKS + chunk) * COLS + vcol,
        column_scale,
        (~is_transform) & (vcol < COLS),
    )


@triton.jit
def _carry(
    P,
    D,
    Bias,
    Initial,
    Carry,
    Final,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
):
    sh, block = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, BR)
    c = block * 32 + tl.arange(0, 32)
    state = tl.load(
        Initial + (sh * ROWS + r[:, None]) * COLS + c[None, :],
        (r[:, None] < ROWS) & (c[None, :] < COLS),
        0.0,
    )
    for chunk in range(CHUNKS):
        base = (sh * CHUNKS + chunk) * ROWS * COLS
        tl.store(
            Carry + base + r[:, None] * COLS + c[None, :],
            state,
            (r[:, None] < ROWS) & (c[None, :] < COLS),
        )
        transform = tl.load(
            P + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * ROWS + r[None, :],
            (r[:, None] < ROWS) & (r[None, :] < ROWS),
            0.0,
        )
        decay = tl.load(D + (sh * CHUNKS + chunk) * COLS + c, c < COLS, 1.0)
        bias = tl.load(
            Bias + base + r[:, None] * COLS + c[None, :],
            (r[:, None] < ROWS) & (c[None, :] < COLS),
            0.0,
        )
        state = (
            tl.dot(transform, state, input_precision="tf32x3") * decay[None, :] + bias
        )
    tl.store(
        Final + (sh * ROWS + r[:, None]) * COLS + c[None, :],
        state,
        (r[:, None] < ROWS) & (c[None, :] < COLS),
    )


@triton.jit
def _replay(
    L,
    R,
    C,
    G,
    IK,
    IV,
    Q,
    Carry,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    BATCH_HEADS: tl.constexpr,
    OPS: tl.constexpr,
    INJECTIONS: tl.constexpr,
    READOUTS: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tl.arange(0, BR)
    c = block * 16 + tl.arange(0, 16)
    state = tl.load(
        Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
        (r[:, None] < ROWS) & (c[None, :] < COLS),
        0.0,
    )
    for step in range(CHUNK):
        token = chunk * CHUNK + step
        flat = (seq * LENGTH + token) * HEADS + head
        for op in tl.static_range(len(OPS)):
            if OPS[op] == 0:
                state *= tl.load(
                    C + op * LENGTH * BATCH_HEADS + flat, token < LENGTH, 1.0
                )
            elif OPS[op] == 1:
                factor = tl.load(
                    L + (op * LENGTH * BATCH_HEADS + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    1.0,
                )
                state *= factor[:, None]
            elif OPS[op] == 3:
                factor = tl.load(
                    G + (op * LENGTH * BATCH_HEADS + flat) * COLS + c,
                    (token < LENGTH) & (c < COLS),
                    1.0,
                )
                state *= factor[None, :]
            else:
                left = tl.load(
                    L + (op * LENGTH * BATCH_HEADS + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    0.0,
                )
                right = tl.load(
                    R + (op * LENGTH * BATCH_HEADS + flat) * ROWS + r,
                    (token < LENGTH) & (r < ROWS),
                    0.0,
                )
                coefficient = tl.load(
                    C + op * LENGTH * BATCH_HEADS + flat, token < LENGTH, 0.0
                )
                state += (
                    coefficient
                    * left[:, None]
                    * tl.sum(state * right[:, None], 0)[None, :]
                )
        for injection in tl.static_range(INJECTIONS):
            key = tl.load(
                IK + (injection * LENGTH * BATCH_HEADS + flat) * ROWS + r,
                (token < LENGTH) & (r < ROWS),
                0.0,
            )
            value = tl.load(
                IV + (injection * LENGTH * BATCH_HEADS + flat) * COLS + c,
                (token < LENGTH) & (c < COLS),
                0.0,
            )
            state += key[:, None] * value[None, :]
        for readout in tl.static_range(READOUTS):
            query = tl.load(
                Q + (readout * LENGTH * BATCH_HEADS + flat) * ROWS + r,
                (token < LENGTH) & (r < ROWS),
                0.0,
            )
            out = tl.sum(state * query[:, None], 0)
            tl.store(
                Output + (readout * LENGTH * BATCH_HEADS + flat) * COLS + c,
                out,
                (token < LENGTH) & (c < COLS),
            )


def execute(
    *,
    operation_codes,
    left_factors,
    right_factors,
    coefficients,
    injection_left,
    injection_right,
    readouts,
    initial_state,
    output_dtypes,
):
    batch, heads, rows, columns = initial_state.shape
    tokens = readouts[0].shape[0]
    length = tokens // batch
    chunks = triton.cdiv(length, 64)
    device = initial_state.device
    codes = tuple(
        {"scale": 0, "row-scale": 1, "rank-one": 2, "column-scale": 3}[code]
        for code in operation_codes
    )

    def binding(part, size):
        if part.numel() == 1:
            return part.reshape(1, 1, 1).expand(tokens, heads, size)
        return part.reshape(tokens, heads, -1).expand(tokens, heads, size)

    # Only the operation-specific buffers that a static Triton branch reads need
    # the full packed shape. Unused pointers remain valid one-element tensors.
    def packed(shape, used):
        return torch.empty(shape if used else (1,), device=device, dtype=torch.float32)

    left_used = any(code in (1, 2) for code in codes)
    right_used = any(code == 2 for code in codes)
    coefficient_used = any(code in (0, 2) for code in codes)
    gate_used = any(code == 3 for code in codes)
    left = packed((len(codes), tokens, heads, rows), left_used)
    right = packed((len(codes), tokens, heads, rows), right_used)
    coefficient = packed((len(codes), tokens, heads), coefficient_used)
    gate = packed((len(codes), tokens, heads, columns), gate_used)
    for i, code in enumerate(codes):
        if code in (1, 2):
            left[i].copy_(binding(left_factors[i], rows))
        if code == 2:
            right[i].copy_(binding(right_factors[i], rows))
        if code in (0, 2):
            coefficient[i].copy_(binding(coefficients[i], 1).squeeze(-1))
        if code == 3:
            gate[i].copy_(binding(left_factors[i], columns))

    def stack_or_bind(parts, size):
        if len(parts) == 1:
            return binding(parts[0], size).float().contiguous()
        return torch.stack([binding(part, size) for part in parts]).float().contiguous()

    # The common single-injection/single-readout case now reuses an already
    # contiguous FP32 binding instead of allocating a one-leading-axis stack.
    ik = stack_or_bind(injection_left, rows)
    iv = stack_or_bind(injection_right, columns)
    query = stack_or_bind(readouts, rows)

    transform = torch.empty(
        batch * heads, chunks, rows, rows, device=device, dtype=torch.float32
    )
    decay = torch.empty(
        batch * heads, chunks, columns, device=device, dtype=torch.float32
    )
    bias = torch.empty(
        batch * heads, chunks, rows, columns, device=device, dtype=torch.float32
    )
    carry = torch.empty_like(bias)
    final = torch.empty_like(initial_state)
    output = torch.empty(
        len(readouts), tokens, heads, columns, device=device, dtype=torch.float32
    )
    _pairs[(batch * heads, chunks, triton.cdiv(rows + columns, 32))](
        left,
        right,
        coefficient,
        gate,
        ik,
        iv,
        transform,
        decay,
        bias,
        length,
        heads,
        rows,
        columns,
        chunks,
        triton.next_power_of_2(rows),
        codes,
        len(injection_left),
    )
    _carry[(batch * heads, triton.cdiv(columns, 32))](
        transform,
        decay,
        bias,
        initial_state,
        carry,
        final,
        rows,
        columns,
        chunks,
        triton.next_power_of_2(rows),
    )
    _replay[(batch * heads, chunks, triton.cdiv(columns, 16))](
        left,
        right,
        coefficient,
        gate,
        ik,
        iv,
        query,
        carry,
        output,
        length,
        heads,
        rows,
        columns,
        chunks,
        triton.next_power_of_2(rows),
        batch * heads,
        codes,
        len(injection_left),
        len(readouts),
    )
    return tuple(
        output[i].reshape(batch, length, heads, columns).permute(0, 2, 1, 3).to(dtype)
        for i, dtype in enumerate(output_dtypes)
    ), final
