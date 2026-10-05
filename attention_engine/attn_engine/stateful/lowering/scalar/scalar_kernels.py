"""Raw-input scalar affine summaries, chunk carries and parallel readout.

Only FP32 chunk-boundary States and one scalar per chunk reach global scratch.
Gate products are formed forwards: zero and signed scalar gates do not need an
inverse, and a log gate is exponentiated before it enters a product.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

CHUNK_TOKENS = 64
_CHUNK = tl.constexpr(CHUNK_TOKENS)


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _load_expression(
    pointer,
    offset,
    mask,
    OPERATION: tl.constexpr,
    SCALE: tl.constexpr,
    LOG_SCALE: tl.constexpr,
    NEUTRAL: tl.constexpr,
):
    value = tl.load(pointer + offset, mask, 0).to(tl.float32)
    if OPERATION == "exp":
        value = tl.exp(value)
    value *= SCALE
    if LOG_SCALE:
        value = tl.exp(value)
    return tl.where(mask, value, NEUTRAL)


@triton.jit
def _chunk_summary(
    Propagation,
    Left,
    Right,
    Scale,
    Bias,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    PROPAGATION_HEADS: tl.constexpr,
    LEFT_HEADS: tl.constexpr,
    RIGHT_HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr,
    PROPAGATION_OPERATION: tl.constexpr,
    LEFT_OPERATION: tl.constexpr,
    RIGHT_OPERATION: tl.constexpr,
    PROPAGATION_SCALE: tl.constexpr,
    LEFT_SCALE: tl.constexpr,
    RIGHT_SCALE: tl.constexpr,
    PROPAGATION_LOG: tl.constexpr,
    LEFT_LOG: tl.constexpr,
    RIGHT_LOG: tl.constexpr,
):
    sequence_head, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence, head = sequence_head // HEADS, sequence_head % HEADS
    row = tile // tl.cdiv(COLUMNS, 32) * 32 + tl.arange(0, 32)
    column = tile % tl.cdiv(COLUMNS, 32) * 32 + tl.arange(0, 32)
    step = tl.arange(0, _CHUNK)
    token = chunk * _CHUNK + step
    left_head = head // (HEADS // LEFT_HEADS)
    right_head = head // (HEADS // RIGHT_HEADS)
    key = _load_expression(
        Left,
        ((sequence * LEFT_HEADS + left_head) * LENGTH + token[None, :]) * ROWS
        + row[:, None],
        (token[None, :] < LENGTH) & (row[:, None] < ROWS),
        LEFT_OPERATION,
        LEFT_SCALE,
        LEFT_LOG,
        0.0,
    )
    value = _load_expression(
        Right,
        ((sequence * RIGHT_HEADS + right_head) * LENGTH + token[:, None]) * COLUMNS
        + column[None, :],
        (token[:, None] < LENGTH) & (column[None, :] < COLUMNS),
        RIGHT_OPERATION,
        RIGHT_SCALE,
        RIGHT_LOG,
        0.0,
    )
    if not IDENTITY:
        propagation_head = head // (HEADS // PROPAGATION_HEADS)
        propagation_base = (sequence * PROPAGATION_HEADS + propagation_head) * LENGTH
        # An injection at step i is multiplied only by gates i+1 through end.
        next_gate = _load_expression(
            Propagation,
            propagation_base + token + 1,
            (step + 1 < _CHUNK) & (token + 1 < LENGTH),
            PROPAGATION_OPERATION,
            PROPAGATION_SCALE,
            PROPAGATION_LOG,
            1.0,
        )
        suffix = tl.associative_scan(next_gate, 0, _multiply, reverse=True)
        key *= suffix[None, :]
        if tile == 0:
            first_gate = _load_expression(
                Propagation,
                propagation_base + chunk * _CHUNK,
                chunk * _CHUNK < LENGTH,
                PROPAGATION_OPERATION,
                PROPAGATION_SCALE,
                PROPAGATION_LOG,
                1.0,
            )
            chunk_scale = first_gate * tl.sum(tl.where(step == 0, suffix, 0.0), 0)
            tl.store(Scale + sequence_head * CHUNKS + chunk, chunk_scale)
    bias = tl.dot(key, value, input_precision="tf32x3")
    tl.store(
        Bias
        + ((sequence_head * CHUNKS + chunk) * ROWS + row[:, None]) * COLUMNS
        + column[None, :],
        bias,
        (row[:, None] < ROWS) & (column[None, :] < COLUMNS),
    )


@triton.jit
def _chunk_prefix(
    Scale,
    Bias,
    Initial,
    Carry,
    Final,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
):
    sequence_head, block = tl.program_id(0), tl.program_id(1)
    element = block * 256 + tl.arange(0, 256)
    mask = element < ROWS * COLUMNS
    state = tl.load(Initial + sequence_head * ROWS * COLUMNS + element, mask, 0)
    for chunk in range(CHUNKS):
        offset = (sequence_head * CHUNKS + chunk) * ROWS * COLUMNS + element
        # Readout needs the prior chunk's State, not its terminal State.
        tl.store(Carry + offset, state, mask)
        bias = tl.load(Bias + offset, mask, 0)
        if IDENTITY:
            state += bias
        else:
            scale = tl.load(Scale + sequence_head * CHUNKS + chunk)
            state = scale * state + bias
    if WRITE_FINAL:
        tl.store(Final + sequence_head * ROWS * COLUMNS + element, state, mask)


@triton.jit
def _chunk_readout(
    Propagation,
    Left,
    Right,
    Readout,
    Carry,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    PROPAGATION_HEADS: tl.constexpr,
    LEFT_HEADS: tl.constexpr,
    RIGHT_HEADS: tl.constexpr,
    READOUT_HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr,
    PROPAGATION_OPERATION: tl.constexpr,
    LEFT_OPERATION: tl.constexpr,
    RIGHT_OPERATION: tl.constexpr,
    READOUT_OPERATION: tl.constexpr,
    PROPAGATION_SCALE: tl.constexpr,
    LEFT_SCALE: tl.constexpr,
    RIGHT_SCALE: tl.constexpr,
    READOUT_SCALE: tl.constexpr,
    PROPAGATION_LOG: tl.constexpr,
    LEFT_LOG: tl.constexpr,
    RIGHT_LOG: tl.constexpr,
    READOUT_LOG: tl.constexpr,
):
    sequence_head, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence, head = sequence_head // HEADS, sequence_head % HEADS
    step = tl.arange(0, _CHUNK)
    token = chunk * _CHUNK + step
    column = block * 32 + tl.arange(0, 32)
    row_lane = tl.arange(0, 32)
    left_head = head // (HEADS // LEFT_HEADS)
    right_head = head // (HEADS // RIGHT_HEADS)
    readout_head = head // (HEADS // READOUT_HEADS)
    scores = tl.full((_CHUNK, _CHUNK), 0.0, tl.float32)
    output = tl.full((_CHUNK, 32), 0.0, tl.float32)
    for row_block in range(tl.cdiv(ROWS, 32)):
        row = row_block * 32 + row_lane
        query = _load_expression(
            Readout,
            ((sequence * READOUT_HEADS + readout_head) * LENGTH + token[:, None])
            * ROWS
            + row[None, :],
            (token[:, None] < LENGTH) & (row[None, :] < ROWS),
            READOUT_OPERATION,
            READOUT_SCALE,
            READOUT_LOG,
            0.0,
        )
        key = _load_expression(
            Left,
            ((sequence * LEFT_HEADS + left_head) * LENGTH + token[None, :]) * ROWS
            + row[:, None],
            (token[None, :] < LENGTH) & (row[:, None] < ROWS),
            LEFT_OPERATION,
            LEFT_SCALE,
            LEFT_LOG,
            0.0,
        )
        state = tl.load(
            Carry
            + ((sequence_head * CHUNKS + chunk) * ROWS + row[:, None]) * COLUMNS
            + column[None, :],
            (row[:, None] < ROWS) & (column[None, :] < COLUMNS),
            0,
        )
        scores += tl.dot(query, key, input_precision="tf32x3")
        output += tl.dot(query, state, input_precision="tf32x3")
    if not IDENTITY:
        propagation_head = head // (HEADS // PROPAGATION_HEADS)
        propagation_base = (sequence * PROPAGATION_HEADS + propagation_head) * LENGTH
        gate = _load_expression(
            Propagation,
            propagation_base + token,
            token < LENGTH,
            PROPAGATION_OPERATION,
            PROPAGATION_SCALE,
            PROPAGATION_LOG,
            1.0,
        )
        prefix = tl.associative_scan(gate, 0, _multiply)
        output *= prefix[:, None]
        next_gate = _load_expression(
            Propagation,
            propagation_base + token + 1,
            (step + 1 < _CHUNK) & (token + 1 < LENGTH),
            PROPAGATION_OPERATION,
            PROPAGATION_SCALE,
            PROPAGATION_LOG,
            1.0,
        )
        # One forward product per destination/source pair. Unlike prefix ratios,
        # this remains defined after an exactly-zero gate and for signed gates.
        factors = tl.where(step[None, :] < step[:, None], next_gate[None, :], 1.0)
        weights = tl.associative_scan(factors, 1, _multiply, reverse=True)
        scores *= weights
    scores = tl.where(step[None, :] <= step[:, None], scores, 0.0)
    value = _load_expression(
        Right,
        ((sequence * RIGHT_HEADS + right_head) * LENGTH + token[:, None]) * COLUMNS
        + column[None, :],
        (token[:, None] < LENGTH) & (column[None, :] < COLUMNS),
        RIGHT_OPERATION,
        RIGHT_SCALE,
        RIGHT_LOG,
        0.0,
    )
    output += tl.dot(scores, value, input_precision="tf32x3")
    tl.store(
        Output
        + ((sequence * HEADS + head) * LENGTH + token[:, None]) * COLUMNS
        + column[None, :],
        output,
        (token[:, None] < LENGTH) & (column[None, :] < COLUMNS),
    )


def scalar_chunked_dense_raw(
    propagation: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
    expressions: tuple[Any, Any, Any, Any],
    *,
    sequence_length: int,
    identity_propagation: bool,
    return_final_state: bool,
    threads: int,
) -> None:
    sequences, heads, rows, columns = initial.shape
    chunks = triton.cdiv(sequence_length, CHUNK_TOKENS)
    bias = torch.empty(
        sequences, heads, chunks, rows, columns, device=initial.device, dtype=torch.float32
    )
    carry = torch.empty_like(bias)
    # Identity never reads Scale. An already-live pointer avoids dummy scratch.
    scale = initial if identity_propagation else torch.empty(
        sequences, heads, chunks, device=initial.device, dtype=torch.float32
    )
    common = dict(
        LENGTH=sequence_length,
        HEADS=heads,
        PROPAGATION_HEADS=heads if identity_propagation else propagation.shape[1],
        LEFT_HEADS=left.shape[1],
        RIGHT_HEADS=right.shape[1],
        ROWS=rows,
        COLUMNS=columns,
        CHUNKS=chunks,
        IDENTITY=identity_propagation,
        PROPAGATION_OPERATION=expressions[0].operation,
        LEFT_OPERATION=expressions[1].operation,
        RIGHT_OPERATION=expressions[2].operation,
        PROPAGATION_SCALE=expressions[0].scale,
        LEFT_SCALE=expressions[1].scale,
        RIGHT_SCALE=expressions[2].scale,
        PROPAGATION_LOG=expressions[0].is_log_scale,
        LEFT_LOG=expressions[1].is_log_scale,
        RIGHT_LOG=expressions[2].is_log_scale,
    )
    num_warps = max(4, threads // 32)
    _chunk_summary[(sequences * heads, chunks, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))](
        propagation, left, right, scale, bias, **common,
        num_warps=num_warps, enable_fp_fusion=False,
    )
    _chunk_prefix[(sequences * heads, triton.cdiv(rows * columns, 256))](
        scale, bias, initial, carry, final,
        ROWS=rows, COLUMNS=columns, CHUNKS=chunks,
        IDENTITY=identity_propagation, WRITE_FINAL=return_final_state,
        num_warps=4, enable_fp_fusion=False,
    )
    _chunk_readout[(sequences * heads, chunks, triton.cdiv(columns, 32))](
        propagation, left, right, readout, carry, output, **common,
        READOUT_HEADS=readout.shape[1],
        READOUT_OPERATION=expressions[3].operation,
        READOUT_SCALE=expressions[3].scale,
        READOUT_LOG=expressions[3].is_log_scale,
        num_warps=num_warps, enable_fp_fusion=False,
    )
