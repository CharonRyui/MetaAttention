"""FP32 compact chunk checkpoints and inverse-free, deterministic local VJPs.

Dense and packed recurrences use the same device-segmented mathematical kernels.
Only the chunk-local replay tensor has a token-State axis (exactly 16 tokens).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 16
_CHUNK = tl.constexpr(CHUNK_SIZE)
_CODES = {"scale": 0, "row-scale": 1, "rank-one": 2, "column-scale": 3}


@triton.jit
def _chunk_prefix(Offsets, Prefix, SEQUENCES: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    start = tl.load(Offsets + index, index < SEQUENCES, 0)
    end = tl.load(Offsets + index + 1, index < SEQUENCES, 0)
    count = tl.where(index < SEQUENCES, tl.cdiv(end - start, _CHUNK), 0)
    inclusive = tl.cumsum(count, 0)
    tl.store(Prefix + index, inclusive - count, index < SEQUENCES)
    tl.store(Prefix + SEQUENCES, tl.sum(count, 0))


@triton.jit
def _chunk_metadata(Offsets, Prefix, Metadata, SEQUENCES: tl.constexpr,
                    CAPACITY: tl.constexpr, BLOCK: tl.constexpr, SEARCH_STEPS: tl.constexpr):
    chunk = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(Prefix + SEQUENCES)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), SEQUENCES, tl.int32)
    for iteration in range(SEARCH_STEPS):
        middle = (low + high) // 2
        boundary = tl.load(Prefix + middle, middle <= SEQUENCES, count)
        before = boundary <= chunk
        low = tl.where(before, middle + 1, low)
        high = tl.where(before, high, middle)
    sequence = low - 1
    valid = (chunk < count) & (chunk < CAPACITY)
    sequence = tl.where(valid, sequence, 0)
    first = tl.load(Prefix + sequence)
    start = tl.load(Offsets + sequence) + (chunk - first) * _CHUNK
    end = tl.load(Offsets + sequence + 1)
    tl.store(Metadata + chunk, tl.where(valid, sequence, -1), chunk < CAPACITY)
    tl.store(Metadata + CAPACITY + chunk, start, chunk < CAPACITY)
    tl.store(Metadata + 2 * CAPACITY + chunk, end, chunk < CAPACITY)


def metadata(token_count, sequence_count, device, sequence_offsets=None):
    """Build segmentation without observing any device offset on the CPU."""
    if sequence_offsets is None:
        if sequence_count == 0 or token_count % sequence_count:
            if token_count:
                raise ValueError("dense token count must divide the State batch")
            length = 0
        else:
            length = token_count // sequence_count
        offsets = torch.arange(sequence_count + 1, device=device, dtype=torch.int64) * length
        capacity = sequence_count * triton.cdiv(length, CHUNK_SIZE)
    else:
        offsets = sequence_offsets.contiguous()
        capacity = triton.cdiv(token_count, CHUNK_SIZE) + sequence_count if token_count else 0
    prefix = torch.empty(sequence_count + 1, device=device, dtype=torch.int32)
    chunks = torch.empty((3, capacity), device=device, dtype=torch.int64)
    _chunk_prefix[(1,)](offsets, prefix, sequence_count,
                         triton.next_power_of_2(max(1, sequence_count)))
    if capacity:
        _chunk_metadata[(triton.cdiv(capacity, 128),)](
            offsets, prefix, chunks, sequence_count, capacity, 128,
            (sequence_count + 1).bit_length())
    return offsets, prefix, chunks


@triton.jit
def _apply(state, L, R, C, flat, op: tl.constexpr, code: tl.constexpr,
           TOTAL: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
           LEFT_WIDTH: tl.constexpr, r, c):
    if code == 0:
        state = state * tl.load(C + op * TOTAL + flat)
    elif code == 1:
        gate = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + r, r < ROWS, 1.0)
        state = state * gate[:, None]
    elif code == 3:
        gate = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + c, c < COLS, 1.0)
        state = state * gate[None, :]
    else:
        left = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + r, r < ROWS, 0.0)
        right = tl.load(R + (op * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
        coefficient = tl.load(C + op * TOTAL + flat)
        projection = tl.sum(state * right[:, None], 0)
        state = state + coefficient * left[:, None] * projection[None, :]
    return state


@triton.jit
def _output_flat(token, head, sequence, Offsets, HEADS: tl.constexpr,
                 LENGTH: tl.constexpr, PACKED: tl.constexpr):
    if PACKED:
        return token * HEADS + head
    start = tl.load(Offsets + sequence)
    return (sequence * HEADS + head) * LENGTH + token - start


@triton.jit
def _summary(L, R, C, IK, IV, Q, DY, Metadata, Offsets, Transform, Decay, Bias,
             HEADS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
             TOTAL: tl.constexpr, LEFT_WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
             BR: tl.constexpr, BC: tl.constexpr, OPS: tl.constexpr,
             INJECTIONS: tl.constexpr, READOUTS: tl.constexpr,
             LENGTH: tl.constexpr, PACKED: tl.constexpr, REVERSE: tl.constexpr):
    chunk, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence = tl.load(Metadata + chunk)
    if sequence >= 0:
        start = tl.load(Metadata + CAPACITY + chunk)
        end = tl.load(Metadata + 2 * CAPACITY + chunk)
        r = tl.arange(0, BR)
        c = block * BC + tl.arange(0, BC)
        linear = block < tl.cdiv(ROWS, BC)
        v = c - tl.cdiv(ROWS, BC) * BC
        matrix = tl.where(linear & (r[:, None] == c[None, :]), 1.0, 0.0)
        decay = tl.full((BC,), 1.0, tl.float32)
        for step in range(_CHUNK):
            if REVERSE:
                token = start + _CHUNK - 1 - step
            else:
                token = start + step
            if token < end:
                flat = token * HEADS + head
                if REVERSE:
                    output_flat = _output_flat(token, head, sequence, Offsets, HEADS, LENGTH, PACKED)
                    for readout in tl.static_range(READOUTS):
                        query = tl.load(Q + (readout * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                        dy = tl.load(DY + (readout * TOTAL + output_flat) * COLS + v,
                                     (~linear) & (v < COLS), 0.0)
                        matrix = matrix + query[:, None] * dy[None, :]
                    for op in tl.static_range(len(OPS) - 1, -1, -1):
                        code = OPS[op]
                        if code == 3:
                            gate = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + v,
                                           (~linear) & (v < COLS), 1.0)
                            matrix = matrix * gate[None, :]
                            decay = decay * gate
                        elif code == 2:
                            left = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + r, r < ROWS, 0.0)
                            right = tl.load(R + (op * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                            coefficient = tl.load(C + op * TOTAL + flat)
                            projection = tl.sum(matrix * left[:, None], 0)
                            matrix = matrix + coefficient * right[:, None] * projection[None, :]
                        else:
                            matrix = _apply(matrix, L, R, C, flat, op, code, TOTAL,
                                            ROWS, COLS, LEFT_WIDTH, r, c)
                else:
                    for forward_op in tl.static_range(len(OPS)):
                        if OPS[forward_op] == 3:
                            gate = tl.load(L + (forward_op * TOTAL + flat) * LEFT_WIDTH + v,
                                           (~linear) & (v < COLS), 1.0)
                            matrix = matrix * gate[None, :]
                            decay = decay * gate
                        else:
                            matrix = _apply(matrix, L, R, C, flat, forward_op, OPS[forward_op], TOTAL,
                                            ROWS, COLS, LEFT_WIDTH, r, c)
                    for injection in tl.static_range(INJECTIONS):
                        key = tl.load(IK + (injection * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                        value = tl.load(IV + (injection * TOTAL + flat) * COLS + v,
                                        (~linear) & (v < COLS), 0.0)
                        matrix = matrix + key[:, None] * value[None, :]
        slot = chunk * HEADS + head
        tl.store(Transform + (slot * ROWS + r[:, None]) * ROWS + c[None, :], matrix,
                 linear & (r[:, None] < ROWS) & (c[None, :] < ROWS))
        tl.store(Bias + (slot * ROWS + r[:, None]) * COLS + v[None, :], matrix,
                 (~linear) & (r[:, None] < ROWS) & (v[None, :] < COLS))
        tl.store(Decay + slot * COLS + v, decay, (~linear) & (v < COLS))


@triton.jit
def _carry(Transform, Decay, Bias, Initial, Prefix, Checkpoints, Final,
           HEADS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
           BR: tl.constexpr, BC: tl.constexpr, REVERSE: tl.constexpr):
    sh, block = tl.program_id(0), tl.program_id(1)
    sequence, head = sh // HEADS, sh % HEADS
    first = tl.load(Prefix + sequence)
    last = tl.load(Prefix + sequence + 1)
    r = tl.arange(0, BR)
    c = block * BC + tl.arange(0, BC)
    mask = (r[:, None] < ROWS) & (c[None, :] < COLS)
    state = tl.load(Initial + (sh * ROWS + r[:, None]) * COLS + c[None, :], mask, 0.0)
    for index in range(last - first):
        if REVERSE:
            chunk = last - 1 - index
        else:
            chunk = first + index
        slot = chunk * HEADS + head
        base = (slot * ROWS + r[:, None]) * COLS + c[None, :]
        tl.store(Checkpoints + base, state, mask)
        if REVERSE or index + 1 < last - first:
            transform = tl.load(Transform + (slot * ROWS + r[:, None]) * ROWS + r[None, :],
                                (r[:, None] < ROWS) & (r[None, :] < ROWS), 0.0)
            decay = tl.load(Decay + slot * COLS + c, c < COLS, 1.0)
            bias = tl.load(Bias + base, mask, 0.0)
            state = tl.dot(transform, state, input_precision="tf32x3") * decay[None, :] + bias
    if REVERSE or first == last:
        tl.store(Final + (sh * ROWS + r[:, None]) * COLS + c[None, :], state, mask)


@triton.jit
def _replay(L, R, C, IK, IV, Q, Metadata, Offsets, Checkpoints, Output, Final,
            HEADS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
            TOTAL: tl.constexpr, LEFT_WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
            BR: tl.constexpr, BC: tl.constexpr, OPS: tl.constexpr,
            INJECTIONS: tl.constexpr, READOUTS: tl.constexpr,
            LENGTH: tl.constexpr, PACKED: tl.constexpr):
    chunk, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence = tl.load(Metadata + chunk)
    if sequence >= 0:
        start = tl.load(Metadata + CAPACITY + chunk)
        end = tl.load(Metadata + 2 * CAPACITY + chunk)
        r = tl.arange(0, BR)
        c = block * BC + tl.arange(0, BC)
        mask = (r[:, None] < ROWS) & (c[None, :] < COLS)
        slot = chunk * HEADS + head
        state = tl.load(Checkpoints + (slot * ROWS + r[:, None]) * COLS + c[None, :], mask, 0.0)
        for step in range(_CHUNK):
            token = start + step
            if token < end:
                flat = token * HEADS + head
                for op in tl.static_range(len(OPS)):
                    state = _apply(state, L, R, C, flat, op, OPS[op], TOTAL,
                                   ROWS, COLS, LEFT_WIDTH, r, c)
                for injection in tl.static_range(INJECTIONS):
                    key = tl.load(IK + (injection * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    value = tl.load(IV + (injection * TOTAL + flat) * COLS + c, c < COLS, 0.0)
                    state = state + key[:, None] * value[None, :]
                output_flat = _output_flat(token, head, sequence, Offsets, HEADS, LENGTH, PACKED)
                for readout in tl.static_range(READOUTS):
                    query = tl.load(Q + (readout * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    output = tl.sum(state * query[:, None], 0)
                    tl.store(Output + (readout * TOTAL + output_flat) * COLS + c, output, c < COLS)
        if start + _CHUNK >= end:
            tl.store(Final + ((sequence * HEADS + head) * ROWS + r[:, None]) * COLS + c[None, :], state, mask)


@triton.jit
def _local_vjp(L, R, C, IK, IV, Q, Metadata, Offsets, Checkpoints, Carry, DY,
               DL, DR, DC, DIK, DIV, DQ,
               HEADS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
               TOTAL: tl.constexpr, LEFT_WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
               BR: tl.constexpr, BC: tl.constexpr, OPS: tl.constexpr,
               INJECTIONS: tl.constexpr, READOUTS: tl.constexpr,
               LENGTH: tl.constexpr, PACKED: tl.constexpr):
    chunk, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence = tl.load(Metadata + chunk)
    if sequence >= 0:
        start = tl.load(Metadata + CAPACITY + chunk)
        end = tl.load(Metadata + 2 * CAPACITY + chunk)
        r = tl.arange(0, BR)
        c = block * BC + tl.arange(0, BC)
        steps = tl.arange(0, _CHUNK)
        mask = (r[:, None] < ROWS) & (c[None, :] < COLS)
        base = ((chunk * HEADS + head) * ROWS + r[:, None]) * COLS + c[None, :]
        state = tl.load(Checkpoints + base, mask, 0.0)
        history = tl.full((_CHUNK, BR, BC), 0.0, tl.float32)
        for step in range(_CHUNK):
            token = start + step
            history = tl.where(steps[:, None, None] == step, state[None, :, :], history)
            if token < end:
                flat = token * HEADS + head
                for op in tl.static_range(len(OPS)):
                    state = _apply(state, L, R, C, flat, op, OPS[op], TOTAL,
                                   ROWS, COLS, LEFT_WIDTH, r, c)
                for injection in tl.static_range(INJECTIONS):
                    key = tl.load(IK + (injection * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    value = tl.load(IV + (injection * TOTAL + flat) * COLS + c, c < COLS, 0.0)
                    state = state + key[:, None] * value[None, :]
        adjoint = tl.load(Carry + base, mask, 0.0)
        for reverse_step in range(_CHUNK):
            step = _CHUNK - 1 - reverse_step
            token = start + step
            if token < end:
                flat = token * HEADS + head
                gather_index = tl.full((1, BR, BC), step, tl.int32)
                previous = tl.gather(history, gather_index, 0).reshape((BR, BC))
                state = previous
                for op in tl.static_range(len(OPS)):
                    state = _apply(state, L, R, C, flat, op, OPS[op], TOTAL,
                                   ROWS, COLS, LEFT_WIDTH, r, c)
                for injection in tl.static_range(INJECTIONS):
                    key = tl.load(IK + (injection * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    value = tl.load(IV + (injection * TOTAL + flat) * COLS + c, c < COLS, 0.0)
                    state = state + key[:, None] * value[None, :]
                output_flat = _output_flat(token, head, sequence, Offsets, HEADS, LENGTH, PACKED)
                for readout in tl.static_range(READOUTS):
                    dy = tl.load(DY + (readout * TOTAL + output_flat) * COLS + c, c < COLS, 0.0)
                    query = tl.load(Q + (readout * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    dq = tl.sum(state * dy[None, :], 1)
                    tl.store(DQ + ((block * READOUTS + readout) * TOTAL + flat) * ROWS + r, dq, r < ROWS)
                    adjoint = adjoint + query[:, None] * dy[None, :]
                for injection in tl.static_range(INJECTIONS):
                    key = tl.load(IK + (injection * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                    value = tl.load(IV + (injection * TOTAL + flat) * COLS + c, c < COLS, 0.0)
                    dk = tl.sum(adjoint * value[None, :], 1)
                    dv = tl.sum(adjoint * key[:, None], 0)
                    tl.store(DIK + ((block * INJECTIONS + injection) * TOTAL + flat) * ROWS + r, dk, r < ROWS)
                    tl.store(DIV + (injection * TOTAL + flat) * COLS + c, dv, c < COLS)
                for op in tl.static_range(len(OPS) - 1, -1, -1):
                    code = OPS[op]
                    state = previous
                    for earlier in tl.static_range(op):
                        state = _apply(state, L, R, C, flat, earlier, OPS[earlier], TOTAL,
                                       ROWS, COLS, LEFT_WIDTH, r, c)
                    if code == 0:
                        coefficient = tl.load(C + op * TOTAL + flat)
                        dc = tl.sum(tl.sum(adjoint * state, 1), 0)
                        tl.store(DC + (block * len(OPS) + op) * TOTAL + flat, dc)
                        adjoint = adjoint * coefficient
                    elif code == 1:
                        gate = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + r, r < ROWS, 1.0)
                        dl_row = tl.sum(adjoint * state, 1)
                        tl.store(DL + ((block * len(OPS) + op) * TOTAL + flat) * LEFT_WIDTH + r, dl_row, r < ROWS)
                        adjoint = adjoint * gate[:, None]
                    elif code == 3:
                        gate = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + c, c < COLS, 1.0)
                        dl_column = tl.sum(adjoint * state, 0)
                        tl.store(DL + ((block * len(OPS) + op) * TOTAL + flat) * LEFT_WIDTH + c, dl_column, c < COLS)
                        adjoint = adjoint * gate[None, :]
                    else:
                        left = tl.load(L + (op * TOTAL + flat) * LEFT_WIDTH + r, r < ROWS, 0.0)
                        right = tl.load(R + (op * TOTAL + flat) * ROWS + r, r < ROWS, 0.0)
                        coefficient = tl.load(C + op * TOTAL + flat)
                        projection = tl.sum(state * right[:, None], 0)
                        adjoint_projection = tl.sum(adjoint * left[:, None], 0)
                        dl = coefficient * tl.sum(adjoint * projection[None, :], 1)
                        dr = coefficient * tl.sum(state * adjoint_projection[None, :], 1)
                        dc = tl.sum(adjoint_projection * projection, 0)
                        tl.store(DL + ((block * len(OPS) + op) * TOTAL + flat) * LEFT_WIDTH + r, dl, r < ROWS)
                        tl.store(DR + ((block * len(OPS) + op) * TOTAL + flat) * ROWS + r, dr, r < ROWS)
                        tl.store(DC + (block * len(OPS) + op) * TOTAL + flat, dc)
                        adjoint = adjoint + coefficient * right[:, None] * adjoint_projection[None, :]


@triton.jit
def _reduce(Partial, Result, SIZE: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for block in range(BLOCKS):
        value = value + tl.load(Partial + block * SIZE + index, index < SIZE, 0.0)
    tl.store(Result + index, value, index < SIZE)


def _layout(readout, initial_state, sequence_offsets):
    sequences, heads, rows, columns = initial_state.shape
    tokens = readout.shape[-4]
    packed = sequence_offsets is not None
    length = 0 if packed or not sequences else tokens // sequences
    count = readout.shape[0] if readout.ndim == 5 else 1
    shape = (tokens, heads, columns) if packed else (sequences, heads, length, columns)
    if readout.ndim == 5:
        shape = (count, *shape)
    return sequences, heads, rows, columns, tokens, length, count, packed, shape


def forward(*, operation_codes, left_factors, right_factors, coefficients,
            injection_left, injection_right, readout, initial_state,
            output_dtype, record=True, sequence_offsets=None, chunk_metadata=None):
    """Return outputs, final State and FP32 entries; segmentation stays on device."""
    sequences, heads, rows, columns, tokens, length, count, packed, shape = _layout(
        readout, initial_state, sequence_offsets)
    offsets, prefix, chunks = (metadata(tokens, sequences, initial_state.device, sequence_offsets)
                               if chunk_metadata is None else chunk_metadata)
    capacity = chunks.shape[1]
    device = initial_state.device
    checkpoints = torch.empty((capacity, heads, rows, columns), device=device, dtype=torch.float32)
    output = torch.empty(shape, device=device, dtype=output_dtype)
    final = torch.empty_like(initial_state)
    if not capacity:
        final.copy_(initial_state)
    else:
        codes = tuple(_CODES[code] for code in operation_codes)
        total, width = tokens * heads, left_factors.shape[-2]
        br = max(16, triton.next_power_of_2(rows))
        transform = torch.empty((capacity, heads, rows, rows), device=device, dtype=torch.float32)
        decay = torch.empty((capacity, heads, columns), device=device, dtype=torch.float32)
        bias = torch.empty_like(checkpoints)
        _summary[(capacity, heads, triton.cdiv(rows, 32) + triton.cdiv(columns, 32))](
            left_factors, right_factors, coefficients, injection_left, injection_right,
            readout, output, chunks, offsets, transform, decay, bias,
            heads, rows, columns, total, width, capacity, br, 32, codes,
            injection_left.shape[0], count, length, packed, False,
            num_warps=4, enable_fp_fusion=False)
        _carry[(sequences * heads, triton.cdiv(columns, 16))](
            transform, decay, bias, initial_state, prefix, checkpoints, final,
            heads, rows, columns, br, 16, False, num_warps=4, enable_fp_fusion=False)
        _replay[(capacity, heads, triton.cdiv(columns, 16))](
            left_factors, right_factors, coefficients, injection_left, injection_right,
            readout, chunks, offsets, checkpoints, output, final,
            heads, rows, columns, total, width, capacity, br, 16, codes,
            injection_left.shape[0], count, length, packed,
            num_warps=4, enable_fp_fusion=False)
    if not packed:
        checkpoints = checkpoints.reshape(sequences, triton.cdiv(length, CHUNK_SIZE), heads, rows, columns).permute(0, 2, 1, 3, 4)
    if not record:
        checkpoints = checkpoints[:0]
    return output, final, checkpoints


def backward(*, operation_codes, left_factors, right_factors, coefficients,
             injection_left, injection_right, readout, checkpoints, initial_state,
             grad_output, grad_final, sequence_offsets=None, chunk_metadata=None):
    """Reverse affine carry followed by parallel, inverse-free chunk-local VJPs."""
    sequences, heads, rows, columns, tokens, length, count, packed, shape = _layout(
        readout, initial_state, sequence_offsets)
    offsets, prefix, chunks = (metadata(tokens, sequences, initial_state.device, sequence_offsets)
                               if chunk_metadata is None else chunk_metadata)
    capacity = chunks.shape[1]
    if not packed:
        checkpoints = checkpoints.permute(0, 2, 1, 3, 4).reshape(capacity, heads, rows, columns)
    factors = (left_factors, right_factors, coefficients, injection_left, injection_right, readout)
    gradients = [torch.zeros_like(value, dtype=torch.float32) for value in factors]
    di = torch.empty_like(initial_state, dtype=torch.float32)
    df = torch.zeros_like(initial_state, dtype=torch.float32) if grad_final is None else grad_final.float().contiguous()
    if not capacity:
        di.copy_(df)
        return (*gradients, di)
    dy = torch.zeros(shape, device=initial_state.device, dtype=torch.float32) if grad_output is None else grad_output.float().contiguous()
    codes = tuple(_CODES[code] for code in operation_codes)
    total, width = tokens * heads, left_factors.shape[-2]
    br, blocks = max(16, triton.next_power_of_2(rows)), triton.cdiv(columns, 16)
    transform = torch.empty((capacity, heads, rows, rows), device=initial_state.device, dtype=torch.float32)
    decay = torch.empty((capacity, heads, columns), device=initial_state.device, dtype=torch.float32)
    source = torch.empty_like(checkpoints)
    carry = torch.empty_like(checkpoints)
    _summary[(capacity, heads, triton.cdiv(rows, 32) + triton.cdiv(columns, 32))](
        left_factors, right_factors, coefficients, injection_left, injection_right,
        readout, dy, chunks, offsets, transform, decay, source,
        heads, rows, columns, total, width, capacity, br, 32, codes,
        injection_left.shape[0], count, length, packed, True,
        num_warps=4, enable_fp_fusion=False)
    _carry[(sequences * heads, blocks)](
        transform, decay, source, df, prefix, carry, di,
        heads, rows, columns, br, 16, True, num_warps=4, enable_fp_fusion=False)
    partial_indices = (0, 1, 2, 3, 5)
    partials = {index: torch.zeros((blocks, *factors[index].shape), device=initial_state.device,
                                 dtype=torch.float32) for index in partial_indices}
    _local_vjp[(capacity, heads, blocks)](
        *factors, chunks, offsets, checkpoints, carry, dy,
        partials[0], partials[1], partials[2], partials[3], gradients[4], partials[5],
        heads, rows, columns, total, width, capacity, br, 16, codes,
        injection_left.shape[0], count, length, packed, num_warps=4, enable_fp_fusion=False)
    for index in partial_indices:
        size = factors[index].numel()
        if size:
            _reduce[(triton.cdiv(size, 256),)](partials[index], gradients[index], size, blocks, 256,
                                             enable_fp_fusion=False)
    return (*gradients, di)
