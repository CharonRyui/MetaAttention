"""Chunked execution for diagonal State recurrences.

A diagonal Transition (whole-State, first-Feature or second-Feature Axis Scale
factors) needs no dense summary. This module runs such a Program through two
kernels: one that walks each sequence once and writes the running affine pair
only where a chunk ends, and one that replays each chunk independently with the
State column in registers and the readout contracted in place. Nothing
token-sized reaches global memory except the output.

The kernels own their symbols and layouts, so they never share a TileLang cache
entry with the per-token prefix kernel of the elementwise family.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _compile_boundary_pair(
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    chunk_tokens: int,
    threads: int,
    device_index: int,
):
    """Write the running pair where each chunk of each sequence ends.

    One thread owns one State element and walks its sequence accumulating
    ``(A_t, B_t)`` with ``S_t = A_t S_0 + B_t``. Only chunk boundaries are
    written, into a compact ``[chunks, heads, rows, columns]`` buffer.
    """
    import tilelang
    import tilelang.language as T

    chunk_count = tilelang.cdiv(sequence_length, chunk_tokens)
    matrix_size = rows * columns
    matrix_blocks = tilelang.cdiv(matrix_size, threads)

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build_kernel():
        @T.prim_func
        def boundary_pair_kernel(
            gate_rows: T.Tensor((sequence_length, heads, rows), "float32"),
            gate_columns: T.Tensor((sequence_length, heads, columns), "float32"),
            key: T.Tensor((sequence_length, heads, rows), "float32"),
            value: T.Tensor((sequence_length, heads, columns), "float32"),
            carry_scale: T.Tensor((chunk_count, heads, rows, columns), "float32"),
            carry_bias: T.Tensor((chunk_count, heads, rows, columns), "float32"),
        ):
            if False:
                _ = (chunk_count, matrix_blocks, rows, columns, threads)
            with T.Kernel(sequence_count * heads, matrix_blocks, threads=threads) as (
                sequence_head,
                matrix_block,
            ):
                sequence = sequence_head // heads
                head = sequence_head % heads
                index = matrix_block * threads + T.get_thread_binding()
                accumulated_scale = T.alloc_local((1,), dtype="float32")
                accumulated_bias = T.alloc_local((1,), dtype="float32")
                if index < matrix_size:
                    row = index // columns
                    column = index % columns
                    accumulated_scale[0] = 1.0
                    accumulated_bias[0] = 0.0
                    for token in T.Serial(sequence_length):
                        # factors are token-major with batch-major flattening
                        flat = sequence * sequence_length + token
                        accumulated_bias[0] = (
                            gate_rows[flat, head, row]
                            * gate_columns[flat, head, column]
                            * accumulated_bias[0]
                            + key[flat, head, row] * value[flat, head, column]
                        )
                        accumulated_scale[0] = (
                            gate_rows[flat, head, row]
                            * gate_columns[flat, head, column]
                            * accumulated_scale[0]
                        )
                        if (
                            token % chunk_tokens == chunk_tokens - 1
                            or token == sequence_length - 1
                        ):
                            carry_scale[
                                token // chunk_tokens, head, row, column
                            ] = accumulated_scale[0]
                            carry_bias[
                                token // chunk_tokens, head, row, column
                            ] = accumulated_bias[0]

        return boundary_pair_kernel

    return build_kernel()


@lru_cache(maxsize=None)
def _compile_chunk_recurrence(
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    output_dtype: torch.dtype,
    chunk_tokens: int,
    thread_columns: int,
    device_index: int,
):
    """Replay one chunk per block, State in registers, readout in place.

    The grid gives every (sequence, head, chunk, column slice) its own block, so
    the block count scales with the chunk count instead of with heads alone.
    """
    import tilelang
    import tilelang.language as T

    chunk_count = tilelang.cdiv(sequence_length, chunk_tokens)
    block_columns = min(columns, thread_columns)
    column_blocks = tilelang.cdiv(columns, block_columns)

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build_kernel():
        @T.prim_func
        def chunk_recurrence_kernel(
            carry_scale: T.Tensor((chunk_count, heads, rows, columns), "float32"),
            carry_bias: T.Tensor((chunk_count, heads, rows, columns), "float32"),
            gate_rows: T.Tensor((sequence_length, heads, rows), "float32"),
            gate_columns: T.Tensor((sequence_length, heads, columns), "float32"),
            key: T.Tensor((sequence_length, heads, rows), "float32"),
            value: T.Tensor((sequence_length, heads, columns), "float32"),
            query: T.Tensor((sequence_length, heads, rows), "float32"),
            initial: T.Tensor((sequence_count, heads, rows, columns), "float32"),
            output: T.Tensor(
                (sequence_count, heads, sequence_length, columns), dtype=output_dtype
            ),
            final: T.Tensor((sequence_count, heads, rows, columns), "float32"),
        ):
            if False:
                _ = (output_dtype, chunk_count, column_blocks, rows, columns)
            with T.Kernel(
                sequence_count * heads, chunk_count, column_blocks, threads=block_columns
            ) as (sequence_head, chunk, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                column = column_block * block_columns + T.get_thread_binding()
                state = T.alloc_local((rows,), dtype="float32")
                accumulated = T.alloc_local((1,), dtype="float32")
                start = chunk * chunk_tokens
                if column < columns:
                    for row in T.serial(rows):
                        if chunk == 0:
                            state[row] = initial[sequence, head, row, column]
                        else:
                            state[row] = (
                                carry_scale[chunk - 1, head, row, column]
                                * initial[sequence, head, row, column]
                                + carry_bias[chunk - 1, head, row, column]
                            )
                for token_in_chunk in T.Serial(chunk_tokens):
                    token = start + token_in_chunk
                    if token < sequence_length:
                        flat = sequence * sequence_length + token
                        if column < columns:
                            for row in T.serial(rows):
                                state[row] = (
                                    gate_rows[flat, head, row]
                                    * gate_columns[flat, head, column]
                                    * state[row]
                                    + key[flat, head, row] * value[flat, head, column]
                                )
                            accumulated[0] = 0.0
                            for row in T.serial(rows):
                                accumulated[0] = (
                                    accumulated[0] + query[flat, head, row] * state[row]
                                )
                            output[sequence, head, token, column] = accumulated[0]
                    if chunk == chunk_count - 1 and token == sequence_length - 1:
                        # the last chunk owns the final State of the sequence
                        for row in T.serial(rows):
                            final[sequence, head, row, column] = state[row]

        return chunk_recurrence_kernel

    return build_kernel()


def chunked_elementwise_dense(
    *,
    gate_rows: torch.Tensor,
    gate_columns: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    initial_state: torch.Tensor,
    output_dtype: torch.dtype,
    chunk_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one diagonal recurrence through the chunked schedule.

    Every per-token tensor is token-major and contiguous. ``gate_columns`` may be
    omitted for a whole-State or first-Feature decay, in which case the column
    factor is one.
    """
    rows, columns = initial_state.shape[2], initial_state.shape[3]
    sequence_count, heads = initial_state.shape[0], initial_state.shape[1]
    token_count = key.shape[0]
    if token_count % chunk_tokens and sequence_count > 1:
        raise ValueError("chunked recurrence needs chunk-aligned dense sequences")
    device = initial_state.device
    rows_gate = gate_rows.reshape(token_count, heads, -1)
    if rows_gate.shape[-1] == 1 and rows > 1:
        # a whole-State decay applies to every row
        rows_gate = rows_gate.expand(token_count, heads, rows)
    rows_gate = rows_gate.float().contiguous()
    # the kernels accumulate in fp32, so the per-token factors are promoted here,
    # at one declared point, rather than inside the kernel
    key = key.reshape(token_count, heads, -1).float().contiguous()
    value = value.reshape(token_count, heads, -1).float().contiguous()
    query = query.reshape(token_count, heads, -1).float().contiguous()
    columns_gate = (
        torch.ones(token_count, heads, columns, device=device, dtype=torch.float32)
        if gate_columns is None
        else gate_columns.reshape(token_count, heads, columns).float().contiguous()
    )
    chunk_count = (token_count + chunk_tokens - 1) // chunk_tokens
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    boundary = _compile_boundary_pair(
        sequence_count,
        token_count,
        heads,
        rows,
        columns,
        chunk_tokens,
        128,
        device_index,
    )
    carry_scale = torch.empty(
        chunk_count, heads, rows, columns, device=device, dtype=torch.float32
    )
    carry_bias = torch.empty_like(carry_scale)
    boundary(
        rows_gate,
        columns_gate,
        key.contiguous(),
        value.contiguous(),
        carry_scale,
        carry_bias,
    )
    kernel = _compile_chunk_recurrence(
        sequence_count,
        token_count,
        heads,
        rows,
        columns,
        output_dtype,
        chunk_tokens,
        64,
        device_index,
    )
    output = torch.empty(
        sequence_count, heads, token_count, columns, device=device, dtype=output_dtype
    )
    final = torch.empty(
        sequence_count, heads, rows, columns, device=device, dtype=torch.float32
    )
    kernel(
        carry_scale,
        carry_bias,
        rows_gate,
        columns_gate,
        key,
        value,
        query,
        initial_state.contiguous(),
        output,
        final,
    )
    return output, final
