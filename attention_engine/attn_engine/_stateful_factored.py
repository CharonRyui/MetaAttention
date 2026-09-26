"""Factored recurrent lowering for matrices States with rank-one transitions.

A matrix State whose per-token transition is a chain of whole-State or
first-feature Axis Scales and first-feature Rank One contractions is applied by
one kernel that keeps the State column in registers and replays the authored
operations per token. Nothing per token is written to global memory, so the
plan's memory is the State plus the output, and the doubling prefix scan and its
per-token factor matrices disappear.
"""
from __future__ import annotations

from functools import lru_cache

import torch


_OPERATION_CODES = {"scale": 0, "row-scale": 1, "rank-one": 2}


def _stack(parts: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if not parts:
        return torch.empty((0, *shape), device="cuda", dtype=torch.float32)
    return torch.stack([part.reshape(shape) for part in parts], dim=0).contiguous()


@lru_cache(maxsize=None)
def _compile_factored_dense(
    operation_codes: tuple[str, ...],
    injection_count: int,
    token_count: int,
    sequence_count: int,
    heads: int,
    rows: int,
    columns: int,
    output_dtype: torch.dtype,
    device_index: int,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    thread_count = int(threads)
    block_columns = min(columns, thread_count)
    column_blocks = tilelang.cdiv(columns, block_columns)
    operation_count = len(operation_codes)

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            opcode: T.Tensor((operation_count,), dtype="int32"),
            left: T.Tensor(
                (operation_count, token_count, heads, rows, 1), dtype="float32"
            ),
            right: T.Tensor(
                (operation_count, token_count, heads, rows, 1), dtype="float32"
            ),
            coefficient: T.Tensor(
                (operation_count, token_count, heads, 1, 1), dtype="float32"
            ),
            injection_left: T.Tensor(
                (injection_count, token_count, heads, rows, 1), dtype="float32"
            ),
            injection_right: T.Tensor(
                (injection_count, token_count, heads, 1, columns), dtype="float32"
            ),
            readout: T.Tensor((token_count, heads, rows, 1), dtype="float32"),
            initial: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
            output: T.Tensor(
                (sequence_count, heads, token_count, columns), dtype=output_dtype
            ),
            final: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
        ):
            if False:
                # tilelang resolves annotations through the kernel's free
                # variables, so names used only in a signature must be referenced.
                _ = (output_dtype, operation_count, injection_count)
            with T.Kernel(
                sequence_count * heads, column_blocks, threads=thread_count
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                column = column_block * block_columns + T.get_thread_binding()
                state = T.alloc_local((rows,), dtype="float32")
                dot = T.alloc_local((1,), dtype="float32")
                accumulated = T.alloc_local((1,), dtype="float32")
                if column < columns:
                    for row in T.serial(rows):
                        state[row] = initial[sequence, head, row, column]
                for token in T.Serial(token_count):
                    for index in T.Serial(operation_count):
                        if opcode[index] == 0:
                            for row in T.serial(rows):
                                state[row] = (
                                    state[row] * coefficient[index, token, head, 0, 0]
                                )
                        elif opcode[index] == 1:
                            for row in T.serial(rows):
                                state[row] = state[row] * left[index, token, head, row, 0]
                        else:
                            dot[0] = 0.0
                            for row in T.serial(rows):
                                dot[0] = (
                                    dot[0]
                                    + right[index, token, head, row, 0] * state[row]
                                )
                            for row in T.serial(rows):
                                state[row] = state[row] + (
                                    coefficient[index, token, head, 0, 0]
                                    * left[index, token, head, row, 0]
                                    * dot[0]
                                )
                    for index in T.Serial(injection_count):
                        if column < columns:
                            for row in T.serial(rows):
                                state[row] = state[row] + (
                                    injection_left[index, token, head, row, 0]
                                    * injection_right[index, token, head, 0, column]
                                )
                    if column < columns:
                        accumulated[0] = 0.0
                        for row in T.serial(rows):
                            accumulated[0] = (
                                accumulated[0] + readout[token, head, row, 0] * state[row]
                            )
                        output[sequence, head, token, column] = accumulated[0]
                if column < columns:
                    for row in T.serial(rows):
                        final[sequence, head, row, column] = state[row]

        return kernel

    return build_kernel()


def factored_recurrence_dense(
    *,
    operation_codes: tuple[str, ...],
    left_factors: list[torch.Tensor],
    right_factors: list[torch.Tensor],
    coefficients: list[torch.Tensor],
    injection_left: list[torch.Tensor],
    injection_right: list[torch.Tensor],
    readout: torch.Tensor,
    initial_state: torch.Tensor,
    output_dtype: torch.dtype,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the authored operations as one factored recurrent kernel.

    ``operation_codes`` is the authored operation sequence: ``"scale"`` for a
    whole-State Axis Scale, ``"row-scale"`` for a first-feature Axis Scale, and
    ``"rank-one"`` for a first-feature Rank One contraction. Every per-token
    tensor keeps the Program's head-first ``[tokens, heads, ...]`` layout.
    """
    sequence_count, heads, _, columns = initial_state.shape
    rows = initial_state.shape[2]
    token_count = readout.shape[0]
    shape = (token_count, heads, rows, 1)
    scalar_shape = (token_count, heads, 1, 1)
    columns_shape = (token_count, heads, 1, columns)
    device_index = initial_state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_factored_dense(
        tuple(operation_codes),
        len(injection_left),
        token_count,
        sequence_count,
        heads,
        rows,
        columns,
        output_dtype,
        device_index,
        threads,
    )
    output = torch.empty(
        sequence_count,
        heads,
        token_count,
        columns,
        device=initial_state.device,
        dtype=output_dtype,
    )
    final = torch.empty_like(initial_state)
    codes = torch.tensor(
        [_OPERATION_CODES[code] for code in operation_codes],
        dtype=torch.int32,
        device=initial_state.device,
    )
    kernel(
        codes,
        _stack(left_factors, shape),
        _stack(right_factors, shape),
        _stack(coefficients, scalar_shape),
        _stack(injection_left, shape),
        _stack(injection_right, columns_shape),
        readout.reshape(shape).float().contiguous(),
        initial_state.contiguous(),
        output,
        final,
    )
    return output, final
