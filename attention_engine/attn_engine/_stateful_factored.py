"""Ordered low-rank recurrence lowering.

Dense inference composes FP32 affine summaries per chunk, scans incoming State
carries and replays independent chunks with all authored readouts. The original
single-readout recurrent kernel and recorded-State reverse walk remain the
compiled gradient path. Neither path reorders noncommuting propagations.
"""

from __future__ import annotations

from functools import lru_cache

import torch


_OPERATION_CODES = {"scale": 0, "row-scale": 1, "column-scale": 3, "rank-one": 2}


def _code_names(codes: tuple[int, ...]) -> tuple[str, ...]:
    names = {value: name for name, value in _OPERATION_CODES.items()}
    return tuple(names[code] for code in codes)


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
    record_states: bool = False,
):
    import tilelang
    import tilelang.language as T

    thread_count = int(threads)
    block_columns = min(columns, thread_count)
    column_blocks = tilelang.cdiv(columns, block_columns)
    operation_count = len(operation_codes)
    # an unused recording buffer keeps a valid (empty) shape rather than a
    # different signature, so one kernel definition serves both modes
    record_sequences = sequence_count if record_states else 0
    record_tokens = token_count if record_states else 0

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
            recorded: T.Tensor(
                (record_sequences, heads, record_tokens, rows, columns),
                dtype="bfloat16",
            ),
        ):
            if False:
                # tilelang resolves annotations through the kernel's free
                # variables, so names used only in a signature must be referenced.
                _ = (
                    output_dtype,
                    operation_count,
                    injection_count,
                    record_sequences,
                    record_tokens,
                )
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
                                state[row] = (
                                    state[row] * left[index, token, head, row, 0]
                                )
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
                                accumulated[0]
                                + readout[token, head, row, 0] * state[row]
                            )
                        output[sequence, head, token, column] = accumulated[0]
                        if record_states:
                            # the reverse walk needs the per-token State; storing it
                            # is exact, while inverting the recurrence is not
                            for row in T.serial(rows):
                                recorded[sequence, head, token, row, column] = state[
                                    row
                                ]
                if column < columns:
                    for row in T.serial(rows):
                        final[sequence, head, row, column] = state[row]

        return kernel

    return build_kernel()


def _run_factored_forward(
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
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
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
    record_states = backward is not None
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
        record_states,
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
    recorded = torch.empty(
        sequence_count if record_states else 0,
        heads,
        token_count if record_states else 0,
        rows,
        columns,
        device=initial_state.device,
        dtype=torch.bfloat16,
    )
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
        recorded,
    )
    return output, final, recorded


def factored_recurrence_chunked(
    *,
    operation_codes: tuple[str, ...],
    left_factors: list[torch.Tensor],
    right_factors: list[torch.Tensor],
    coefficients: list[torch.Tensor],
    injection_left: list[torch.Tensor],
    injection_right: list[torch.Tensor],
    readouts: list[torch.Tensor],
    initial_state: torch.Tensor,
    output_dtypes: tuple[torch.dtype, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Execute ordered affine summaries, FP32 chunk carries and parallel replay."""
    from ._stateful_lowrank_kernels import execute

    return execute(
        operation_codes=operation_codes,
        left_factors=left_factors,
        right_factors=right_factors,
        coefficients=coefficients,
        injection_left=injection_left,
        injection_right=injection_right,
        readouts=readouts,
        initial_state=initial_state,
        output_dtypes=output_dtypes,
    )


@lru_cache(maxsize=None)
def _compile_factored_backward(
    operation_codes: tuple[str, ...],
    injection_count: int,
    token_count: int,
    sequence_count: int,
    heads: int,
    rows: int,
    columns: int,
    device_index: int,
    threads: int = 128,
):
    """Reverse walk over stored token States.

    The walk reads the per-token States recorded by the forward and replays each
    token's authored chain forward to recover an operation's input State, so no
    gradient depends on inverting a decaying recurrence.
    """
    import tilelang
    import tilelang.language as T

    thread_count = int(threads)
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
            recorded: T.Tensor(
                (sequence_count, heads, token_count, rows, columns), dtype="bfloat16"
            ),
            initial: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
            grad_output: T.Tensor(
                (sequence_count, heads, token_count, columns), dtype="float32"
            ),
            grad_final: T.Tensor(
                (sequence_count, heads, rows, columns), dtype="float32"
            ),
            grad_left: T.Tensor(
                (operation_count, token_count, heads, rows, 1), dtype="float32"
            ),
            grad_right: T.Tensor(
                (operation_count, token_count, heads, rows, 1), dtype="float32"
            ),
            grad_coefficient: T.Tensor(
                (operation_count, token_count, heads, 1, 1), dtype="float32"
            ),
            grad_injection_left: T.Tensor(
                (injection_count, token_count, heads, rows, 1), dtype="float32"
            ),
            grad_injection_right: T.Tensor(
                (injection_count, token_count, heads, 1, columns), dtype="float32"
            ),
            grad_readout: T.Tensor((token_count, heads, rows, 1), dtype="float32"),
            grad_initial: T.Tensor(
                (sequence_count, heads, rows, columns), dtype="float32"
            ),
        ):
            if False:
                _ = (rows, columns, operation_count, injection_count)
            with T.Kernel(sequence_count * heads, threads=thread_count) as (
                sequence_head,
            ):
                sequence = sequence_head // heads
                head = sequence_head % heads
                state = T.alloc_shared((rows, columns), dtype="float32")
                previous = T.alloc_shared((rows, columns), dtype="float32")
                adjoint = T.alloc_shared((rows, columns), dtype="float32")
                factor_row = T.alloc_shared((rows,), dtype="float32")
                second_row = T.alloc_shared((rows,), dtype="float32")
                column_factor = T.alloc_shared((columns,), dtype="float32")
                second_column = T.alloc_shared((columns,), dtype="float32")
                row_scratch = T.alloc_shared((rows,), dtype="float32")
                scalars = T.alloc_shared((2,), dtype="float32")
                local = T.alloc_local((1,), dtype="float32")

                for row, column in T.Parallel(rows, columns):
                    adjoint[row, column] = grad_final[sequence, head, row, column]
                T.sync_threads()

                for step in T.Serial(token_count):
                    token = token_count - 1 - step
                    for row, column in T.Parallel(rows, columns):
                        state[row, column] = recorded[
                            sequence, head, token, row, column
                        ]
                        if token == 0:
                            previous[row, column] = initial[sequence, head, row, column]
                        else:
                            previous[row, column] = recorded[
                                sequence, head, token - 1, row, column
                            ]
                    for column in T.Parallel(columns):
                        column_factor[column] = grad_output[
                            sequence, head, token, column
                        ]
                    for row in T.Parallel(rows):
                        second_row[row] = readout[token, head, row, 0]
                    T.sync_threads()

                    # readout: out_t = q_t . s_t
                    for row, column in T.Parallel(rows, columns):
                        adjoint[row, column] = (
                            adjoint[row, column]
                            + column_factor[column] * second_row[row]
                        )
                    for row in T.Parallel(rows):
                        local[0] = 0.0
                        for column in T.serial(columns):
                            local[0] = (
                                local[0] + column_factor[column] * state[row, column]
                            )
                        grad_readout[token, head, row, 0] = local[0]
                    T.sync_threads()

                    # additive injections have no State dependence
                    for index in T.Serial(injection_count):
                        for row in T.Parallel(rows):
                            factor_row[row] = injection_left[index, token, head, row, 0]
                        for column in T.Parallel(columns):
                            second_column[column] = injection_right[
                                index, token, head, 0, column
                            ]
                        T.sync_threads()
                        for row in T.Parallel(rows):
                            local[0] = 0.0
                            for column in T.serial(columns):
                                local[0] = (
                                    local[0]
                                    + adjoint[row, column] * second_column[column]
                                )
                            grad_injection_left[index, token, head, row, 0] = local[0]
                        for column in T.Parallel(columns):
                            local[0] = 0.0
                            for row in T.serial(rows):
                                local[0] = (
                                    local[0] + adjoint[row, column] * factor_row[row]
                                )
                            grad_injection_right[index, token, head, 0, column] = local[
                                0
                            ]
                        T.sync_threads()

                    # authored operations invert in reverse order, each replaying
                    # the earlier operations from the incoming State
                    for index in T.Serial(operation_count):
                        operation = operation_count - 1 - index
                        for row, column in T.Parallel(rows, columns):
                            state[row, column] = previous[row, column]
                        T.sync_threads()
                        for earlier in T.Serial(operation_count):
                            if earlier < operation:
                                for row in T.Parallel(rows):
                                    factor_row[row] = left[earlier, token, head, row, 0]
                                for row in T.Parallel(rows):
                                    second_row[row] = right[
                                        earlier, token, head, row, 0
                                    ]
                                if T.get_thread_binding() == 0:
                                    scalars[0] = coefficient[earlier, token, head, 0, 0]
                                T.sync_threads()
                                for row, column in T.Parallel(rows, columns):
                                    state[row, column] = (
                                        state[row, column] * scalars[0]
                                        if opcode[earlier] == 0
                                        else state[row, column] * factor_row[row]
                                        if opcode[earlier] == 1
                                        else state[row, column]
                                    )
                                T.sync_threads()
                                if opcode[earlier] == 2:
                                    for column in T.Parallel(columns):
                                        local[0] = 0.0
                                        for row in T.serial(rows):
                                            local[0] = (
                                                local[0]
                                                + second_row[row] * state[row, column]
                                            )
                                        second_column[column] = local[0]
                                    T.sync_threads()
                                    for row, column in T.Parallel(rows, columns):
                                        state[row, column] = state[row, column] + (
                                            scalars[0]
                                            * factor_row[row]
                                            * second_column[column]
                                        )
                                    T.sync_threads()
                        for row in T.Parallel(rows):
                            factor_row[row] = left[operation, token, head, row, 0]
                        if opcode[operation] == 2:
                            for row in T.Parallel(rows):
                                second_row[row] = right[operation, token, head, row, 0]
                        if T.get_thread_binding() == 0:
                            scalars[0] = coefficient[operation, token, head, 0, 0]
                        T.sync_threads()
                        if opcode[operation] == 2:
                            for column in T.Parallel(columns):
                                local[0] = 0.0
                                for row in T.serial(rows):
                                    local[0] = (
                                        local[0] + second_row[row] * state[row, column]
                                    )
                                column_factor[column] = local[0]
                            for column in T.Parallel(columns):
                                local[0] = 0.0
                                for row in T.serial(rows):
                                    local[0] = (
                                        local[0]
                                        + factor_row[row] * adjoint[row, column]
                                    )
                                second_column[column] = local[0]
                            T.sync_threads()
                            if T.get_thread_binding() == 0:
                                local[0] = 0.0
                                for column in T.serial(columns):
                                    local[0] = (
                                        local[0]
                                        + second_column[column] * column_factor[column]
                                    )
                                scalars[1] = local[0]
                            T.sync_threads()
                            grad_coefficient[operation, token, head, 0, 0] = scalars[1]
                            for row in T.Parallel(rows):
                                local[0] = 0.0
                                for column in T.serial(columns):
                                    local[0] = (
                                        local[0]
                                        + adjoint[row, column] * column_factor[column]
                                    )
                                grad_left[operation, token, head, row, 0] = (
                                    local[0] * scalars[0]
                                )
                            for row in T.Parallel(rows):
                                local[0] = 0.0
                                for column in T.serial(columns):
                                    local[0] = (
                                        local[0]
                                        + second_column[column] * state[row, column]
                                    )
                                grad_right[operation, token, head, row, 0] = (
                                    local[0] * scalars[0]
                                )
                            T.sync_threads()
                            for row, column in T.Parallel(rows, columns):
                                adjoint[row, column] = adjoint[row, column] + (
                                    scalars[0] * second_row[row] * second_column[column]
                                )
                            T.sync_threads()
                        elif opcode[operation] == 1:
                            for row in T.Parallel(rows):
                                local[0] = 0.0
                                for column in T.serial(columns):
                                    local[0] = (
                                        local[0]
                                        + adjoint[row, column] * state[row, column]
                                    )
                                grad_left[operation, token, head, row, 0] = local[0]
                            T.sync_threads()
                            for row, column in T.Parallel(rows, columns):
                                adjoint[row, column] = (
                                    adjoint[row, column] * factor_row[row]
                                )
                            T.sync_threads()
                        else:
                            for row in T.Parallel(rows):
                                local[0] = 0.0
                                for column in T.serial(columns):
                                    local[0] = (
                                        local[0]
                                        + adjoint[row, column] * state[row, column]
                                    )
                                row_scratch[row] = local[0]
                            T.sync_threads()
                            if T.get_thread_binding() == 0:
                                local[0] = 0.0
                                for row in T.serial(rows):
                                    local[0] = local[0] + row_scratch[row]
                                grad_coefficient[operation, token, head, 0, 0] = local[
                                    0
                                ]
                            T.sync_threads()
                            for row, column in T.Parallel(rows, columns):
                                adjoint[row, column] = adjoint[row, column] * scalars[0]
                            T.sync_threads()

                for row, column in T.Parallel(rows, columns):
                    grad_initial[sequence, head, row, column] = adjoint[row, column]

        return kernel

    return build_kernel()


def factored_recurrence_backward(
    *,
    operation_codes: tuple[str, ...],
    opcode: torch.Tensor,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    coefficients: torch.Tensor,
    injection_left: torch.Tensor,
    injection_right: torch.Tensor,
    readout: torch.Tensor,
    recorded: torch.Tensor,
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final: torch.Tensor,
    threads: int = 128,
) -> tuple[torch.Tensor, ...]:
    """Return gradients for the stacked factors, the readout, and the initial State."""
    sequence_count, heads, rows, columns = initial_state.shape
    token_count = readout.shape[0]
    device_index = initial_state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_factored_backward(
        tuple(operation_codes),
        injection_left.shape[0],
        token_count,
        sequence_count,
        heads,
        rows,
        columns,
        device_index,
        threads,
    )
    grad_left = torch.zeros_like(left_factors)
    grad_right = torch.zeros_like(right_factors)
    grad_coefficient = torch.zeros_like(coefficients)
    grad_injection_left = torch.zeros_like(injection_left)
    grad_injection_right = torch.zeros_like(injection_right)
    grad_readout = torch.zeros_like(readout)
    grad_initial = torch.zeros_like(initial_state)
    kernel(
        opcode,
        left_factors,
        right_factors,
        coefficients,
        injection_left,
        injection_right,
        readout,
        recorded,
        initial_state,
        grad_output.contiguous(),
        grad_final.contiguous(),
        grad_left,
        grad_right,
        grad_coefficient,
        grad_injection_left,
        grad_injection_right,
        grad_readout,
        grad_initial,
    )
    return (
        grad_left,
        grad_right,
        grad_coefficient,
        grad_injection_left,
        grad_injection_right,
        grad_readout,
        grad_initial,
    )


class _TileLangFactoredRecurrence(torch.autograd.Function):
    """Autograd transport for the factored recurrence and its reverse walk."""

    @staticmethod
    def forward(
        ctx,
        opcode,
        left,
        right,
        coefficient,
        injection_left,
        injection_right,
        readout,
        initial_state,
        output_dtype,
        backward,
        config,
    ):
        codes = _code_names(tuple(opcode.tolist()))
        output, final, recorded = _run_factored_forward(
            operation_codes=codes,
            left_factors=list(left),
            right_factors=list(right),
            coefficients=list(coefficient),
            injection_left=list(injection_left),
            injection_right=list(injection_right),
            readout=readout,
            initial_state=initial_state,
            output_dtype=output_dtype,
            backward=backward,
            config=config,
            threads=dict(config).get("threads", 128),
        )
        ctx.backward_binding = backward
        ctx.operation_codes = codes
        ctx.save_for_backward(
            opcode,
            left,
            right,
            coefficient,
            injection_left,
            injection_right,
            readout,
            recorded,
            initial_state,
            final,
        )
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (
            opcode,
            left,
            right,
            coefficient,
            injection_left,
            injection_right,
            readout,
            recorded,
            initial_state,
            final,
        ) = ctx.saved_tensors
        if ctx.backward_binding is None:
            raise RuntimeError("factored recurrence has no compiler-selected backward")
        context = {
            "factored.operation_codes": ctx.operation_codes,
            "factored.opcode": opcode,
            "factored.left": left,
            "factored.right": right,
            "factored.coefficient": coefficient,
            "factored.injection_left": injection_left,
            "factored.injection_right": injection_right,
            "factored.readout": readout,
            "factored.recorded": recorded,
            "factored.initial_state": initial_state,
            "factored.final_state": final,
        }
        gradients = ctx.backward_binding(context, (grad_output, grad_final))
        # opcode, output_dtype, backward, and config take no gradient
        return (None, *gradients, None, None, None)


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
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the authored operations as one factored recurrent kernel."""
    shape = (readout.shape[0], readout.shape[1], initial_state.shape[2], 1)
    scalar_shape = (readout.shape[0], readout.shape[1], 1, 1)
    columns_shape = (readout.shape[0], readout.shape[1], 1, initial_state.shape[3])
    opcode = torch.tensor(
        [_OPERATION_CODES[code] for code in operation_codes],
        dtype=torch.int32,
        device=initial_state.device,
    )
    return _TileLangFactoredRecurrence.apply(
        opcode,
        _stack(left_factors, shape),
        _stack(right_factors, shape),
        _stack(coefficients, scalar_shape),
        _stack(injection_left, shape),
        _stack(injection_right, columns_shape),
        readout.reshape(shape).float().contiguous(),
        initial_state.contiguous(),
        output_dtype,
        backward,
        tuple(config),
    )
