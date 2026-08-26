from __future__ import annotations

from functools import lru_cache

import torch


def tilelang_elementwise_prefix(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Compose affine summaries and apply sequence initial State in one compiler seam."""
    if segment.shape != (bias.shape[0],):
        raise ValueError("invalid segmented affine prefix shapes")
    if scale.shape not in (bias.shape, (*bias.shape[:2], 1, 1)):
        raise ValueError("invalid affine scale shape")
    if bias.numel() == 0:
        return bias
    return _TileLangElementwisePrefix.apply(scale, bias, segment, initial_state)


@lru_cache(maxsize=None)
def _compile_elementwise_prefix(
    token_count: int,
    heads: int,
    scale_rows: int,
    scale_columns: int,
    rows: int,
    columns: int,
    dtype: torch.dtype,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    block_tokens = 128
    matrix_lanes = 4
    matrix_size = rows * columns
    matrix_blocks = tilelang.cdiv(matrix_size, matrix_lanes)
    chunk_count = tilelang.cdiv(token_count, block_tokens)

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            scale: T.Tensor(
                (token_count, heads, scale_rows, scale_columns), dtype=dtype
            ),
            bias: T.Tensor((token_count, heads, rows, columns), dtype=dtype),
            segment: T.Tensor((token_count,), dtype="int32"),
            output_scale: T.Tensor(
                (token_count, heads, scale_rows, scale_columns), dtype=dtype
            ),
            output_bias: T.Tensor((token_count, heads, rows, columns), dtype=dtype),
        ):
            with T.Kernel(heads, matrix_blocks, threads=block_tokens) as (
                head,
                matrix_block,
            ):
                thread = T.get_thread_binding()
                local_scale = T.alloc_local((matrix_lanes,), dtype="float32")
                local_bias = T.alloc_local((matrix_lanes,), dtype="float32")
                previous_scale = T.alloc_local((matrix_lanes,), dtype="float32")
                previous_bias = T.alloc_local((matrix_lanes,), dtype="float32")
                shared_scale = T.alloc_shared(
                    (block_tokens, matrix_lanes), dtype="float32"
                )
                shared_bias = T.alloc_shared(
                    (block_tokens, matrix_lanes), dtype="float32"
                )
                shared_segment = T.alloc_shared((block_tokens,), dtype="int32")
                carry_scale = T.alloc_shared((matrix_lanes,), dtype="float32")
                carry_bias = T.alloc_shared((matrix_lanes,), dtype="float32")
                carry_segment = T.alloc_shared((1,), dtype="int32")

                if thread == 0:
                    carry_segment[0] = -1
                    for lane in T.Serial(matrix_lanes):
                        carry_scale[lane] = 1.0
                        carry_bias[lane] = 0.0
                T.sync_threads()

                for chunk in T.Serial(chunk_count):
                    token = chunk * block_tokens + thread
                    if token < token_count:
                        shared_segment[thread] = segment[token]
                    else:
                        shared_segment[thread] = -2
                    for lane in T.Serial(matrix_lanes):
                        matrix_index = matrix_block * matrix_lanes + lane
                        if token < token_count and matrix_index < matrix_size:
                            row = matrix_index // columns
                            column = matrix_index % columns
                            scale_row = 0 if scale_rows == 1 else row
                            scale_column = 0 if scale_columns == 1 else column
                            shared_scale[thread, lane] = scale[
                                token, head, scale_row, scale_column
                            ]
                            shared_bias[thread, lane] = bias[token, head, row, column]
                        else:
                            shared_scale[thread, lane] = 1.0
                            shared_bias[thread, lane] = 0.0
                    T.sync_threads()

                    for stage in T.Serial(7):
                        offset = 1 << stage
                        for lane in T.Serial(matrix_lanes):
                            local_scale[lane] = shared_scale[thread, lane]
                            local_bias[lane] = shared_bias[thread, lane]
                            if (
                                thread >= offset
                                and shared_segment[thread]
                                == shared_segment[thread - offset]
                            ):
                                previous_scale[lane] = shared_scale[
                                    thread - offset, lane
                                ]
                                previous_bias[lane] = shared_bias[thread - offset, lane]
                                local_bias[lane] = (
                                    local_scale[lane] * previous_bias[lane]
                                    + local_bias[lane]
                                )
                                local_scale[lane] = (
                                    local_scale[lane] * previous_scale[lane]
                                )
                        T.sync_threads()
                        for lane in T.Serial(matrix_lanes):
                            shared_scale[thread, lane] = local_scale[lane]
                            shared_bias[thread, lane] = local_bias[lane]
                        T.sync_threads()

                    if token < token_count:
                        for lane in T.Serial(matrix_lanes):
                            matrix_index = matrix_block * matrix_lanes + lane
                            if matrix_index < matrix_size:
                                row = matrix_index // columns
                                column = matrix_index % columns
                                local_scale[lane] = shared_scale[thread, lane]
                                local_bias[lane] = shared_bias[thread, lane]
                                if shared_segment[thread] == carry_segment[0]:
                                    local_bias[lane] = (
                                        local_scale[lane] * carry_bias[lane]
                                        + local_bias[lane]
                                    )
                                    local_scale[lane] = (
                                        local_scale[lane] * carry_scale[lane]
                                    )
                                if (
                                    scale_rows != 1
                                    or scale_columns != 1
                                    or (matrix_block == 0 and lane == 0)
                                ):
                                    output_scale[
                                        token, head, scale_row, scale_column
                                    ] = local_scale[lane]
                                output_bias[token, head, row, column] = local_bias[lane]
                    T.sync_threads()

                    if thread == 0:
                        last = T.min(
                            block_tokens - 1,
                            token_count - chunk * block_tokens - 1,
                        )
                        for lane in T.Serial(matrix_lanes):
                            matrix_index = matrix_block * matrix_lanes + lane
                            if matrix_index < matrix_size:
                                row = matrix_index // columns
                                column = matrix_index % columns
                                carry_bias[lane] = output_bias[
                                    chunk * block_tokens + last,
                                    head,
                                    row,
                                    column,
                                ]
                                if scale_rows == 1 and scale_columns == 1:
                                    carry_scale[lane] = output_scale[
                                        chunk * block_tokens + last, head, 0, 0
                                    ]
                        carry_segment[0] = shared_segment[last]
                    T.sync_threads()

        return kernel

    return build_kernel()


def _tilelang_segmented_elementwise_scan(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    *,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_scale = torch.empty_like(scale)
    output_bias = bias if inplace else torch.empty_like(bias)
    device_index = scale.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_elementwise_prefix(
        scale.shape[0],
        scale.shape[1],
        scale.shape[2],
        scale.shape[3],
        bias.shape[2],
        bias.shape[3],
        scale.dtype,
        device_index,
    )
    kernel(scale, bias, segment, output_scale, output_bias)
    return output_scale, output_bias


class _TileLangElementwisePrefix(torch.autograd.Function):
    """Native TileLang scalar-affine composition with compiler-derived VJP."""

    @staticmethod
    def forward(
        ctx,
        scale: torch.Tensor,
        bias: torch.Tensor,
        segment: torch.Tensor,
        initial_state: torch.Tensor,
    ):
        inference = not any(
            tensor.requires_grad for tensor in (scale, bias, initial_state)
        )
        output_scale, output_bias = _tilelang_segmented_elementwise_scan(
            scale,
            bias,
            segment,
            inplace=inference,
        )
        initial = initial_state[segment.long()]
        if inference:
            states = output_bias.addcmul_(output_scale, initial)
        else:
            states = output_scale * initial + output_bias
        ctx.save_for_backward(scale, segment, output_scale, output_bias, initial_state)
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor | None):
        scale, segment, output_scale, output_bias, initial_state = ctx.saved_tensors
        grad_states = (
            torch.zeros_like(output_bias) if grad_states is None else grad_states
        )
        grad_output_scale = grad_states * initial_state[segment.long()]
        continuation = torch.zeros_like(output_scale)
        continuation[:-1] = torch.where(
            (segment[:-1] == segment[1:]).view(-1, 1, 1, 1),
            scale[1:].expand_as(output_scale[1:]),
            torch.zeros_like(output_scale[1:]),
        )
        reverse_segment = segment.flip(0).contiguous()
        adjoint_scale = _tilelang_segmented_elementwise_scan(
            continuation.flip(0).contiguous(),
            grad_output_scale.flip(0).contiguous(),
            reverse_segment,
        )[1].flip(0)
        adjoint_bias = _tilelang_segmented_elementwise_scan(
            continuation.flip(0).contiguous(),
            grad_states.flip(0).contiguous(),
            reverse_segment,
        )[1].flip(0)
        previous_scale = torch.ones_like(output_scale)
        previous_bias = torch.zeros_like(output_bias)
        same_segment = (segment[1:] == segment[:-1]).view(-1, 1, 1, 1)
        previous_scale[1:] = torch.where(
            same_segment, output_scale[:-1], previous_scale[1:]
        )
        previous_bias[1:] = torch.where(
            same_segment, output_bias[:-1], previous_bias[1:]
        )
        scale_gradient = _sum_to_shape(
            adjoint_scale * previous_scale + adjoint_bias * previous_bias,
            scale.shape,
        )
        state_gradient = torch.zeros_like(initial_state)
        state_gradient.index_add_(0, segment.long(), grad_states * output_scale)
        return scale_gradient, adjoint_bias, None, state_gradient


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected == 1 and actual != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value
