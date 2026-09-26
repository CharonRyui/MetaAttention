from __future__ import annotations

from functools import lru_cache

import torch


def tilelang_elementwise_prefix(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    initial_state: torch.Tensor,
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
) -> torch.Tensor:
    """Compose affine summaries through a compiler-selected VJP binding."""
    if segment.shape != (bias.shape[0],):
        raise ValueError("invalid segmented affine prefix shapes")
    if scale.ndim != 4 or scale.shape[:2] != bias.shape[:2]:
        raise ValueError("invalid affine scale shape")
    rows, columns = bias.shape[2], bias.shape[3]
    if scale.shape[2:] not in ((1, 1), (rows, 1), (1, columns), (rows, columns)):
        # a per-axis factor, a per-token scalar, or a full elementwise map
        raise ValueError(f"invalid affine scale shape {tuple(scale.shape)}")
    if bias.numel() == 0:
        return bias
    if isinstance(config, int):
        config = (("block_tokens", config),)
    if backward is None:
        from ._stateful_backward import default_backward
        backward = default_backward("elementwise", config)
    return _TileLangElementwisePrefix.apply(
        scale, bias, segment, initial_state, backward, tuple(config)
    )




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
    block_tokens: int = 128,
):
    import tilelang
    import tilelang.language as T

    block_tokens = int(block_tokens)
    if block_tokens <= 0 or block_tokens & (block_tokens - 1):
        raise ValueError("block_tokens must be a positive power of two")
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
            if False:
                # tilelang resolves annotations through the kernel's free
                # variables, so names used only in a signature must be referenced.
                _ = (dtype, rows)
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

                    for stage in T.Serial(block_tokens.bit_length() - 1):
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
                                write_scale_row = 0 if scale_rows == 1 else row
                                write_scale_column = 0 if scale_columns == 1 else column
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
                                        token, head, write_scale_row, write_scale_column
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
                                # The carry must hold the previous chunk's prefix
                                # factor for this matrix element: a per-axis factor
                                # keeps one entry per row or per column instead of
                                # one entry for the whole State.
                                carry_scale[lane] = output_scale[
                                    chunk * block_tokens + last,
                                    head,
                                    0 if scale_rows == 1 else row,
                                    0 if scale_columns == 1 else column,
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
    block_tokens: int = 128,
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
        block_tokens,
    )
    kernel(scale, bias, segment, output_scale, output_bias)
    return output_scale, output_bias


class _TileLangElementwisePrefix(torch.autograd.Function):
    """Autograd transport for a compiler-selected elementwise VJP binding."""

    @staticmethod
    def forward(ctx, scale, bias, segment, initial_state, backward, config):
        config = tuple(config)
        block_tokens = dict(config).get("block_tokens", 128)
        inference = not any(tensor.requires_grad for tensor in (scale, bias, initial_state))
        output_scale, output_bias = _tilelang_segmented_elementwise_scan(
            scale, bias, segment, inplace=inference, block_tokens=block_tokens
        )
        initial = initial_state[segment.long()]
        states = output_bias.addcmul_(output_scale, initial) if inference else output_scale * initial + output_bias
        ctx.backward_binding = backward
        ctx.save_for_backward(scale, segment, output_scale, output_bias, initial_state)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        scale, segment, output_scale, output_bias, initial_state = ctx.saved_tensors
        context = {
            "transition.scale": scale,
            "transition.segment": segment,
            "transition.output_scale": output_scale,
            "transition.output_bias": output_bias,
            "transition.initial_state": initial_state,
        }
        return (*ctx.backward_binding(context, grad_states), None, None)



def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected == 1 and actual != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value
