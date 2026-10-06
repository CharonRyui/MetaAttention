"""TileLang chunked lowering for raw-input scalar affine operators.

The kernel keeps the state tile resident while processing 64-token chunks.  All
products are formed forwards (rather than reconstructed from prefix ratios), so
signed and zero gates have the same semantics as the reference recurrence.
"""

from functools import lru_cache
from typing import Any

import torch


CHUNK_TOKENS = 64


def _metadata(tensor: torch.Tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype).removeprefix("torch.")


@lru_cache(maxsize=None)
def _compile_chunked(
    propagation_meta,
    left_meta,
    right_meta,
    readout_meta,
    initial_meta,
    output_meta,
    final_meta,
    *,
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    propagation_heads: int,
    left_heads: int,
    right_heads: int,
    readout_heads: int,
    identity: bool,
    propagation_operation: str,
    left_operation: str,
    right_operation: str,
    readout_operation: str,
    propagation_scale: float,
    left_scale: float,
    right_scale: float,
    readout_scale: float,
    propagation_log: bool,
    left_log: bool,
    right_log: bool,
    readout_log: bool,
    write_final: bool,
    threads: int,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    chunks = tilelang.cdiv(sequence_length, CHUNK_TOKENS)
    block_columns = threads

    def transform(value, operation, scale, is_log):
        if operation == "exp":
            value = T.exp(value)
        value = value * scale
        if is_log:
            value = T.exp(value)
        return value

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Propagation: T.StridedTensor(*propagation_meta),
            Left: T.StridedTensor(*left_meta),
            Right: T.StridedTensor(*right_meta),
            Readout: T.StridedTensor(*readout_meta),
            Initial: T.StridedTensor(*initial_meta),
            Output: T.StridedTensor(*output_meta),
            Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(
                sequence_count * heads,
                T.ceildiv(columns, block_columns),
                threads=threads,
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                left_head = head // (heads // left_heads)
                right_head = head // (heads // right_heads)
                readout_head = head // (heads // readout_heads)
                propagation_head = head // (heads // propagation_heads) if not identity else 0
                column_start = column_block * block_columns

                column = column_start + T.get_thread_binding()
                state = T.alloc_local((rows,), "float32")
                result = T.alloc_local((1,), "float32")
                if column < columns:
                    for row in T.serial(rows):
                        state[row] = T.cast(Initial[sequence, head, row, column], "float32")
                    for chunk in T.serial(chunks):
                        for token in T.serial(CHUNK_TOKENS):
                            logical_token = chunk * CHUNK_TOKENS + token
                            if logical_token < sequence_length:
                                if identity:
                                    gate = 1.0
                                else:
                                    gate = transform(
                                        T.cast(Propagation[sequence, propagation_head, logical_token], "float32"),
                                        propagation_operation, propagation_scale, propagation_log,
                                    )
                                value = transform(
                                    T.cast(Right[sequence, right_head, logical_token, column], "float32"),
                                    right_operation, right_scale, right_log,
                                )
                                result[0] = 0.0
                                for row in T.serial(rows):
                                    key = transform(
                                        T.cast(Left[sequence, left_head, logical_token, row], "float32"),
                                        left_operation, left_scale, left_log,
                                    )
                                    query = transform(
                                        T.cast(Readout[sequence, readout_head, logical_token, row], "float32"),
                                        readout_operation, readout_scale, readout_log,
                                    )
                                    state[row] = gate * state[row] + key * value
                                    result[0] += query * state[row]
                                Output[sequence, head, logical_token, column] = result[0]
                    if write_final:
                        for row in T.serial(rows):
                            Final[sequence, head, row, column] = state[row]

        return kernel

    return build_kernel()


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
    """Launch the chunked TileLang scalar path using the established raw ABI."""
    sequences, heads, rows, columns = initial.shape
    propagation_heads = heads if identity_propagation else propagation.shape[1]
    kernel = _compile_chunked(
        _metadata(propagation), _metadata(left), _metadata(right), _metadata(readout),
        _metadata(initial), _metadata(output), _metadata(final),
        sequence_count=sequences, sequence_length=sequence_length, heads=heads,
        rows=rows, columns=columns, propagation_heads=propagation_heads,
        left_heads=left.shape[1], right_heads=right.shape[1], readout_heads=readout.shape[1],
        identity=identity_propagation,
        propagation_operation=expressions[0].operation,
        left_operation=expressions[1].operation,
        right_operation=expressions[2].operation,
        readout_operation=expressions[3].operation,
        propagation_scale=expressions[0].scale,
        left_scale=expressions[1].scale,
        right_scale=expressions[2].scale,
        readout_scale=expressions[3].scale,
        propagation_log=expressions[0].is_log_scale,
        left_log=expressions[1].is_log_scale,
        right_log=expressions[2].is_log_scale,
        readout_log=expressions[3].is_log_scale,
        write_final=return_final_state, threads=threads,
        device_index=initial.device.index if initial.device.index is not None else torch.cuda.current_device(),
    )
    kernel(propagation, left, right, readout, initial, output, final)
