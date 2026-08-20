
from dataclasses import asdict, dataclass
from functools import lru_cache
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

import torch


@dataclass(frozen=True)
class TileLangPlan:
    """Portable identity and environment descriptor for one specialization."""

    semantic_identity: str
    binary_identity: str
    tuning_identity: str
    mode: str
    feature_sizes: tuple[int, int]
    uses_parallel_summary: bool = True

    @property
    def backend(self) -> str:
        return "tilelang"


@dataclass(frozen=True)
class TileLangExecutable:
    """Compiler-owned forward/backward dispatch for one analyzed specialization."""

    plan: TileLangPlan
    backward_ir: Any
    dense_summary: bool
    scalar_factorized: Any | None

    def prefix(
        self,
        transform: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
        segment: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.dense_summary:
            assert isinstance(transform, tuple)
            return tilelang_dense_affine_prefix(
                transform[0], transform[1], bias, segment, initial_state
            )
        assert isinstance(transform, torch.Tensor)
        return tilelang_elementwise_prefix(transform, bias, segment, initial_state)
    def scalar_factorized_dense(
        self,
        scale: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        readout: torch.Tensor,
        initial_state: torch.Tensor,
        *,
        sequence_count: int,
        sequence_length: int,
        output_dtype: torch.dtype,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_dense(
            scale,
            left,
            right,
            readout,
            initial_state,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
            output_dtype=output_dtype,
            return_final_state=return_final_state,
        )


def compile_plan(
    *,
    source_identity: str,
    mode: str,
    feature_sizes: tuple[int, int],
    device: torch.device,
    backward_ir: Any,
    dense_summary: bool,
    scalar_factorized: Any | None,
) -> TileLangExecutable:
    """Compile an executable and atomically publish its portable descriptor."""
    toolchain = _toolchain_fingerprint()
    environment = _environment_fingerprint(device)
    binary_identity = _digest((source_identity, mode, feature_sizes, toolchain, environment))
    tuning_identity = _digest((binary_identity, environment, _clock_fingerprint(device)))
    plan = TileLangPlan(
        source_identity,
        binary_identity,
        tuning_identity,
        mode,
        feature_sizes,
    )
    if device.type == "cuda":
        try:
            import tilelang  # noqa: F401
        except ImportError as error:
            raise RuntimeError("TileLang is required for CUDA Stateful Operator plans") from error
        _publish_manifest(plan)
    return TileLangExecutable(plan, backward_ir, dense_summary, scalar_factorized)

def _toolchain_fingerprint() -> tuple[str, ...]:
    try:
        tilelang_version = version("tilelang")
    except PackageNotFoundError:
        tilelang_version = "unavailable"
    return (
        platform.python_implementation(),
        platform.python_version(),
        torch.__version__,
        str(torch.version.cuda),
        tilelang_version,
    )


def _environment_fingerprint(device: torch.device) -> tuple[Any, ...]:
    if device.type != "cuda":
        return (device.type,)
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return (
        properties.name,
        tuple(torch.cuda.get_device_capability(index)),
        properties.total_memory,
        properties.multi_processor_count,
        _nvidia_smi_fingerprint(index, "driver_version"),
        _toolchain_fingerprint(),
    )


def _clock_fingerprint(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _nvidia_smi_fingerprint(
        index,
        "clocks.applications.graphics,clocks.applications.memory,persistence_mode,power.limit",
    )


def _nvidia_smi_fingerprint(index: int, fields: str) -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to fingerprint the CUDA driver and clock policy") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _publish_manifest(plan: TileLangPlan) -> None:
    root = Path(
        os.environ.get(
            "STATEFUL_OPERATOR_CACHE_DIR",
            Path.home() / ".cache" / "meta-attention" / "stateful-operator",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{plan.binary_identity}.json"
    lock_path = root / f"{plan.binary_identity}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".plan-", dir=root)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(asdict(plan), stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)



@lru_cache(maxsize=None)
def _compile_scalar_factorized_dense_forward(
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    return_final_state: bool,
):
    import tilelang
    import tilelang.language as T

    chunk_tokens = 32
    block_columns = 32
    chunk_count = tilelang.cdiv(sequence_length, chunk_tokens)

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build_kernel():
        @T.prim_func
        def kernel(
            scale: T.Tensor((heads, sequence_count * sequence_length), dtype="float32"),
            left: T.Tensor(
                (heads, sequence_count * sequence_length, rows), dtype=input_dtype
            ),
            right: T.Tensor(
                (heads, sequence_count * sequence_length, columns), dtype=input_dtype
            ),
            readout: T.Tensor(
                (heads, sequence_count * sequence_length, rows), dtype=input_dtype
            ),
            initial: T.Tensor(
                (sequence_count, heads, rows, columns), dtype="float32"
            ),
            output: T.Tensor(
                (sequence_count, heads, sequence_length, columns), dtype=output_dtype
            ),
            final: T.Tensor(
                (sequence_count, heads, rows, columns), dtype="float32"
            ),
        ):
            with T.Kernel(
                sequence_count * heads,
                T.ceildiv(columns, block_columns),
                threads=128,
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                column_start = column_block * block_columns
                state = T.alloc_shared((rows, block_columns), dtype="float32")
                state_cast = T.alloc_shared(
                    (rows, block_columns), dtype=input_dtype
                )
                query_shared = T.alloc_shared(
                    (chunk_tokens, rows), dtype=input_dtype
                )
                key_shared = T.alloc_shared(
                    (chunk_tokens, rows), dtype=input_dtype
                )
                value_shared = T.alloc_shared(
                    (chunk_tokens, block_columns), dtype=input_dtype
                )
                weighted_key_shared = T.alloc_shared(
                    (chunk_tokens, rows), dtype=input_dtype
                )
                scores_shared = T.alloc_shared(
                    (chunk_tokens, chunk_tokens), dtype=input_dtype
                )
                log_prefix = T.alloc_shared((chunk_tokens,), dtype="float32")
                scores = T.alloc_fragment(
                    (chunk_tokens, chunk_tokens), dtype="float32"
                )
                output_fragment = T.alloc_fragment(
                    (chunk_tokens, block_columns), dtype="float32"
                )
                update_fragment = T.alloc_fragment(
                    (rows, block_columns), dtype="float32"
                )

                for row, column in T.Parallel(rows, block_columns):
                    state[row, column] = initial[
                        sequence, head, row, column_start + column
                    ]
                T.sync_threads()

                for chunk in T.Serial(chunk_count):
                    chunk_start = chunk * chunk_tokens
                    for token, row in T.Parallel(chunk_tokens, rows):
                        logical_token = chunk_start + token
                        physical_token = sequence * sequence_length + logical_token
                        if logical_token < sequence_length:
                            query_shared[token, row] = readout[
                                head, physical_token, row
                            ]
                            key_shared[token, row] = left[
                                head, physical_token, row
                            ]
                        else:
                            query_shared[token, row] = 0.0
                            key_shared[token, row] = 0.0
                    for token, column in T.Parallel(
                        chunk_tokens, block_columns
                    ):
                        logical_token = chunk_start + token
                        physical_token = sequence * sequence_length + logical_token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            value_shared[token, column] = right[
                                head, physical_token, column_start + column
                            ]
                        else:
                            value_shared[token, column] = 0.0
                    if T.get_thread_binding() == 0:
                        running_log = T.alloc_local((1,), dtype="float32")
                        running_log[0] = 0.0
                        for token in T.Serial(chunk_tokens):
                            logical_token = chunk_start + token
                            if logical_token < sequence_length:
                                physical_token = (
                                    sequence * sequence_length + logical_token
                                )
                                running_log[0] += T.log(
                                    scale[head, physical_token]
                                )
                            log_prefix[token] = running_log[0]
                    for row, column in T.Parallel(rows, block_columns):
                        state_cast[row, column] = state[row, column]
                    T.sync_threads()

                    T.clear(output_fragment)
                    T.gemm(
                        query_shared,
                        state_cast,
                        output_fragment,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, column in T.Parallel(
                        chunk_tokens, block_columns
                    ):
                        output_fragment[token, column] *= T.exp(
                            log_prefix[token]
                        )

                    T.clear(scores)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, source in T.Parallel(
                        chunk_tokens, chunk_tokens
                    ):
                        scores[token, source] = T.if_then_else(
                            source <= token,
                            scores[token, source]
                            * T.exp(log_prefix[token] - log_prefix[source]),
                            0.0,
                        )
                    T.copy(scores, scores_shared)
                    T.gemm(
                        scores_shared,
                        value_shared,
                        output_fragment,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, column in T.Parallel(
                        chunk_tokens, block_columns
                    ):
                        logical_token = chunk_start + token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            output[
                                sequence,
                                head,
                                logical_token,
                                column_start + column,
                            ] = output_fragment[token, column]

                    end_log = log_prefix[
                        T.min(chunk_tokens - 1, sequence_length - chunk_start - 1)
                    ]
                    for token, row in T.Parallel(chunk_tokens, rows):
                        weighted_key_shared[token, row] = (
                            key_shared[token, row]
                            * T.exp(end_log - log_prefix[token])
                        )
                    T.clear(update_fragment)
                    T.gemm(
                        weighted_key_shared,
                        value_shared,
                        update_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for row, column in T.Parallel(rows, block_columns):
                        state[row, column] = (
                            T.exp(end_log) * state[row, column]
                            + update_fragment[row, column]
                        )
                    T.sync_threads()
                if return_final_state:
                    for row, column in T.Parallel(rows, block_columns):
                        if column_start + column < columns:
                            final[
                                sequence, head, row, column_start + column
                            ] = state[row, column]

        return kernel

    return build_kernel()


def tilelang_scalar_factorized_dense(
    scale: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    sequence_count: int,
    sequence_length: int,
    output_dtype: torch.dtype,
    return_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    heads, rows, columns = initial_state.shape[1:]
    output = torch.empty(
        sequence_count,
        heads,
        sequence_length,
        columns,
        device=left.device,
        dtype=output_dtype,
    )
    final = torch.empty_like(initial_state) if return_final_state else initial_state
    device_index = left.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_scalar_factorized_dense_forward(
        sequence_count,
        sequence_length,
        heads,
        rows,
        columns,
        left.dtype,
        output_dtype,
        device_index,
        return_final_state,
    )
    kernel(
        scale.reshape(sequence_count * sequence_length, heads)
        .transpose(0, 1)
        .contiguous(),
        left.reshape(sequence_count * sequence_length, heads, rows)
        .transpose(0, 1)
        .contiguous(),
        right.reshape(sequence_count * sequence_length, heads, columns)
        .transpose(0, 1)
        .contiguous(),
        readout.reshape(sequence_count * sequence_length, heads, rows)
        .transpose(0, 1)
        .contiguous(),
        initial_state,
        output,
        final,
    )
    return output, final if return_final_state else None


def is_parallel_plan(plan: Any) -> bool:
    return isinstance(plan, TileLangExecutable) and plan.plan.uses_parallel_summary

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
            output_bias: T.Tensor(
                (token_count, heads, rows, columns), dtype=dtype
            ),
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
                                previous_bias[lane] = shared_bias[
                                    thread - offset, lane
                                ]
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
                                if scale_rows != 1 or scale_columns != 1 or (
                                    matrix_block == 0 and lane == 0
                                ):
                                    output_scale[
                                        token, head, scale_row, scale_column
                                    ] = local_scale[lane]
                                output_bias[token, head, row, column] = local_bias[
                                    lane
                                ]
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
        ctx.save_for_backward(
            scale, segment, output_scale, output_bias, initial_state
        )
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
        state_gradient.index_add_(
            0, segment.long(), grad_states * output_scale
        )
        return scale_gradient, adjoint_bias, None, state_gradient


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected == 1 and actual != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value




@lru_cache(maxsize=None)
def _compile_dense_matmul_step(
    token_count: int,
    heads: int,
    rows: int,
    inner: int,
    columns: int,
    validity_offset: int,
    operand_offset: int,
    left_from_previous: bool,
    passthrough: int,
    addend: bool,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    block_rows = 32
    block_columns = 32
    block_inner = 32

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            left: T.Tensor((token_count, heads, rows, inner), dtype="float32"),
            right: T.Tensor(
                (token_count, heads, inner, columns), dtype="float32"
            ),
            segment: T.Tensor((token_count,), dtype="int32"),
            extra: T.Tensor(
                (token_count, heads, rows, columns), dtype="float32"
            ),
            output: T.Tensor(
                (token_count, heads, rows, columns), dtype="float32"
            ),
        ):
            with T.Kernel(
                token_count * heads,
                T.ceildiv(rows, block_rows),
                T.ceildiv(columns, block_columns),
                threads=128,
            ) as (token_head, row_block, column_block):
                token = token_head // heads
                head = token_head % heads
                row_start = row_block * block_rows
                column_start = column_block * block_columns
                valid = (token >= validity_offset) & (
                    segment[token] == segment[token - validity_offset]
                )
                accumulator = T.alloc_fragment(
                    (block_rows, block_columns), dtype="float32"
                )
                left_shared = T.alloc_shared(
                    (block_rows, block_inner), dtype="float32"
                )
                right_shared = T.alloc_shared(
                    (block_inner, block_columns), dtype="float32"
                )
                T.clear(accumulator)
                if valid:
                    left_token = (
                        token - operand_offset if left_from_previous else token
                    )
                    right_token = (
                        token if left_from_previous else token - operand_offset
                    )
                    for inner_block in T.Pipelined(
                        T.ceildiv(inner, block_inner), num_stages=1
                    ):
                        inner_start = inner_block * block_inner
                        T.copy(
                            left[
                                left_token,
                                head,
                                row_start : row_start + block_rows,
                                inner_start : inner_start + block_inner,
                            ],
                            left_shared,
                        )
                        T.copy(
                            right[
                                right_token,
                                head,
                                inner_start : inner_start + block_inner,
                                column_start : column_start + block_columns,
                            ],
                            right_shared,
                        )
                        T.gemm(left_shared, right_shared, accumulator)
                    if addend:
                        for row, column in T.Parallel(
                            block_rows, block_columns
                        ):
                            accumulator[row, column] += extra[
                                token,
                                head,
                                row_start + row,
                                column_start + column,
                            ]
                    T.copy(
                        accumulator,
                        output[
                            token,
                            head,
                            row_start : row_start + block_rows,
                            column_start : column_start + block_columns,
                        ],
                    )
                else:
                    if passthrough == 1:
                        T.copy(
                            left[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )
                    elif passthrough == 2:
                        T.copy(
                            right[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )
                    else:
                        T.copy(
                            extra[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )

        return kernel

    return build_kernel()


def _tilelang_dense_affine_scan(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count, heads, rows, _ = left.shape
    columns = right.shape[-1]
    device_index = left.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    offset = 1
    while offset < token_count:
        next_left = torch.empty_like(left)
        next_right = torch.empty_like(right)
        temporary = torch.empty_like(bias)
        next_bias = torch.empty_like(bias)
        empty_bias = torch.empty_like(bias)
        empty_left = torch.empty_like(left)
        empty_right = torch.empty_like(right)
        _compile_dense_matmul_step(
            token_count, heads, rows, rows, rows, offset, offset, False, 1, False, device_index
        )(left, left, segment, empty_left, next_left)
        _compile_dense_matmul_step(
            token_count, heads, columns, columns, columns, offset, offset, True, 2, False, device_index
        )(right, right, segment, empty_right, next_right)
        _compile_dense_matmul_step(
            token_count, heads, rows, rows, columns, offset, offset, False, 0, False, device_index
        )(left, bias, segment, empty_bias, temporary)
        _compile_dense_matmul_step(
            token_count, heads, rows, columns, columns, offset, 0, False, 0, True, device_index
        )(temporary, right, segment, bias, next_bias)
        left, right, bias = next_left, next_right, next_bias
        offset *= 2
    return left, right, bias


def tilelang_dense_affine_prefix(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Compose dense affine summaries with an explicit parallel reverse VJP."""
    if (
        left.ndim != 4
        or right.ndim != 4
        or bias.ndim != 4
        or left.shape[:2] != bias.shape[:2]
        or right.shape[:2] != bias.shape[:2]
        or segment.shape != (bias.shape[0],)
    ):
        raise ValueError("invalid dense affine prefix shapes")
    return _TileLangDenseAffinePrefix.apply(
        left, right, bias, segment, initial_state
    )


class _TileLangDenseAffinePrefix(torch.autograd.Function):
    """Explicit recurrence VJP derived from dense affine Backward IR."""

    @staticmethod
    def forward(
        ctx,
        left: torch.Tensor,
        right: torch.Tensor,
        bias: torch.Tensor,
        segment: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        prefix_left, prefix_right, prefix_bias = _tilelang_dense_affine_scan(
            left, right, bias, segment
        )
        states = (
            prefix_left
            @ initial_state[segment.long()]
            @ prefix_right
            + prefix_bias
        )
        ctx.save_for_backward(left, right, segment, initial_state, states)
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor | None):
        left, right, segment, initial_state, states = ctx.saved_tensors
        grad_states = (
            torch.zeros_like(states) if grad_states is None else grad_states
        )
        same_next = (segment[:-1] == segment[1:]).view(-1, 1, 1, 1)
        continuation_left = torch.zeros_like(left)
        continuation_right = torch.zeros_like(right)
        continuation_left[:-1] = torch.where(
            same_next,
            left[1:].transpose(-1, -2),
            continuation_left[:-1],
        )
        continuation_right[:-1] = torch.where(
            same_next,
            right[1:].transpose(-1, -2),
            continuation_right[:-1],
        )
        reverse_segment = segment.flip(0).contiguous()
        _, _, adjoint = _tilelang_dense_affine_scan(
            continuation_left.flip(0).contiguous(),
            continuation_right.flip(0).contiguous(),
            grad_states.flip(0).contiguous(),
            reverse_segment,
        )
        adjoint = adjoint.flip(0)

        previous_state = initial_state[segment.long()].clone()
        previous_state[1:] = torch.where(
            same_next,
            states[:-1],
            previous_state[1:],
        )
        grad_left = (
            adjoint
            @ right.transpose(-1, -2)
            @ previous_state.transpose(-1, -2)
        )
        grad_right = (
            previous_state.transpose(-1, -2)
            @ left.transpose(-1, -2)
            @ adjoint
        )
        grad_initial = torch.zeros_like(initial_state)
        sequence_starts = torch.ones_like(segment, dtype=torch.bool)
        sequence_starts[1:] = segment[1:] != segment[:-1]
        start_indices = sequence_starts.nonzero().flatten()
        start_segments = segment[start_indices].long()
        initial_adjoint = (
            left[start_indices].transpose(-1, -2)
            @ adjoint[start_indices]
            @ right[start_indices].transpose(-1, -2)
        )
        grad_initial.index_add_(0, start_segments, initial_adjoint)
        return grad_left, grad_right, adjoint, None, grad_initial







