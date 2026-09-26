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
from typing import Any, Mapping

import torch

# the chunk length the recurrence kernel and the published workspace bound share
CHUNK_TOKENS = 64


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


def _chunked_state(
    *,
    gate_rows: torch.Tensor,
    gate_columns: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor | None,
    query: torch.Tensor,
    initial_state: torch.Tensor,
    output_dtype: torch.dtype,
    chunk_tokens: int = CHUNK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one diagonal recurrence through the chunked schedule.

    Every per-token tensor is token-major and contiguous. ``gate_columns`` may be
    omitted for a whole-State decay or one that scales the rows alone, in which
    case the column factor is one. The readout contracts the rows and writes the
    columns, so ``query`` spans the rows. A rank-1 State is passed as one column
    per token and promotes to a single column of the rows axis, where its
    injection factor is the key and ``value`` is the constant one.
    """
    promoted = initial_state.dim() == 3
    if promoted:
        # [sequences, heads, columns] is a single column per token: the State's
        # one Feature lands on the rows, where a first-Feature readout contracts
        # it, and the injection factor becomes the key. This matches the promoted
        # matrix view the reference lowering already uses for rank-1 States.
        initial_state = initial_state.unsqueeze(-1)
    rows, columns = initial_state.shape[2], initial_state.shape[3]
    sequence_count, heads = initial_state.shape[0], initial_state.shape[1]
    token_count = value.shape[0] if value is not None else key.shape[0]
    if value is None:
        assert columns == 1, "a rank-1 State is one column per token"
        value = torch.ones(
            token_count, heads, 1, device=key.device, dtype=key.dtype
        )
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
    if promoted:
        # a rank-1 State keeps its declared rank
        final = final.squeeze(-1)
    return output, final


def chunked_elementwise_dense(
    *,
    gate_rows: tuple[torch.Tensor, ...],
    gate_columns: tuple[torch.Tensor | None, ...],
    key: tuple[torch.Tensor | None, ...],
    value: tuple[torch.Tensor, ...],
    query: tuple[torch.Tensor, ...],
    initial_state: tuple[torch.Tensor, ...],
    output_dtype: torch.dtype,
    chunk_tokens: int = CHUNK_TOKENS,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Run every diagonal recurrence of a program through the chunked schedule.

    One ``(output, final)`` pair per State, in the order the States were given.
    The States are independent here: a program whose readouts compose them must
    apply that composition to the returned outputs. Each State compiles its own
    kernels, since the sizes are part of their cache key, so a rank-1 State and a
    dense one coexist without sharing a kernel instance.
    """
    state_count = len(initial_state)
    per_state = (gate_rows, gate_columns, key, value, query)
    if any(len(entry) != state_count for entry in per_state):
        raise ValueError("every per-State argument must name the same States")
    outputs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    for index in range(state_count):
        output, final = _chunked_state(
            gate_rows=gate_rows[index],
            gate_columns=gate_columns[index],
            key=key[index],
            value=value[index],
            query=query[index],
            initial_state=initial_state[index],
            output_dtype=output_dtype,
            chunk_tokens=chunk_tokens,
        )
        outputs.append(output)
        finals.append(final)
    return tuple(outputs), tuple(finals)


def _contracted_terms(expression: Any) -> list[Any] | None:
    """Collect the State Terms a readout expression carries, or ``None``."""
    from .stateful_operator import Add, Constant, Divide, Multiply, StateTerm

    if isinstance(expression, StateTerm):
        return [expression]
    if isinstance(expression, Constant):
        return []
    if isinstance(expression, (Add, Multiply, Divide)):
        left = _contracted_terms(expression.left)
        right = _contracted_terms(expression.right)
        return None if left is None or right is None else left + right
    return None


def _compose_readout(
    expression: Any, contracted: Mapping[str, torch.Tensor]
) -> torch.Tensor | None:
    """Evaluate a readout whose State Terms the kernels already contracted.

    ``None`` means the expression carries a node this epilogue cannot evaluate
    over contracted outputs, which keeps the Program on the reference lowering.
    """
    from .stateful_operator import Add, Constant, Divide, Multiply, StateTerm

    if isinstance(expression, StateTerm):
        return contracted.get(expression.state)
    if isinstance(expression, Constant):
        device = next(iter(contracted.values())).device
        return torch.full((), expression.value, device=device, dtype=expression.dtype)
    if isinstance(expression, (Add, Multiply, Divide)):
        left = _compose_readout(expression.left, contracted)
        right = _compose_readout(expression.right, contracted)
        if left is None or right is None:
            return None
        if isinstance(expression, Add):
            return left + right
        if isinstance(expression, Multiply):
            return left * right
        return left / right
    return None


def chunked_states_eligible(algorithm: Any) -> bool:
    """Whether ``execute_chunked_states`` can express this Program's algebra.

    Static, from the IR alone. The runtime gate additionally requires a CUDA
    device, a chunk-aligned uniform length, and no autograd, so a Program that
    passes here can still fall back to the reference lowering.
    """
    from .stateful_operator import AxisScale, StateContraction, StateTerm

    from ._stateful_reference import _matrix_view

    terms_by_state: dict[str, int] = {state.name: 0 for state in algorithm.states}
    for readout in algorithm.readouts:
        expression = (
            StateTerm(readout.state, readout.operand, readout.role)
            if isinstance(readout, StateContraction)
            else readout.expression
        )
        terms = _contracted_terms(expression)
        if terms is None:
            return False
        for term in terms:
            if term.prior or term.role is None or term.state not in terms_by_state:
                return False
            terms_by_state[term.state] += 1
    # the kernels write one output per State, so each State may be read once
    if any(count != 1 for count in terms_by_state.values()):
        return False
    for state in algorithm.states:
        transition = next(
            (item for item in algorithm.transitions if item.state == state.name), None
        )
        if transition is None or len(transition.injections) != 1:
            return False
        if any(not isinstance(node, AxisScale) for node in transition.propagations):
            return False
        view = _matrix_view(state)
        first_role = view.feature_roles[0]
        for factor in transition.injections[0].factors:
            roles = tuple(factor.roles)
            if len(roles) > 1:
                return False
            if roles and roles[0] not in (first_role, view.feature_roles[-1]):
                return False
    return True


def execute_chunked_states(
    algorithm: Any,
    *,
    values: Mapping[str, torch.Tensor],
    initial_states: Mapping[str, torch.Tensor],
    segment: torch.Tensor,
    role_sizes: Mapping[Any, int],
    heads: int,
    token_count: int,
    device: torch.device,
    return_final_state: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None:
    """Run a multi-State Program through the chunked schedule, or decline.

    Declining (``None``) leaves the caller on the reference lowering. This path
    takes a Program whose States all have diagonal Transitions with one separable
    injection, and whose readout is arithmetic over State Terms that each contract
    their State's first Feature. That last condition is the kernels' orientation:
    they contract the rows and write the columns, so a term contracting the second
    Feature, or reducing the whole State (``role=None``, as HGRN's identity readout
    does), needs an output these kernels do not produce.

    The States stay in global memory only as chunk-boundary carries, which is the
    point: the reference lowering materializes one State per token per State.
    """
    if device.type != "cuda" or token_count % CHUNK_TOKENS:
        return None
    if any(
        tensor.requires_grad for tensor in (*values.values(), *initial_states.values())
    ):
        return None
    from .stateful_operator import AxisScale, StateContraction, StateTerm, _evaluate

    from ._stateful_reference import (
        _affine_summary,
        _matrix_view,
        _role_vector,
    )

    def evaluate(
        expression: Any, bound: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return _evaluate(expression, bound, feature_axes=None, segment=segment)

    if not chunked_states_eligible(algorithm):
        return None
    terms_by_state: dict[str, list[Any]] = {
        state.name: [] for state in algorithm.states
    }
    expressions: list[tuple[str, Any]] = []
    for readout in algorithm.readouts:
        expression = (
            StateTerm(readout.state, readout.operand, readout.role)
            if isinstance(readout, StateContraction)
            else readout.expression
        )
        terms = _contracted_terms(expression)
        assert terms is not None
        for term in terms:
            terms_by_state[term.state].append(term)
        expressions.append((readout.name, expression))
    gate_rows: list[torch.Tensor] = []
    gate_columns: list[torch.Tensor | None] = []
    keys: list[torch.Tensor] = []
    values_per_state: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    initials: list[torch.Tensor] = []
    for state in algorithm.states:
        view = _matrix_view(state)
        transition = next(
            item for item in algorithm.transitions if item.state == state.name
        )
        if len(transition.injections) != 1:
            return None
        if any(not isinstance(node, AxisScale) for node in transition.propagations):
            return None
        first_role = view.feature_roles[0]
        left_size = role_sizes[first_role]
        right_size = 1 if state.rank == 1 else role_sizes[view.feature_roles[1]]
        left, right, elementwise = _affine_summary(
            transition.propagations,
            view=view,
            evaluate=evaluate,
            values=values,
            token_count=token_count,
            heads=heads,
            left_size=left_size,
            right_size=right_size,
            device=device,
            dense=False,
        )
        if left is not None or right is not None or elementwise is None:
            return None
        term = terms_by_state[state.name][0]
        if term.role != first_role:
            return None
        # take the injection's own factors rather than slicing the assembled
        # matrix: slicing bias[..., 0] also carries the other factor's first
        # element, which is a per-token scalar that then rides along in the key
        key = None
        column = None
        for factor in transition.injections[0].factors:
            factor_value = evaluate(factor.expression, values).float()
            roles = tuple(factor.roles)
            if len(roles) == 1 and roles[0] == first_role:
                key = _role_vector(factor_value, token_count, heads)
            elif roles and roles[0] == view.feature_roles[-1]:
                column = _role_vector(factor_value, token_count, heads)
            elif not roles:
                # a factor without Feature roles adds the same value to every row
                column = factor_value.reshape(token_count, heads, -1)
            else:
                return None
        if key is None:
            return None
        if column is None:
            column = torch.ones(token_count, heads, 1, device=device)
        # a whole-State scale is a single factor over both axes, so only the row
        # gate carries it; the kernels expand an extent-one row gate themselves
        gate_rows.append(elementwise[:, :, :, 0])
        gate_columns.append(
            None if elementwise.shape[3] == 1 else elementwise[:, :, 0, :]
        )
        keys.append(key)
        values_per_state.append(column)
        queries.append(
            _role_vector(evaluate(term.operand, values), token_count, heads)
        )
        initials.append(initial_states[state.name])
    outputs, finals = chunked_elementwise_dense(
        gate_rows=tuple(gate_rows),
        gate_columns=tuple(gate_columns),
        key=tuple(keys),
        value=tuple(values_per_state),
        query=tuple(queries),
        initial_state=tuple(initials),
        output_dtype=algorithm.readouts[0].output_dtype,
    )
    # the kernels write [sequence, head, token, column]; the space the reference
    # lowering's caller consumes is token before head, since its readouts come back
    # from the token-major internal layout
    outputs = tuple(output.permute(0, 2, 1, 3) for output in outputs)
    contracted = {
        state.name: output for state, output in zip(algorithm.states, outputs)
    }
    result: dict[str, torch.Tensor] = {}
    for name, expression in expressions:
        composed = _compose_readout(expression, contracted)
        if composed is None:
            return None
        readout = next(item for item in algorithm.readouts if item.name == name)
        result[name] = composed.to(readout.output_dtype)
    if not return_final_state:
        return result, None
    return result, {
        state.name: final for state, final in zip(algorithm.states, finals)
    }
