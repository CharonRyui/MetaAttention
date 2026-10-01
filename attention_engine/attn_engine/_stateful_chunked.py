"""Dense diagonal-State lowering with FP32 chunk carries and typed readouts.

Identity recurrence uses chunk GEMMs, synthesizing Product Features in tiles.
Other diagonal transitions keep State in distributed registers for one chunk;
dependent readouts consume contracted vectors, never per-token matrix States.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

# the chunk length the recurrence kernel and the published workspace bound share
CHUNK_TOKENS = 64


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
    log_gate_rows: bool = False,
    unit_gate_columns: bool = False,
    scalar_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a diagonal recurrence through distributed chunk replay."""
    from ._stateful_diagonal_kernels import prepare_state, read_state

    if chunk_tokens != CHUNK_TOKENS:
        raise ValueError("diagonal recurrence uses 64-token chunks")
    promoted = initial_state.dim() == 3
    initial = initial_state.unsqueeze(-1) if promoted else initial_state
    batch, heads, rows, columns = initial.shape
    tokens = key.shape[0]
    length = tokens // batch
    if tokens != batch * length:
        raise ValueError("dense tokens must divide evenly between sequences")
    if value is None:
        value = torch.ones(
            tokens, heads, columns, device=key.device, dtype=torch.float32
        )
    rows_gate = gate_rows.reshape(tokens, heads, -1).float()
    if log_gate_rows:
        rows_gate = rows_gate.exp()
    rows_gate = rows_gate.expand(tokens, heads, rows)
    columns_gate = None if unit_gate_columns else gate_columns
    carry, final, factors = prepare_state(
        rows_gate=rows_gate,
        columns_gate=columns_gate,
        key=key,
        value=value,
        initial=initial,
        sequence_length=length,
        return_final_state=True,
    )
    output = read_state(carry, factors, query, sequence_length=length)
    output = (
        output.reshape(batch, length, heads, columns)
        .permute(0, 2, 1, 3)
        .to(output_dtype)
    )
    return output, final.squeeze(-1) if promoted else final


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
    log_gate_rows: tuple[bool, ...] = (),
    unit_gate_columns: tuple[bool, ...] = (),
    scalar_gate: tuple[bool, ...] = (),
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Run every diagonal recurrence through the chunked schedule."""
    state_count = len(initial_state)
    per_state = (gate_rows, gate_columns, key, value, query)
    if any(len(entry) != state_count for entry in per_state):
        raise ValueError("every per-State argument must name the same States")
    if log_gate_rows and len(log_gate_rows) != state_count:
        raise ValueError("log_gate_rows must name every State when provided")
    if unit_gate_columns and len(unit_gate_columns) != state_count:
        raise ValueError("unit_gate_columns must name every State when provided")
    if scalar_gate and len(scalar_gate) != state_count:
        raise ValueError("scalar_gate must name every State when provided")
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
            log_gate_rows=bool(log_gate_rows and log_gate_rows[index]),
            unit_gate_columns=bool(unit_gate_columns and unit_gate_columns[index]),
            scalar_gate=bool(scalar_gate and scalar_gate[index]),
        )
        outputs.append(output)
        finals.append(final)
    return tuple(outputs), tuple(finals)


def _contracted_terms(expression: Any) -> list[Any] | None:
    """Collect nested State Terms; reject unimplemented expression nodes."""
    from .stateful_operator import (
        Add,
        Clamp,
        Constant,
        Contract,
        Divide,
        Exp,
        Input,
        Log,
        Multiply,
        Negate,
        Outer,
        Reciprocal,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
    )

    if isinstance(expression, StateTerm):
        dependencies = _contracted_terms(expression.operand)
        return None if dependencies is None else dependencies + [expression]
    if isinstance(expression, (Input, Constant)):
        return []
    if isinstance(expression, (Add, Multiply, Divide, Outer, Contract)):
        left = _contracted_terms(expression.left)
        right = _contracted_terms(expression.right)
        return None if left is None or right is None else left + right
    if isinstance(
        expression, (Negate, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp, Softmax)
    ):
        return _contracted_terms(expression.operand)
    return None


def _axis_factors(
    propagations: Any,
    *,
    view: Any,
    evaluate: Any,
    values: Mapping[str, torch.Tensor],
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """A diagonal Transition's per-axis decay factors, kept separate.

    ``_affine_summary`` returns their outer product, which costs
    ``token_count * heads * rows * columns`` elements -- 268 MB for a 64x64 State
    at T=2048, and 1 GiB for the 256x64 one `rebased` carries. The chunked kernels
    take one gate per axis, so that product is pure overhead here. ``None`` means a
    propagation is not a diagonal scale, which sends the caller to the reference.
    """
    from .stateful_operator import AxisScale, Identity

    from ._stateful_reference import _role_vector

    def per_token(expression: Any, size: int | None) -> torch.Tensor | None:
        factor = evaluate(expression, values).float()
        if factor.numel() == 1:
            return factor.reshape(1, 1, 1).expand(token_count, heads, size or 1)
        if size is None:
            broadcast = factor.reshape(token_count, heads, -1)
            return broadcast if broadcast.shape[-1] == 1 else None
        return _role_vector(factor, token_count, heads).reshape(
            token_count, heads, size
        )

    rows = torch.ones(token_count, heads, left_size, device=device)
    columns = (
        None
        if right_size == 1
        else torch.ones(token_count, heads, right_size, device=device)
    )
    for node in propagations:
        if isinstance(node, Identity):
            continue
        if not isinstance(node, AxisScale):
            return None
        roles = tuple(node.roles)
        whole = len(roles) == 2 or not roles
        if whole:
            scale = per_token(node.factor, None)
            if scale is None:
                return None
            rows = rows * scale
            continue
        if roles[0] == view.feature_roles[0]:
            row_factor = per_token(node.factor, left_size)
            if row_factor is None:
                return None
            rows = rows * row_factor
        elif roles[0] == view.feature_roles[-1]:
            if columns is None:
                continue
            column_factor = per_token(node.factor, right_size)
            if column_factor is None:
                return None
            columns = columns * column_factor
        else:
            return None
    return rows, columns


def _readout_expression(readout: Any) -> Any:
    from .stateful_operator import StateContraction, StateTerm

    return (
        StateTerm(readout.state, readout.operand, readout.role)
        if isinstance(readout, StateContraction)
        else readout.expression
    )


def _expression_features(expression: Any, algorithm: Any) -> tuple[Any, ...] | None:
    """Prove the squeezed value carries at most one Feature axis."""
    from .stateful_operator import (
        Constant,
        Contract,
        FeatureRole,
        Input,
        Outer,
        StateTerm,
    )

    if isinstance(expression, Constant):
        return ()
    if isinstance(expression, Input):
        spec = next(
            (item for item in algorithm.inputs if item.name == expression.name), None
        )
        if spec is None:
            return None
        roles = tuple(role for role in spec.roles if isinstance(role, FeatureRole))
        return roles if len(roles) <= 1 else None
    if isinstance(expression, StateTerm):
        spec = next(
            (item for item in algorithm.states if item.name == expression.state), None
        )
        if spec is None:
            return None
        operand = _expression_features(expression.operand, algorithm)
        if expression.role is None:
            return spec.feature_roles if operand == () and spec.rank == 1 else None
        if operand != (expression.role,):
            return None
        return tuple(role for role in spec.feature_roles if role != expression.role)
    if hasattr(expression, "left"):
        left = _expression_features(expression.left, algorithm)
        right = _expression_features(expression.right, algorithm)
        if left is None or right is None:
            return None
        if isinstance(expression, Outer):
            return (
                (expression.role,)
                if left == (expression.role.left,) and right == (expression.role.right,)
                else None
            )
        if isinstance(expression, Contract):
            return (
                ()
                if left in ((), (expression.role,))
                and right in ((), (expression.role,))
                else None
            )
        # The reference readout value space also aligns its single surviving
        # Feature positionally (including an extent-one authored unit Feature).
        return left or right
    if hasattr(expression, "operand"):
        return _expression_features(expression.operand, algorithm)
    return None


def chunked_states_eligible(algorithm: Any, *, packed: bool = False) -> bool:
    """Static proof for dense diagonal recurrence with dependent readout passes."""
    from .stateful_operator import AxisScale, Identity

    if packed:
        return False
    specs = {state.name: state for state in algorithm.states}
    modes: dict[str, set[bool]] = {name: set() for name in specs}
    for readout in algorithm.readouts:
        terms = _contracted_terms(_readout_expression(readout))
        if terms is None:
            return False
        if _expression_features(_readout_expression(readout), algorithm) is None:
            return False
        for term in terms:
            state = specs.get(term.state)
            if state is None:
                return False
            if term.role is None:
                if state.rank != 1:
                    return False
            elif term.role != state.feature_roles[0]:
                return False
            modes[state.name].add(term.role is None)
    if any(len(mode) > 1 for mode in modes.values()):
        return False
    for state in algorithm.states:
        transition = next(
            (item for item in algorithm.transitions if item.state == state.name), None
        )
        if transition is None or len(transition.injections) != 1:
            return False
        for node in transition.propagations:
            if not isinstance(node, (AxisScale, Identity)):
                return False
            if isinstance(node, AxisScale):
                if _contracted_terms(node.factor) != []:
                    return False
                features = _expression_features(node.factor, algorithm)
                if features is None or (features and features != tuple(node.roles)):
                    return False
                if any(role not in state.feature_roles for role in node.roles):
                    return False
                if len(node.roles) > 1 and features:
                    return False
        seen: set[Any] = set()
        scalar_count = 0
        for factor in transition.injections[0].factors:
            roles = tuple(factor.roles)
            scalar_count += not roles
            if scalar_count > 1:
                return False
            if len(roles) > 1 or any(
                role not in state.feature_roles or role in seen for role in roles
            ):
                return False
            if _contracted_terms(factor.expression) != []:
                return False
            features = _expression_features(factor.expression, algorithm)
            if features is None or (features and features != roles):
                return False
            seen.update(roles)
        if seen != set(state.feature_roles):
            return False
    return True


def chunked_states_workspace(
    algorithm: Any, role_sizes: Mapping[Any, int], heads: int
) -> tuple[int, int]:
    """Conservative ``(bytes/token, bytes/chunk/sequence)`` scratch bound.

    Charge B*T times the first and B*(ceil(T/64)+1) times the second, including
    final States. Covers factor evaluation, contracted FP32 outputs and authored
    arithmetic. Identity Product Features are synthesized inside GEMM tiles.
    """
    from .stateful_operator import Identity, ProductFeatureRole

    per_token = 0
    per_chunk = 0
    for state in algorithm.states:
        dims = [role_sizes[role] for role in state.feature_roles]
        elements = dims[0] * (dims[1] if len(dims) == 2 else 1)
        transition = next(
            item for item in algorithm.transitions if item.state == state.name
        )
        identity = all(isinstance(node, Identity) for node in transition.propagations)
        per_chunk += 4 * heads * elements * (2 if identity else 3)
        factor_dims = [
            role_sizes[role.left] + role_sizes[role.right]
            if identity and isinstance(role, ProductFeatureRole)
            else role_sizes[role]
            for role in state.feature_roles
        ]
        per_token += 32 * heads * sum(factor_dims)
    for readout in algorithm.readouts:
        for term in _contracted_terms(_readout_expression(readout)) or ():
            state = next(
                state for state in algorithm.states if state.name == term.state
            )
            output = (
                role_sizes[state.feature_roles[-1]]
                if state.rank == 2 or term.role is None
                else 1
            )
            per_token += 64 * heads * output
    return per_token, per_chunk


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
    sequence_length: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None:
    """Dense inference with boundary carries, GEMMs and dependent readout passes."""
    if device.type != "cuda" or sequence_length is None or sequence_length <= 0:
        return None
    if any(
        tensor.requires_grad for tensor in (*values.values(), *initial_states.values())
    ):
        return None
    if not chunked_states_eligible(algorithm):
        return None
    from .stateful_operator import (
        Add,
        Clamp,
        Contract,
        Divide,
        Exp,
        Identity,
        Log,
        Multiply,
        Negate,
        Outer,
        Reciprocal,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
        _evaluate,
    )
    from ._stateful_reference import _matrix_view
    from ._stateful_diagonal_kernels import prepare_state, read_state

    specs = {state.name: state for state in algorithm.states}
    transitions = {transition.state: transition for transition in algorithm.transitions}
    identity_read = {
        term.state
        for readout in algorithm.readouts
        for term in (_contracted_terms(_readout_expression(readout)) or ())
        if term.role is None
    }
    prepared: dict[str, Any] = {}
    final_states: dict[str, torch.Tensor] = {}
    term_values: dict[Any, torch.Tensor] = {}
    input_values: dict[Any, torch.Tensor] = {}

    def vector(value: torch.Tensor, size: int) -> torch.Tensor:
        if value.numel() == 1:
            return value.reshape(1, 1, 1).expand(token_count, heads, size)
        return value.reshape(token_count, heads, -1).expand(token_count, heads, size)

    def input_expression(expression: Any) -> torch.Tensor:
        if expression not in input_values:
            result = _evaluate(
                expression, values, segment=segment, squeeze_features=True
            )
            input_values[expression] = (
                result.unsqueeze(-1) if result.ndim == 2 else result
            )
        return input_values[expression]

    def row_expression(expression: Any, size: int, lazy: bool) -> Any:
        if lazy and isinstance(expression, Outer):
            return (
                input_expression(expression.left).reshape(token_count, heads, -1),
                input_expression(expression.right).reshape(token_count, heads, -1),
            )
        return vector(input_expression(expression), size)

    def prepare(name: str):
        if name in prepared:
            return prepared[name]
        state = specs[name]
        transition = transitions[name]
        rows = role_sizes[state.feature_roles[0]]
        columns = 1 if state.rank == 1 else role_sizes[state.feature_roles[1]]
        identity = all(isinstance(node, Identity) for node in transition.propagations)
        if identity:
            row_gate, column_gate = None, None
        else:
            row_gate, column_gate = _axis_factors(
                transition.propagations,
                view=_matrix_view(state),
                evaluate=lambda expr, _: input_expression(expr),
                values=values,
                token_count=token_count,
                heads=heads,
                left_size=rows,
                right_size=columns,
                device=device,
            )
        key = None
        value = None
        scalar = None
        for factor in transition.injections[0].factors:
            if not factor.roles:
                scalar = input_expression(factor.expression)
            elif factor.roles[0] == state.feature_roles[0]:
                key = row_expression(
                    factor.expression, rows, identity and name not in identity_read
                )
            else:
                value = vector(input_expression(factor.expression), columns)
        if value is None:
            value = torch.ones(
                token_count, heads, columns, device=device, dtype=torch.float32
            )
        if scalar is not None:
            value = value * vector(scalar, 1)
        initial = initial_states[name]
        if state.rank == 1:
            if name in identity_read:
                initial = initial.unsqueeze(-2)
                value, key = key, value
                row_gate, column_gate = None, row_gate
            else:
                initial = initial.unsqueeze(-1)
        carry, final, factors = prepare_state(
            rows_gate=row_gate,
            columns_gate=column_gate,
            key=key,
            value=value,
            initial=initial,
            sequence_length=sequence_length,
            return_final_state=return_final_state,
            identity=identity,
        )
        if state.rank == 1:
            final = final.squeeze(-2 if name in identity_read else -1)
        final_states[name] = final
        prepared[name] = carry, factors, identity
        return prepared[name]

    def evaluate(expression: Any) -> torch.Tensor:
        if _contracted_terms(expression) == []:
            return input_expression(expression)
        if isinstance(expression, StateTerm):
            if expression not in term_values:
                carry, factors, identity = prepare(expression.state)
                size = carry.shape[2]
                if expression.role is None:
                    query = vector(evaluate(expression.operand), 1)
                elif (
                    identity
                    and isinstance(expression.operand, Outer)
                    and not _contracted_terms(expression.operand)
                ):
                    query = row_expression(expression.operand, size, True)
                else:
                    query = vector(evaluate(expression.operand), size)
                term_values[expression] = read_state(
                    carry,
                    factors,
                    query,
                    sequence_length=sequence_length,
                    prior=expression.prior,
                    identity=identity,
                )
            return term_values[expression]
        if isinstance(expression, (Add, Multiply, Divide, Contract, Outer)):
            left, right = evaluate(expression.left), evaluate(expression.right)
            if isinstance(expression, Add):
                return left + right
            if isinstance(expression, Multiply):
                return left * right
            if isinstance(expression, Divide):
                return left / right
            if isinstance(expression, Contract):
                return (left.float() * right.float()).sum(-1, keepdim=True)
            return (left.float().unsqueeze(-1) * right.float().unsqueeze(-2)).flatten(2)
        operand = evaluate(expression.operand).float()
        if isinstance(expression, Negate):
            return -operand
        if isinstance(expression, Exp):
            return operand.exp()
        if isinstance(expression, Log):
            return operand.log()
        if isinstance(expression, Sigmoid):
            return operand.sigmoid()
        if isinstance(expression, Sqrt):
            return operand.sqrt()
        if isinstance(expression, Reciprocal):
            return operand.reciprocal()
        if isinstance(expression, Clamp):
            return operand.clamp(expression.minimum, expression.maximum)
        if isinstance(expression, Softmax):
            return operand.softmax(-1)
        raise AssertionError("eligibility admitted an unsupported readout")

    outputs = {
        readout.name: evaluate(_readout_expression(readout)).to(readout.output_dtype)
        for readout in algorithm.readouts
    }
    if return_final_state:
        for state in algorithm.states:
            prepare(state.name)
        return outputs, final_states
    return outputs, None
