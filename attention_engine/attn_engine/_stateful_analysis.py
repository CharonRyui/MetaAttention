from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .stateful_operator import AlgorithmIR, InputExpression


class ExecutionClass(Enum):
    SCANABLE = "scanable"
    RECURRENT = "recurrent"


@dataclass(frozen=True)
class ScalarExpressionIR:
    """One-input expression recipe accepted by the raw dense lowering."""

    input_name: str
    operation: str
    scale: float = 1.0
    is_log_scale: bool = False


@dataclass(frozen=True)
class DenseInputIR:
    """Physical axes needed to load one dense Tensor Input directly."""

    name: str
    batch_axis: int
    head_axis: int
    sequence_axis: int
    feature_axis: int | None


@dataclass(frozen=True)
class DenseScalarFactorizedIR:
    """Target-independent recipe for one-launch dense scalar lowering."""

    propagation: tuple[DenseInputIR, ScalarExpressionIR]
    left_factor: tuple[DenseInputIR, ScalarExpressionIR]
    right_factor: tuple[DenseInputIR, ScalarExpressionIR]
    readout: tuple[DenseInputIR, ScalarExpressionIR]


@dataclass(frozen=True)
class ScalarFactorizedIR:
    """Structural proof for scalar propagation with rank-one injections."""

    injection_count: int
    readout_count: int
    propagation_count: int
    dense: DenseScalarFactorizedIR | None


@dataclass(frozen=True)
class TypedExpression:
    expression: InputExpression
    dtype: torch.dtype
    roles: tuple[Any, ...]

@dataclass(frozen=True)
class AffineSummaryIR:
    """Canonical semantic summary for vec(State): M @ state + additive."""

    state: str
    transform: str = "dense"
    additive: str = "state"


@dataclass(frozen=True)
class VJPStep:
    path: str
    operation: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class BackwardIR:
    output_cotangents: tuple[str, ...]
    final_state_cotangent: str
    reverse_steps: tuple[VJPStep, ...]
    input_gradients: tuple[str, ...]
    initial_state_gradient: str


@dataclass(frozen=True)
class Analysis:
    execution_class: ExecutionClass
    affine_summary: AffineSummaryIR
    backward: BackwardIR
    expression_types: tuple[tuple[str, TypedExpression], ...]
    scalar_factorized: ScalarFactorizedIR | None
    canonical_identity: str


def analyze_algorithm(algorithm: AlgorithmIR) -> Analysis:
    """Validate typing and derive target-independent forward/backward semantics."""
    from .stateful_operator import (
        Add,
        AxisScale,
        Constant,
        Exp,
        Identity,
        Input,
        Multiply,
        Negate,
        RankOnePropagation,
        StatefulCompilationError,
        _canonical_identity,
    )

    input_types = {item.name: (item.dtype, item.roles) for item in algorithm.inputs}
    state = algorithm.states[0]
    feature_roles = state.feature_roles
    types: list[tuple[str, TypedExpression]] = []

    def type_expression(expression: InputExpression, path: str) -> TypedExpression:
        if isinstance(expression, Input):
            try:
                dtype, roles = input_types[expression.name]
            except KeyError as error:
                raise StatefulCompilationError(
                    "IR_TYPE", path, {"unknown_input": expression.name}
                ) from error
            typed = TypedExpression(
                expression,
                dtype,
                _canonical_roles(roles, algorithm),
            )
        elif isinstance(expression, Constant):
            typed = TypedExpression(expression, expression.dtype, ())
        elif isinstance(expression, (Add, Multiply)):
            left = type_expression(expression.left, f"{path}.left")
            right = type_expression(expression.right, f"{path}.right")
            if not _broadcastable(left.roles, right.roles, algorithm):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    path,
                    {"left_roles": _role_names(left.roles), "right_roles": _role_names(right.roles)},
                )
            typed = TypedExpression(
                expression,
                _promote(left.dtype, right.dtype),
                _broadcast_roles(left.roles, right.roles, algorithm),
            )
        elif isinstance(expression, Negate):
            operand = type_expression(expression.operand, f"{path}.operand")
            typed = TypedExpression(expression, operand.dtype, operand.roles)
        elif isinstance(expression, Exp):
            operand = type_expression(expression.operand, f"{path}.operand")
            typed = TypedExpression(expression, torch.float32, operand.roles)
        else:
            raise StatefulCompilationError(
                "IR_TYPE", path, {"node": type(expression).__name__}
            )
        types.append((path, typed))
        return typed

    reverse: list[VJPStep] = []
    for index, node in enumerate(algorithm.transition.propagations):
        path = f"transition.propagations[{index}]"
        if isinstance(node, Identity):
            continue
        if isinstance(node, AxisScale):
            typed = type_expression(node.factor, f"{path}.factor")
            expected = tuple(role for role in typed.roles if role in feature_roles)
            if set(expected) != set(node.roles):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.roles",
                    {"factor_features": _role_names(expected), "declared": _role_names(node.roles)},
                )
            reverse.append(VJPStep(path, "axis_scale", _input_names(node.factor)))
        elif isinstance(node, RankOnePropagation):
            left = type_expression(node.left, f"{path}.left")
            right = type_expression(node.right, f"{path}.right")
            coefficient = type_expression(node.coefficient, f"{path}.coefficient")
            for name, typed in (("left", left), ("right", right)):
                if tuple(role for role in typed.roles if role in feature_roles) != (node.role,):
                    raise StatefulCompilationError(
                        "IR_TYPE", f"{path}.{name}", {"expected_feature": str(node.role)}
                    )
            if any(role in feature_roles for role in coefficient.roles):
                raise StatefulCompilationError(
                    "IR_TYPE", f"{path}.coefficient", {"expected": "feature scalar"}
                )
            reverse.append(
                VJPStep(path, "rank_one", _input_names(node.left, node.right, node.coefficient))
            )
        else:
            raise StatefulCompilationError(
                "RECURRENT", path, {"node": type(node).__name__}
            )

    for index, injection in enumerate(algorithm.transition.injections):
        path = f"transition.injections[{index}]"
        for factor_index, factor in enumerate(injection.factors):
            typed = type_expression(
                factor.expression, f"{path}.factors[{factor_index}].expression"
            )
            actual = tuple(role for role in typed.roles if role in feature_roles)
            if set(actual) != set(factor.roles):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.factors[{factor_index}].roles",
                    {"expression_features": _role_names(actual), "declared": _role_names(factor.roles)},
                )
        reverse.append(
            VJPStep(path, "product_injection", tuple(name for factor in injection.factors for name in _input_names(factor.expression)))
        )

    for index, readout in enumerate(algorithm.readouts):
        path = f"readouts[{index}]"
        operand = type_expression(readout.operand, f"{path}.operand")
        actual = tuple(role for role in operand.roles if role in feature_roles)
        if actual != (readout.role,):
            raise StatefulCompilationError(
                "IR_TYPE", f"{path}.operand", {"expected_feature": str(readout.role)}
            )
        reverse.append(VJPStep(path, "state_contraction", _input_names(readout.operand)))

    reverse.extend(
        VJPStep(f"head_mapping[{index}]", "head_reduce", (str(source),))
        for index, (source, _) in enumerate(algorithm.head_mapping.mappings)
    )
    reverse.append(
        VJPStep(
            "transition.state",
            "state_recurrence",
            (state.name, f"d_{state.name}_final"),
        )
    )
    backward = BackwardIR(
        tuple(readout.name for readout in algorithm.readouts),
        f"d_{state.name}_final",
        tuple(reversed(reverse)),
        tuple(item.name for item in algorithm.inputs if item.differentiable),
        f"d_{state.name}_initial",
    )
    scalar_factorized = None
    scalar_propagations = all(
        isinstance(node, Identity)
        or isinstance(node, AxisScale) and not node.roles
        for node in algorithm.transition.propagations
    )
    factorized_injections = bool(algorithm.transition.injections) and all(
        len(injection.factors) == 2
        and {role for factor in injection.factors for role in factor.roles}
        == set(feature_roles)
        and all(len(factor.roles) == 1 for factor in injection.factors)
        for injection in algorithm.transition.injections
    )
    same_orientation = bool(algorithm.readouts) and all(
        readout.role == feature_roles[0] for readout in algorithm.readouts
    )
    if scalar_propagations and factorized_injections and same_orientation:
        dense_scalar = None
        if (
            len(algorithm.transition.propagations) == 1
            and len(algorithm.transition.injections) == 1
            and len(algorithm.readouts) == 1
        ):
            propagation_node = algorithm.transition.propagations[0]
            injection = algorithm.transition.injections[0]
            left_factor = next(
                factor for factor in injection.factors if factor.roles == (feature_roles[0],)
            )
            right_factor = next(
                factor for factor in injection.factors if factor.roles == (feature_roles[1],)
            )
            propagation = _dense_scalar_binding(
                propagation_node.factor, algorithm, feature_role=None
            )
            left = _dense_scalar_binding(
                left_factor.expression, algorithm, feature_role=feature_roles[0]
            )
            right = _dense_scalar_binding(
                right_factor.expression, algorithm, feature_role=feature_roles[1]
            )
            readout = _dense_scalar_binding(
                algorithm.readouts[0].operand,
                algorithm,
                feature_role=feature_roles[0],
            )
            if all(recipe is not None for recipe in (propagation, left, right, readout)):
                dense_scalar = DenseScalarFactorizedIR(
                    propagation, left, right, readout
                )
        scalar_factorized = ScalarFactorizedIR(
            len(algorithm.transition.injections),
            len(algorithm.readouts),
            len(algorithm.transition.propagations),
            dense_scalar,
        )
    return Analysis(
        ExecutionClass.SCANABLE,
        AffineSummaryIR(state.name),
        backward,
        tuple(types),
        scalar_factorized,
        _canonical_identity(algorithm),
    )


def _promote(left: torch.dtype, right: torch.dtype) -> torch.dtype:
    return torch.float32 if torch.float32 in (left, right) else torch.bfloat16


def _broadcastable(
    left: tuple[Any, ...], right: tuple[Any, ...], algorithm: AlgorithmIR
) -> bool:
    from .stateful_operator import HeadRole

    left_heads = {
        _mapped_head(role, algorithm) for role in left if isinstance(role, HeadRole)
    }
    right_heads = {
        _mapped_head(role, algorithm) for role in right if isinstance(role, HeadRole)
    }
    return not left_heads or not right_heads or left_heads == right_heads


def _broadcast_roles(
    left: tuple[Any, ...], right: tuple[Any, ...], algorithm: AlgorithmIR
) -> tuple[Any, ...]:
    from .stateful_operator import HeadRole, One

    combined = left + tuple(role for role in right if role not in left)
    result: list[Any] = []
    for role in combined:
        if role is One:
            continue
        mapped = _mapped_head(role, algorithm) if isinstance(role, HeadRole) else role
        if mapped not in result:
            result.append(mapped)
    return tuple(result) or combined


def _canonical_roles(roles: tuple[Any, ...], algorithm: AlgorithmIR) -> tuple[Any, ...]:
    from .stateful_operator import HeadRole

    return tuple(
        _mapped_head(role, algorithm) if isinstance(role, HeadRole) else role
        for role in roles
    )


def _mapped_head(role: Any, algorithm: AlgorithmIR) -> Any:
    return algorithm.head_mapping.target_for(role) or role


def _role_names(roles: tuple[Any, ...]) -> list[str]:
    return [str(role) for role in roles]


def _input_names(*expressions: InputExpression) -> tuple[str, ...]:
    from .stateful_operator import Add, Exp, Input, Multiply, Negate

    result: list[str] = []
    for expression in expressions:
        if isinstance(expression, Input):
            result.append(expression.name)
        elif isinstance(expression, (Add, Multiply)):
            result.extend(_input_names(expression.left, expression.right))
        elif isinstance(expression, (Negate, Exp)):
            result.extend(_input_names(expression.operand))
    return tuple(result)


def _dense_scalar_binding(
    expression: InputExpression,
    algorithm: AlgorithmIR,
    *,
    feature_role: Any | None,
    require_exp: bool = False,
) -> tuple[DenseInputIR, ScalarExpressionIR] | None:
    from .stateful_operator import Batch, Constant, Exp, HeadRole, Input, Multiply, Negate, Sequence

    operation = "identity"
    scale = 1.0
    is_log_scale = False
    if isinstance(expression, Exp):
        expression = expression.operand
        is_log_scale = True
    elif require_exp:
        return None
    if isinstance(expression, Negate):
        expression = expression.operand
        scale = -scale
    if isinstance(expression, Multiply):
        if isinstance(expression.left, Constant):
            scale *= expression.left.value
            expression = expression.right
        elif isinstance(expression.right, Constant):
            scale *= expression.right.value
            expression = expression.left
        else:
            return None
    if not isinstance(expression, Input):
        return None
    spec = next(item for item in algorithm.inputs if item.name == expression.name)
    if Batch not in spec.roles or Sequence not in spec.roles:
        return None
    head_axis = next(
        (index for index, role in enumerate(spec.roles) if isinstance(role, HeadRole)),
        None,
    )
    if head_axis is None:
        return None
    expected_roles = (
        (Batch, spec.roles[head_axis], Sequence)
        if feature_role is None
        else (Batch, spec.roles[head_axis], Sequence, feature_role)
    )
    if spec.roles != expected_roles:
        return None
    present_features = tuple(
        role for role in algorithm.states[0].feature_roles if role in spec.roles
    )
    if present_features != (() if feature_role is None else (feature_role,)):
        return None
    feature_axis = None if feature_role is None else spec.roles.index(feature_role)
    binding = DenseInputIR(
        spec.name,
        spec.roles.index(Batch),
        head_axis,
        spec.roles.index(Sequence),
        feature_axis,
    )
    return binding, ScalarExpressionIR(spec.name, operation, scale, is_log_scale)
