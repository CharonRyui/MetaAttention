from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
import math
import os
import sys
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

SCHEMA_VERSION = "stateful-operator-ir/1"
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


class StatefulCompilationError(Exception):
    """Stable public failure from Stateful Operator validation or specialization."""

    def __init__(
        self,
        category: str,
        path: str,
        details: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        self.category = category
        self.code = category
        self.path = path
        self.details = dict(details or {})
        super().__init__(message or f"{category} at {path}")

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "path": self.path, "details": self.details}


class _NamedTensorTuple:
    __slots__ = ("_names", "_values")

    def __init__(self, names: Sequence[str], values: Sequence[torch.Tensor]) -> None:
        if len(names) != len(values):
            raise ValueError("names and values must have equal length")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        object.__setattr__(self, "_names", tuple(names))
        object.__setattr__(self, "_values", tuple(values))

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def values(self) -> tuple[torch.Tensor, ...]:
        return self._values

    def __getitem__(self, key: int | str) -> torch.Tensor:
        if isinstance(key, int):
            return self._values[key]
        try:
            return self._values[self._names.index(key)]
        except ValueError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(zip(self._names, self._values))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.as_dict()!r})"


class NamedOutputTuple(_NamedTensorTuple):
    pass


class StateTuple(_NamedTensorTuple):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    outputs: NamedOutputTuple
    final_state: StateTuple | None = None


class InputExpression:
    def __add__(self, other: InputExpression | float | int) -> Add:
        return Add(self, as_expression(other))

    def __radd__(self, other: InputExpression | float | int) -> Add:
        return Add(as_expression(other), self)

    def __mul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(self, as_expression(other))

    def __rmul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(as_expression(other), self)

    def __neg__(self) -> Negate:
        return Negate(self)


@dataclass(frozen=True)
class Input(InputExpression):
    name: str


@dataclass(frozen=True, init=False)
class Constant(InputExpression):
    value: float
    dtype: torch.dtype
    bits: int

    def __init__(self, value: float | int, dtype: torch.dtype = torch.float32) -> None:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError("constants support only torch.float32 and torch.bfloat16")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("constants reject NaN and infinity")
        rounded = torch.tensor(numeric, dtype=dtype)
        integer_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        object.__setattr__(self, "value", float(rounded.item()))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "bits", int(rounded.view(integer_dtype).item()))


@dataclass(frozen=True)
class Add(InputExpression):
    left: InputExpression
    right: InputExpression

    def __post_init__(self) -> None:
        left, right = self.left, self.right
        if _identity(left) > _identity(right):
            object.__setattr__(self, "left", right)
            object.__setattr__(self, "right", left)


@dataclass(frozen=True)
class Multiply(InputExpression):
    left: InputExpression
    right: InputExpression

    def __post_init__(self) -> None:
        left, right = self.left, self.right
        if _identity(left) > _identity(right):
            object.__setattr__(self, "left", right)
            object.__setattr__(self, "right", left)


@dataclass(frozen=True)
class Negate(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Exp(InputExpression):
    operand: InputExpression


def as_expression(value: InputExpression | float | int) -> InputExpression:
    if isinstance(value, InputExpression):
        return value
    if isinstance(value, (int, float)):
        return Constant(value)
    raise TypeError(
        f"expected an input expression or numeric constant, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TensorInput:
    name: str
    roles: tuple[str, ...]
    dtype: torch.dtype
    differentiable: bool = True

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid tensor input name {self.name!r}")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError(f"tensor input '{self.name}' requires unique logical roles")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("Tensor Inputs support only BF16 and FP32")


@dataclass(frozen=True, init=False)
class StateSpec:
    name: str
    roles: tuple[str, str, str, str]

    def __init__(self, name: str, roles: Sequence[str]) -> None:
        roles = tuple(roles)
        if len(roles) != 4:
            raise ValueError("matrix State roles are batch, head, and two Features")
        if roles[-2] == roles[-1]:
            raise ValueError("State Feature roles must be distinct and ordered")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "roles", roles)

    @property
    def head_role(self) -> str:
        return self.roles[1]

    @property
    def feature_roles(self) -> tuple[str, str]:
        return self.roles[2], self.roles[3]


@dataclass(frozen=True)
class Identity:
    pass


@dataclass(frozen=True, init=False)
class AxisScale:
    factor: InputExpression
    roles: tuple[str, ...]

    def __init__(
        self, factor: InputExpression | float | int, roles: Sequence[str] = ()
    ) -> None:
        object.__setattr__(self, "factor", as_expression(factor))
        object.__setattr__(self, "roles", tuple(roles))


@dataclass(frozen=True, init=False)
class RankOnePropagation:
    role: str
    left: InputExpression
    right: InputExpression
    coefficient: InputExpression

    def __init__(
        self,
        role: str,
        left: InputExpression,
        right: InputExpression,
        coefficient: InputExpression | float | int = 1.0,
    ) -> None:
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "left", as_expression(left))
        object.__setattr__(self, "right", as_expression(right))
        object.__setattr__(self, "coefficient", as_expression(coefficient))


@dataclass(frozen=True, init=False)
class ProductFactor:
    expression: InputExpression
    roles: tuple[str, ...]

    def __init__(
        self, expression: InputExpression, roles: Sequence[str] = ()
    ) -> None:
        object.__setattr__(self, "expression", as_expression(expression))
        object.__setattr__(self, "roles", tuple(roles))


@dataclass(frozen=True)
class ProductInjection:
    factors: tuple[ProductFactor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        if not self.factors:
            raise ValueError("Product Injection requires at least one factor")


@dataclass(frozen=True, init=False)
class StateContraction:
    name: str
    state: str
    operand: InputExpression
    role: str
    output_dtype: torch.dtype

    def __init__(
        self,
        name: str,
        state: str,
        operand: InputExpression,
        role: str,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if output_dtype not in _SUPPORTED_DTYPES:
            raise ValueError("State Contraction output supports only BF16 and FP32")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "output_dtype", output_dtype)


@dataclass(frozen=True, init=False)
class HeadMapping:
    mappings: tuple[tuple[str, str], ...]

    def __init__(
        self,
        mappings: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    ) -> None:
        values = mappings.items() if isinstance(mappings, Mapping) else mappings
        pairs = tuple((str(source), str(target)) for source, target in values)
        if len({source for source, _ in pairs}) != len(pairs):
            raise ValueError("Head Mapping source roles must be unique")
        object.__setattr__(self, "mappings", pairs)

    def target_for(self, source: str) -> str | None:
        return dict(self.mappings).get(source)


@dataclass(frozen=True, init=False)
class StateTransition:
    state: str
    propagations: tuple[Identity | AxisScale | RankOnePropagation, ...]
    injections: tuple[ProductInjection, ...]

    def __init__(
        self,
        state: str,
        *,
        propagations: Sequence[Identity | AxisScale | RankOnePropagation] = (),
        injections: Sequence[ProductInjection] = (),
    ) -> None:
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "propagations", tuple(propagations))
        object.__setattr__(self, "injections", tuple(injections))


@dataclass(frozen=True, init=False)
class AlgorithmIR:
    inputs: tuple[TensorInput, ...]
    states: tuple[StateSpec, ...]
    transition: StateTransition
    readouts: tuple[StateContraction, ...]
    head_mapping: HeadMapping

    def __init__(
        self,
        inputs: Sequence[TensorInput],
        states: Sequence[StateSpec],
        transition: StateTransition,
        readouts: Sequence[StateContraction],
        head_mapping: HeadMapping,
    ) -> None:
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "readouts", tuple(readouts))
        object.__setattr__(self, "head_mapping", head_mapping)
        self._validate()

    def _validate(self) -> None:
        _reject_duplicate_names(self.inputs, "inputs")
        _reject_duplicate_names(self.states, "states")
        if len(self.states) != 1:
            raise StatefulCompilationError("IR_SCHEMA", "states", {"expected": 1})
        state = self.states[0]
        if self.transition.state != state.name:
            raise StatefulCompilationError(
                "IR_TYPE", "transition.state", {"unknown_state": self.transition.state}
            )
        if not self.readouts:
            raise StatefulCompilationError(
                "IR_SCHEMA", "readouts", {"expected": "at least one"}
            )
        _reject_duplicate_names(self.readouts, "readouts")
        input_names = {item.name for item in self.inputs}
        for expression in _expressions(self):
            if isinstance(expression, Input) and expression.name not in input_names:
                raise StatefulCompilationError(
                    "IR_TYPE", "expression", {"unknown_input": expression.name}
                )
        features = set(state.feature_roles)
        for index, propagation in enumerate(self.transition.propagations):
            if isinstance(propagation, AxisScale):
                if not set(propagation.roles) <= features:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"transition.propagations[{index}].roles",
                        {"roles": propagation.roles},
                    )
            elif isinstance(propagation, RankOnePropagation):
                if propagation.role not in features:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"transition.propagations[{index}].role",
                        {"role": propagation.role},
                    )
            elif not isinstance(propagation, Identity):
                raise StatefulCompilationError(
                    "IR_SCHEMA",
                    f"transition.propagations[{index}]",
                    {"node": type(propagation).__name__},
                )
        for index, injection in enumerate(self.transition.injections):
            covered = {role for factor in injection.factors for role in factor.roles}
            if not features <= covered:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"transition.injections[{index}]",
                    {"missing_roles": sorted(features - covered)},
                )
        for index, readout in enumerate(self.readouts):
            if readout.state != state.name or readout.role not in features:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"readouts[{index}]",
                    {"state": readout.state, "role": readout.role},
                )

    @property
    def structural_identity(self) -> str:
        return hashlib.sha256(
            json.dumps(_canonical(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Target:
    device_type: str = "cuda"
    product_family: str = "H20"
    capability: tuple[int, int] = (9, 0)


@dataclass(frozen=True)
class CompileOptions:
    tune: bool = False
    tune_filename: str = "tune_result"
    tune_backward: bool = False


@dataclass(frozen=True)
class _Runtime:
    packed: bool
    device: torch.device
    sequence_count: int
    token_count: int
    uniform_length: int | None
    state_heads: int
    feature_sizes: tuple[int, int]
    offsets: torch.Tensor | None


class StatefulOperator:
    def __init__(
        self,
        algorithm: AlgorithmIR,
        *,
        target: Target | None = None,
        compile_options: CompileOptions | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.target = target
        self.compile_options = compile_options or CompileOptions()
        self._specializations: set[str] = set()

    @property
    def structural_identity(self) -> str:
        payload = (
            SCHEMA_VERSION,
            _canonical(self.algorithm),
            _canonical(self.target),
            _canonical(self.compile_options),
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def __call__(
        self,
        *,
        initial_state: StateTuple | None = None,
        sequence_offsets: torch.Tensor | None = None,
        return_final_state: bool = False,
        **inputs: torch.Tensor,
    ) -> ExecutionResult:
        expected = {item.name for item in self.algorithm.inputs}
        missing = expected - inputs.keys()
        unknown = inputs.keys() - expected
        if missing:
            raise StatefulCompilationError(
                "RUNTIME_BINDING", "inputs", {"missing": sorted(missing)}
            )
        if unknown:
            raise StatefulCompilationError(
                "RUNTIME_BINDING", "inputs", {"unknown": sorted(unknown)}
            )
        runtime, role_sizes = self._validate_runtime(inputs, sequence_offsets)
        self._validate_target(runtime.device)
        self._validate_aliases(inputs)
        state = self._validate_state(initial_state, runtime)
        self._specializations.add(
            self._specialization_key(inputs, runtime, return_final_state)
        )
        values = self._normalize_inputs(inputs, runtime, role_sizes)
        outputs, final_state = self._lower_parallel_scan(
            values, state, runtime, return_final_state
        )
        return ExecutionResult(
            NamedOutputTuple(
                tuple(readout.name for readout in self.algorithm.readouts), outputs
            ),
            final_state,
        )

    def _validate_runtime(
        self,
        inputs: Mapping[str, torch.Tensor],
        offsets: torch.Tensor | None,
    ) -> tuple[_Runtime, dict[str, int]]:
        packed = offsets is not None
        device: torch.device | None = None
        role_sizes: dict[str, int] = {}
        token_count: int | None = None
        uniform_length: int | None = None
        for spec in self.algorithm.inputs:
            tensor = inputs[spec.name]
            path = f"inputs.{spec.name}"
            if not isinstance(tensor, torch.Tensor):
                raise StatefulCompilationError(
                    "RUNTIME_BINDING", path, {"expected": "torch.Tensor"}
                )
            if tensor.dtype != spec.dtype:
                raise StatefulCompilationError(
                    "RUNTIME_DTYPE",
                    path,
                    {"expected": str(spec.dtype), "actual": str(tensor.dtype)},
                )
            if not tensor.is_contiguous():
                raise StatefulCompilationError(
                    "RUNTIME_LAYOUT", path, {"expected": "contiguous"}
                )
            if not spec.differentiable and tensor.requires_grad:
                raise StatefulCompilationError(
                    "RUNTIME_GRAD", path, {"differentiable": False}
                )
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise StatefulCompilationError("RUNTIME_DEVICE", "inputs", {})
            if packed and "sequence" in spec.roles:
                physical_roles = tuple(
                    role for role in spec.roles if role not in ("batch", "sequence")
                )
                if tensor.ndim != len(physical_roles) + 1:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"mode": "packed"}
                    )
                token_count = tensor.shape[0] if token_count is None else token_count
                if tensor.shape[0] != token_count:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"role": "sequence"}
                    )
                dimensions = zip(physical_roles, tensor.shape[1:])
            else:
                if tensor.ndim != len(spec.roles):
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"expected_ndim": len(spec.roles)}
                    )
                dimensions = zip(spec.roles, tensor.shape)
                if "sequence" in spec.roles:
                    length = tensor.shape[spec.roles.index("sequence")]
                    uniform_length = length if uniform_length is None else uniform_length
                    if length != uniform_length or length <= 0:
                        raise StatefulCompilationError(
                            "RUNTIME_SHAPE", path, {"role": "sequence"}
                        )
            for role, size in dimensions:
                if role == "one":
                    if size != 1:
                        raise StatefulCompilationError(
                            "RUNTIME_SHAPE", path, {"role": "one", "expected": 1}
                        )
                    continue
                previous = role_sizes.setdefault(role, size)
                if previous != size:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE",
                        path,
                        {"role": role, "sizes": [previous, size]},
                    )
        assert device is not None
        if packed:
            assert offsets is not None
            token_count = token_count or 0
            if (
                offsets.dtype != torch.int32
                or offsets.ndim != 1
                or not offsets.is_contiguous()
                or offsets.device != device
                or offsets.numel() < 1
            ):
                raise StatefulCompilationError(
                    "RUNTIME_OFFSETS",
                    "sequence_offsets",
                    {"expected": "device-local contiguous int32 vector"},
                )
            torch._assert_async(offsets[0] == 0, "sequence_offsets must start at zero")
            torch._assert_async(
                offsets[-1] == token_count,
                "sequence_offsets must end at total token count",
            )
            torch._assert_async(
                torch.all(offsets[1:] >= offsets[:-1]),
                "sequence_offsets must be nondecreasing",
            )
            sequence_count = offsets.shape[0] - 1
            role_sizes["batch"] = sequence_count
        else:
            if uniform_length is None:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", "inputs", {"sequence": "required"}
                )
            sequence_count = role_sizes["batch"]
            token_count = sequence_count * uniform_length
        state_spec = self.algorithm.states[0]
        try:
            state_heads = role_sizes[state_spec.head_role]
            feature_sizes = tuple(role_sizes[role] for role in state_spec.feature_roles)
        except KeyError as error:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE", "states[0].roles", {"missing_role": error.args[0]}
            ) from error
        if device.type == "cuda" and (
            feature_sizes[0] not in (64, 128) or feature_sizes[1] not in (64, 128)
        ):
            raise StatefulCompilationError(
                "SPECIALIZATION",
                "states[0].roles",
                {"feature_sizes": feature_sizes, "supported": [64, 128]},
            )
        for source, target in self.algorithm.head_mapping.mappings:
            if target != state_spec.head_role or source not in role_sizes:
                raise StatefulCompilationError(
                    "IR_TYPE", "head_mapping", {"source": source, "target": target}
                )
            if state_heads % role_sizes[source]:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", "head_mapping", {"source": source}
                )
        return (
            _Runtime(
                packed,
                device,
                sequence_count,
                token_count,
                uniform_length,
                state_heads,
                (feature_sizes[0], feature_sizes[1]),
                offsets,
            ),
            role_sizes,
        )
    def _validate_target(self, device: torch.device) -> None:
        if device.type == "cpu":
            return
        if device.type != "cuda" or torch.version.hip is not None:
            raise StatefulCompilationError(
                "TARGET", "target", {"expected": "CUDA H20", "actual": device.type}
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        product = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "MIG" in product.upper() or "H20" not in product.upper() or capability != (9, 0):
            raise StatefulCompilationError(
                "TARGET",
                "target",
                {"expected_product": "H20", "actual_product": product, "capability": capability},
            )
        if self.target is not None and (
            self.target.device_type != "cuda"
            or self.target.product_family.upper() not in product.upper()
            or self.target.capability != capability
        ):
            raise StatefulCompilationError(
                "TARGET_CONSTRAINT", "target", {"constraint": _canonical(self.target)}
            )

    @staticmethod
    def _validate_aliases(inputs: Mapping[str, torch.Tensor]) -> None:
        tensors = list(inputs.items())
        for index, (left_name, left) in enumerate(tensors):
            for right_name, right in tensors[index + 1 :]:
                if left is right or left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr():
                    continue
                left_start = left.storage_offset() * left.element_size()
                right_start = right.storage_offset() * right.element_size()
                left_end = left_start + left.numel() * left.element_size()
                right_end = right_start + right.numel() * right.element_size()
                if max(left_start, right_start) < min(left_end, right_end):
                    raise StatefulCompilationError(
                        "RUNTIME_ALIAS",
                        "inputs",
                        {"overlapping": [left_name, right_name]},
                    )

    def _validate_state(
        self, state: StateTuple | None, runtime: _Runtime
    ) -> torch.Tensor:
        spec = self.algorithm.states[0]
        shape = (
            runtime.sequence_count,
            runtime.state_heads,
            runtime.feature_sizes[0],
            runtime.feature_sizes[1],
        )
        if state is None:
            return torch.zeros(shape, device=runtime.device, dtype=torch.float32)
        if not isinstance(state, StateTuple) or state.names != (spec.name,):
            raise StatefulCompilationError(
                "RUNTIME_STATE", "initial_state", {"expected": [spec.name]}
            )
        tensor = state[0]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device != runtime.device
            or not tensor.is_contiguous()
        ):
            raise StatefulCompilationError(
                "RUNTIME_STATE",
                f"initial_state.{spec.name}",
                {"shape": shape, "dtype": "torch.float32", "device": str(runtime.device)},
            )
        return tensor

    def _normalize_inputs(
        self,
        inputs: Mapping[str, torch.Tensor],
        runtime: _Runtime,
        role_sizes: Mapping[str, int],
    ) -> dict[str, torch.Tensor]:
        state = self.algorithm.states[0]
        features = state.feature_roles
        result: dict[str, torch.Tensor] = {}
        for spec in self.algorithm.inputs:
            tensor = inputs[spec.name]
            if runtime.packed and "sequence" in spec.roles:
                physical_roles = tuple(
                    role for role in spec.roles if role not in ("batch", "sequence")
                )
                value = tensor
                token_axis = 0
            elif "sequence" in spec.roles:
                head_role = _head_role(spec.roles, features)
                order = [spec.roles.index("batch"), spec.roles.index("sequence")]
                if head_role is not None:
                    order.append(spec.roles.index(head_role))
                order.extend(spec.roles.index(role) for role in features if role in spec.roles)
                value = tensor.permute(order).reshape(
                    runtime.token_count, *[tensor.shape[index] for index in order[2:]]
                )
                physical_roles = tuple(spec.roles[index] for index in order[2:])
                token_axis = 0
            else:
                value = tensor
                physical_roles = tuple(role for role in spec.roles if role != "one")
                token_axis = None
            head_role = _head_role(physical_roles, features)
            if token_axis is None:
                if head_role is None:
                    value = value.reshape(1, 1, *value.shape[len(spec.roles) :])
                else:
                    head_axis = physical_roles.index(head_role)
                    value = value.movedim(head_axis, 0).unsqueeze(0)
                value = value.expand(runtime.token_count, *value.shape[1:])
            elif head_role is None:
                value = value.unsqueeze(1)
            else:
                head_axis = 1 + physical_roles.index(head_role)
                if head_axis != 1:
                    value = value.movedim(head_axis, 1)
            source_heads = value.shape[1]
            if source_heads != runtime.state_heads:
                if runtime.state_heads % source_heads:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", f"inputs.{spec.name}", {"role": head_role}
                    )
                value = value.repeat_interleave(runtime.state_heads // source_heads, dim=1)
            present_features = tuple(role for role in features if role in spec.roles)
            if present_features == ():
                value = value.reshape(runtime.token_count, runtime.state_heads, 1, 1)
            elif present_features == (features[0],):
                value = value.reshape(runtime.token_count, runtime.state_heads, -1, 1)
            elif present_features == (features[1],):
                value = value.reshape(runtime.token_count, runtime.state_heads, 1, -1)
            else:
                value = value.reshape(
                    runtime.token_count,
                    runtime.state_heads,
                    runtime.feature_sizes[0],
                    runtime.feature_sizes[1],
                )
            result[spec.name] = value
        return result

    def _lower_parallel_scan(
        self,
        values: Mapping[str, torch.Tensor],
        initial_state: torch.Tensor,
        runtime: _Runtime,
        return_final_state: bool,
    ) -> tuple[tuple[torch.Tensor, ...], StateTuple | None]:
        token_count = runtime.token_count
        heads = runtime.state_heads
        left_size, right_size = runtime.feature_sizes
        device = runtime.device
        state_spec = self.algorithm.states[0]
        if runtime.packed:
            assert runtime.offsets is not None
            lengths = runtime.offsets[1:] - runtime.offsets[:-1]
            segment = torch.repeat_interleave(
                torch.arange(runtime.sequence_count, device=device), lengths.long()
            )
        else:
            assert runtime.uniform_length is not None
            segment = torch.arange(runtime.sequence_count, device=device).repeat_interleave(
                runtime.uniform_length
            )
        if token_count == 0:
            outputs = tuple(
                torch.empty(
                    (0, heads, right_size if readout.role == state_spec.feature_roles[0] else left_size),
                    device=device,
                    dtype=readout.output_dtype,
                )
                for readout in self.algorithm.readouts
            )
            final = StateTuple((state_spec.name,), (initial_state,)) if return_final_state else None
            return outputs, final

        identity_left = torch.eye(left_size, device=device).expand(
            token_count, heads, left_size, left_size
        )
        identity_right = torch.eye(right_size, device=device).expand(
            token_count, heads, right_size, right_size
        )
        left = identity_left
        right = identity_right
        elementwise: torch.Tensor | None = None
        has_rank = any(
            isinstance(node, RankOnePropagation)
            for node in self.algorithm.transition.propagations
        )
        for index, node in enumerate(self.algorithm.transition.propagations):
            if isinstance(node, Identity):
                continue
            if isinstance(node, AxisScale):
                factor = _evaluate(node.factor, values).float()
                if len(node.roles) == 2:
                    if has_rank:
                        raise StatefulCompilationError(
                            "SPECIALIZATION",
                            f"transition.propagations[{index}]",
                            {"reason": "dense affine resource limit"},
                        )
                    elementwise = factor if elementwise is None else elementwise * factor
                elif not node.roles:
                    left = factor * left
                elif node.roles[0] == state_spec.feature_roles[0]:
                    left = torch.diag_embed(factor[..., :, 0]) @ left
                else:
                    right = right @ torch.diag_embed(factor[..., 0, :])
            elif isinstance(node, RankOnePropagation):
                left_vector = _feature_vector(
                    _evaluate(node.left, values), node.role, state_spec
                )
                right_vector = _feature_vector(
                    _evaluate(node.right, values), node.role, state_spec
                )
                coefficient = _evaluate(node.coefficient, values).float()[..., :1, :1]
                if node.role == state_spec.feature_roles[0]:
                    update = identity_left + coefficient * (
                        left_vector.unsqueeze(-1) * right_vector.unsqueeze(-2)
                    )
                    left = update @ left
                else:
                    update = identity_right + coefficient * (
                        right_vector.unsqueeze(-1) * left_vector.unsqueeze(-2)
                    )
                    right = right @ update

        injection = torch.zeros(
            token_count, heads, left_size, right_size, device=device
        )
        for term in self.algorithm.transition.injections:
            product: torch.Tensor | float = 1.0
            for factor in term.factors:
                product = product * _evaluate(factor.expression, values).float()
            injection = injection + product

        if elementwise is not None:
            scale = elementwise
            scale, injection = _segmented_elementwise_scan(scale, injection, segment)
            states = scale * initial_state[segment] + injection
        else:
            left, right, injection = _segmented_affine_scan(
                left, right, injection, segment
            )
            states = left @ initial_state[segment] @ right + injection

        output_values: list[torch.Tensor] = []
        for readout in self.algorithm.readouts:
            operand = _feature_vector(
                _evaluate(readout.operand, values), readout.role, state_spec
            ).float()
            if readout.role == state_spec.feature_roles[0]:
                output = torch.einsum("nhf,nhfg->nhg", operand, states)
            else:
                output = torch.einsum("nhg,nhfg->nhf", operand, states)
            output = output.to(readout.output_dtype)
            if not runtime.packed:
                assert runtime.uniform_length is not None
                output = output.reshape(
                    runtime.sequence_count,
                    runtime.uniform_length,
                    heads,
                    output.shape[-1],
                ).permute(0, 2, 1, 3)
            output_values.append(output)

        final_state = None
        if return_final_state:
            if runtime.packed:
                assert runtime.offsets is not None
                lengths = runtime.offsets[1:] - runtime.offsets[:-1]
                last = (runtime.offsets[1:] - 1).clamp_min(0).long()
                gathered = states[last]
                final = torch.where(
                    (lengths == 0).view(-1, 1, 1, 1), initial_state, gathered
                )
            else:
                assert runtime.uniform_length is not None
                final = states.reshape(
                    runtime.sequence_count,
                    runtime.uniform_length,
                    heads,
                    left_size,
                    right_size,
                )[:, -1]
            final_state = StateTuple((state_spec.name,), (final,))
        return tuple(output_values), final_state

    def _specialization_key(
        self,
        inputs: Mapping[str, torch.Tensor],
        runtime: _Runtime,
        return_final_state: bool,
    ) -> str:
        active = sorted(
            name
            for name, tensor in inputs.items()
            if torch.is_grad_enabled() and tensor.requires_grad
        )
        metadata = [
            (name, tuple(tensor.shape), str(tensor.dtype), tensor.device.type)
            for name, tensor in sorted(inputs.items())
        ]
        offset_metadata = (
            None
            if runtime.offsets is None
            else (tuple(runtime.offsets.shape), str(runtime.offsets.dtype))
        )
        payload = (
            self.structural_identity,
            metadata,
            runtime.packed,
            offset_metadata,
            active,
            return_final_state,
            "cuda-h20-sm90",
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _segmented_affine_scan(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = 1
    token_count = left.shape[0]
    while offset < token_count:
        old_left, old_right, old_bias = left, right, bias
        valid = segment[offset:] == segment[:-offset]
        combined_left = old_left[offset:] @ old_left[:-offset]
        combined_right = old_right[:-offset] @ old_right[offset:]
        combined_bias = (
            old_left[offset:] @ old_bias[:-offset] @ old_right[offset:]
            + old_bias[offset:]
        )
        mask = valid.view(-1, 1, 1, 1)
        left = torch.cat((old_left[:offset], torch.where(mask, combined_left, old_left[offset:])))
        right = torch.cat((old_right[:offset], torch.where(mask, combined_right, old_right[offset:])))
        bias = torch.cat((old_bias[:offset], torch.where(mask, combined_bias, old_bias[offset:])))
        offset *= 2
    return left, right, bias


def _segmented_elementwise_scan(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = 1
    token_count = scale.shape[0]
    while offset < token_count:
        old_scale, old_bias = scale, bias
        valid = (segment[offset:] == segment[:-offset]).view(-1, 1, 1, 1)
        combined_scale = old_scale[offset:] * old_scale[:-offset]
        combined_bias = old_scale[offset:] * old_bias[:-offset] + old_bias[offset:]
        scale = torch.cat((old_scale[:offset], torch.where(valid, combined_scale, old_scale[offset:])))
        bias = torch.cat((old_bias[:offset], torch.where(valid, combined_bias, old_bias[offset:])))
        offset *= 2
    return scale, bias


def _evaluate(
    expression: InputExpression, values: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    if isinstance(expression, Input):
        return values[expression.name]
    if isinstance(expression, Constant):
        sample = next(iter(values.values()))
        return torch.tensor(expression.value, device=sample.device, dtype=expression.dtype)
    if isinstance(expression, Add):
        return _evaluate(expression.left, values) + _evaluate(expression.right, values)
    if isinstance(expression, Multiply):
        return _evaluate(expression.left, values) * _evaluate(expression.right, values)
    if isinstance(expression, Negate):
        return -_evaluate(expression.operand, values)
    if isinstance(expression, Exp):
        return _evaluate(expression.operand, values).float().exp()
    raise StatefulCompilationError(
        "IR_TYPE", "expression", {"node": type(expression).__name__}
    )


def _feature_vector(
    value: torch.Tensor, role: str, state: StateSpec
) -> torch.Tensor:
    if role == state.feature_roles[0]:
        return value[..., :, 0]
    return value[..., 0, :]


def _head_role(roles: Sequence[str], features: Sequence[str]) -> str | None:
    return next(
        (
            role
            for role in roles
            if role not in ("batch", "sequence", "one") and role not in features
        ),
        None,
    )


def _reject_duplicate_names(items: Iterable[Any], path: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.name in seen:
            raise StatefulCompilationError(
                "IR_SCHEMA", path, {"duplicate": item.name}
            )
        seen.add(item.name)


def _expressions(value: Any) -> Iterable[InputExpression]:
    if isinstance(value, InputExpression):
        yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _expressions(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _expressions(item)


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            **{
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _identity(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))


def load_generated_callable(source: str, callable_name: str) -> Callable[..., Any]:
    """Load legacy LinearAttentionEngine generated source without changing its API."""
    code_hash = hashlib.sha256(source.encode()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    file_path = os.path.join(cache_dir, f"{code_hash}.py")
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(file_path):
        temporary = f"{file_path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as output:
            output.write(source)
        os.replace(temporary, file_path)
    if cache_dir not in sys.path:
        sys.path.append(cache_dir)
    module = sys.modules.get(code_hash)
    if module is None:
        spec = importlib.util.spec_from_file_location(code_hash, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load generated stateful module {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[code_hash] = module
        spec.loader.exec_module(module)
    return getattr(module, callable_name)
