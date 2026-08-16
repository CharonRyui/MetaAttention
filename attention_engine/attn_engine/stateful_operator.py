from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Iterable

import torch

from core import CustomIO, meta_tensor


_ROLE_ALIASES = {
    "one": "1",
    "batch": "batch",
    "query_heads": "heads",
    "key_heads": "heads",
    "value_heads": "heads",
    "state_heads": "heads",
    "sequence": "seq_len",
    "key_dim": "dimqk",
    "value_dim": "dimv",
}


class InputExpression:
    def __mul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(self, as_expression(other))

    def __rmul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(as_expression(other), self)


@dataclass(frozen=True)
class Input(InputExpression):
    name: str


@dataclass(frozen=True)
class Constant(InputExpression):
    value: float


@dataclass(frozen=True)
class Multiply(InputExpression):
    left: InputExpression
    right: InputExpression


@dataclass(frozen=True)
class Exp(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Log(InputExpression):
    operand: InputExpression


def as_expression(value: InputExpression | float | int) -> InputExpression:
    if isinstance(value, InputExpression):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(f"expected an input expression or numeric constant, got {type(value).__name__}")


@dataclass(frozen=True)
class TensorInput:
    name: str
    roles: tuple[str, ...]
    dtype: torch.dtype
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid tensor input name {self.name!r}")
        if not self.roles:
            raise ValueError(f"tensor input '{self.name}' must declare shape roles")
        unknown = set(self.roles) - set(_ROLE_ALIASES)
        if unknown:
            raise ValueError(f"tensor input '{self.name}' has unknown shape role {sorted(unknown)[0]!r}")


@dataclass(frozen=True)
class StateSpec:
    name: str
    roles: tuple[str, ...]
    storage_dtype: torch.dtype = torch.float32
    accumulation_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class Identity:
    pass


@dataclass(frozen=True)
class ElementwiseScale:
    scale: InputExpression


@dataclass(frozen=True)
class DiagonalScale:
    scale: InputExpression


@dataclass(frozen=True)
class RankOneDelta:
    key: InputExpression
    beta: InputExpression




@dataclass(frozen=True)
class PropagationComposition:
    nodes: tuple[Identity | ElementwiseScale | DiagonalScale | RankOneDelta, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("propagation composition must contain at least one node")
Propagation = Identity | ElementwiseScale | DiagonalScale | RankOneDelta | PropagationComposition


@dataclass(frozen=True)
class OuterProduct:
    left: InputExpression
    right: InputExpression


@dataclass(frozen=True)
class ElementwiseProduct:
    left: InputExpression
    right: InputExpression


@dataclass(frozen=True)
class DirectInput:
    value: InputExpression


@dataclass(frozen=True)
class ZeroInjection:
    pass


Injection = OuterProduct | ElementwiseProduct | DirectInput | ZeroInjection


@dataclass(frozen=True)
class MatrixReadout:
    state: str
    query: InputExpression


@dataclass(frozen=True)
class ElementwiseReadout:
    state: str
    value: InputExpression


Readout = MatrixReadout | ElementwiseReadout


@dataclass(frozen=True)
class StateTransition:
    state: str
    propagation: Propagation
    injection: Injection


@dataclass(frozen=True)
class HeadMapping:
    query: str
    key: str
    value: str
    state: str

    def validate(self, role_sizes: dict[str, int]) -> None:
        for role in (self.query, self.key, self.value, self.state):
            if role not in role_sizes:
                raise ValueError(f"head mapping references unavailable role '{role}'")
            if role_sizes[role] <= 0:
                raise ValueError(f"head role '{role}' must be positive")
        query_heads = role_sizes[self.query]
        key_heads = role_sizes[self.key]
        value_heads = role_sizes[self.value]
        state_heads = role_sizes[self.state]
        if query_heads != key_heads:
            raise ValueError("query heads must equal key heads")
        if value_heads != state_heads:
            raise ValueError("value heads must equal state heads")
        if state_heads % query_heads:
            raise ValueError(
                f"state heads ({state_heads}) must be divisible by query heads ({query_heads})"
            )


@dataclass(frozen=True)
class AlgorithmIR:
    inputs: tuple[TensorInput, ...]
    states: tuple[StateSpec, ...]
    transition: StateTransition
    readout: Readout
    head_mapping: HeadMapping

    def __post_init__(self) -> None:
        _reject_duplicate_names(self.inputs, "tensor input")
        _reject_duplicate_names(self.states, "state")
        if not self.states:
            raise ValueError("Algorithm IR requires at least one state")
        state_names = {state.name for state in self.states}
        if self.transition.state not in state_names:
            raise ValueError(f"transition references unknown state '{self.transition.state}'")
        if self.readout.state not in state_names:
            raise ValueError(f"readout references unknown state '{self.readout.state}'")
        input_names = {tensor.name for tensor in self.inputs}
        for expression in _expressions(self):
            if isinstance(expression, Input) and expression.name not in input_names:
                raise ValueError(f"expression references unknown input '{expression.name}'")

    @property
    def structural_identity(self) -> str:
        payload = json.dumps(_canonical(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _reject_duplicate_names(items: Iterable[Any], kind: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.name in seen:
            raise ValueError(f"duplicate {kind} name '{item.name}'")
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
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            **{field.name: _canonical(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class CompileOptions:
    backend: str = "tilelang"
    tune: bool = False
    tune_filename: str = "tune_result"
    tune_backward: bool = False


@dataclass(frozen=True)
class StateTuple:
    names: tuple[str, ...]
    values: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("state names and values must have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("state names must be unique")

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self.values[self.names.index(name)]
        except ValueError as error:
            raise KeyError(name) from error

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(zip(self.names, self.values))


class StatefulOperator:
    def __init__(
        self,
        algorithm: AlgorithmIR,
        *,
        compile_options: CompileOptions | None = None,
        normalization: str | None = None,
    ) -> None:
        if normalization is not None:
            raise ValueError("normalization is not supported by StatefulOperator")
        self.algorithm = algorithm
        self.compile_options = compile_options or CompileOptions()
        if self.compile_options.backend != "tilelang":
            raise ValueError(f"unsupported stateful backend '{self.compile_options.backend}'")
        self._compiled: dict[str, Callable[..., Any]] = {}

    @property
    def structural_identity(self) -> str:
        payload = (self.algorithm.structural_identity, _canonical(self.compile_options))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def __call__(
        self,
        *,
        initial_state: StateTuple | dict[str, torch.Tensor] | None = None,
        return_final_state: bool = False,
        **inputs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, StateTuple]:
        expected = {item.name for item in self.algorithm.inputs}
        missing = expected - inputs.keys()
        unknown = inputs.keys() - expected
        if missing:
            raise TypeError(f"missing required input: {sorted(missing)[0]}")
        if unknown:
            raise TypeError(f"unknown input: {sorted(unknown)[0]}")
        role_sizes = self._validate_inputs(inputs)
        self.algorithm.head_mapping.validate(role_sizes)
        state = self._validate_state(initial_state, role_sizes, inputs)
        key = self._specialization_key(inputs) + (":stateful" if (state is not None or return_final_state) else ":stateless")
        lowering = self._compiled.get(key)
        if lowering is None:
            if (state is not None or return_final_state) and not _contains_rank_one_delta(
                self.algorithm.transition.propagation
            ):
                lowering = _compile_linear_stateful_lowering(self.algorithm)
            else:
                lowering = self._compile(inputs)
            self._compiled[key] = lowering
        result = lowering(inputs, state, return_final_state)
        if return_final_state:
            output, state_values = result
            return output, StateTuple(tuple(item.name for item in self.algorithm.states), state_values)
        return result

    def _validate_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, int]:
        role_sizes: dict[str, int] = {}
        device: torch.device | None = None
        for spec in self.algorithm.inputs:
            tensor = inputs[spec.name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"input '{spec.name}' must be a torch.Tensor")
            if tensor.ndim != len(spec.roles):
                raise ValueError(f"input '{spec.name}' must have {len(spec.roles)} dimensions")
            if tensor.dtype != spec.dtype:
                raise ValueError(f"input '{spec.name}' must have dtype {spec.dtype}, got {tensor.dtype}")
            if not tensor.is_contiguous():
                raise ValueError(f"input '{spec.name}' must be contiguous")
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise ValueError("all tensor inputs must be on the same device")
            for role, size in zip(spec.roles, tensor.shape):
                prior = role_sizes.setdefault(role, size)
                if prior != size:
                    raise ValueError(f"shape role '{role}' has conflicting sizes {prior} and {size}")
        return role_sizes

    def _validate_state(
        self,
        state: StateTuple | dict[str, torch.Tensor] | None,
        role_sizes: dict[str, int],
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...] | None:
        if state is None:
            return None
        values = state.as_dict() if isinstance(state, StateTuple) else state
        expected_names = {item.name for item in self.algorithm.states}
        if set(values) != expected_names:
            missing = expected_names - values.keys()
            unknown = values.keys() - expected_names
            detail = f"missing {sorted(missing)}" if missing else f"unknown {sorted(unknown)}"
            raise ValueError(f"invalid initial state names: {detail}")
        device = next(iter(inputs.values())).device
        result: list[torch.Tensor] = []
        for spec in self.algorithm.states:
            tensor = values[spec.name]
            expected_shape = tuple(role_sizes[role] for role in spec.roles)
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"initial state '{spec.name}' must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != spec.storage_dtype:
                raise ValueError(f"initial state '{spec.name}' must have dtype {spec.storage_dtype}")
            if tensor.device != device:
                raise ValueError(f"initial state '{spec.name}' must be on device {device}")
            if not tensor.is_contiguous():
                raise ValueError(f"initial state '{spec.name}' must be contiguous")
            result.append(tensor)
        return tuple(result)

    def _specialization_key(self, inputs: dict[str, torch.Tensor]) -> str:
        metadata = [
            (name, tuple(tensor.shape), str(tensor.dtype), tensor.device.type)
            for name, tensor in sorted(inputs.items())
        ]
        return hashlib.sha256(
            json.dumps((self.structural_identity, metadata), sort_keys=True).encode()
        ).hexdigest()

    def _compile(self, inputs: dict[str, torch.Tensor]) -> Callable[..., Any]:
        if _contains_rank_one_delta(self.algorithm.transition.propagation):
            return _compile_gdn_lowering(self.algorithm)
        return _compile_linear_lowering(self.algorithm, inputs, self.compile_options)


def load_generated_callable(source: str, callable_name: str) -> Callable[..., Any]:
    code_hash = hashlib.sha256(source.encode()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    file_path = os.path.join(cache_dir, f"{code_hash}.py")
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as output:
            output.write(source)
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


def _contains_rank_one_delta(propagation: Propagation) -> bool:
    if isinstance(propagation, RankOneDelta):
        return True
    if isinstance(propagation, PropagationComposition):
        return any(isinstance(node, RankOneDelta) for node in propagation.nodes)
    return False


def _compile_gdn_lowering(algorithm: AlgorithmIR) -> Callable[..., Any]:
    propagation = algorithm.transition.propagation
    if not isinstance(propagation, PropagationComposition) or len(propagation.nodes) != 2:
        raise ValueError("rank-one delta lowering requires scale then rank-one propagation")
    scale_node, delta_node = propagation.nodes
    if not isinstance(scale_node, ElementwiseScale) or not isinstance(delta_node, RankOneDelta):
        raise ValueError("rank-one delta lowering requires ordered scale then rank-one propagation")
    if not isinstance(algorithm.transition.injection, OuterProduct):
        raise ValueError("rank-one delta lowering requires outer-product injection")
    if not isinstance(algorithm.readout, MatrixReadout):
        raise ValueError("rank-one delta lowering requires matrix readout")
    gate = _match_exp_input(scale_node.scale)
    key = _match_input(delta_node.key)
    beta = _match_input(delta_node.beta)
    injection_key = _match_input(algorithm.transition.injection.left)
    value, injection_beta = _match_product_inputs(algorithm.transition.injection.right)
    query, scale = _match_scaled_input(algorithm.readout.query)
    if key != injection_key or beta != injection_beta:
        raise ValueError("rank-one delta key and beta must match the injection operands")

    def lowering(bound, initial_state, return_final_state):
        from .gdn_engine import validate_gdn_inputs
        from .gdn_tilelang import gated_delta_rule

        state = initial_state[0] if initial_state is not None else None
        validate_gdn_inputs(
            bound[query], bound[key], bound[value], bound[gate], bound[beta],
            scale=scale, initial_state=state,
        )
        output, final = gated_delta_rule(
            bound[query], bound[key], bound[value], bound[gate], bound[beta],
            scale=scale, initial_state=state, output_final_state=return_final_state,
        )
        return (output, (final,)) if return_final_state else output

    return lowering

def _compile_linear_lowering(
    algorithm: AlgorithmIR,
    inputs: dict[str, torch.Tensor],
    options: CompileOptions,
) -> Callable[..., Any]:
    if len(algorithm.states) != 1:
        raise ValueError("linear lowering supports exactly one matrix state")
    if not isinstance(algorithm.transition.injection, OuterProduct):
        raise ValueError("linear lowering supports only outer-product injection")
    if not isinstance(algorithm.readout, MatrixReadout):
        raise ValueError("linear lowering supports only matrix readout")
    if isinstance(algorithm.transition.propagation, (PropagationComposition, RankOneDelta, DiagonalScale)):
        raise ValueError("linear lowering does not support this propagation structure")

    query_name = _single_input_name(algorithm.readout.query, "readout query")
    key_name = _single_input_name(algorithm.transition.injection.left, "injection key")
    value_name = _first_input_name(algorithm.transition.injection.right)
    primary = {query_name, key_name, value_name}
    propagation = algorithm.transition.propagation
    if isinstance(propagation, Identity):
        decay_name = None
        decay_mod = None
    elif isinstance(propagation, ElementwiseScale):
        decay_name = _first_input_name(propagation.scale)
        primary.add(decay_name)
        decay_mod = _expression_callback(_logarithm_for_linear_backend(propagation.scale), decay_name)
    else:
        raise ValueError("unsupported linear propagation")

    custom_specs = [spec for spec in algorithm.inputs if spec.name not in primary]
    custom_io = CustomIO(
        {spec.name: tuple(_ROLE_ALIASES[role] for role in spec.roles) for spec in custom_specs}
    )
    q_mod = _expression_callback(algorithm.readout.query, query_name)
    k_mod = _expression_callback(algorithm.transition.injection.left, key_name)
    v_mod = _expression_callback(algorithm.transition.injection.right, value_name)
    query = inputs[query_name]
    key = inputs[key_name]
    value = inputs[value_name]
    qkv_meta = tuple(
        meta_tensor(*tensor.shape, dtype=tensor.dtype) for tensor in (query, key, value)
    )
    from core.lower.lower_linear import lower_tl

    source = lower_tl(
        qkv_meta,
        q_mod,
        k_mod,
        v_mod,
        decay_mod,
        custom_io,
        tune=options.tune,
        tune_filename=options.tune_filename,
        tune_bwd=options.tune_backward,
    )
    callable_ = load_generated_callable(source, "linear_attention")
    ordered_names = [query_name, key_name, value_name]
    if decay_name is not None:
        ordered_names.append(decay_name)
    ordered_names.extend(spec.name for spec in custom_specs)

    def lowering(bound, initial_state, return_final_state):
        return callable_(*(bound[name] for name in ordered_names))

    return lowering


def _compile_linear_stateful_lowering(algorithm: AlgorithmIR) -> Callable[..., Any]:
    if len(algorithm.states) != 1:
        raise ValueError("linear stateful lowering supports exactly one state")
    if not isinstance(algorithm.transition.injection, OuterProduct):
        raise ValueError("linear stateful lowering supports only outer-product injection")
    if not isinstance(algorithm.readout, MatrixReadout):
        raise ValueError("linear stateful lowering supports only matrix readout")
    state_name = algorithm.transition.state
    query = _single_input_name(algorithm.readout.query, "readout query")

    def lowering(bound, initial_state, return_final_state):
        state = initial_state[0] if initial_state is not None else None
        output, final = _run_linear_stateful(
            algorithm, bound, state, query, state_name
        )
        return (output, (final,)) if return_final_state else output

    return lowering

def _run_linear_stateful(algorithm, bound, state, query_name, state_name):
    state_spec = next(spec for spec in algorithm.states if spec.name == state_name)
    query_value = _evaluate_runtime_expression(algorithm.readout.query, bound)
    injection = algorithm.transition.injection
    left = _evaluate_runtime_expression(injection.left, bound)
    right = _evaluate_runtime_expression(injection.right, bound)
    propagation = algorithm.transition.propagation
    if isinstance(propagation, Identity):
        scale = None
    elif isinstance(propagation, ElementwiseScale):
        scale = _evaluate_runtime_expression(propagation.scale, bound)
    else:
        raise ValueError("unsupported linear stateful propagation")

    batch, _, length, _ = query_value.shape
    state_heads = left.shape[1]
    if state is None:
        state = torch.zeros(
            batch,
            state_heads,
            left.shape[-1],
            right.shape[-1],
            device=query_value.device,
            dtype=state_spec.accumulation_dtype,
        )
    outputs = []
    for token in range(length):
        token_scale = 1 if scale is None else scale[..., token]
        current = state * token_scale[..., None, None]
        current = current + torch.einsum(
            "bhk,bhv->bhkv", left[..., token, :].float(), right[..., token, :].float()
        )
        query_token = query_value[..., token, :].float()
        outputs.append(torch.einsum("bhk,bhkv->bhv", query_token, current).to(query_value.dtype))
        state = current
    return torch.stack(outputs, dim=2), state




def _evaluate_runtime_expression(expression, bound):
    if isinstance(expression, Input):
        return bound[expression.name]
    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Multiply):
        return _evaluate_runtime_expression(expression.left, bound) * _evaluate_runtime_expression(expression.right, bound)
    if isinstance(expression, Exp):
        return _evaluate_runtime_expression(expression.operand, bound).exp()
    if isinstance(expression, Log):
        return _evaluate_runtime_expression(expression.operand, bound).log()
    raise ValueError(f"unsupported runtime input expression {type(expression).__name__}")


def _expression_callback(expression: InputExpression, root_name: str) -> Callable[..., Any] | None:
    if expression == Input(root_name):
        return None

    def callback(root, custom_io):
        return _evaluate_symbolic(expression, root_name, root, custom_io)

    return callback


def _evaluate_symbolic(expression, root_name, root, custom_io):
    if isinstance(expression, Input):
        if expression.name == root_name:
            return root
        return custom_io.input_tensors[expression.name]
    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Multiply):
        return _evaluate_symbolic(expression.left, root_name, root, custom_io) * _evaluate_symbolic(
            expression.right, root_name, root, custom_io
        )
    if isinstance(expression, Exp):
        return _evaluate_symbolic(expression.operand, root_name, root, custom_io).exp()
    if isinstance(expression, Log):
        return _evaluate_symbolic(expression.operand, root_name, root, custom_io).log()
    raise ValueError(f"unsupported input expression {type(expression).__name__}")


def _logarithm_for_linear_backend(expression: InputExpression) -> InputExpression:
    if isinstance(expression, Exp):
        return expression.operand
    return Log(expression)


def _single_input_name(expression: InputExpression, description: str) -> str:
    names = {item.name for item in _expressions(expression) if isinstance(item, Input)}
    if len(names) != 1:
        raise ValueError(f"{description} must derive from exactly one tensor input")
    return next(iter(names))


def _first_input_name(expression: InputExpression) -> str:
    names = [item.name for item in _expressions(expression) if isinstance(item, Input)]
    if not names:
        raise ValueError("propagation scale must reference a tensor input")
    return names[0]


def _match_input(expression: InputExpression) -> str:
    if not isinstance(expression, Input):
        raise ValueError("expected a direct tensor input")
    return expression.name


def _match_exp_input(expression: InputExpression) -> str:
    if not isinstance(expression, Exp):
        raise ValueError("GDN scale propagation must explicitly apply Exp")
    return _match_input(expression.operand)


def _match_product_inputs(expression: InputExpression) -> tuple[str, str]:
    if not isinstance(expression, Multiply):
        raise ValueError("GDN injection value must be beta-scaled")
    return _match_input(expression.left), _match_input(expression.right)


def _match_scaled_input(expression: InputExpression) -> tuple[str, float]:
    if isinstance(expression, Input):
        return expression.name, 1.0
    if not isinstance(expression, Multiply):
        raise ValueError("GDN readout query must be a scaled tensor input")
    if isinstance(expression.left, Input) and isinstance(expression.right, Constant):
        return expression.left.name, expression.right.value
    if isinstance(expression.right, Input) and isinstance(expression.left, Constant):
        return expression.right.name, expression.left.value
    raise ValueError("GDN readout scale must be a Python numeric constant")
