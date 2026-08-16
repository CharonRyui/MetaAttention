from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
import warnings

import torch

from .stateful_operator import (
    AlgorithmIR,
    CompileOptions,
    ElementwiseScale,
    Exp,
    HeadMapping,
    Input,
    MatrixReadout,
    OuterProduct,
    PropagationComposition,
    RankOneDelta,
    StateSpec,
    StateTransition,
    StatefulOperator,
    StateTuple,
    TensorInput,
)


CHUNK_SIZE = 64
HEAD_DIM = 128
DEFAULT_SCALE = HEAD_DIM**-0.5


def _validate_tensor(name: str, tensor: torch.Tensor, *, ndim: int, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_gdn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    check_device: bool = True,
) -> tuple[int, int, int, int, float]:
    """Validate the first-release H20 GDN public contract."""
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _validate_tensor(name, tensor, ndim=4, dtype=torch.bfloat16)
    for name, tensor in (("gate", gate), ("beta", beta)):
        _validate_tensor(name, tensor, ndim=3, dtype=torch.float32)

    batch, query_heads, length, query_dim = query.shape
    key_batch, key_heads, key_length, key_dim = key.shape
    value_batch, value_heads, value_length, value_dim = value.shape
    if query_heads <= 0 or key_heads <= 0 or value_heads <= 0:
        raise ValueError("head counts must be positive")
    if query_heads != key_heads:
        raise ValueError("query and key head counts must match")
    if value_heads % query_heads:
        raise ValueError("Hv must be divisible by Hk")
    if query_dim != HEAD_DIM or key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("query, key, and value head dimensions must be 128")
    if batch != key_batch or batch != value_batch:
        raise ValueError("query, key, and value batch dimensions must match")
    if length != key_length or length != value_length:
        raise ValueError("query, key, and value sequence lengths must match")
    if length <= 0 or length % CHUNK_SIZE:
        raise ValueError("sequence length must be a positive multiple of 64")
    if gate.shape != (batch, value_heads, length):
        raise ValueError("gate must have shape [B, Hv, T]")
    if beta.shape != (batch, value_heads, length):
        raise ValueError("beta must have shape [B, Hv, T]")

    tensors = (query, key, value, gate, beta)
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("all GDN inputs must be on the same device")
    if initial_state is not None:
        _validate_tensor("initial_state", initial_state, ndim=4, dtype=torch.float32)
        expected_state = (batch, value_heads, HEAD_DIM, HEAD_DIM)
        if initial_state.shape != expected_state:
            raise ValueError(f"initial_state must have shape {expected_state}")
        if initial_state.device != query.device:
            raise ValueError("initial_state must be on the same device as query")

    if scale is None:
        resolved_scale = DEFAULT_SCALE
    elif not isinstance(scale, float):
        raise TypeError("scale must be a Python float")
    else:
        resolved_scale = scale

    if check_device:
        if query.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("GDN Engine requires an NVIDIA CUDA H20/sm89 device")
        capability = torch.cuda.get_device_capability(query.device)
        name = torch.cuda.get_device_name(query.device)
        if "H20" not in name.upper():
            raise RuntimeError(
                f"GDN Engine requires NVIDIA H20/sm89; found {name} with capability {capability}"
            )

    return batch, query_heads, value_heads, length, resolved_scale


def _build_gdn_operator(*, scale: float | None = None) -> StatefulOperator:
    resolved_scale = DEFAULT_SCALE if scale is None else scale
    if not isinstance(resolved_scale, float):
        raise TypeError("scale must be a Python float")
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), torch.bfloat16),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), torch.bfloat16),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), torch.bfloat16),
            TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
            TensorInput("beta", ("batch", "state_heads", "sequence"), torch.float32),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            PropagationComposition(
                (ElementwiseScale(Exp(Input("gate"))), RankOneDelta(Input("key"), Input("beta")))
            ),
            OuterProduct(Input("key"), Input("value") * Input("beta")),
        ),
        readout=MatrixReadout("memory", Input("query") * resolved_scale),
        head_mapping=HeadMapping("query_heads", "key_heads", "value_heads", "state_heads"),
    )
    return StatefulOperator(algorithm, compile_options=CompileOptions())


@dataclass(frozen=True)
class GDNEngine:
    """Deprecated positional adapter for the IR-first Gated Delta Rule operator."""

    device: torch.device | str | int | None = None
    _operators: dict[float, StatefulOperator] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        elif isinstance(self.device, Integral):
            device = torch.device("cuda", int(self.device))
        else:
            device = torch.device(self.device)
        object.__setattr__(self, "device", device)
        warnings.warn(
            "GDNEngine is deprecated; construct Algorithm IR and StatefulOperator with named inputs instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if device.type != "cuda" or torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("GDN Engine requires an NVIDIA CUDA H20/sm89 device")
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        if "H20" not in name.upper():
            raise RuntimeError(
                f"GDN Engine requires NVIDIA H20/sm89; found {name} with capability {capability}"
            )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _, _, _, _, resolved_scale = validate_gdn_inputs(
            query, key, value, gate, beta, scale=scale, initial_state=initial_state
        )
        operator = self._operators.get(resolved_scale)
        if operator is None:
            operator = _build_gdn_operator(scale=resolved_scale)
            self._operators[resolved_scale] = operator
        state = None if initial_state is None else StateTuple(("memory",), (initial_state,))
        result = operator(
            query=query,
            key=key,
            value=value,
            gate=gate,
            beta=beta,
            initial_state=state,
            return_final_state=output_final_state,
        )
        if output_final_state:
            output, final_state = result
            return output, final_state["memory"]
        return result
