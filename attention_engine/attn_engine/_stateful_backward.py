from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


# Context entries are deliberately plain tuples so the planner can copy them into
# its serializable ContextContract without importing this runtime module.
ContextEntry = tuple[str, str, tuple[str, ...], str]


_DENSE_CONTEXT: tuple[ContextEntry, ...] = (
    ("transition.left", "runtime", ("head", "row", "feature"), "[tokens,heads,rows,inner]"),
    ("transition.right", "runtime", ("head", "feature", "column"), "[tokens,heads,inner,columns]"),
    ("transition.segment", "int32", ("token",), "[tokens]"),
    ("transition.initial_state", "runtime", ("head", "row", "column"), "[sequences,heads,rows,columns]"),
    ("transition.states", "runtime", ("token", "head", "row", "column"), "[tokens,heads,rows,columns]"),
)
_ELEMENTWISE_CONTEXT: tuple[ContextEntry, ...] = (
    ("transition.scale", "runtime", ("head", "row", "column"), "[tokens,heads,scale_rows,scale_columns]"),
    ("transition.segment", "int32", ("token",), "[tokens]"),
    ("transition.output_scale", "runtime", ("token", "head", "row", "column"), "[tokens,heads,scale_rows,scale_columns]"),
    ("transition.output_bias", "runtime", ("token", "head", "row", "column"), "[tokens,heads,rows,columns]"),
    ("transition.initial_state", "runtime", ("sequence", "head", "row", "column"), "[sequences,heads,rows,columns]"),
)


def _config_values(
    config: tuple[tuple[str, int], ...] | int,
    family: str | None = None,
) -> tuple[tuple[str, int], ...]:
    if isinstance(config, int):
        return (("block_tokens" if family == "elementwise" else "threads", config),)
    return tuple((str(name), int(value)) for name, value in config)


def _config(config: tuple[tuple[str, int], ...], name: str, default: int) -> int:
    values = dict(config)
    result = values.get(name, default)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def backward_eligibility(backward_ir: Any, family: str) -> tuple[bool, str | None]:
    """Check that a BackwardIR contains operations this executable transports.

    This is intentionally an IR check, rather than a model/profile check.  The
    current summary kernels consume every canonical VJP operation and reject a
    future operation until its executable is implemented.
    """
    if family not in {"dense", "elementwise"}:
        return False, f"unsupported backward runtime family: {family}"
    if backward_ir is None:
        return False, "missing BackwardIR"
    steps = getattr(backward_ir, "reverse_steps", None)
    if steps is None:
        return False, "BackwardIR has no reverse_steps"
    supported = {
        "axis_scale",
        "rank_one",
        "product_injection",
        "state_contraction",
        "readout",
        "head_reduce",
        "state_recurrence",
    }
    unsupported = [step.operation for step in steps if step.operation not in supported]
    if unsupported:
        return False, f"unsupported BackwardIR operations: {tuple(unsupported)!r}"
    return True, None


@dataclass(frozen=True)
class BackwardBinding:
    """Compiler-selected VJP executable used by a generated autograd wrapper.

    ``saved_context`` is a named mapping.  ``__call__`` returns gradients in
    the forward wrapper's positional input order; the segment gradient is None.
    The binding owns its kernel configuration and never reads forward state.
    """

    backward_ir: Any
    family: str
    config: tuple[tuple[str, int], ...]
    context_values: tuple[ContextEntry, ...]

    def __call__(
        self,
        saved_context: Mapping[str, torch.Tensor],
        grad_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        missing = [name for name, *_ in self.context_values if name not in saved_context]
        if missing:
            raise RuntimeError(f"compiler backward context missing: {tuple(missing)!r}")
        if self.family == "dense":
            return _dense_vjp(saved_context, grad_states, self.config)
        if self.family == "elementwise":
            return _elementwise_vjp(saved_context, grad_states, self.config)
        raise RuntimeError(f"unknown backward runtime family: {self.family}")



def default_backward(family: str, config: tuple[tuple[str, int], ...] | int = ()) -> BackwardBinding:
    """Construct the direct-prefix binding when no compiler plan is supplied."""
    class _DirectBackwardIR:
        reverse_steps = ()

    return BackwardBinding(
        _DirectBackwardIR(), family, _config_values(config, family),
        _DENSE_CONTEXT if family == "dense" else _ELEMENTWISE_CONTEXT,
    )
def bind_backward(
    backward_ir: Any,
    family: str,
    config: tuple[tuple[str, int], ...] | int = (),
) -> BackwardBinding:
    """Lower a BackwardIR to a callable VJP binding with fixed configuration."""
    eligible, reason = backward_eligibility(backward_ir, family)
    if not eligible:
        raise ValueError(reason)
    values = _config_values(config)
    if family == "dense":
        _config(values, "threads", 128)
        context_values = _DENSE_CONTEXT
    else:
        _config(values, "block_tokens", 128)
        context_values = _ELEMENTWISE_CONTEXT
    return BackwardBinding(backward_ir, family, values, context_values)


def _dense_vjp(
    context: Mapping[str, torch.Tensor],
    grad_states: torch.Tensor | None,
    config: tuple[tuple[str, int], ...],
) -> tuple[torch.Tensor | None, ...]:
    from ._stateful_affine import _tilelang_dense_affine_scan

    left = context["transition.left"]
    right = context["transition.right"]
    segment = context["transition.segment"]
    initial_state = context["transition.initial_state"]
    states = context["transition.states"]
    grad_states = torch.zeros_like(states) if grad_states is None else grad_states
    same_next = (segment[:-1] == segment[1:]).view(-1, 1, 1, 1)
    continuation_left = torch.zeros_like(left)
    continuation_right = torch.zeros_like(right)
    continuation_left[:-1] = torch.where(
        same_next, left[1:].transpose(-1, -2), continuation_left[:-1]
    )
    continuation_right[:-1] = torch.where(
        same_next, right[1:].transpose(-1, -2), continuation_right[:-1]
    )
    reverse_segment = segment.flip(0).contiguous()
    _, _, adjoint = _tilelang_dense_affine_scan(
        continuation_left.flip(0).contiguous(),
        continuation_right.flip(0).contiguous(),
        grad_states.flip(0).contiguous(),
        reverse_segment,
        threads=_config(config, "threads", 128),
    )
    adjoint = adjoint.flip(0)

    previous_state = initial_state[segment.long()].clone()
    previous_state[1:] = torch.where(same_next, states[:-1], previous_state[1:])
    grad_left = adjoint @ right.transpose(-1, -2) @ previous_state.transpose(-1, -2)
    grad_right = previous_state.transpose(-1, -2) @ left.transpose(-1, -2) @ adjoint
    grad_bias = adjoint
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
    if start_segments.numel():
        grad_initial.index_add_(0, start_segments, initial_adjoint)
    return grad_left, grad_right, grad_bias, None, grad_initial


def _elementwise_vjp(
    context: Mapping[str, torch.Tensor],
    grad_states: torch.Tensor | None,
    config: tuple[tuple[str, int], ...],
) -> tuple[torch.Tensor | None, ...]:
    from ._stateful_elementwise import _tilelang_segmented_elementwise_scan

    scale = context["transition.scale"]
    segment = context["transition.segment"]
    output_scale = context["transition.output_scale"]
    output_bias = context["transition.output_bias"]
    initial_state = context["transition.initial_state"]
    grad_states = torch.zeros_like(output_bias) if grad_states is None else grad_states
    block_tokens = _config(config, "block_tokens", 128)
    grad_output_scale = grad_states * initial_state[segment.long()]
    continuation = torch.zeros_like(output_scale)
    continuation[:-1] = torch.where(
        (segment[:-1] == segment[1:]).view(-1, 1, 1, 1),
        scale[1:].expand_as(output_scale[1:]),
        torch.zeros_like(output_scale[1:]),
    )
    # The reverse pass walks the tokens from the end inside the kernel, so no
    # flipped copies of these token-sized tensors are needed.
    adjoint_scale = _tilelang_segmented_elementwise_scan(
        continuation,
        grad_output_scale,
        segment,
        block_tokens=block_tokens,
        reverse=True,
    )[1]
    adjoint_bias = _tilelang_segmented_elementwise_scan(
        continuation,
        grad_states,
        segment,
        block_tokens=block_tokens,
        reverse=True,
    )[1]
    previous_scale = torch.ones_like(output_scale)
    previous_bias = torch.zeros_like(output_bias)
    same_segment = (segment[1:] == segment[:-1]).view(-1, 1, 1, 1)
    previous_scale[1:] = torch.where(same_segment, output_scale[:-1], previous_scale[1:])
    previous_bias[1:] = torch.where(same_segment, output_bias[:-1], previous_bias[1:])
    scale_gradient = _sum_to_shape(
        adjoint_scale * previous_scale + adjoint_bias * previous_bias, scale.shape
    )
    state_gradient = torch.zeros_like(initial_state)
    if segment.numel():
        state_gradient.index_add_(0, segment.long(), grad_states * output_scale)
    return scale_gradient, adjoint_bias, None, state_gradient


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected == 1 and actual != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value
