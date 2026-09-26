"""Runtime implementations for structurally justified stateful plans.

The functions in this module deliberately operate on typed IR objects supplied by
``stateful_operator``.  They do not identify models or profiles; eligibility is
only a property of the authored role structure.  The operator remains the owner
of segment handling and output layout while this module owns the small algebraic
kernels used by each structured runtime family.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch


Eligibility = tuple[bool, str | None]


def axis_scale_eligibility(algorithm: Any) -> Eligibility:
    """Return whether every propagation is an AxisScale/Identity node.

    Axis scales are applied in their authored order.  A full-feature scale and
    scalar scale are elementwise; a one-feature scale is represented by a
    diagonal transform, including for non-square State features.
    """
    propagations = tuple(
        node for transition in algorithm.transitions for node in transition.propagations
    )
    if not propagations:
        return False, "no-axis-scale-propagation"
    if not all(type(node).__name__ == "AxisScale" for node in propagations):
        return False, "propagation-sequence-contains-non-axis-scale"
    return True, None


def product_injection_eligibility(algorithm: Any) -> Eligibility:
    """Return whether injections have explicit role-compatible GEMM factors."""
    injections = tuple(
        injection
        for transition in algorithm.transitions
        for injection in transition.injections
    )
    if not injections:
        return False, "no-product-injection"
    features = {
        role for state in algorithm.states for role in state.feature_roles
    }
    for index, injection in enumerate(injections):
        roles = [tuple(factor.roles) for factor in injection.factors]
        if any(set(item) - features for item in roles):
            return False, f"injection-{index}-unknown-feature-role"
        # A full-state factor is already a GEMM operand.  Multiple factors are
        # valid, but only disjoint single-feature factors get the outer-product
        # fast path.
        if len(injection.factors) == 0:
            return False, f"injection-{index}-empty"
    return True, None


def rank_one_eligibility(algorithm: Any) -> Eligibility:
    """Return whether ordered RankOne propagations can be factorized."""
    propagations = tuple(
        node for transition in algorithm.transitions for node in transition.propagations
    )
    if not propagations or not any(type(node).__name__ == "RankOnePropagation" for node in propagations):
        return False, "no-rank-one-propagation"
    if not all(type(node).__name__ in {"RankOnePropagation", "AxisScale"} for node in propagations):
        return False, "propagation-sequence-contains-unsupported-node"
    return True, None


def shared_state_contraction_eligibility(algorithm: Any) -> Eligibility:
    """Return whether at least two independent readouts share one State."""
    from .stateful_operator import StateContraction

    readouts = [readout for readout in algorithm.readouts if isinstance(readout, StateContraction)]
    if len(readouts) != len(algorithm.readouts):
        return False, "readout-expressions-are-not-state-contractions"
    if len(readouts) < 2:
        return False, "fewer-than-two-readouts"
    states = {readout.state for readout in readouts}
    if len(states) != 1:
        return False, "readouts-observe-different-states"
    return True, None


def factored_recurrence_eligibility(algorithm: Any) -> Eligibility:
    """Return whether the factored recurrent kernel can replay the Program."""
    from ._stateful_analysis import rank_one_factored_eligible

    if rank_one_factored_eligible(algorithm) is None:
        return False, "not-a-factored-recurrence"
    return True, None


RUNTIME_FAMILY_ELIGIBILITY: dict[str, Callable[[Any], Eligibility]] = {
    "factored-recurrence": factored_recurrence_eligibility,
    "axis-scale": axis_scale_eligibility,
    "product-injection-gemm": product_injection_eligibility,
    "rank-one-factorized": rank_one_eligibility,
    "shared-state-contraction": shared_state_contraction_eligibility,
}


def _axis_scale_summary(
    propagations: Sequence[Any],
    *,
    evaluate: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
    values: Mapping[str, torch.Tensor],
    state_spec: Any,
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
    dense: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build authored AxisScale transforms without reassociation.

    A dense request materializes one diagonal matrix per token, which is only
    needed when a propagation on the same axis is not diagonal. A pure diagonal
    transform stays an elementwise factor of shape ``[tokens, heads, size, 1]``
    for a first-feature scale and ``[tokens, heads, 1, size]`` for a
    second-feature scale, so the prefix kernel never materializes ``[T,H,K,K]``.
    """
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None
    elementwise: torch.Tensor | None = (
        torch.ones(token_count, heads, 1, 1, device=device) if not dense else None
    )
    identity_left = (
        torch.eye(left_size, device=device)
        .expand(token_count, heads, left_size, left_size)
        .contiguous()
        if dense
        else None
    )
    identity_right = (
        torch.eye(right_size, device=device)
        .expand(token_count, heads, right_size, right_size)
        .contiguous()
        if dense
        else None
    )
    if dense:
        left, right = identity_left, identity_right
    elementwise_roles: set[Any] = set()
    for node in propagations:
        if type(node).__name__ != "AxisScale":
            raise ValueError("axis-scale runtime received an unsupported propagation")
        factor = evaluate(node.factor, values).float()
        if len(node.roles) == 2 or not node.roles:
            if dense:
                assert left is not None
                left = factor * left
            else:
                elementwise = factor if elementwise is None else elementwise * factor
        elif node.roles[0] == state_spec.feature_roles[0]:
            if dense:
                assert left is not None
                left = torch.diag_embed(factor[..., :, 0]) @ left
            else:
                elementwise_roles.add(node.roles[0])
                elementwise = factor if elementwise is None else elementwise * factor
        elif node.roles[0] == state_spec.feature_roles[1]:
            if dense:
                assert right is not None
                right = right @ torch.diag_embed(factor[..., 0, :])
            else:
                elementwise_roles.add(node.roles[0])
                elementwise = factor if elementwise is None else elementwise * factor
        else:
            raise ValueError("AxisScale role is not a State Feature role")
        if len(elementwise_roles) > 1:
            raise ValueError(
                "two single-feature AxisScales compose both axes: dense transform required"
            )
    return left, right, elementwise


def rank_one_summary(
    propagations: Sequence[Any],
    *,
    evaluate: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
    feature_vector: Callable[[torch.Tensor, Any, Any], torch.Tensor],
    values: Mapping[str, torch.Tensor],
    state_spec: Any,
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compose ordered rank-one updates and AxisScales as dense affine factors."""
    left = torch.eye(left_size, device=device).expand(token_count, heads, left_size, left_size).contiguous()
    right = torch.eye(right_size, device=device).expand(token_count, heads, right_size, right_size).contiguous()
    elementwise: torch.Tensor | None = None
    for node in propagations:
        kind = type(node).__name__
        if kind == "AxisScale":
            factor = evaluate(node.factor, values).float()
            if len(node.roles) == 2 or not node.roles:
                # A full-feature scale is equivalent to a left multiplication;
                # retaining it as a dense diagonal preserves authored order.
                left = factor * left
            elif node.roles[0] == state_spec.feature_roles[0]:
                left = torch.diag_embed(factor[..., :, 0]) @ left
            elif node.roles[0] == state_spec.feature_roles[1]:
                right = right @ torch.diag_embed(factor[..., 0, :])
            else:
                raise ValueError("AxisScale role is not a State Feature role")
        elif kind == "RankOnePropagation":
            left_vector = feature_vector(evaluate(node.left, values), node.role, state_spec)
            right_vector = feature_vector(evaluate(node.right, values), node.role, state_spec)
            coefficient = evaluate(node.coefficient, values).float()[..., :1, :1]
            if node.role == state_spec.feature_roles[0]:
                update = torch.eye(left_size, device=device).expand(token_count, heads, left_size, left_size)
                update = update + coefficient * left_vector.unsqueeze(-1) * right_vector.unsqueeze(-2)
                left = update @ left
            else:
                update = torch.eye(right_size, device=device).expand(token_count, heads, right_size, right_size)
                update = update + coefficient * right_vector.unsqueeze(-1) * left_vector.unsqueeze(-2)
                right = right @ update
        else:
            raise ValueError("rank-one runtime received an unsupported propagation")
    return left, right, elementwise


def product_injection_terms(
    injections: Sequence[Any],
    *,
    evaluate: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
    values: Mapping[str, torch.Tensor],
    feature_roles: tuple[Any, Any],
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Materialize additive injections, using explicit GEMM-shaped products."""
    result: torch.Tensor | None = None
    for injection in injections:
        factors = [evaluate(factor.expression, values).float() for factor in injection.factors]
        product: torch.Tensor | float = 1.0
        if len(factors) == 2 and {tuple(factor.roles) for factor in injection.factors} == {
            (feature_roles[0],), (feature_roles[1],)
        }:
            first = factors[0]
            second = factors[1]
            first_role = injection.factors[0].roles[0]
            if first_role == feature_roles[0]:
                product = first[..., :, 0].unsqueeze(-1) * second[..., 0, :].unsqueeze(-2)
            else:
                product = second[..., :, 0].unsqueeze(-1) * first[..., 0, :].unsqueeze(-2)
        else:
            for factor in factors:
                product = product * factor
        product = torch.broadcast_to(product, (token_count, heads, left_size, right_size))
        if result is None:
            result = product.contiguous()
        else:
            result = result + product
    return result


def shared_readouts(
    readouts: Sequence[Any],
    *,
    evaluate: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
    feature_vector: Callable[[torch.Tensor, Any, Any], torch.Tensor],
    values: Mapping[str, torch.Tensor],
    states: torch.Tensor,
    state_spec: Any,
) -> dict[str, torch.Tensor]:
    """Contract shared State in role groups, preserving declared readout order."""
    grouped: dict[Any, list[tuple[Any, torch.Tensor]]] = {}
    for readout in readouts:
        operand = feature_vector(evaluate(readout.operand, values), readout.role, state_spec).float()
        grouped.setdefault(readout.role, []).append((readout, operand))
    outputs: dict[str, torch.Tensor] = {}
    for role, members in grouped.items():
        if role == state_spec.feature_roles[0]:
            operands = torch.stack([item[1] for item in members], dim=0)
            contracted = torch.einsum("rnhf,nhfg->rnhg", operands, states)
        else:
            operands = torch.stack([item[1] for item in members], dim=0)
            contracted = torch.einsum("rnhg,nhfg->rnhf", operands, states)
        for index, (readout, _) in enumerate(members):
            outputs[readout.name] = contracted[index].to(readout.output_dtype)
    return outputs
