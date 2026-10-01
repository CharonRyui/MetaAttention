"""Candidate ranking by a shape-aware estimate.

`select_candidate` must rank by what a plan actually costs for the concrete
Feature sizes and IR structure, not by a per-lowerer constant, so a Program
whose summary is diagonal or whose propagation is a rank-one contraction can
never fall back to a dense scan.
"""

from __future__ import annotations

import pytest

from attn_engine import HeadRole, ProductFeatureRole, Target
from examples.gated_delta_rule import gated_delta_rule
from examples.linear_attention import (
    dual_gate_gla_attention,
    gla_attention,
    linear_attention,
    simple_gla_attention,
)

pytestmark = pytest.mark.unit

HEADS = 8
DENSE_SCAN = {"tilelang-generic", "reference-stateful"}


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _sizes(operator, rows: int, columns: int) -> dict:
    sizes: dict = {operator.algorithm.states[0].head_role: HEADS}
    for index, role in enumerate(operator.algorithm.states[0].feature_roles):
        sizes[role] = rows if index == 0 else columns
    for spec in operator.algorithm.inputs:
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes.setdefault(role, HEADS)
            elif isinstance(role, ProductFeatureRole):
                sizes[role] = sizes[role.left] * sizes[role.right]
    return sizes


def _plan(factory, *, rows: int = 128, columns: int = 128, **options):
    head_kwargs = {"HQ": 1} if factory is gated_delta_rule else {}
    operator = factory(1, HEADS, 512, rows, columns, **head_kwargs)
    compiled = operator.compile(
        role_sizes=_sizes(operator, rows, columns),
        target=options.pop("target", None) or Target(),
        mode="dense",
        **options,
    )
    return compiled.operator._plans[compiled.specialization].plan


def test_feature_vector_gla_selects_vector_propagation():
    plan = _plan(gla_attention, rows=64, columns=64)
    assert plan.candidate.lowerer == "tilelang-vector-propagation"
    representations = {item.representation for item in plan.mathematical_plans}
    assert "vector-propagation" in representations
    assert "axis-scale" in representations


def test_non_exp_feature_scale_uses_generic_axis_scale_candidate():
    from attn_engine import (
        AlgorithmIR,
        AxisScale,
        HeadMapping,
        Input,
        ProductFactor,
        ProductInjection,
    )
    from attn_engine import StateSpec, StateTransition, StateContraction, TensorInput
    from attn_engine import Batch, FeatureRole, HeadRole, Sequence, StatefulOperator
    import torch

    key = FeatureRole("key_dim")
    value = FeatureRole("value_dim")
    heads = HeadRole("heads")
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("query", (Batch, heads, Sequence, key), torch.bfloat16),
                TensorInput("key", (Batch, heads, Sequence, key), torch.bfloat16),
                TensorInput("value", (Batch, heads, Sequence, value), torch.bfloat16),
                TensorInput("gate", (Batch, heads, Sequence, key), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, heads, key, value)),),
            transition=StateTransition(
                "memory",
                propagations=(AxisScale(Input("gate"), (key,)),),
                injections=(
                    ProductInjection(
                        (ProductFactor(Input("key"), (key,)), ProductFactor(Input("value"), (value,)))
                    ),
                ),
            ),
            readouts=(StateContraction("output", "memory", Input("query"), key, torch.bfloat16),),
            head_mapping=HeadMapping({heads: heads}),
        )
    )
    sizes = {heads: 1, key: 8, value: 8}
    compiled = operator.compile(role_sizes=sizes, target=Target(), mode="dense")
    plan = compiled.operator._plans[compiled.specialization].plan
    assert plan.candidate.lowerer == "tilelang-elementwise"

def test_feature_vector_plan_remains_vector_candidate_for_chunked_runtime():
    plan = _plan(gla_attention, rows=64, columns=64)
    assert plan.candidate.lowerer == "tilelang-vector-propagation"
    assert plan.workspace.bytes_per_token == 0

def test_diagonal_decay_never_selects_a_dense_scan():
    plan = _plan(gla_attention)
    assert plan.candidate.lowerer not in DENSE_SCAN
    selected = dict(plan.candidate_estimates)[plan.candidate.identity]
    assert selected.cost == min(estimate.cost for _, estimate in plan.candidate_estimates)

def test_rank_one_propagation_never_selects_a_dense_scan():
    plan = _plan(gated_delta_rule)
    assert plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert plan.candidate.lowerer not in DENSE_SCAN
    selected = dict(plan.candidate_estimates)[plan.candidate.identity]
    assert selected.cost == min(estimate.cost for _, estimate in plan.candidate_estimates)


def test_scalar_decay_keeps_the_factorized_selection():
    plan = _plan(simple_gla_attention)
    assert plan.candidate.lowerer == "tilelang-scalar-factorized"
    costs = [estimate.cost for _, estimate in plan.candidate_estimates]
    assert costs == sorted(costs)


def test_estimates_are_inspectable_with_all_three_components():
    plan = _plan(linear_attention)
    assert plan.candidate_estimates
    for identity, estimate in plan.candidate_estimates:
        assert identity
        assert estimate.bytes_per_token > 0
        assert estimate.flops_per_token > 0
        assert estimate.launches >= 1
    assert plan.candidate_estimates[0][0] == plan.candidate.identity


def test_infeasible_candidate_leaves_a_lower_ranked_plan():
    # an envelope below one candidate's per-token buffer removes it before
    # ranking while the remaining candidates still lower
    from attn_engine._stateful_plan import (
        EstimationShape,
        LOWERERS,
        MathematicalPlan,
        select_candidate,
    )

    shape = EstimationShape.of(heads=8, feature_sizes=(128, 64), dense_summary=False)
    lowerers = {lowerer.name: lowerer for lowerer in LOWERERS}
    plans = {
        name: MathematicalPlan(name, lowerer.representations[0], (), ("t",))
        for name, lowerer in lowerers.items()
    }
    budget = 786_432  # fits the elementwise summary, not the generic identities
    lowered = {}
    estimates = {}
    for name in ("tilelang-generic", "tilelang-elementwise"):
        lowerer = lowerers[name]
        estimate = lowerer.estimate(plans[name], shape)
        estimates[name] = estimate
        assert estimate.workspace_bytes_per_token > 0
        candidate = lowerer.lower(
            plans[name], shape, maximum_workspace_bytes=budget
        )
        if candidate is not None:
            lowered[name] = candidate
    assert "tilelang-generic" not in lowered
    assert "tilelang-elementwise" in lowered
    selected = select_candidate(
        tuple(lowered.values()), {lowered[name]: estimates[name] for name in lowered}
    )
    assert selected is lowered["tilelang-elementwise"]


def test_optimization_disabled_selects_generic_execution_only():
    plan = _plan(gla_attention)
    assert plan.candidate.lowerer != "tilelang-generic"
    unoptimized = _plan(gla_attention, optimized=False)
    assert unoptimized.candidate.lowerer == "tilelang-generic"


def test_two_scaled_axes_lower_without_a_dense_summary():
    """Both axes scaled is still diagonal: one factor per axis, not a summary pair."""
    from attn_engine.stateful_operator import _dense_summary_required

    operator = dual_gate_gla_attention(1, HEADS, 32, 64, 64)
    assert _dense_summary_required(operator.algorithm) is False
    compiled = operator.compile(
        role_sizes=_sizes(operator, 64, 64),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    executable = compiled.operator._plans[compiled.specialization]
    assert executable.dense_summary is False
    assert executable.plan.candidate.lowerer == "tilelang-elementwise"

    # a compilation that asks for a backward uses the same pair, whose VJP
    # contracts each factor along the axis the other spans
    backward_compiled = operator.compile(
        role_sizes=_sizes(operator, 64, 64),
        target=Target(),
        mode="dense",
        optimized=True,
        backward=True,
    )
    assert (
        backward_compiled.operator._plans[backward_compiled.specialization].dense_summary
        is False
    )
