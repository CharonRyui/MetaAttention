from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from attn_engine import (
    Add,
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
    ExecutionResult,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    Multiply,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulCompilationError,
    StatefulOperator,
    StateTuple,
    TensorInput,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")
ROLE_SIZES = {
    QUERY_HEADS: 1,
    KEY_HEADS: 1,
    VALUE_HEADS: 1,
    STATE_HEADS: 1,
    KEY: 2,
    VALUE: 3,
}


def _compile(
    operator,
    *,
    mode="dense",
    backward=False,
    role_sizes=None,
    optimized=True,
    autotune=False,
):
    from attn_engine import Target

    return operator.compile(
        role_sizes=ROLE_SIZES if role_sizes is None else role_sizes,
        target=Target(),
        mode=mode,
        optimized=optimized,
        autotune=autotune,
        backward=backward,
    )


def test_metadata_only_dense_compile_reuses_plan_across_dynamic_workloads():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator)

    first = _inputs(length=1)
    second = {
        name: value.repeat(2, 1, 5 // value.shape[2] + 1, *([1] * (value.ndim - 3)))[
            :, :, :5
        ].contiguous()
        for name, value in _inputs(length=1).items()
    }
    first_result = compiled(**first)
    second_result = compiled(**second)

    torch.testing.assert_close(first_result.outputs[0], _reference(first)[0])
    torch.testing.assert_close(second_result.outputs[0], _reference(second)[0])
    assert len(operator._plans) == 1


def test_dynamic_workloads_reuse_binary_and_workspace_buffers():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator)
    executable = operator._plans[compiled.specialization]
    binary_identity = executable.plan.binary_identity

    compiled(**_inputs(length=3))
    first = tuple(executable.workspace_allocator._buffers.values())
    compiled(**_inputs(length=3))
    second = tuple(executable.workspace_allocator._buffers.values())
    compiled(**_inputs(length=5))

    assert first == second
    assert len(executable.workspace_allocator._buffers) == 2
    assert executable.plan.binary_identity == binary_identity


def test_empty_packed_workload_allocates_no_workspace():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator, mode="packed")
    inputs = {
        "query": torch.empty(0, 1, 2),
        "key": torch.empty(0, 1, 2),
        "value": torch.empty(0, 1, 3),
        "gate": torch.empty(0, 1),
    }
    offsets = torch.tensor([0, 0, 0], dtype=torch.int32)

    compiled(**inputs, sequence_offsets=offsets)

    executable = operator._plans[compiled.specialization]
    assert executable.workspace_allocator._buffers == {}


def test_runtime_workspace_uses_explicit_target_resource_envelope():
    from attn_engine import Target

    operator = StatefulOperator(_accumulator_ir())
    compiled = operator.compile(
        role_sizes=ROLE_SIZES,
        target=Target(workspace_bytes=23),
        mode="dense",
    )

    with pytest.raises(StatefulCompilationError) as captured:
        compiled(**_inputs(length=1))

    assert captured.value.category == "RUNTIME_RESOURCE"
    assert captured.value.path == "workspace"


def test_backward_compilation_publishes_compiler_owned_backward_candidate():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator, backward=True)
    executable = operator._plans[compiled.specialization]

    assert executable.backward_ir is operator.algorithm._analysis.backward
    assert executable.plan.backward_candidate is not None
    assert executable.plan.backward_candidate.direction == "backward"
    assert executable.plan.backward_kernel_config is not None
    assert executable.plan.context_contract.required == executable.plan.context_contract.produced


def test_context_contract_rejects_dtype_role_and_shape_mismatches():
    from attn_engine._stateful_plan import ContextContract, ContextValue

    produced = ContextValue(
        "inputs.query", "torch.float32", ("batch", "heads", "sequence", "key"), "runtime"
    )
    mismatches = (
        ContextValue(produced.name, "torch.bfloat16", produced.roles, produced.shape_rule),
        ContextValue(produced.name, produced.dtype, ("batch", "sequence"), produced.shape_rule),
        ContextValue(produced.name, produced.dtype, produced.roles, "token-major"),
    )

    for required in mismatches:
        with pytest.raises(ValueError, match="incompatible context value"):
            ContextContract((produced,), (required,), ()).validate()


def test_context_contract_identity_separates_backward_behavior():
    operator = StatefulOperator(_accumulator_ir())
    forward = _compile(operator, backward=False)
    backward = _compile(operator, backward=True)
    forward_plan = operator._plans[forward.specialization].plan
    backward_plan = operator._plans[backward.specialization].plan

    assert forward_plan.context_contract.identity != backward_plan.context_contract.identity
    assert forward_plan.binary_identity != backward_plan.binary_identity



def test_static_selector_prefers_specialized_candidate_and_keeps_generic_fallback():
    from attn_engine._stateful_plan import LoweredCandidate, select_candidate

    generic = LoweredCandidate("g", "p0", "tilelang-generic", "forward")
    specialized = LoweredCandidate("s", "p1", "tilelang-elementwise", "forward")
    assert select_candidate((generic, specialized)) is specialized
    assert select_candidate((generic,)) is generic



def test_workload_policy_covers_dense_and_packed_shapes_and_scores_geomean():
    from attn_engine._stateful_plan import WORKLOAD_POLICY, normalized_geometric_mean

    assert {point.distribution for point in WORKLOAD_POLICY.points("packed")} >= {
        "uniform",
        "long-tail",
        "empty-containing",
    }
    assert len(WORKLOAD_POLICY.points("dense")) >= 2
    scores = normalized_geometric_mean(
        {"first": (1.0, 4.0), "second": (2.0, 2.0)}
    )
    assert scores["first"] == pytest.approx(2**0.5)
    assert scores["second"] == pytest.approx(2**0.5)


def test_autotune_config_is_derived_only_for_selected_candidate():
    from attn_engine._stateful_plan import LoweredCandidate, choose_kernel_config

    selected = LoweredCandidate("selected", "plan", "tilelang-elementwise", "forward")
    config = choose_kernel_config(
        selected,
        {"selected": (1.0, 2.0), "loser": (2.0, 1.0)},
        autotune=True,
    )
    stable = choose_kernel_config(selected, None, autotune=False)

    assert config.identity != stable.identity
    assert config.values[0][0] == "probe-score-millionths"
    assert stable.values == ()



def test_successful_artifacts_publish_atomically_by_layer(tmp_path, monkeypatch):
    from attn_engine._stateful_plan import publish_successful_artifact

    monkeypatch.setenv("STATEFUL_OPERATOR_CACHE_DIR", str(tmp_path))
    destination = publish_successful_artifact("artifact", {"ok": True}, layer="plans")
    same = publish_successful_artifact("artifact", {"ok": False}, layer="plans")

    assert destination == same
    assert json.loads(destination.read_text()) == {"ok": True}


def test_h20_evidence_remains_pending_until_idle_hardware_measurement():
    from attn_engine._stateful_plan import H20_EVIDENCE, WORKLOAD_POLICY

    assert H20_EVIDENCE.target == "NVIDIA H20"
    assert H20_EVIDENCE.workload_policy == WORKLOAD_POLICY.version
    assert H20_EVIDENCE.status == "pending"


def test_metadata_only_compile_rejects_runtime_roles_and_tensor_arguments():
    from attn_engine import Target

    operator = StatefulOperator(_accumulator_ir())
    with pytest.raises(StatefulCompilationError) as captured:
        operator.compile(
            role_sizes={**ROLE_SIZES, Batch: 1}, target=Target(), mode="dense"
        )
    assert captured.value.category == "COMPILE_ROLE_SIZE"

    with pytest.raises(TypeError):
        operator.compile(
            role_sizes=ROLE_SIZES,
            target=Target(),
            mode="dense",
            query=_inputs()["query"],
        )


def _accumulator_ir(dtype: torch.dtype = torch.float32) -> AlgorithmIR:
    return AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("gate"))),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), KEY, dtype),),
        head_mapping=HeadMapping(
            mappings={
                QUERY_HEADS: STATE_HEADS,
                KEY_HEADS: STATE_HEADS,
                VALUE_HEADS: STATE_HEADS,
            }
        ),
    )


def _inputs(length: int = 3, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(7)
    return {
        "query": torch.randn(
            1, 1, length, 2, generator=generator, requires_grad=requires_grad
        ),
        "key": torch.randn(
            1, 1, length, 2, generator=generator, requires_grad=requires_grad
        ),
        "value": torch.randn(
            1, 1, length, 3, generator=generator, requires_grad=requires_grad
        ),
        "gate": torch.randn(
            1, 1, length, generator=generator, requires_grad=requires_grad
        ),
    }


def _reference(inputs, initial=None):
    state = torch.zeros(1, 1, 2, 3) if initial is None else initial
    outputs = []
    for token in range(inputs["query"].shape[2]):
        state = state * inputs["gate"][..., token].exp().unsqueeze(-1).unsqueeze(-1)
        state = state + torch.einsum(
            "bhf,bhg->bhfg",
            inputs["key"][..., token, :],
            inputs["value"][..., token, :],
        )
        outputs.append(
            torch.einsum("bhf,bhfg->bhg", inputs["query"][..., token, :], state)
        )
    return torch.stack(outputs, dim=2), state


def test_batch_static_inputs_preserve_per_sequence_values():
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("key", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("weight", (Batch, STATE_HEADS, VALUE), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("weight"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction("output", "memory", Input("query"), KEY, torch.float32),
        ),
        head_mapping=HeadMapping(),
    )
    query = torch.ones(2, 1, 2, 2)
    key = torch.ones(2, 1, 2, 2)
    weight = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    operator = StatefulOperator(algorithm)
    compiled = _compile(
        operator,
        role_sizes={STATE_HEADS: 1, KEY: 2, VALUE: 3},
    )
    output = compiled(query=query, key=key, weight=weight).outputs[0]
    expected = torch.tensor(
        [
            [[[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]]],
            [[[8.0, 10.0, 12.0], [16.0, 20.0, 24.0]]],
        ]
    )
    torch.testing.assert_close(output, expected)


def test_constant_bits_and_commutative_identity_are_canonical():
    assert Constant(0.0).bits != Constant(-0.0).bits
    with pytest.raises(ValueError, match="NaN and infinity"):
        Constant(float("nan"))
    assert Add(Input("x"), Input("y")) == Add(Input("y"), Input("x"))


def test_mandatory_simplification_removes_safe_zero_and_one_constants():
    from attn_engine._stateful_analysis import simplify_expression

    x = Input("x")
    assert simplify_expression(x + Constant(0.0)) == x
    assert simplify_expression(x * Constant(1.0)) == x
    assert simplify_expression(Exp(Constant(0.0))) == Constant(1.0)


def test_simplification_preserves_authored_association():
    from attn_engine._stateful_analysis import simplify_expression

    x, y, z = Input("x"), Input("y"), Input("z")
    assert simplify_expression((x + y) + z) != simplify_expression(x + (y + z))
    assert Multiply(Input("x"), Input("y")) == Multiply(Input("y"), Input("x"))
    assert Add(Add(Input("x"), Input("y")), Input("z")) != Add(
        Input("x"), Add(Input("y"), Input("z"))
    )


def test_direct_invocation_requires_explicit_compile():
    operator = StatefulOperator(_accumulator_ir())

    with pytest.raises(RuntimeError, match="explicitly compiled before invocation"):
        operator(**_inputs())


def test_dense_invocation_returns_immutable_named_result_and_gradients():
    inputs = _inputs(requires_grad=True)
    expected_output, expected_state = _reference(inputs)
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator, backward=True)
    result = compiled(**inputs, return_final_state=True)

    assert isinstance(result, ExecutionResult)
    torch.testing.assert_close(result.outputs["output"], expected_output)
    torch.testing.assert_close(result.outputs[0], expected_output)
    assert result.final_state is not None
    torch.testing.assert_close(result.final_state["memory"], expected_state)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.final_state = None
    loss = result.outputs[0].square().sum() + result.final_state[0].square().sum()
    gradients = torch.autograd.grad(loss, tuple(inputs.values()))
    assert all(gradient is not None for gradient in gradients)


def test_omitted_state_and_continuation_match_full_invocation():
    inputs = _inputs(length=4)
    operator = StatefulOperator(_accumulator_ir())
    prefix_inputs = {name: value[:, :, :2] for name, value in inputs.items()}
    suffix_inputs = {name: value[:, :, 2:] for name, value in inputs.items()}
    compiled = _compile(operator)
    full = compiled(**inputs, return_final_state=True)
    prefix_compiled = compiled
    prefix = prefix_compiled(**prefix_inputs, return_final_state=True)
    assert prefix.final_state is not None
    suffix_compiled = compiled
    suffix = suffix_compiled(
        **suffix_inputs,
        initial_state=prefix.final_state,
    )
    torch.testing.assert_close(
        torch.cat((prefix.outputs[0], suffix.outputs[0]), dim=2), full.outputs[0]
    )


def test_packed_empty_sequences_preserve_state_and_offsets_are_runtime_data():
    operator = StatefulOperator(_accumulator_ir())
    dense = _inputs(length=3)
    packed = {
        name: value.squeeze(0).transpose(0, 1).contiguous()
        for name, value in dense.items()
    }
    packed = {
        name: value.transpose(0, 1).contiguous() if value.shape[0] == 1 else value
        for name, value in packed.items()
    }
    offsets = torch.tensor([0, 0, 3, 3], dtype=torch.int32)
    initial = torch.randn(3, 1, 2, 3)
    initial_state = StateTuple(("memory",), (initial,))
    compiled = _compile(operator, mode="packed")
    result = compiled(
        **packed,
        sequence_offsets=offsets,
        initial_state=initial_state,
        return_final_state=True,
    )
    assert result.final_state is not None
    torch.testing.assert_close(result.final_state[0][0], initial[0])
    torch.testing.assert_close(result.final_state[0][2], initial[2])
    assert result.outputs[0].shape == (3, 1, 3)


def test_specialization_identity_reuses_offset_metadata_and_tracks_training_modes():
    operator = StatefulOperator(_accumulator_ir())
    dense = _inputs(length=3)
    packed = {
        name: value.squeeze(0).transpose(0, 1).contiguous()
        for name, value in dense.items()
    }
    packed = {
        name: value.transpose(0, 1).contiguous() if value.shape[0] == 1 else value
        for name, value in packed.items()
    }
    first_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
    compiled = _compile(operator, mode="packed")
    compiled(**packed, sequence_offsets=first_offsets)
    second_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    compiled(**packed, sequence_offsets=second_offsets)
    assert len(operator._specializations) == 1

    backward_compiled = _compile(operator, mode="packed", backward=True)
    initial = torch.zeros(2, 1, 2, 3, requires_grad=True)
    initial_state = StateTuple(("memory",), (initial,))
    backward_compiled(
        **packed,
        sequence_offsets=second_offsets,
        initial_state=initial_state,
    )
    assert len(operator._specializations) == 2


def test_simplified_identity_ignores_removed_expression_identities():
    original = _accumulator_ir()
    rewritten = AlgorithmIR(
        inputs=original.inputs,
        states=original.states,
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Add(Input("gate"), Constant(0.0)))),),
            injections=original.transition.injections,
        ),
        readouts=original.readouts,
        head_mapping=original.head_mapping,
    )

    assert original._analysis.canonical_identity != rewritten._analysis.canonical_identity
    assert original._analysis.simplified.identity == rewritten._analysis.simplified.identity

def test_analysis_derives_scanable_affine_summary_and_backward_ir():
    analysis = _accumulator_ir()._analysis
    assert analysis.execution_class.value == "scanable"
    assert analysis.affine_summary.transform == "dense"
    assert analysis.backward.output_cotangents == ("output",)
    assert analysis.backward.initial_state_gradient == "d_memory_initial"
    assert {step.operation for step in analysis.backward.reverse_steps} >= {
        "axis_scale",
        "product_injection",
        "state_contraction",
        "head_reduce",
    }


def test_compiled_plan_owns_parallel_forward_and_derived_backward():
    operator = StatefulOperator(_accumulator_ir())
    inputs = _inputs()
    compiled = _compile(operator, backward=True)
    compiled(**inputs)
    executable = next(iter(operator._plans.values()))

    assert executable.plan.uses_parallel_summary
    assert executable.backward_ir is operator.algorithm._analysis.backward
    assert any(
        step.operation == "state_recurrence"
        for step in executable.backward_ir.reverse_steps
    )


def test_structured_plans_and_kernel_config_have_separate_identities():
    operator = StatefulOperator(_accumulator_ir())
    generic = _compile(operator, optimized=False)
    structured = _compile(operator, optimized=True)
    generic_plan = operator._plans[generic.specialization].plan
    structured_plan = operator._plans[structured.specialization].plan

    assert generic_plan.mathematical_plan.representation == "generic-dense-affine"
    assert structured_plan.mathematical_plan.rewrites == (
        "axis-scale",
        "product-injection-gemm",
    )
    assert structured_plan.mathematical_plan.identity != structured_plan.candidate.identity
    assert structured_plan.candidate.identity != structured_plan.kernel_config.identity


def test_planner_enumerates_generic_baseline_and_registered_structured_candidates():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator, optimized=True)
    plan = operator._plans[compiled.specialization].plan

    representations = {item.representation for item in plan.mathematical_plans}
    assert "generic-dense-affine" in representations
    assert "axis-scale" in representations
    assert "product-injection-gemm" in representations
    assert "scalar-factorized" in representations
    assert any(
        candidate.plan_identity
        == next(
            item.identity
            for item in plan.mathematical_plans
            if item.representation == "generic-dense-affine"
        )
        for candidate in plan.lowered_candidates
    )


def test_optimization_disabled_enumerates_only_generic_plan():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator, optimized=False)
    plan = operator._plans[compiled.specialization].plan

    assert tuple(item.representation for item in plan.mathematical_plans) == (
        "generic-dense-affine",
    )
    assert plan.mathematical_plan == plan.mathematical_plans[0]


def test_lowerer_registry_allows_multiple_implementations_per_plan():
    from attn_engine._stateful_plan import CandidateLowerer, MathematicalPlan

    plan = MathematicalPlan(
        "semantic", "axis-scale", ("axis-scale",), ("state-transition",)
    )
    first = CandidateLowerer("first", ("axis-scale",), 3).lower(plan, (64, 128))
    second = CandidateLowerer("second", ("axis-scale",), 4).lower(plan, (64, 128))
    mismatch = CandidateLowerer("other", ("generic-dense-affine",), 1).lower(
        plan, (64, 128)
    )

    assert first is not None and second is not None
    assert first.plan_identity == second.plan_identity
    assert first.identity != second.identity
    assert mismatch is None


def test_shared_readouts_and_rank_one_are_model_independent_planner_features():
    from examples.gated_delta_rule import compositional_pressure_profile

    operator = compositional_pressure_profile()
    roles = {
        role: 4 if isinstance(role, HeadRole) else 64
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if isinstance(role, (HeadRole, FeatureRole))
    }
    roles[operator.algorithm.states[0].feature_roles[1]] = 128
    compiled = _compile(operator, role_sizes=roles)
    plan = operator._plans[compiled.specialization].plan
    rewrites = {
        rewrite
        for mathematical_plan in plan.mathematical_plans
        for rewrite in mathematical_plan.rewrites
    }
    assert "rank-one-factorized" in rewrites
    assert "shared-state-contraction" in rewrites
    assert plan.mathematical_plan.representation == "generic-dense-affine"



def test_analysis_maps_expression_head_roles_to_state_heads():
    analysis = _accumulator_ir()._analysis
    injection_type = dict(analysis.expression_types)[
        "transition.injections[0].factors[0].expression"
    ]
    assert STATE_HEADS in injection_type.roles
    assert KEY_HEADS not in injection_type.roles


def test_analysis_derives_raw_dense_recipe_from_structure():
    scalar = _accumulator_ir()._analysis.scalar_factorized
    assert scalar is not None
    dense = scalar.dense
    assert dense is not None
    assert dense.propagation[0].name == "gate"
    assert dense.propagation[1].is_log_scale
    assert dense.left_factor[0].name == "key"
    assert dense.right_factor[0].name == "value"
    assert dense.readout[0].name == "query"


def test_runtime_revalidates_metadata():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator)
    invalid = _inputs()
    invalid["query"] = invalid["query"].transpose(-1, -2)
    with pytest.raises(StatefulCompilationError) as captured:
        compiled(**invalid)
    assert captured.value.category in ("RUNTIME_LAYOUT", "RUNTIME_SHAPE")


def test_analysis_rejects_unmapped_incompatible_head_roles():
    other_heads = HeadRole("other_heads")
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(
                TensorInput("left", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
                TensorInput(
                    "right", (Batch, other_heads, Sequence, KEY), torch.float32
                ),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("left") * Input("right"), (KEY,)),
                            ProductFactor(Constant(1.0), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(StateContraction("output", "memory", Input("left"), KEY),),
            head_mapping=HeadMapping({KEY_HEADS: STATE_HEADS}),
        )
    assert captured.value.category == "IR_TYPE"
    assert captured.value.path == "transition.injections[0].factors[0].expression"


@pytest.mark.parametrize("loss_mode", ("output", "state", "joint"))
def test_output_and_final_state_cotangent_modes_match_reference(loss_mode):
    actual_inputs = _inputs(length=4, requires_grad=True)
    reference_inputs = {
        name: value.detach().clone().requires_grad_()
        for name, value in actual_inputs.items()
    }
    actual_initial = torch.randn(1, 1, 2, 3, requires_grad=True)
    reference_initial = actual_initial.detach().clone().requires_grad_()
    operator = StatefulOperator(_accumulator_ir())
    initial_state = StateTuple(("memory",), (actual_initial,))
    compiled = _compile(operator, backward=True)
    result = compiled(
        **actual_inputs,
        initial_state=initial_state,
        return_final_state=True,
    )
    expected_output, expected_state = _reference(reference_inputs, reference_initial)
    assert result.final_state is not None
    actual_loss = torch.zeros((), dtype=torch.float32)
    expected_loss = torch.zeros((), dtype=torch.float32)
    if loss_mode in ("output", "joint"):
        actual_loss = actual_loss + result.outputs[0].square().sum()
        expected_loss = expected_loss + expected_output.square().sum()
    if loss_mode in ("state", "joint"):
        actual_loss = actual_loss + result.final_state[0].square().sum()
        expected_loss = expected_loss + expected_state.square().sum()
    actual_gradients = torch.autograd.grad(
        actual_loss, (*actual_inputs.values(), actual_initial), allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (*reference_inputs.values(), reference_initial),
        allow_unused=True,
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-3)


def test_untyped_roles_are_rejected_without_compatibility_shim():
    with pytest.raises(TypeError, match="typed logical roles"):
        TensorInput("query", ("batch", "heads", "sequence", "feature"), torch.float32)


def test_structured_errors_are_json_safe_and_have_canonical_paths():
    operator = StatefulOperator(_accumulator_ir())
    compiled = _compile(operator)
    with pytest.raises(StatefulCompilationError) as captured:
        compiled(query=torch.empty(0), key=torch.empty(0), value=torch.empty(0))
    error = captured.value
    assert error.category == "RUNTIME_BINDING"
    assert error.path == "inputs"
    json.dumps(error.as_dict())


def test_structured_errors_with_typed_roles_are_json_safe():
    error = StatefulCompilationError(
        "IR_TYPE", "roles", {"role": KEY, "mapping": {QUERY_HEADS: STATE_HEADS}}
    )
    assert json.loads(json.dumps(error.as_dict())) == {
        "category": "IR_TYPE",
        "path": "roles",
        "details": {
            "mapping": {"query_heads": {"name": "state_heads", "node": "HeadRole"}},
            "role": {"name": "key_dim", "node": "FeatureRole"},
        },
    }


def test_ir_requires_exactly_one_matrix_state():
    ir = _accumulator_ir()
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=ir.inputs,
            states=(),
            transition=ir.transition,
            readouts=ir.readouts,
            head_mapping=ir.head_mapping,
        )
    assert captured.value.category == "IR_SCHEMA"
    assert captured.value.path == "states"
