from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

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
    output = StatefulOperator(algorithm)(query=query, key=key, weight=weight).outputs[0]
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
    assert Multiply(Input("x"), Input("y")) == Multiply(Input("y"), Input("x"))
    assert Add(Add(Input("x"), Input("y")), Input("z")) != Add(
        Input("x"), Add(Input("y"), Input("z"))
    )


def test_dense_invocation_returns_immutable_named_result_and_gradients():
    inputs = _inputs(requires_grad=True)
    expected_output, expected_state = _reference(inputs)
    result = StatefulOperator(_accumulator_ir())(**inputs, return_final_state=True)

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
    full = operator(**inputs, return_final_state=True)
    prefix = operator(
        **{name: value[:, :, :2] for name, value in inputs.items()},
        return_final_state=True,
    )
    assert prefix.final_state is not None
    suffix = operator(
        **{name: value[:, :, 2:] for name, value in inputs.items()},
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
    result = operator(
        **packed,
        sequence_offsets=offsets,
        initial_state=StateTuple(("memory",), (initial,)),
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
    operator(**packed, sequence_offsets=torch.tensor([0, 1, 3], dtype=torch.int32))
    operator(**packed, sequence_offsets=torch.tensor([0, 2, 3], dtype=torch.int32))
    assert len(operator._specializations) == 1

    initial = torch.zeros(2, 1, 2, 3, requires_grad=True)
    operator(
        **packed,
        sequence_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        initial_state=StateTuple(("memory",), (initial,)),
    )
    assert len(operator._specializations) == 2


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
    operator(**_inputs())
    executable = next(iter(operator._plans.values()))

    assert executable.plan.uses_parallel_summary
    assert executable.backward_ir is operator.algorithm._analysis.backward
    assert any(
        step.operation == "state_recurrence"
        for step in executable.backward_ir.reverse_steps
    )


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


def test_cached_dense_call_revalidates_runtime_metadata(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "0")
    operator = StatefulOperator(_accumulator_ir())
    operator._dense_cached_call = operator._build_dense_cached_call(
        _inputs(),
        SimpleNamespace(uniform_length=3, sequence_count=1),
        torch.zeros(1, 1, 2, 3),
        object(),
    )
    invalid = _inputs()
    invalid["query"] = invalid["query"].transpose(-1, -2)
    with pytest.raises(StatefulCompilationError) as captured:
        operator(**invalid)
    assert captured.value.category == "RUNTIME_SHAPE"


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
    result = StatefulOperator(_accumulator_ir())(
        **actual_inputs,
        initial_state=StateTuple(("memory",), (actual_initial,)),
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
    with pytest.raises(StatefulCompilationError) as captured:
        operator(query=torch.empty(0), key=torch.empty(0), value=torch.empty(0))
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
