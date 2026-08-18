from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from attn_engine import (
    Add,
    AlgorithmIR,
    AxisScale,
    Constant,
    ExecutionResult,
    Exp,
    HeadMapping,
    Input,
    Multiply,
    ProductFactor,
    ProductInjection,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulCompilationError,
    StatefulOperator,
    StateTuple,
    TensorInput,
)

pytestmark = pytest.mark.unit


def _accumulator_ir(dtype: torch.dtype = torch.float32) -> AlgorithmIR:
    return AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), dtype),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), dtype),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), dtype),
            TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("gate"))),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), ("key_dim",)),
                        ProductFactor(Input("value"), ("value_dim",)),
                    )
                ),
            ),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), "key_dim", dtype),),
        head_mapping=HeadMapping(
            mappings={
                "query_heads": "state_heads",
                "key_heads": "state_heads",
                "value_heads": "state_heads",
            }
        ),
    )


def _inputs(length: int = 3, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "query": torch.randn(1, 1, length, 2, generator=generator, requires_grad=requires_grad),
        "key": torch.randn(1, 1, length, 2, generator=generator, requires_grad=requires_grad),
        "value": torch.randn(1, 1, length, 3, generator=generator, requires_grad=requires_grad),
        "gate": torch.randn(1, 1, length, generator=generator, requires_grad=requires_grad),
    }
    return tensors


def _reference(inputs, initial=None):
    state = torch.zeros(1, 1, 2, 3) if initial is None else initial
    outputs = []
    for token in range(inputs["query"].shape[2]):
        state = state * inputs["gate"][..., token].exp().unsqueeze(-1).unsqueeze(-1)
        state = state + torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][..., token, :], inputs["value"][..., token, :]
        )
        outputs.append(torch.einsum("bhf,bhfg->bhg", inputs["query"][..., token, :], state))
    return torch.stack(outputs, dim=2), state


def test_constant_bits_and_commutative_identity_are_canonical():
    assert Constant(0.0).bits != Constant(-0.0).bits
    with pytest.raises(ValueError, match="NaN and infinity"):
        Constant(float("nan"))
    assert Add(Input("x"), Input("y")) == Add(Input("y"), Input("x"))
    assert Multiply(Input("x"), Input("y")) == Multiply(Input("y"), Input("x"))
    assert Add(Add(Input("x"), Input("y")), Input("z")) != Add(Input("x"), Add(Input("y"), Input("z")))


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
    prefix = operator(**{name: value[:, :, :2] for name, value in inputs.items()}, return_final_state=True)
    assert prefix.final_state is not None
    suffix = operator(
        **{name: value[:, :, 2:] for name, value in inputs.items()},
        initial_state=prefix.final_state,
    )
    torch.testing.assert_close(
        torch.cat((prefix.outputs[0], suffix.outputs[0]), dim=2),
        full.outputs[0],
    )


def test_packed_empty_sequences_preserve_state_and_offsets_are_runtime_data():
    operator = StatefulOperator(_accumulator_ir())
    dense = _inputs(length=3)
    packed = {name: value.squeeze(0).transpose(0, 1).contiguous() for name, value in dense.items()}
    # canonical packed tensors are [tokens, heads, features...]
    packed = {name: value.transpose(0, 1).contiguous() if value.shape[0] == 1 else value for name, value in packed.items()}
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
    assert result.outputs[0].shape[0] == 3


def test_structured_errors_are_json_safe_and_have_canonical_paths():
    operator = StatefulOperator(_accumulator_ir())
    with pytest.raises(StatefulCompilationError) as captured:
        operator(query=torch.empty(0), key=torch.empty(0), value=torch.empty(0))
    error = captured.value
    assert error.category == "RUNTIME_BINDING"
    assert error.path == "inputs"
    json.dumps(error.as_dict())


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
