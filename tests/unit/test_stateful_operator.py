from __future__ import annotations

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
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
    TensorInput,
)


pytestmark = pytest.mark.unit


def _gla_ir() -> AlgorithmIR:
    inputs = (
        TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), torch.bfloat16),
        TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), torch.bfloat16),
        TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), torch.bfloat16),
        TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
    )
    return AlgorithmIR(
        inputs=inputs,
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            state="memory",
            propagation=ElementwiseScale(Input("gate")),
            injection=OuterProduct(Input("key"), Input("value")),
        ),
        readout=MatrixReadout("memory", Input("query")),
        head_mapping=HeadMapping(query="query_heads", key="key_heads", value="value_heads", state="state_heads"),
    )


def test_structurally_equivalent_ir_has_stable_identity():
    first = _gla_ir()
    second = _gla_ir()

    assert first.structural_identity == second.structural_identity
    assert first == second


def test_semantic_change_changes_structural_identity():
    first = _gla_ir()
    second = AlgorithmIR(
        inputs=first.inputs,
        states=first.states,
        transition=StateTransition(
            state="memory",
            propagation=ElementwiseScale(Exp(Input("gate"))),
            injection=first.transition.injection,
        ),
        readout=first.readout,
        head_mapping=first.head_mapping,
    )

    assert first.structural_identity != second.structural_identity


def test_duplicate_state_names_are_rejected():
    ir = _gla_ir()
    with pytest.raises(ValueError, match="duplicate state name 'memory'"):
        AlgorithmIR(
            inputs=ir.inputs,
            states=ir.states + ir.states,
            transition=ir.transition,
            readout=ir.readout,
            head_mapping=ir.head_mapping,
        )


def test_unknown_state_reference_is_rejected():
    ir = _gla_ir()
    with pytest.raises(ValueError, match="unknown state 'other'"):
        AlgorithmIR(
            inputs=ir.inputs,
            states=ir.states,
            transition=StateTransition(
                state="other",
                propagation=ir.transition.propagation,
                injection=ir.transition.injection,
            ),
            readout=ir.readout,
            head_mapping=ir.head_mapping,
        )


def test_head_mapping_rejects_non_divisible_runtime_heads_before_lowering():
    operator = StatefulOperator(_gla_ir())
    tensors = {
        "query": torch.empty(1, 2, 64, 128, dtype=torch.bfloat16),
        "key": torch.empty(1, 2, 64, 128, dtype=torch.bfloat16),
        "value": torch.empty(1, 3, 64, 128, dtype=torch.bfloat16),
        "gate": torch.empty(1, 3, 64, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="state heads .* divisible by query heads"):
        operator(**tensors)


def test_runtime_inputs_bind_by_keyword_name():
    operator = StatefulOperator(_gla_ir())

    with pytest.raises(TypeError, match="missing required input: gate"):
        operator(
            query=torch.empty(0),
            key=torch.empty(0),
            value=torch.empty(0),
        )
    with pytest.raises(TypeError, match="unknown input: typo"):
        operator(
            query=torch.empty(0),
            key=torch.empty(0),
            value=torch.empty(0),
            gate=torch.empty(0),
            typo=torch.empty(0),
        )


def test_normalization_is_rejected_explicitly():
    with pytest.raises(ValueError, match="normalization is not supported"):
        StatefulOperator(_gla_ir(), normalization="sum")


def test_gdn_ordered_propagation_is_valid_and_order_sensitive():
    inputs = (
        TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), torch.bfloat16),
        TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), torch.bfloat16),
        TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), torch.bfloat16),
        TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
        TensorInput("beta", ("batch", "state_heads", "sequence"), torch.float32),
    )
    state = (StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),)
    mapping = HeadMapping(query="query_heads", key="key_heads", value="value_heads", state="state_heads")
    propagation = PropagationComposition(
        (ElementwiseScale(Exp(Input("gate"))), RankOneDelta(Input("key"), Input("beta")))
    )
    transition = StateTransition(
        "memory",
        propagation,
        OuterProduct(Input("key"), Input("value") * Input("beta")),
    )
    forward = AlgorithmIR(
        inputs,
        state,
        transition,
        MatrixReadout("memory", Input("query") * (128**-0.5)),
        mapping,
    )
    reversed_ir = AlgorithmIR(
        inputs,
        state,
        StateTransition(
            "memory",
            PropagationComposition(tuple(reversed(propagation.nodes))),
            transition.injection,
        ),
        forward.readout,
        mapping,
    )

    assert forward.structural_identity != reversed_ir.structural_identity
