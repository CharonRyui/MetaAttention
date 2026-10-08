"""Behavioral regression tests for the Stateful performance measurement contract."""

from dataclasses import replace

import pytest
import torch

from testing.benchmark_stateful_plans import (
    CASES,
    _correctness,
    _split_inputs,
    _token_oracle,
    inputs_for,
    role_sizes_for,
)

pytestmark = pytest.mark.unit


def _workload(name, length=5):
    case = replace(next(case for case in CASES if case.name == name),
                   batch=2, heads=2, dim=3, dim_value=2)
    operator = case.factory(**case.kwargs(length))
    sizes = role_sizes_for(operator, case, length)
    torch.manual_seed(37)
    inputs = inputs_for(operator, case, length, torch.device("cpu"))
    return operator, sizes, inputs


def test_missing_final_state_is_invalid_only_for_state_contracts():
    expected = {"output": torch.tensor([[[[2.0]]]])}
    states = {"memory": torch.tensor([[[[3.0]]]])}
    assert _correctness(expected, None, expected, states, "output-only")["status"] == "VALID"
    assert _correctness(expected, None, expected, states, "final-state")["status"] == "INVALID"
    assert _correctness(expected, None, expected, states, "continuation")["status"] == "INVALID"




@pytest.mark.parametrize("name", ["linear", "simple-gla", "gla", "gdn", "gsa", "based", "rebased", "rwkv6", "rwkv7", "hgrn"])
def test_token_recurrence_continuation_preserves_all_states(name):
    operator, sizes, inputs = _workload(name)
    full_outputs, full_states = _token_oracle(operator, inputs, sizes)
    prefix_inputs, suffix_inputs = _split_inputs(operator, inputs, 2)
    prefix_outputs, prefix_states = _token_oracle(operator, prefix_inputs, sizes)
    suffix_outputs, suffix_states = _token_oracle(
        operator, suffix_inputs, sizes, initial_state=prefix_states,
    )
    for output_name, expected in full_outputs.items():
        torch.testing.assert_close(
            torch.cat((prefix_outputs[output_name], suffix_outputs[output_name]), dim=2), expected,
            rtol=0, atol=0,
        )
    for state_name, expected in full_states.items():
        torch.testing.assert_close(suffix_states[state_name], expected, rtol=0, atol=0)


def test_rank_one_oracle_applies_decay_before_removal():
    operator, sizes, inputs = _workload("gdn", length=1)
    initial = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3, 1).expand(-1, -1, -1, 2).clone()
    _, actual = _token_oracle(operator, inputs, sizes, initial_state={"memory": initial})
    key = inputs["key"][:, :, 0].float()
    value = inputs["value"][:, :, 0].float()
    beta = inputs["beta"][:, :, 0].float()
    decayed = initial * inputs["gate"][:, :, 0].float().exp()[:, :, None, None]
    removed = torch.einsum("bhk,bhkv->bhv", key, decayed)
    expected = decayed + beta[:, :, None, None] * key.unsqueeze(-1) * (value - removed).unsqueeze(-2)
    torch.testing.assert_close(actual["memory"], expected, rtol=1e-6, atol=1e-6)


def test_based_oracle_matches_polynomial_attention():
    operator, sizes, inputs = _workload("based")
    actual, _ = _token_oracle(operator, inputs, sizes)
    q, k, v = (inputs[name].float() for name in ("query", "key", "value"))
    scores = torch.einsum("bhtk,bhsk->bhts", q * (q.shape[-1] ** -0.5), k)
    weights = (1 + scores + 0.5 * scores.square()).tril()
    expected = torch.einsum("bhts,bhsv->bhtv", weights, v) / (weights.sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual["output"], expected, rtol=1e-6, atol=1e-6)
