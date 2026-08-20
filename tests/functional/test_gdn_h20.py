from __future__ import annotations

import pytest
import torch

from attention_engine.attn_engine.gdn_reference import gated_delta_rule_reference
from attn_engine import StateTuple
from examples.gated_delta_rule import (
    compositional_pressure_profile,
    gated_delta_rule,
)


pytestmark = [
    pytest.mark.functional,
    pytest.mark.gpu,
    pytest.mark.h20,
    pytest.mark.usefixtures("gpu_device", "seed"),
]


def _bindings(device: torch.device, *, length: int = 64):
    batch, query_heads, state_heads, dim = 1, 1, 2, 128
    query = torch.randn(
        batch,
        query_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = (
        torch.nn.functional.normalize(torch.randn_like(query), p=2, dim=-1)
        .detach()
        .requires_grad_()
    )
    value = torch.randn(
        batch,
        state_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = (-torch.rand(batch, state_heads, length, device=device)).requires_grad_()
    beta = torch.rand(batch, state_heads, length, device=device, requires_grad=True)
    return query, key, value, gate, beta


def test_gdn_generic_profile_matches_independent_reference(gpu_device):
    query, key, value, gate, beta = _bindings(gpu_device)
    initial = torch.randn(
        1, 2, 128, 128, device=gpu_device, dtype=torch.float32, requires_grad=True
    )
    operator = gated_delta_rule(1, 2, 64, 128, 128, HQ=1)
    result = operator(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected_output, expected_state = gated_delta_rule_reference(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
    )
    assert result.final_state is not None
    assert expected_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.1, atol=0.1
    )


def test_gdn_continuation_matches_full_invocation(gpu_device):
    query, key, value, gate, beta = _bindings(gpu_device, length=128)
    operator = gated_delta_rule(1, 2, 128, 128, 128, HQ=1)
    full = operator(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        return_final_state=True,
    )
    prefix = operator(
        query=query[:, :, :64].contiguous(),
        key=key[:, :, :64].contiguous(),
        value=value[:, :, :64].contiguous(),
        gate=gate[:, :, :64].contiguous(),
        beta=beta[:, :, :64].contiguous(),
        return_final_state=True,
    )
    assert prefix.final_state is not None
    suffix = operator(
        query=query[:, :, 64:].contiguous(),
        key=key[:, :, 64:].contiguous(),
        value=value[:, :, 64:].contiguous(),
        gate=gate[:, :, 64:].contiguous(),
        beta=beta[:, :, 64:].contiguous(),
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    assert full.final_state is not None
    assert suffix.final_state is not None
    torch.testing.assert_close(
        torch.cat((prefix.outputs[0], suffix.outputs[0]), dim=2),
        full.outputs[0],
        rtol=0.1,
        atol=0.1,
    )
    torch.testing.assert_close(
        suffix.final_state[0], full.final_state[0], rtol=0.1, atol=0.1
    )


def _pressure_bindings(device: torch.device, lengths: tuple[int, ...]):
    generator = torch.Generator(device=device).manual_seed(29)
    token_count = sum(lengths)

    def sample(heads: int, features: int, dtype=torch.bfloat16):
        return (
            0.05
            * torch.randn(
                token_count,
                heads,
                features,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        ).requires_grad_()

    return {
        "query": sample(2, 64),
        "auxiliary_query": sample(2, 64),
        "left": sample(2, 64),
        "right": sample(2, 64),
        "left_gate": (
            -0.01 * torch.rand(token_count, 4, 64, device=device, generator=generator)
        ).requires_grad_(),
        "right_gate": (
            -0.01 * torch.rand(token_count, 4, 128, device=device, generator=generator)
        ).requires_grad_(),
        "coefficient": (
            0.001 * torch.randn(token_count, 4, device=device, generator=generator)
        ).requires_grad_(),
        "value": sample(2, 128),
    }


def _pressure_reference(bindings, lengths, initial):
    repeated = {
        name: value.repeat_interleave(2, dim=1) if value.shape[1] == 2 else value
        for name, value in bindings.items()
    }
    outputs = []
    auxiliaries = []
    finals = []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence]
        for token in range(offset, offset + length):
            left_gate = repeated["left_gate"][token].float().exp()
            right_gate = repeated["right_gate"][token].float().exp()
            left = repeated["left"][token].float()
            right = repeated["right"][token].float()
            value = repeated["value"][token].float()
            coefficient = repeated["coefficient"][token].float()
            state = left_gate[:, :, None] * state
            contraction = torch.einsum("hd,hdf->hf", right, state)
            state = (
                state
                + coefficient[:, None, None]
                * left[:, :, None]
                * contraction[:, None, :]
            )
            state = state * right_gate[:, None, :]
            state = state + left[:, :, None] * value[:, None, :]
            state = (
                state
                + right[:, :, None] * value[:, None, :] * coefficient[:, None, None]
            )
            outputs.append(
                torch.einsum("hd,hdf->hf", repeated["query"][token].float(), state).to(
                    torch.bfloat16
                )
            )
            auxiliaries.append(
                torch.einsum(
                    "hd,hdf->hf",
                    repeated["auxiliary_query"][token].float(),
                    state,
                ).to(torch.bfloat16)
            )
        finals.append(state)
        offset += length
    output_shape = (0, 4, 128)
    return (
        torch.stack(outputs)
        if outputs
        else initial.new_empty(output_shape).to(torch.bfloat16),
        torch.stack(auxiliaries)
        if auxiliaries
        else initial.new_empty(output_shape).to(torch.bfloat16),
        torch.stack(finals),
    )


def test_compositional_pressure_profile_packed_forward_backward(gpu_device):
    lengths = (0, 65, 0, 64)
    offsets = torch.tensor((0, 0, 65, 65, 129), device=gpu_device, dtype=torch.int32)
    actual_bindings = _pressure_bindings(gpu_device, lengths)
    reference_bindings = {
        name: value.detach().clone().requires_grad_()
        for name, value in actual_bindings.items()
    }
    actual_initial = (
        0.05 * torch.randn(4, 4, 64, 128, device=gpu_device, dtype=torch.float32)
    ).requires_grad_()
    reference_initial = actual_initial.detach().clone().requires_grad_()
    operator = compositional_pressure_profile()
    result = operator(
        **actual_bindings,
        sequence_offsets=offsets,
        initial_state=StateTuple(("memory",), (actual_initial,)),
        return_final_state=True,
    )
    expected_output, expected_auxiliary, expected_state = _pressure_reference(
        reference_bindings, lengths, reference_initial
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.04, atol=0.1
    )
    torch.testing.assert_close(
        result.outputs["auxiliary"], expected_auxiliary, rtol=0.04, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state[0], expected_state, rtol=0.04, atol=0.1
    )
    assert torch.equal(result.final_state[0][0], actual_initial[0])
    assert torch.equal(result.final_state[0][2], actual_initial[2])

    actual_loss = sum(output.float().square().mean() for output in result.outputs)
    actual_loss = actual_loss + result.final_state[0].square().mean()
    expected_loss = expected_output.float().square().mean()
    expected_loss = expected_loss + expected_auxiliary.float().square().mean()
    expected_loss = expected_loss + expected_state.square().mean()
    actual_gradients = torch.autograd.grad(
        actual_loss, (*actual_bindings.values(), actual_initial)
    )
    expected_gradients = torch.autograd.grad(
        expected_loss, (*reference_bindings.values(), reference_initial)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.2)


def test_compositional_pressure_profile_dense_unaligned_tail(gpu_device):
    lengths = (129,)
    packed = _pressure_bindings(gpu_device, lengths)
    dense = {
        name: value.transpose(0, 1).unsqueeze(0).contiguous()
        for name, value in packed.items()
    }
    initial = 0.05 * torch.randn(1, 4, 64, 128, device=gpu_device)
    result = compositional_pressure_profile()(
        **dense,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected_output, expected_auxiliary, expected_state = _pressure_reference(
        packed, lengths, initial
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"],
        expected_output.permute(1, 0, 2).unsqueeze(0),
        rtol=0.04,
        atol=0.1,
    )
    torch.testing.assert_close(
        result.outputs["auxiliary"],
        expected_auxiliary.permute(1, 0, 2).unsqueeze(0),
        rtol=0.04,
        atol=0.1,
    )
    torch.testing.assert_close(
        result.final_state[0], expected_state, rtol=0.04, atol=0.1
    )


def test_compositional_pressure_profile_all_empty(gpu_device):
    operator = compositional_pressure_profile()
    bindings = _pressure_bindings(gpu_device, (0, 0))
    initial = torch.randn(2, 4, 64, 128, device=gpu_device)
    result = operator(
        **bindings,
        sequence_offsets=torch.tensor((0, 0, 0), device=gpu_device, dtype=torch.int32),
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    assert result.outputs["output"].shape == (0, 4, 128)
    assert result.final_state is not None
    assert torch.equal(result.final_state[0], initial)
