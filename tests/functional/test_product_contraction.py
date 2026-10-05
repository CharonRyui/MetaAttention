"""ProductFeature contraction order, explicit carries and polynomial readouts.

Oracles replay independently in FP64; no FLA results or optimized helper output
is used as the correctness reference.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import ProductFeatureRole
from attn_engine.stateful.lowering.chunked import execute_chunked_states
from attn_engine.stateful.lowering.diagonal.diagonal_kernels import prepare_state
from attn_engine.stateful.lowering.readout.product_kernels import product_chunk_scores, read_product_state
from examples.based_attention import based_attention
from examples.rebased_attention import rebased_attention

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_product_factors_order_carry_and_continuation(gpu_device, prior, reverse):
    batch, heads, length, columns = 2, 2, 131, 35
    left, right = (7, 35) if reverse else (35, 7)
    generator = torch.Generator(device=gpu_device).manual_seed(101)

    def random(size):
        return torch.randn(
            batch, length, heads, size, device=gpu_device, generator=generator
        ) * 0.2

    kl, kr, ql, qr = random(left), random(right), random(left), random(right)
    value = random(columns)
    initial = torch.randn(
        batch, heads, left * right, columns, device=gpu_device, generator=generator
    ) * 0.1
    denominator_initial = torch.randn(
        batch, heads, left * right, 1, device=gpu_device, generator=generator
    ) * 0.1

    from attn_engine import (
        AlgorithmIR, Batch, FeatureRole, HeadMapping, HeadRole, Input, Outer,
        ProductFactor, ProductInjection, Readout, Sequence, StateSpec, StateTerm,
        StateTransition, TensorInput,
    )

    left_role, right_role, value_role = (FeatureRole(name) for name in ("left", "right", "value"))
    pair_role = ProductFeatureRole(left_role, right_role)
    head_role = HeadRole("head")
    key_pair = Outer(Input("kl"), Input("kr"), pair_role)
    query_pair = Outer(Input("ql"), Input("qr"), pair_role)
    algorithm = AlgorithmIR(
        inputs=tuple(TensorInput(name, (Batch, Sequence, head_role, role), torch.float32)
                     for name, role in (("kl", left_role), ("kr", right_role),
                                        ("ql", left_role), ("qr", right_role), ("v", value_role))),
        states=(StateSpec("memory", (Batch, head_role, pair_role, value_role)),),
        transition=(StateTransition("memory", injections=(ProductInjection((
            ProductFactor(key_pair, (pair_role,)), ProductFactor(Input("v"), (value_role,)),
        )),)),),
        readouts=(Readout("out", StateTerm("memory", query_pair, pair_role, prior=prior), torch.float32),),
        head_mapping=HeadMapping({head_role: head_role}),
    )

    def run(start, end, state, denominator_state, reuse):
        count = end - start
        key = tuple(part[:, start:end].reshape(batch * count, heads, -1).contiguous()
                    for part in (kl, kr))
        query = tuple(part[:, start:end].reshape(batch * count, heads, -1).contiguous()
                      for part in (ql, qr))
        values = value[:, start:end].reshape(batch * count, heads, columns).contiguous()
        carry, final, factors = prepare_state(
            rows_gate=None, columns_gate=None, key=key, value=values, initial=state,
            sequence_length=count, return_final_state=True, identity=True,
        )
        ones = torch.ones(batch * count, heads, 1, device=gpu_device)
        den_carry, den_final, den_factors = prepare_state(
            rows_gate=None, columns_gate=None, key=key, value=ones,
            initial=denominator_state, sequence_length=count,
            return_final_state=True, identity=True,
        )
        scores = product_chunk_scores(key, query, sequence_length=count, prior=prior) if reuse else None
        output = read_product_state(
            carry, (factors[2], factors[3]), query, factors[4],
            sequence_length=count, prior=prior, scores=scores,
        )
        denominator = read_product_state(
            den_carry, (den_factors[2], den_factors[3]), query, den_factors[4],
            sequence_length=count, prior=prior, scores=scores,
        )
        return (output.reshape(batch, count, heads, columns),
                denominator.reshape(batch, count, heads, 1), final, den_final)

    with torch.no_grad():
        actual, actual_den, final, den_final = run(0, length, initial, denominator_initial, False)
        shared, shared_den, _, _ = run(0, length, initial, denominator_initial, True)
        first, first_den, middle, middle_den = run(0, 65, initial, denominator_initial, True)
        second, second_den, continued, continued_den = run(65, length, middle, middle_den, True)
        typed_output, typed_final = execute_chunked_states(
            algorithm,
            values={name: tensor.reshape(batch * length, heads, -1)
                    for name, tensor in {"kl": kl, "kr": kr, "ql": ql, "qr": qr, "v": value}.items()},
            initial_states={"memory": initial},
            segment=torch.arange(batch, device=gpu_device).repeat_interleave(length),
            role_sizes={left_role: left, right_role: right, pair_role: left * right, value_role: columns},
            heads=heads, token_count=batch * length, device=gpu_device,
            return_final_state=True, sequence_length=length,
        )

    state = initial.double().reshape(batch, heads, left, right, columns).clone()
    den_state = denominator_initial.double().reshape(batch, heads, left, right, 1).clone()
    expected, expected_den = [], []
    for token in range(length):
        if not prior:
            state += torch.einsum("bhi,bhj,bhc->bhijc", kl[:, token].double(), kr[:, token].double(), value[:, token].double())
            den_state += torch.einsum("bhi,bhj->bhij", kl[:, token].double(), kr[:, token].double()).unsqueeze(-1)
        expected.append(torch.einsum("bhi,bhj,bhijc->bhc", ql[:, token].double(), qr[:, token].double(), state))
        expected_den.append(torch.einsum("bhi,bhj,bhijc->bhc", ql[:, token].double(), qr[:, token].double(), den_state))
        if prior:
            state += torch.einsum("bhi,bhj,bhc->bhijc", kl[:, token].double(), kr[:, token].double(), value[:, token].double())
            den_state += torch.einsum("bhi,bhj->bhij", kl[:, token].double(), kr[:, token].double()).unsqueeze(-1)
    expected = torch.stack(expected, 1).float()
    expected_den = torch.stack(expected_den, 1).float()
    for output in (actual, shared, torch.cat((first, second), 1),
                   typed_output["out"].reshape(batch, length, heads, columns)):
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
    for output in (actual_den, shared_den, torch.cat((first_den, second_den), 1)):
        torch.testing.assert_close(output, expected_den, atol=2e-5, rtol=2e-4)
    for output in (final, continued, typed_final["memory"]):
        torch.testing.assert_close(output, state.flatten(2, 3).float(), atol=2e-5, rtol=2e-4)
    for output in (den_final, continued_den):
        torch.testing.assert_close(output, den_state.flatten(2, 3).float(), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("family", ["based", "rebased"])
def test_polynomial_readout_initial_final_and_continuation(gpu_device, family):
    batch, heads, length, dim, columns = 2, 2, 131, 5, 3
    factory = based_attention if family == "based" else rebased_attention
    algorithm = factory(batch, heads, length, dim, columns, dtype=torch.float32).algorithm
    generator = torch.Generator(device=gpu_device).manual_seed(109)
    # Positive Q/K keep the normalizer away from cancellation, while values and
    # nonzero initial numerators exercise signed arithmetic and expanded carry.
    query = torch.rand(batch, length, heads, dim, device=gpu_device, generator=generator) * 0.5
    key = torch.rand(batch, length, heads, dim, device=gpu_device, generator=generator) * 0.5
    value = torch.randn(batch, length, heads, columns, device=gpu_device, generator=generator) * 0.2
    roles = {}
    for state in algorithm.states:
        for role in state.feature_roles:
            if isinstance(role, ProductFeatureRole):
                roles[role] = dim * dim
                roles[role.left] = roles[role.right] = dim
            elif role.name == "key_dim":
                roles[role] = dim
            elif role.name == "value_dim":
                roles[role] = columns
            else:
                roles[role] = 1
    initial = {}
    for state in algorithm.states:
        shape = (batch, heads, *(roles[role] for role in state.feature_roles))
        if state.name.startswith("denominator"):
            initial[state.name] = torch.full(shape, 0.25, device=gpu_device)
        else:
            initial[state.name] = torch.randn(shape, device=gpu_device, generator=generator) * 0.1

    def run(start, end, state):
        count = end - start
        outputs, final = execute_chunked_states(
            algorithm,
            values={name: tensor[:, start:end].reshape(batch * count, heads, -1)
                    for name, tensor in {"query": query, "key": key, "value": value}.items()},
            initial_states=state,
            segment=torch.arange(batch, device=gpu_device).repeat_interleave(count),
            role_sizes=roles, heads=heads, token_count=batch * count,
            device=gpu_device, return_final_state=True, sequence_length=count,
        )
        return outputs["output"].reshape(batch, count, heads, columns), final

    with torch.no_grad():
        actual, final = run(0, length, initial)
        first, middle = run(0, 65, initial)
        second, continued = run(65, length, middle)
    oracle = {name: tensor.double().clone() for name, tensor in initial.items()}
    expected = []
    for token in range(length):
        # Authored scale is applied in FP32 before the product map is built.
        q = (query[:, token].float() * dim**-0.5).double()
        k, v = key[:, token].double(), value[:, token].double()
        pair = torch.einsum("bhi,bhj->bhij", k, k).flatten(-2)
        q_pair = torch.einsum("bhi,bhj->bhij", q, q).flatten(-2)
        if family == "based":
            oracle["numerator_0"] += v
            oracle["denominator_0"] += 1
            oracle["numerator_1"] += torch.einsum("bhi,bhc->bhic", k, v)
            oracle["denominator_1"] += k
            oracle["numerator_2"] += torch.einsum("bhi,bhc->bhic", pair, v)
            oracle["denominator_2"] += pair
            numerator = (oracle["numerator_0"]
                         + torch.einsum("bhi,bhic->bhc", q, oracle["numerator_1"])
                         + 0.5 * torch.einsum("bhi,bhic->bhc", q_pair, oracle["numerator_2"]))
            denominator = (oracle["denominator_0"]
                           + torch.einsum("bhi,bhi->bh", q, oracle["denominator_1"]).unsqueeze(-1)
                           + 0.5 * torch.einsum("bhi,bhi->bh", q_pair, oracle["denominator_2"]).unsqueeze(-1)
                           + 1e-6)
        else:
            oracle["numerator"] += torch.einsum("bhi,bhc->bhic", pair, v)
            oracle["denominator"] += pair
            numerator = torch.einsum("bhi,bhic->bhc", q_pair, oracle["numerator"])
            denominator = torch.einsum("bhi,bhi->bh", q_pair, oracle["denominator"]).unsqueeze(-1) + 1e-5
        expected.append(numerator / denominator)
    expected = torch.stack(expected, 1).float()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(torch.cat((first, second), 1), expected, atol=2e-5, rtol=2e-4)
    for name, state in oracle.items():
        torch.testing.assert_close(final[name], state.float(), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(continued[name], state.float(), atol=2e-5, rtol=2e-4)
