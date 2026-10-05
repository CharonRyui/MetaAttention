"""Behaviour regressions for the private pointwise expression DAG seam."""
from __future__ import annotations
import pytest
import torch

from attn_engine import Batch, Constant, FeatureRole, HeadRole, Input, Sequence, StateTerm, TensorInput
from attn_engine.stateful.lowering.expression.expression import compile_expression


@pytest.fixture
def isolated_expression_kernel_cache(monkeypatch):
    from attn_engine.stateful.lowering.expression import expression_kernels as kernels

    cache: dict[tuple[object, ...], object] = {}
    monkeypatch.setattr(kernels, "_CACHE", cache)
    return cache


def test_shared_subexpression_is_evaluated_as_one_dag_node():
    x, y = Input("x"), Input("y")
    shared = x + y
    compiled = compile_expression({"square": shared * shared, "shifted": shared + 1.0})

    result = compiled({"x": torch.tensor([2.0]), "y": torch.tensor([3.0])})

    torch.testing.assert_close(result["square"], torch.tensor([25.0]))
    torch.testing.assert_close(result["shifted"], torch.tensor([6.0]))


def test_dag_preserves_authored_association_and_intermediate_dtype_rounding():
    x = Input("x")
    # BF16's first addition rounds before the multiplication. Comparing with a
    # separately authored FP32 reassociation would hide this observable contract.
    expression = (x + Constant(0.1, torch.bfloat16)) * Constant(1.0001)
    value = torch.tensor([1.003], dtype=torch.bfloat16)
    result = compile_expression(expression)({"x": value})

    expected = (value + torch.tensor(0.1, dtype=torch.bfloat16)) * torch.tensor(
        1.0001, dtype=torch.bfloat16
    )
    reassociated = value.float() + 0.1 * 1.0001
    torch.testing.assert_close(result, expected)
    assert not torch.equal(result.float(), reassociated)


def test_strided_and_broadcast_leaves_are_loaded_without_flattening():
    x, bias = Input("x"), Input("bias")
    base = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    strided = base[:, ::2]
    result = compile_expression(x / bias)(
        {"x": strided, "bias": torch.tensor([[1.0, 2.0, 4.0]])}
    )

    torch.testing.assert_close(result, strided / torch.tensor([[1.0, 2.0, 4.0]]))
    assert result.shape == (2, 3)


def test_state_terms_are_materialized_leaves_not_recursive_input_evaluation():
    query = Input("query")
    term = StateTerm("memory", query, FeatureRole("key"))
    compiled = compile_expression(term + term)
    materialized = torch.tensor([[1.5, 2.5]], dtype=torch.float32)

    result = compiled({}, {term: materialized})
    torch.testing.assert_close(result, materialized + materialized)


def test_divide_and_unary_nodes_keep_pointwise_torch_order():
    x = Input("x")
    expression = ((x + 2.0).sqrt() / (x + 3.0)).reciprocal()
    value = torch.tensor([1.0, 3.0], dtype=torch.float32)
    result = compile_expression(expression)({"x": value})

    expected = ((value + 2.0).sqrt() / (value + 3.0)).reciprocal()
    torch.testing.assert_close(result, expected)

@pytest.mark.gpu
def test_cuda_pointwise_dag_uses_fused_multi_output_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = torch.full((16,), 2.0, device="cuda", dtype=torch.float32)
    expression = compile_expression(
        {"sum": Input("x") + Input("y"), "scaled": (Input("x") + Input("y")) * 3.0}
    )
    actual = expression({"x": x, "y": y})
    torch.cuda.synchronize()
    torch.testing.assert_close(actual["sum"], x + y)
    torch.testing.assert_close(actual["scaled"], (x + y) * 3.0)

@pytest.mark.gpu
def test_cuda_expression_cache_distinguishes_operator_constants_and_root_order(
    isolated_expression_kernel_cache,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    first = compile_expression({"left": Input("x") + 1.0, "right": Input("x") * 2.0})
    second = compile_expression({"left": Input("x") * 2.0, "right": Input("x") + 3.0})

    first_actual = first({"x": x})
    second_actual = second({"x": x})
    torch.cuda.synchronize()
    torch.testing.assert_close(first_actual["left"], x + 1.0)
    torch.testing.assert_close(first_actual["right"], x * 2.0)
    torch.testing.assert_close(second_actual["left"], x * 2.0)
    torch.testing.assert_close(second_actual["right"], x + 3.0)


@pytest.mark.gpu
def test_cuda_expression_cache_distinguishes_descriptor_address_formulas(
    isolated_expression_kernel_cache,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from attn_engine.stateful.runtime.bindings import build_binding

    feature = FeatureRole("feature")
    first_head = HeadRole("first_head")
    second_head = HeadRole("second_head")
    first_spec = TensorInput("x", (Batch, Sequence, first_head, feature), torch.float32)
    second_spec = TensorInput("x", (Batch, second_head, Sequence, feature), torch.float32)
    first_raw = torch.arange(16, device="cuda", dtype=torch.float32).reshape(1, 4, 2, 2)
    second_raw = torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 2)
    first = build_binding(
        first_spec, first_raw, role_sizes={first_head: 4, feature: 2}, mode="dense",
        state_heads=4, feature_roles=(feature,),
    )
    second = build_binding(
        second_spec, second_raw, role_sizes={second_head: 4, feature: 2}, mode="dense",
        state_heads=4, feature_roles=(feature,),
    )
    expression = compile_expression(Input("x"))

    first_actual = expression({"x": first})
    second_actual = expression({"x": second})
    first_expected = first_raw[0].reshape(4, 2, 2)[:, torch.arange(4, device="cuda") // 2]
    second_expected = second_raw[0, 0].unsqueeze(1).expand(4, 4, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(first_actual, first_expected)
    torch.testing.assert_close(second_actual, second_expected)

    strided_storage = torch.arange(32, device="cuda", dtype=torch.float32).reshape(1, 4, 2, 4)
    strided_raw = strided_storage[..., ::2]
    strided = build_binding(
        first_spec, strided_raw, role_sizes={first_head: 4, feature: 2}, mode="dense",
        state_heads=4, feature_roles=(feature,),
    )
    strided_actual = expression({"x": strided})
    strided_expected = strided_raw[0].reshape(4, 2, 2)[:, torch.arange(4, device="cuda") // 2]
    torch.cuda.synchronize()
    torch.testing.assert_close(strided_actual, strided_expected)
