"""Scalar/identity dense schedules against an independent token recurrence."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StateTuple,
    StatefulOperator,
    Target,
    TensorInput,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

KEY = FeatureRole("scalar_test_key")
VALUE = FeatureRole("scalar_test_value")
QUERY_HEADS = HeadRole("scalar_test_query_heads")
KEY_HEADS = HeadRole("scalar_test_key_heads")
VALUE_HEADS = HeadRole("scalar_test_value_heads")
STATE_HEADS = HeadRole("scalar_test_state_heads")


def _compile(*, heads, key_heads, value_heads, rows, columns, dtype, gate_kind):
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
    ]
    propagations = ()
    if gate_kind is not None:
        inputs.append(
            TensorInput("gate", (Batch, VALUE_HEADS, Sequence), torch.float32)
        )
        gate = Input("gate")
        propagations = (AxisScale(Exp(gate) if gate_kind == "log" else gate),)
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=tuple(inputs),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                propagations=propagations,
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                StateContraction(
                    "output", "memory", Input("query") * rows**-0.5, KEY, dtype
                ),
            ),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )
    return operator.compile(
        role_sizes={
            QUERY_HEADS: heads,
            KEY_HEADS: key_heads,
            VALUE_HEADS: value_heads,
            STATE_HEADS: heads,
            KEY: rows,
            VALUE: columns,
        },
        target=Target(),
        mode="dense",
        optimized=True,
    )


def _inputs(batch, heads, key_heads, value_heads, length, rows, columns, dtype, gate_kind, device):
    generator = torch.Generator().manual_seed(31 + batch + length)
    inputs = {
        "query": torch.randn(batch, heads, length, rows, generator=generator).to(dtype),
        "key": torch.randn(batch, key_heads, length, rows, generator=generator).to(dtype),
        "value": torch.randn(batch, value_heads, length, columns, generator=generator).to(dtype),
    }
    if gate_kind is not None:
        gate = -torch.rand(batch, value_heads, length, generator=generator) * 0.05
        gate[:, :, :2] = 0.0
        for edge in range(64, length, 64):
            gate[:, :, edge - 1 : edge + 2] = 0.0
        if gate_kind == "direct":
            gate = gate.exp()
            # Products must remain valid across zero and negative factors. The
            # reset is deliberately before/at/after a chunk boundary.
            gate[:, :, 30] = 0.0
            gate[:, :, 31::17] *= -1.0
            gate[:, :, 63:66] = 0.0
        inputs["gate"] = gate
    initial = torch.randn(batch, heads, rows, columns, generator=generator) * 0.2
    return {name: tensor.to(device) for name, tensor in inputs.items()}, initial.to(device)


def _reference(inputs, initial, gate_kind):
    # This oracle does not use a chunk summary, prefix ratios, or the runtime's
    # lowering helpers. Each authored token transition writes FP32 State.
    query = inputs["query"].cpu().float()
    heads = query.shape[1]
    key = inputs["key"].cpu().float()
    value = inputs["value"].cpu().float()
    key = key.repeat_interleave(heads // key.shape[1], dim=1)
    value = value.repeat_interleave(heads // value.shape[1], dim=1)
    gate = None
    if gate_kind is not None:
        gate = inputs["gate"].cpu().float()
        gate = gate.repeat_interleave(heads // gate.shape[1], dim=1)
        if gate_kind == "log":
            gate = gate.exp()
    state = initial.cpu().clone()
    output = []
    for token in range(query.shape[2]):
        if gate is not None:
            state = gate[:, :, token, None, None] * state
        state = state + key[:, :, token, :, None] * value[:, :, token, None, :]
        output.append(
            torch.einsum(
                "bhr,bhrc->bhc", query[:, :, token] * query.shape[-1]**-0.5, state
            )
        )
    return torch.stack(output, dim=2).to(inputs["query"].dtype), state


def _assert_result(result, expected, expected_final):
    torch.testing.assert_close(
        result.outputs["output"].cpu(), expected, rtol=0.02, atol=0.05
    )
    assert result.final_state is not None
    assert result.final_state["memory"].dtype == torch.float32
    torch.testing.assert_close(
        result.final_state["memory"].cpu(), expected_final, rtol=0.01, atol=0.02
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("gate_kind", [None, "log"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_scalar_lengths_final_state_continuation_and_graph(batch, gate_kind, dtype, gpu_device):
    """One public executable crosses streaming, chunk-grid and tail workloads."""
    heads, rows, columns = 2, 64, 64
    compiled = _compile(
        heads=heads, key_heads=heads, value_heads=heads,
        rows=rows, columns=columns, dtype=dtype, gate_kind=gate_kind,
    )
    for length in (63, 64, 65, 130, 2048):
        inputs, initial = _inputs(
            batch, heads, heads, heads, length, rows, columns, dtype, gate_kind, gpu_device
        )
        initial_before = initial.clone()
        expected, expected_final = _reference(inputs, initial, gate_kind)
        with torch.no_grad():
            result = compiled(
                **inputs, initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
            output_only = compiled(
                **inputs, initial_state=StateTuple(("memory",), (initial,))
            )
        _assert_result(result, expected, expected_final)
        assert output_only.final_state is None
        torch.testing.assert_close(
            output_only.outputs["output"].cpu(), expected, rtol=0.02, atol=0.05
        )
        torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)

        # A nonaligned prefix moves the injection at a former chunk boundary
        # into another chunk, so a mistaken carry or tail changes continuation.
        split = min(65, length - 1)
        prefix_inputs = {name: value[:, :, :split].contiguous() for name, value in inputs.items()}
        suffix_inputs = {name: value[:, :, split:].contiguous() for name, value in inputs.items()}
        with torch.no_grad():
            prefix = compiled(
                **prefix_inputs, initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
            suffix = compiled(
                **suffix_inputs, initial_state=prefix.final_state, return_final_state=True
            )
        continued = torch.cat((prefix.outputs["output"], suffix.outputs["output"]), dim=2)
        torch.testing.assert_close(continued.cpu(), expected, rtol=0.02, atol=0.05)
        assert suffix.final_state is not None
        torch.testing.assert_close(
            suffix.final_state["memory"].cpu(), expected_final, rtol=0.01, atol=0.02
        )

        # Kernel compilation happens above, outside capture. Fresh output and
        # scratch allocations during capture must remain valid for replay.
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            captured = compiled(
                **inputs, initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
        graph.replay()
        _assert_result(captured, expected, expected_final)


@pytest.mark.parametrize("length", [65, 130, 2048])
def test_scalar_signed_zero_gates_and_grouped_heads(length, gpu_device):
    """Zero factors cannot be handled by an inverse gate or log(prefix) ratio."""
    heads, key_heads, value_heads, rows, columns = 4, 2, 1, 128, 64
    compiled = _compile(
        heads=heads, key_heads=key_heads, value_heads=value_heads,
        rows=rows, columns=columns, dtype=torch.float32, gate_kind="direct",
    )
    inputs, initial = _inputs(
        2, heads, key_heads, value_heads, length, rows, columns,
        torch.float32, "direct", gpu_device,
    )
    expected, expected_final = _reference(inputs, initial, "direct")
    with torch.no_grad():
        result = compiled(
            **inputs, initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    _assert_result(result, expected, expected_final)


def test_log_gate_underflow_does_not_destroy_later_injections(gpu_device):
    length, heads, rows, columns = 130, 2, 64, 128
    compiled = _compile(
        heads=heads, key_heads=heads, value_heads=heads,
        rows=rows, columns=columns, dtype=torch.float32, gate_kind="log",
    )
    inputs, initial = _inputs(
        1, heads, heads, heads, length, rows, columns,
        torch.float32, "log", gpu_device,
    )
    inputs["gate"][:, :, 63:65] = -1000.0
    expected, expected_final = _reference(inputs, initial, "log")
    with torch.no_grad():
        result = compiled(
            **inputs, initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    _assert_result(result, expected, expected_final)


def test_one_scalar_executable_handles_changing_batch_occupancy(gpu_device):
    heads, rows, columns, length = 2, 64, 64, 130
    compiled = _compile(
        heads=heads, key_heads=heads, value_heads=heads,
        rows=rows, columns=columns, dtype=torch.float32, gate_kind="log",
    )
    for batch in (1, 16, 2):
        inputs, initial = _inputs(
            batch, heads, heads, heads, length, rows, columns,
            torch.float32, "log", gpu_device,
        )
        expected, expected_final = _reference(inputs, initial, "log")
        with torch.no_grad():
            result = compiled(
                **inputs, initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
        _assert_result(result, expected, expected_final)


@pytest.mark.parametrize("gate_kind", [None, "direct", "log"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_scalar_parallel_row_column_tails_and_continuation(gate_kind, dtype, gpu_device):
    """Exercise padded GEMM tiles even for a short continuation suffix."""
    from attn_engine.stateful.lowering.scalar.scalar import tilelang_scalar_factorized_dense_raw

    heads, rows, columns, length = 4, 35, 67, 130
    inputs, initial = _inputs(
        1, heads, 2, 1, length, rows, columns, dtype, gate_kind, gpu_device
    )
    def binding(name, scale=1.0, log=False):
        return (SimpleNamespace(name=name),
                SimpleNamespace(operation="identity", scale=scale, is_log_scale=log))
    recipe = SimpleNamespace(
        propagation=binding("__identity_propagation__" if gate_kind is None else "gate",
                            log=gate_kind == "log"),
        left_factor=binding("key"), right_factor=binding("value"),
        readout=binding("query", rows**-0.5),
    )
    def execute(values, state, write_final):
        return tilelang_scalar_factorized_dense_raw(
            values, state, recipe, sequence_count=1, sequence_length=values["query"].shape[2],
            output_dtype=dtype, return_final_state=write_final,
        )
    expected, expected_final = _reference(inputs, initial, gate_kind)
    before = initial.clone()
    with torch.no_grad():
        output, final = execute(inputs, initial, True)
        output_only, absent = execute(inputs, initial, False)
        prefix_inputs = {name: value[:, :, :65].contiguous() for name, value in inputs.items()}
        suffix_inputs = {name: value[:, :, 65:].contiguous() for name, value in inputs.items()}
        prefix, prefix_final = execute(prefix_inputs, initial, True)
        suffix, suffix_final = execute(suffix_inputs, prefix_final, True)
    assert absent is None
    assert final.dtype == torch.float32 and suffix_final.dtype == torch.float32
    for result in (output, output_only, torch.cat((prefix, suffix), dim=2)):
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.05)
    for state in (final, suffix_final):
        torch.testing.assert_close(state.cpu(), expected_final, rtol=0.01, atol=0.02)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
