from __future__ import annotations

import pytest

from testing.benchmark_fla_stateful import (
    P0_FAMILIES,
    P0_HEADS,
    P0_SEQLENS,
    benchmark_matrix,
    choose_idle_gpu,
    GPUInfo,
)

pytestmark = pytest.mark.unit


def test_first_phase_matrix_is_deterministic_and_complete():
    matrix = benchmark_matrix(
        families=("linear", "gla"),
        batches=(1, 2),
        heads=(1, 8),
        seqlens=(63, 64),
    )
    assert len(matrix) == 2 * 2 * 2 * 2
    assert matrix == benchmark_matrix(
        families=("linear", "gla"),
        batches=(1, 2),
        heads=(1, 8),
        seqlens=(63, 64),
    )
    assert matrix[0] == ("linear", 1, 1, 63, "output-only")


def test_first_phase_defaults_only_cover_supported_families():
    matrix = benchmark_matrix()
    assert {row[0] for row in matrix} == set(P0_FAMILIES)
    assert {row[2] for row in matrix} == set(P0_HEADS)
    assert {row[3] for row in matrix} == set(P0_SEQLENS)


def test_matrix_rejects_unsupported_or_invalid_axes():
    with pytest.raises(ValueError, match="linear"):
        benchmark_matrix(families=("unknown",))
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark_matrix(seqlens=())
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_matrix(heads=(0,))
    with pytest.raises(ValueError, match="continuation"):
        benchmark_matrix(seqlens=(1,), contracts=("continuation",))


def test_extended_family_defaults_are_registered():
    assert set(P0_FAMILIES) == {
        "linear",
        "normalized-linear",
        "simple-gla",
        "gla",
        "retention",
        "retnet",
        "rwkv6",
        "rwkv7",
        "gdn",
        "gsa",
        "hgrn",
        "based",
        "rebased",
    }


def test_idle_gpu_selection_rejects_busy_and_selects_most_free():
    gpus = (
        GPUInfo(0, "busy", 100, 1000, 10, (123,)),
        GPUInfo(1, "idle-small", 100, 9000, 0),
        GPUInfo(2, "idle-large", 100, 12000, 0),
    )
    assert choose_idle_gpu(gpus).index == 2
    with pytest.raises(RuntimeError, match="no GPU"):
        choose_idle_gpu((gpus[0],))


def test_rwkv7_adapter_maps_decay_before_removal():
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "rwkv7")
    operator = case.factory(**case.kwargs(2))
    inputs = benchmark.inputs_for(operator, case, 2, torch.device("cpu"))
    adapter = _fla_call(case, inputs)
    assert adapter is not None
    assert adapter.comparable is True
    values = adapter.prepare(inputs)
    native = {name: value.transpose(1, 2) for name, value in inputs.items()}
    torch.testing.assert_close(
        values["a"], -native["removal"].float() * native["decay"].exp()
    )
    torch.testing.assert_close(
        values["b"], native["key"].float() * native["beta"].unsqueeze(-1)
    )
    torch.testing.assert_close(
        values["value"], native["value"].float() * native["beta"].unsqueeze(-1)
    )


def test_gdn_adapter_passes_gate_and_beta_by_name(monkeypatch):
    import torch
    import fla.ops.gated_delta_rule as fla_gdn
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "gdn")
    operator = case.factory(**case.kwargs(2))
    inputs = benchmark.inputs_for(operator, case, 2, torch.device("cpu"))

    def recurrent(q, k, v, g=None, gk=None, gv=None, beta=None, **kwargs):
        assert gk is None and gv is None
        torch.testing.assert_close(g, inputs["gate"].transpose(1, 2))
        torch.testing.assert_close(beta, inputs["beta"].transpose(1, 2))
        return v, None

    monkeypatch.setattr(fla_gdn, "fused_recurrent_gated_delta_rule", recurrent)
    adapter = _fla_call(case, inputs)
    adapter.invoke(adapter.prepare(inputs), None, False)


def test_meta_failure_status_distinguishes_runtime_resource():
    from testing.benchmark_fla_stateful import _meta_failure_status

    assert (
        _meta_failure_status(RuntimeError("RUNTIME_RESOURCE at workspace"))
        == "META_RUNTIME_REJECTED"
    )
    assert (
        _meta_failure_status(RuntimeError("COMPILE_RESOURCE at workspace"))
        == "META_COMPILE_REJECTED"
    )
    assert _meta_failure_status(ValueError("bad input")) == "META_FAILED"


def test_valid_pair_requires_all_correctness_and_statuses():
    from testing.benchmark_fla_stateful import _valid_pair

    record = {
        "status": "OK",
        "fla_status": "OK",
        "fla_contract_comparable": True,
        "meta_correctness": {"status": "VALID"},
        "fla_correctness": {"status": "VALID"},
        "pair_correctness": {"status": "VALID"},
    }
    assert _valid_pair(record)
    for field in ("meta_correctness", "fla_correctness", "pair_correctness"):
        invalid = {**record, field: {"status": "INVALID"}}
        assert not _valid_pair(invalid)
    assert not _valid_pair({**record, "status": "OK", "fla_status": "FAILED"})


def test_csv_rows_hide_measurements_for_invalid_pairs():
    from testing.benchmark_fla_stateful import _csv_rows

    record = {
        "schema_version": 5,
        "family": "linear",
        "batch": 1,
        "heads": 1,
        "seqlen": 3,
        "contract": "output-only",
        "status": "OK",
        "fla_status": "FAILED",
        "meta": {"public_cuda_event_ms": 1.0},
        "fla": {"public_cuda_event_ms": 2.0},
        "speedup_public_cuda_event_ms": 2.0,
    }
    row = _csv_rows([record])[0]
    assert row["meta_ms"] is None
    assert row["fla_ms"] is None
    assert row["speedup"] is None


@pytest.mark.gpu
def test_retention_long_final_state_and_continuation(gpu_device):
    import torch
    from dataclasses import replace
    from testing import benchmark_stateful_plans as benchmark

    torch.set_num_threads(1)
    torch.manual_seed(42)
    case = replace(
        next(item for item in benchmark.CASES if item.name == "retention"),
        batch=2,
        heads=16,
    )
    operator = case.factory(**case.kwargs(8192))
    sizes = benchmark.role_sizes_for(operator, case, 8192)
    compiled = operator.compile(
        role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True
    )
    inputs = benchmark.inputs_for(operator, case, 8192, gpu_device)
    expected, expected_states = benchmark._token_oracle(operator, inputs, sizes)
    for contract in ("final-state", "continuation"):
        parts = (
            benchmark._split_inputs(operator, inputs, 4096)
            if contract == "continuation"
            else [inputs]
        )
        outputs, states = benchmark._meta_call(compiled, parts, contract)()
        correctness = benchmark._correctness(
            outputs, states, expected, expected_states, contract
        )
        assert correctness["status"] == "VALID", correctness


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["linear", "retention", "retnet"])
def test_long_scalar_families_use_safe_schedule(gpu_device, family):
    import torch
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == family)
    case = case.__class__(**{**case.__dict__, "batch": 2, "heads": 16})
    operator = case.factory(**case.kwargs(8192))
    sizes = benchmark.role_sizes_for(operator, case, 8192)
    compiled = operator.compile(
        role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True
    )
    inputs = benchmark.inputs_for(operator, case, 8192, gpu_device)
    actual, _ = benchmark._meta_call(compiled, [inputs], "output-only")()
    expected, _ = benchmark._token_oracle(operator, inputs, sizes)
    assert torch.allclose(
        actual["output"].cpu().float(), expected["output"], rtol=0.02, atol=0.05
    )
