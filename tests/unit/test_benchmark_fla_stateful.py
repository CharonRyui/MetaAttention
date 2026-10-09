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
        "linear", "normalized-linear", "simple-gla", "gla", "retention",
        "retnet", "rwkv6", "rwkv7", "gdn", "gsa", "hgrn", "based", "rebased",
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
def test_fla_baseline_records_non_comparable_family(monkeypatch):
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "rwkv7")
    operator = case.factory(**case.kwargs(2))
    inputs = benchmark.inputs_for(operator, case, 2, torch.device("cpu"))
    adapter = _fla_call(case, inputs)
    assert adapter is not None
    assert adapter.comparable is False
    assert "native a/b" in adapter.note

def test_meta_failure_status_distinguishes_runtime_resource():
    from testing.benchmark_fla_stateful import _meta_failure_status

    assert _meta_failure_status(RuntimeError("RUNTIME_RESOURCE at workspace")) == "META_RUNTIME_REJECTED"
    assert _meta_failure_status(RuntimeError("COMPILE_RESOURCE at workspace")) == "META_COMPILE_REJECTED"
    assert _meta_failure_status(ValueError("bad input")) == "META_FAILED"
def test_hgrn_fla_adapter_flattens_heads(monkeypatch):
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "hgrn")
    case = case.__class__(**{**case.__dict__, "batch": 2, "heads": 8})
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cuda"))
    adapter = _fla_call(case, inputs)
    prepared = adapter.prepare(inputs)
    assert prepared[0].shape == (16, 3, 64)
    output, state = adapter.invoke(prepared, None, True)
    assert output.shape == (2, 8, 3, 64)
    assert state.shape == (2, 8, 64)

@pytest.mark.gpu
@pytest.mark.parametrize("family", ["linear", "retention", "retnet"])
def test_long_scalar_families_use_safe_schedule(gpu_device, family):
    import torch
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == family)
    case = case.__class__(**{**case.__dict__, "batch": 2, "heads": 16})
    operator = case.factory(**case.kwargs(8192))
    sizes = benchmark.role_sizes_for(operator, case, 8192)
    compiled = operator.compile(role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True)
    inputs = benchmark.inputs_for(operator, case, 8192, gpu_device)
    actual, _ = benchmark._meta_call(compiled, [inputs], "output-only")()
    expected, _ = benchmark._token_oracle(operator, inputs, sizes)
    assert torch.allclose(actual["output"].cpu().float(), expected["output"], rtol=0.02, atol=0.05)
