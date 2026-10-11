"""Compare example Stateful Operator linear-attention families with FLA.

The runner is restored from the removed ``benchmark_fla_meta_stateful.py`` and
keeps its idle-GPU selection and measurement machinery.  FLA is optional; the
default registry covers every directly mapped example family, while each
record still requires independent-oracle and pairwise correctness before a
speedup is emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

FLA_FAMILIES = (
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
)
P0_FAMILIES = FLA_FAMILIES
P0_BATCHES = (1, 2)
P0_HEADS = (1, 8, 16)
P0_SEQLENS = (63, 64, 65, 130, 2048, 8192)
P0_CONTRACTS = ("output-only", "final-state", "continuation")
RESULT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    utilization_percent: int | None
    compute_pids: tuple[int, ...] = ()
    query_error: str | None = None

    @property
    def free_memory_mib(self) -> int | None:
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        return self.memory_total_mib - self.memory_used_mib

    @property
    def idle(self) -> bool:
        return (
            self.query_error is None
            and not self.compute_pids
            and (self.utilization_percent is None or self.utilization_percent == 0)
        )


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; refusing to guess an idle GPU")
    try:
        return subprocess.run(
            [executable, *args], check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to query nvidia-smi: {error}") from error


def _provenance(torch, fla_source: Path | None) -> dict:
    import fla

    fla_module = Path(fla.__file__).resolve()
    root = Path(__file__).resolve().parents[1]
    revision = _git_revision(root)
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for distribution in ("tilelang", "flash-linear-attention", "fla-core"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    digest = hashlib.sha256()
    # Identify actual measured sources, including uncommitted implementation.
    for directory in (
        root / "attention_engine" / "attn_engine",
        root / "testing",
        root / "examples",
    ):
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return {
        "git_commit": revision,
        "source_sha256": digest.hexdigest(),
        "versions": versions,
        "fla_source": str(fla_source) if fla_source else None,
        "fla_module": str(fla_module),
        "fla_git_commit": _git_revision(fla_source) if fla_source else None,
    }


def _integer(value: str) -> int | None:
    value = value.strip().replace("[N/A]", "")
    try:
        return int(value)
    except ValueError:
        return None


def _visible_physical_indices() -> set[int] | None:
    """Return physical IDs allowed by an existing CUDA visibility mask."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or visible == "all":
        return None
    if visible == "-1":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES=-1 disables CUDA; no GPU can be selected"
        )
    indices: set[int] = set()
    for item in visible.split(","):
        item = item.strip()
        if not item or item.startswith("MIG-"):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must contain physical numeric GPU IDs "
                "for safe automatic selection"
            )
        try:
            indices.add(int(item))
        except ValueError as error:
            raise RuntimeError(
                f"unsupported CUDA_VISIBLE_DEVICES entry: {item!r}"
            ) from error
    return indices


def list_gpus() -> tuple[GPUInfo, ...]:
    result = _run_nvidia_smi(
        [
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"nvidia-smi GPU query failed: {detail}")
    gpus = []
    rows = csv.reader(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    for row in rows:
        if len(row) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
        index = _integer(row[0])
        if index is None:
            raise RuntimeError(f"invalid GPU index from nvidia-smi: {row[0]!r}")
        process_result = _run_nvidia_smi(
            [
                f"--id={index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        query_error = None
        pids: tuple[int, ...] = ()
        if process_result.returncode == 0:
            pids = tuple(
                pid
                for pid in (
                    _integer(line) for line in process_result.stdout.splitlines()
                )
                if pid is not None
            )
        else:
            query_error = (
                process_result.stderr.strip() or "compute-process query failed"
            )
        gpus.append(
            GPUInfo(
                index=index,
                name=row[1].strip(),
                memory_used_mib=_integer(row[2]),
                memory_total_mib=_integer(row[3]),
                utilization_percent=_integer(row[4]),
                compute_pids=pids,
                query_error=query_error,
            )
        )
    return tuple(gpus)


def choose_idle_gpu(
    gpus: Iterable[GPUInfo],
    *,
    requested: int | None = None,
    max_utilization: int = 0,
    max_used_memory_mib: int = 1024,
    min_free_memory_mib: int = 8192,
    allowed_indices: set[int] | None = None,
) -> GPUInfo:
    """Choose the largest-free GPU satisfying a conservative idle policy."""
    candidates = []
    for gpu in gpus:
        if allowed_indices is not None and gpu.index not in allowed_indices:
            continue
        if requested is not None and gpu.index != requested:
            continue
        if gpu.query_error is not None or gpu.compute_pids:
            continue
        if (
            gpu.utilization_percent is not None
            and gpu.utilization_percent > max_utilization
        ):
            continue
        if (
            gpu.memory_used_mib is not None
            and gpu.memory_used_mib > max_used_memory_mib
        ):
            continue
        if (
            gpu.free_memory_mib is not None
            and gpu.free_memory_mib < min_free_memory_mib
        ):
            continue
        candidates.append(gpu)
    if not candidates:
        details = "; ".join(
            f"GPU {gpu.index}: used={gpu.memory_used_mib}MiB free={gpu.free_memory_mib}MiB "
            f"util={gpu.utilization_percent}% pids={list(gpu.compute_pids)} error={gpu.query_error}"
            for gpu in gpus
        )
        raise RuntimeError(f"no GPU satisfies the idle policy ({details})")
    return max(candidates, key=lambda gpu: (gpu.free_memory_mib or -1, -gpu.index))


def _configure_fla_source(source: Path | None) -> Path | None:
    """Use the pinned release unless a checkout is explicitly requested."""
    if source is None:
        return None
    source = source.expanduser().resolve()
    if not (source / "fla").is_dir():
        raise RuntimeError(f"FLA source does not contain a fla package: {source}")
    source_string = str(source)
    if source_string in sys.path:
        sys.path.remove(source_string)
    sys.path.insert(0, source_string)
    return source


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_benchmark_api():
    """Import torch-dependent benchmark helpers after CUDA visibility is fixed."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from testing import benchmark_stateful_plans as benchmark

    return torch, benchmark


@dataclass(frozen=True)
class FLABaseline:
    prepare: object
    invoke: object
    state_names: tuple[str, ...]
    state_contract: bool = True
    comparable: bool = True
    note: str = ""
    preprocessing: str = "transpose/contiguous; timed in public call"

    def split(self, result):
        output, state = (
            (result[0], result[1])
            if isinstance(result, (tuple, list))
            else (result, None)
        )
        if state is None:
            return {"output": output}, None
        values = (state,) if len(self.state_names) == 1 else tuple(state)
        return {"output": output}, dict(zip(self.state_names, values, strict=True))


def _baseline_call(torch, adapter, parts, contract):
    def run():
        first = adapter.prepare(parts[0])
        first_result = adapter.invoke(first, None, contract != "output-only")
        outputs, states = adapter.split(first_result)
        if contract == "continuation":
            second = adapter.prepare(parts[1])
            initial = first_result[1]
            if initial is None:
                raise RuntimeError("FLA did not return a continuation State")
            suffix_outputs, states = adapter.split(
                adapter.invoke(second, initial, True)
            )
            outputs = {
                name: torch.cat((outputs[name], suffix_outputs[name]), dim=2)
                for name in outputs
            }
        return outputs, states

    return run


def _fla_call(case, inputs):
    """Return an FLA 0.5.2 adapter for a directly mapped example."""
    name = case.name
    if name == "hgrn":
        from fla.ops.hgrn import fused_recurrent_hgrn

        def prepare_hgrn(values):
            value, gate = values["value"], values["gate"]
            batch, heads, length, width = value.shape
            # HGRN has a headless gate/state.  Replicate that single authored
            # gate across grouped value heads before flattening B and H for FLA.
            if gate.shape[1] == 1 and heads != 1:
                gate = gate.expand(batch, heads, length, width)
            return (
                value.reshape(batch * heads, length, width),
                gate.reshape(batch * heads, length, width),
            )

        def invoke_hgrn(values, initial, final):
            value, gate = values
            batch_heads, length, width = value.shape
            if initial is not None:
                initial = initial.reshape(batch_heads, width)
            output, state = fused_recurrent_hgrn(
                value, gate, initial_state=initial, output_final_state=final
            )
            batch = batch_heads // case.heads
            output = output.reshape(batch, case.heads, length, width)
            if state is not None:
                state = state.reshape(batch, case.heads, width)
            return output, state

        return FLABaseline(
            prepare_hgrn,
            invoke_hgrn,
            ("hidden",),
            preprocessing="flatten B,H into FLA batch; timed in public call",
        )

    state_contract = name not in {"based", "rebased"}
    state_names = (
        ("memory", "key_sum")
        if name == "normalized-linear"
        else ("key_slot", "slot_value")
        if name == "gsa"
        else ("memory",)
    )

    def prepare(values):
        native = {
            key: value.transpose(1, 2).contiguous() for key, value in values.items()
        }
        if name in {"linear", "simple-gla", "gla"}:
            # These examples inject (key + 0.1) * value in FP32.
            native["key"] = native["key"].float() + 0.1
        if name == "retnet":
            native["gate"] = native["gate"].log()
        if name == "rwkv6":
            native["bonus"] = values["bonus"][0, :, 0].float().contiguous()
        if name == "rwkv7":
            beta = native["beta"].unsqueeze(-1)
            # FLA contracts a with the prior State; the authored propagation
            # contracts removal with the already decayed State.
            native["a"] = -native["removal"].float() * native["decay"].exp()
            native["b"] = native["key"].float() * beta
            native["value"] = native["value"].float() * beta
        return native

    if name == "linear":
        from fla.ops.linear_attn import fused_recurrent_linear_attn as function
    elif name == "normalized-linear":
        from fla.ops.linear_attn import fused_recurrent_linear_attn as function
    elif name == "retention":
        # Use the simple-GLA kernel with the prepared decay tensor.  The
        # retention wrapper rebuilds its decay with a host-side arange, which
        # introduces a CPU->CUDA copy during Graph capture.
        from fla.ops.simple_gla import fused_recurrent_simple_gla as function
    elif name in ("simple-gla", "retnet"):
        from fla.ops.simple_gla import fused_recurrent_simple_gla as function
    elif name == "gla":
        from fla.ops.gla import fused_recurrent_gla as function
    elif name == "rwkv6":
        from fla.ops.rwkv6 import fused_recurrent_rwkv6 as function
    elif name == "gsa":
        from fla.ops.gsa import fused_recurrent_gsa as function
    elif name == "gdn":
        from fla.ops.gated_delta_rule import (
            fused_recurrent_gated_delta_rule as function,
        )
    elif name == "rwkv7":
        from fla.ops.generalized_delta_rule.dplr import (
            fused_recurrent_dplr_delta_rule as function,
        )
    elif name == "based":
        from fla.ops.based import fused_chunk_based as function
    elif name == "rebased":
        from fla.ops.rebased import parallel_rebased as function
    elif name == "hgrn":
        return None
    else:
        return None

    def invoke(values, initial, final):
        query, key, value = (values[key] for key in ("query", "key", "value"))
        # Current FLA recurrent APIs use B,T,H,... inputs directly. Older
        # releases accepted head_first, but passing it breaks the checkout
        # selected by --fla-source.
        kwargs = {}
        if state_contract:
            memory = (
                initial[0]
                if name == "normalized-linear" and initial is not None
                else initial
            )
            kwargs.update(initial_state=memory, output_final_state=final)
        if name in ("linear", "normalized-linear"):
            result = function(
                query,
                key,
                value,
                normalize=False,
                **kwargs,
            )
        elif name == "retention":
            result = function(
                query,
                key,
                value,
                g_gamma=values["decay"][0, 0],
                scale=query.shape[-1] ** -0.5,
                **kwargs,
            )
        elif name in ("simple-gla", "retnet"):
            result = function(
                query, key, value, values.get("gate", values.get("decay")), **kwargs
            )
        elif name == "gla":
            result = function(query, key, value, values["gate"], **kwargs)
        elif name in ("based", "rebased"):
            result = function(query, key, value, **kwargs)
        elif name == "rwkv6":
            result = function(
                query, key, value, values["decay"], values["bonus"], **kwargs
            )
        elif name == "gsa":
            result = function(
                query, key, value, values["slot"], values["gate"], **kwargs
            )
        elif name == "gdn":
            result = function(
                query,
                key,
                value,
                g=values["gate"],
                beta=values["beta"],
                use_qk_l2norm_in_kernel=False,
                **kwargs,
            )
        else:
            result = function(
                query,
                key,
                value,
                values["a"],
                values["b"],
                values["decay"],
                scale=1.0,
                **kwargs,
            )
        output, state = (
            (result[0], result[1])
            if isinstance(result, (tuple, list))
            else (result, None)
        )
        if name == "normalized-linear":
            key_sum = key.float().cumsum(dim=1)
            if initial is not None:
                key_sum = key_sum + initial[1].unsqueeze(1)
            denominator = (query.float() * query.shape[-1] ** -0.5 * key_sum).sum(
                -1, keepdim=True
            )
            output = (output.float() / (denominator + 1e-10)).to(query.dtype)
            state = (state, key_sum[:, -1]) if final else None
        return output.transpose(1, 2), state

    preprocessing = "transpose/contiguous"
    if name in {"linear", "simple-gla", "gla"}:
        preprocessing += "; FP32 authored key + 0.1"
    elif name == "normalized-linear":
        preprocessing += "; FP32 key cumsum/denominator with continuation key_sum"
    elif name == "rwkv7":
        preprocessing += (
            "; FP32 DPLR factors with decay-before-removal and beta injection"
        )
    elif name == "rwkv6":
        preprocessing += "; FP32 bonus materialization"
    elif name == "retnet":
        preprocessing += "; gate log"
    note = (
        "" if state_contract else "FLA does not expose all equivalent recurrent States"
    )
    return FLABaseline(
        prepare,
        invoke,
        state_names,
        state_contract=state_contract,
        note=note,
        preprocessing=preprocessing,
    )


def _pair_correctness(torch, meta_outputs, fla_outputs):
    errors = {}
    valid = set(meta_outputs) == set(fla_outputs)
    for name in sorted(set(meta_outputs) | set(fla_outputs)):
        actual = meta_outputs.get(name)
        reference = fla_outputs.get(name)
        if actual is None or reference is None or actual.shape != reference.shape:
            errors[name] = "missing or wrong shape"
            valid = False
            continue
        difference = (actual.detach().float() - reference.detach().float()).abs()
        errors[name] = difference.max().item()
        valid = valid and torch.allclose(
            actual.float(), reference.float(), rtol=0.02, atol=0.05
        )
    return {"status": "VALID" if valid else "INVALID", "max_abs_errors": errors}


def _valid_pair(record: dict) -> bool:
    """Whether a record is eligible to expose a Meta/FLA comparison."""
    if record.get("status") != "OK" or record.get("fla_status") != "OK":
        return False
    if record.get("fla_contract_comparable") is not True:
        return False
    return all(
        record.get(name, {}).get("status") == "VALID"
        for name in ("meta_correctness", "fla_correctness", "pair_correctness")
    )


def benchmark_matrix(
    families=P0_FAMILIES,
    batches=P0_BATCHES,
    heads=P0_HEADS,
    seqlens=P0_SEQLENS,
    contracts=("output-only",),
):
    """Return the validated extended matrix without importing FLA."""
    if not families or not batches or not heads or not seqlens or not contracts:
        raise ValueError("matrix axes must not be empty")
    if set(families) - set(P0_FAMILIES):
        raise ValueError(f"families must be one of: {', '.join(P0_FAMILIES)}")
    if set(contracts) - set(P0_CONTRACTS):
        raise ValueError(f"contracts must be one of: {', '.join(P0_CONTRACTS)}")
    if any(value < 1 for value in (*batches, *heads, *seqlens)):
        raise ValueError("matrix axes must be positive")
    if "continuation" in contracts and min(seqlens) < 2:
        raise ValueError("continuation requires sequence length >= 2")
    return tuple(
        (family, batch, head, seqlen, contract)
        for family in families
        for batch in batches
        for head in heads
        for seqlen in seqlens
        for contract in contracts
    )


def _workloads(
    benchmark,
    families: list[str],
    batches: list[int],
    heads: list[int],
    seqlens: list[int],
    contracts: tuple[str, ...],
):
    unknown = set(families) - set(P0_FAMILIES)
    if unknown:
        raise ValueError(f"unknown or unsupported FLA families: {sorted(unknown)}")
    unknown_contracts = set(contracts) - set(P0_CONTRACTS)
    if unknown_contracts:
        raise ValueError(f"unknown contracts: {sorted(unknown_contracts)}")
    if not batches or not heads or not seqlens or not contracts:
        raise ValueError("matrix axes must not be empty")
    if any(value < 1 for value in (*batches, *heads, *seqlens)):
        raise ValueError("matrix axes must be positive")
    if "continuation" in contracts and min(seqlens) < 2:
        raise ValueError("continuation requires sequence length >= 2")
    return tuple(
        (replace(case, batch=batch, heads=head), seqlen, contract)
        for case in benchmark.CASES
        if case.name in families
        for batch in batches
        for head in heads
        for seqlen in seqlens
        for contract in contracts
    )


def _meta_failure_status(error: Exception) -> str:
    """Classify failures after compilation without conflating workspace rejects."""
    message = str(error)
    if "COMPILE_RESOURCE" in message:
        return "META_COMPILE_REJECTED"
    if "RUNTIME_RESOURCE" in message or "workspace" in message.lower():
        return "META_RUNTIME_REJECTED"
    if "compile" in message.lower():
        return "META_COMPILE_REJECTED"
    return "META_FAILED"


def _measure_pair(
    benchmark,
    torch,
    case,
    seqlen,
    device,
    warmup,
    repetitions,
    contract,
    timing,
    workspace_bytes=1 << 30,
):
    operator = case.factory(**case.kwargs(seqlen))
    sizes = benchmark.role_sizes_for(operator, case, seqlen)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "family": case.name,
        "batch": case.batch,
        "heads": case.heads,
        "seqlen": seqlen,
        "contract": contract,
        "device": torch.cuda.get_device_name(device),
        "status": "OK",
        "fla_status": "NOT_RUN",
        "timing_units": "milliseconds",
        "compile_timed": False,
        "workspace_limit_bytes": workspace_bytes,
        "ratio_direction": "FLA / MetaAttention (>1 means MetaAttention faster)",
        "measurement_contract": {
            "public_cuda_event": "median warmed invocation",
            "cpu_wall": "median synchronized invocation",
            "memory": "peak allocator stats around one warmed invocation",
            "launches": "CUPTI activity events; foreign launches may be unavailable",
            "preprocessing": "included in both public-call intervals",
        },
    }
    try:
        compiled = operator.compile(
            role_sizes=sizes,
            target=benchmark.Target(workspace_bytes=workspace_bytes),
            mode="dense",
            optimized=True,
        )
        plan = compiled.operator._plans[compiled.specialization]
        estimate = next(
            item
            for identity, item in plan.plan.candidate_estimates
            if identity == plan.plan.candidate.identity
        )
        result.update(
            meta_lowerer=plan.plan.candidate.lowerer,
            mathematical_representation=plan.plan.mathematical_plan.representation.value,
            rewrites=list(plan.plan.mathematical_plan.rewrites),
            static_estimate=asdict(estimate),
        )
        inputs = benchmark.inputs_for(operator, case, seqlen, device)
        parts = (
            benchmark._split_inputs(operator, inputs, seqlen // 2)
            if contract == "continuation"
            else [inputs]
        )
        meta_call = benchmark._meta_call(compiled, parts, contract)
        meta_outputs, meta_states = meta_call()
        expected, expected_states = benchmark._token_oracle(operator, inputs, sizes)
        result["meta_correctness"] = benchmark._correctness(
            meta_outputs, meta_states, expected, expected_states, contract
        )
        result["meta"] = benchmark._measure_call(meta_call, warmup, repetitions, timing)
        result["status"] = (
            "OK" if result["meta_correctness"]["status"] == "VALID" else "META_INVALID"
        )
        if result["status"] != "OK":
            return result
    except Exception as error:
        result.update(status=_meta_failure_status(error), error=str(error))
        return result

    try:
        adapter = _fla_call(case, inputs)
        if adapter is None:
            result.update(
                fla_status="UNAVAILABLE",
                status="NOT_COMPARABLE",
                fla_error="no FLA adapter",
            )
            return result
        fla_contract = (
            contract
            if contract == "output-only" or adapter.state_contract
            else "output-only"
        )
        fla_parts = parts if fla_contract == contract else [inputs]
        baseline = _baseline_call(torch, adapter, fla_parts, fla_contract)
        fla_outputs, fla_states = baseline()
        result["fla_correctness"] = benchmark._correctness(
            fla_outputs, fla_states, expected, expected_states, fla_contract
        )
        result["pair_correctness"] = _pair_correctness(torch, meta_outputs, fla_outputs)
        result.update(
            fla_contract=fla_contract,
            fla_contract_comparable=adapter.comparable and fla_contract == contract,
            fla_preprocessing=adapter.preprocessing,
            fla_preprocessing_timed=True,
        )
        if not adapter.comparable:
            result.update(
                fla_status="NOT_COMPARABLE",
                status="NOT_COMPARABLE",
                fla_note=adapter.note,
            )
            return result
        if result["fla_correctness"]["status"] != "VALID":
            result.update(fla_status="INVALID", status="FLA_INVALID")
            return result
        if result["pair_correctness"]["status"] != "VALID":
            result.update(fla_status="INVALID", status="PAIR_INVALID")
            return result
        if contract != fla_contract:
            result.update(
                fla_status="NOT_COMPARABLE",
                status="NOT_COMPARABLE",
                fla_note=adapter.note,
            )
            return result
        result["fla"] = benchmark._measure_call(baseline, warmup, repetitions, timing)
        result["fla_status"] = "OK"
        for metric in ("cpu_wall_ms", "public_cuda_event_ms", "graph_ms", "kernel_ms"):
            meta_value = result["meta"].get(metric)
            fla_value = result["fla"].get(metric)
            if (
                isinstance(meta_value, (int, float))
                and isinstance(fla_value, (int, float))
                and meta_value > 0
            ):
                result[f"speedup_{metric}"] = fla_value / meta_value
        if "speedup_public_cuda_event_ms" in result:
            result["winner"] = (
                "MetaAttention" if result["speedup_public_cuda_event_ms"] > 1 else "FLA"
            )
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        result.update(fla_status="FAILED", status="FLA_FAILED", fla_error=str(error))
    return result


def _csv_rows(records: list[dict]) -> list[dict]:
    fields = (
        "schema_version",
        "family",
        "batch",
        "heads",
        "seqlen",
        "contract",
        "status",
        "fla_status",
        "fla_contract_comparable",
        "meta_lowerer",
        "mathematical_representation",
        "meta_ms",
        "fla_ms",
        "speedup",
        "meta_graph_ms",
        "fla_graph_ms",
        "meta_kernel_ms",
        "fla_kernel_ms",
        "meta_launches",
        "fla_launches",
        "meta_peak_incremental_bytes",
        "fla_peak_incremental_bytes",
        "meta_peak_reserved_bytes",
        "fla_peak_reserved_bytes",
    )
    rows = []
    for record in records:
        valid_pair = _valid_pair(record)
        meta = record.get("meta", {})
        fla = record.get("fla", {})
        row = {
            "schema_version": record.get("schema_version"),
            "family": record.get("family"),
            "batch": record.get("batch"),
            "heads": record.get("heads"),
            "seqlen": record.get("seqlen"),
            "contract": record.get("contract"),
            "status": record.get("status"),
            "fla_status": record.get("fla_status", "not-run"),
            "fla_contract_comparable": record.get("fla_contract_comparable"),
            "meta_lowerer": record.get("meta_lowerer"),
            "mathematical_representation": record.get("mathematical_representation"),
            "meta_ms": meta.get("public_cuda_event_ms") if valid_pair else None,
            "fla_ms": fla.get("public_cuda_event_ms") if valid_pair else None,
            "speedup": record.get("speedup_public_cuda_event_ms")
            if valid_pair
            else None,
            "meta_graph_ms": meta.get("graph_ms") if valid_pair else None,
            "fla_graph_ms": fla.get("graph_ms") if valid_pair else None,
            "meta_kernel_ms": meta.get("kernel_ms") if valid_pair else None,
            "fla_kernel_ms": fla.get("kernel_ms") if valid_pair else None,
            "meta_launches": meta.get("total_launches") if valid_pair else None,
            "fla_launches": fla.get("total_launches") if valid_pair else None,
            "meta_peak_incremental_bytes": meta.get("peak_incremental_bytes")
            if valid_pair
            else None,
            "fla_peak_incremental_bytes": fla.get("peak_incremental_bytes")
            if valid_pair
            else None,
            "meta_peak_reserved_bytes": meta.get("peak_reserved_bytes")
            if valid_pair
            else None,
            "fla_peak_reserved_bytes": fla.get("peak_reserved_bytes")
            if valid_pair
            else None,
        }
        rows.append({field: row[field] for field in fields})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    usable = [
        row
        for row in rows
        if row["speedup"] is not None and row["meta_ms"] and row["fla_ms"]
    ]
    if not usable:
        raise RuntimeError("no successful Meta/FLA pairs available for plotting")
    labels = [
        f"{row['family']}\nB{row['batch']} H{row['heads']} T{row['seqlen']}"
        for row in usable
    ]
    positions = np.arange(len(usable))
    width = 0.38
    figure, (latency, speedup) = plt.subplots(
        2, 1, figsize=(min(48, max(12, len(labels) * 0.45)), 9), constrained_layout=True
    )
    latency.bar(
        positions - width / 2,
        [row["meta_ms"] for row in usable],
        width,
        label="MetaAttn DSL",
    )
    latency.bar(
        positions + width / 2, [row["fla_ms"] for row in usable], width, label="FLA"
    )
    latency.set_ylabel("Public-call latency (ms)")
    latency.set_title(title)
    latency.set_xticks(positions, labels, rotation=75, ha="right", fontsize=8)
    latency.legend()
    speedup.bar(positions, [row["speedup"] for row in usable], color="#4c78a8")
    speedup.axhline(1.0, color="black", linewidth=0.8)
    speedup.set_ylabel("FLA / MetaAttn")
    speedup.set_xlabel("Workload")
    speedup.set_xticks(positions, labels, rotation=75, ha="right", fontsize=8)
    speedup.set_title("Speedup (>1 means MetaAttn DSL is faster)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_matrix(output_dir: Path, rows: list[dict], title: str) -> None:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    usable = [row for row in rows if row["speedup"] is not None]
    for contract in sorted({row["contract"] for row in usable}):
        selected = [row for row in usable if row["contract"] == contract]
        for family in sorted({row["family"] for row in selected}):
            family_rows = [row for row in selected if row["family"] == family]
            labels = [
                f"B{row['batch']} H{row['heads']} T{row['seqlen']}"
                for row in family_rows
            ]
            values = np.asarray([row["speedup"] for row in family_rows], dtype=float)[
                None, :
            ]
            figure, axis = plt.subplots(
                figsize=(min(24, max(6, len(labels) * 0.45)), 2.8),
                constrained_layout=True,
            )
            finite = values[np.isfinite(values) & (values > 0)]
            if finite.size == 0:
                plt.close(figure)
                continue
            vmax = max(2.0, float(finite.max()), 1.0 / float(finite.min()))
            image = axis.imshow(
                values,
                aspect="auto",
                cmap="RdYlGn",
                norm=colors.LogNorm(vmin=1.0 / vmax, vmax=vmax),
            )
            axis.set_title(f"{title}: FLA / MetaAttn ({family}, {contract})")
            axis.set_yticks([0], ["speedup"])
            axis.set_xticks(
                range(len(labels)), labels, rotation=75, ha="right", fontsize=7
            )
            figure.colorbar(
                image, ax=axis, label=">1 means MetaAttn faster (1x center)"
            )
            figure.savefig(output_dir / f"speedup_{contract}_{family}.png", dpi=160)
            plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", default=",".join(P0_FAMILIES))
    parser.add_argument("--seqlens", type=int, nargs="+", default=list(P0_SEQLENS))
    parser.add_argument("--batches", type=int, nargs="+", default=list(P0_BATCHES))
    parser.add_argument("--heads", type=int, nargs="+", default=list(P0_HEADS))
    parser.add_argument(
        "--contract", choices=(*P0_CONTRACTS, "all"), default="output-only"
    )
    parser.add_argument("--timing", choices=("e2e", "graph", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=1.0,
        help="Meta workspace budget in GiB (default: 1)",
    )
    parser.add_argument(
        "--fla-source",
        type=Path,
        default=None,
        help="optional FLA checkout override; defaults to installed FLA 0.5.2",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="request a physical GPU, still subject to idle checks",
    )
    parser.add_argument("--max-utilization", type=int, default=0)
    parser.add_argument("--max-used-memory-mib", type=int, default=1024)
    parser.add_argument("--min-free-memory-mib", type=int, default=8192)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show GPU and matrix without importing torch",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0 < args.workspace_gib < float("inf"):
        raise SystemExit("workspace-gib must be finite and positive")
    if (
        args.warmup < 0
        or args.repetitions < 1
        or any(value < 1 for value in (*args.seqlens, *args.batches, *args.heads))
    ):
        raise SystemExit(
            "matrix axes must be positive; warmup nonnegative; repetitions positive"
        )
    contracts = P0_CONTRACTS if args.contract == "all" else (args.contract,)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    try:
        selected = choose_idle_gpu(
            list_gpus(),
            requested=args.gpu,
            max_utilization=args.max_utilization,
            max_used_memory_mib=args.max_used_memory_mib,
            min_free_memory_mib=args.min_free_memory_mib,
            allowed_indices=_visible_physical_indices(),
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    matrix = {
        "families": families,
        "batches": args.batches,
        "heads": args.heads,
        "seqlens": args.seqlens,
        "contracts": contracts,
    }
    print(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "selected_gpu": asdict(selected),
                "matrix": matrix,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.index)
    try:
        fla_source = _configure_fla_source(args.fla_source)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    torch, benchmark = _load_benchmark_api()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA became unavailable after selecting the GPU")
    torch.set_num_threads(1)
    try:
        workloads = _workloads(
            benchmark, families, args.batches, args.heads, args.seqlens, contracts
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not workloads:
        raise SystemExit("no workloads selected")
    records = []
    for case, seqlen, contract in workloads:
        torch.manual_seed(42)
        try:
            record = _measure_pair(
                benchmark,
                torch,
                case,
                seqlen,
                torch.device("cuda"),
                args.warmup,
                args.repetitions,
                contract,
                args.timing,
                workspace_bytes=int(args.workspace_gib * (1 << 30)),
            )
        except Exception as error:
            record = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "family": case.name,
                "batch": case.batch,
                "heads": case.heads,
                "seqlen": seqlen,
                "contract": contract,
                "status": "HARNESS_ERROR",
                "fla_correctness": "ignored",
                "error": repr(error),
            }
        records.append(record)
        print(
            json.dumps(
                {
                    "family": case.name,
                    "B": case.batch,
                    "H": case.heads,
                    "T": seqlen,
                    "contract": contract,
                    "meta_ms": record.get("meta", {}).get("public_cuda_event_ms"),
                    "fla_ms": record.get("fla", {}).get("public_cuda_event_ms"),
                    "speedup": record.get("speedup_public_cuda_event_ms"),
                    "status": record.get("status"),
                    "fla_status": record.get("fla_status"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output_dir = (
        args.output_dir
        or Path("benchmark_results")
        / f"stateful_fla_meta_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "selected_gpu": asdict(selected),
        "matrix": matrix,
        "expected_records": len(workloads),
        "actual_records": len(records),
        "timing": args.timing,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "workspace_limit_bytes": int(args.workspace_gib * (1 << 30)),
        "provenance": _provenance(torch, fla_source),
        "correctness": {
            "meta": "measured against independent authored-order FP32 oracle",
            "fla": "measured when adapter execution succeeds",
            "pair": "measured when both outputs are available",
            "valid_pair": "status OK, FLA status OK, comparable contract, and all three correctness statuses VALID",
        },
        "valid_pairs": sum(_valid_pair(record) for record in records),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    rows = _csv_rows(records)
    _write_csv(output_dir / "summary.csv", rows)
    plot_status = "not attempted"
    try:
        _plot(
            output_dir / "comparison.png", rows, f"Stateful Operator on {selected.name}"
        )
        _plot_matrix(output_dir, rows, f"Stateful Operator on {selected.name}")
        plot_status = str(output_dir / "comparison.png")
    except (ImportError, RuntimeError) as error:
        plot_status = f"plot unavailable: {error}"
    print(
        json.dumps(
            {
                "records": str(output_dir / "records.json"),
                "manifest": str(output_dir / "manifest.json"),
                "csv": str(output_dir / "summary.csv"),
                "plot": plot_status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
