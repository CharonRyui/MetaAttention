"""Compare FLA and MetaAttention Stateful Operator performance.

The runner selects a genuinely idle NVIDIA GPU before importing torch. FLA is
an execution baseline only: no FLA output is compared with an oracle and FLA
correctness never gates a measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

P0_BATCHES = (1, 2)
P0_HEADS = (1, 8, 16)
P0_SEQLENS = (63, 64, 65, 130, 2048, 8192)
P0_CONTRACTS = ("output-only", "final-state", "continuation")
RESULT_SCHEMA_VERSION = 2


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
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 disables CUDA; no GPU can be selected")
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
            raise RuntimeError(f"unsupported CUDA_VISIBLE_DEVICES entry: {item!r}") from error
    return indices


def list_gpus() -> tuple[GPUInfo, ...]:
    result = _run_nvidia_smi(
        ["--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"nvidia-smi GPU query failed: {detail}")
    gpus = []
    rows = csv.reader(line.strip() for line in result.stdout.splitlines() if line.strip())
    for row in rows:
        if len(row) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
        index = _integer(row[0])
        if index is None:
            raise RuntimeError(f"invalid GPU index from nvidia-smi: {row[0]!r}")
        process_result = _run_nvidia_smi(
            [f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        )
        query_error = None
        pids: tuple[int, ...] = ()
        if process_result.returncode == 0:
            pids = tuple(
                pid for pid in (_integer(line) for line in process_result.stdout.splitlines())
                if pid is not None
            )
        else:
            query_error = process_result.stderr.strip() or "compute-process query failed"
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
        if gpu.utilization_percent is not None and gpu.utilization_percent > max_utilization:
            continue
        if gpu.memory_used_mib is not None and gpu.memory_used_mib > max_used_memory_mib:
            continue
        if gpu.free_memory_mib is not None and gpu.free_memory_mib < min_free_memory_mib:
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


def _load_benchmark_api():
    """Import torch-dependent benchmark helpers after CUDA visibility is fixed."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from testing import benchmark_stateful_plans as benchmark

    return torch, benchmark


def _workloads(benchmark, families: list[str], batches: list[int], heads: list[int], seqlens: list[int], contracts: tuple[str, ...]):
    wanted = set(families)
    known = {case.name for case in benchmark.CASES}
    unknown = wanted - known
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    if not batches or not heads or not seqlens or not contracts:
        raise ValueError("matrix axes must not be empty")
    if any(value < 1 for value in (*batches, *heads, *seqlens)):
        raise ValueError("matrix axes must be positive")
    if "continuation" in contracts and min(seqlens) < 2:
        raise ValueError("continuation requires sequence length >= 2")
    return tuple(
        (replace(case, batch=batch, heads=head), seqlen, contract)
        for case in benchmark.CASES
        if case.name in wanted
        for batch in batches
        for head in heads
        for seqlen in seqlens
        for contract in contracts
    )


def _measure_pair(benchmark, torch, case, seqlen, device, warmup, repetitions, contract, timing):
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
        "fla_correctness": "ignored",
        "timing_units": "milliseconds",
        "compile_timed": False,
        "measurement_contract": {
            "public_cuda_event": "median warmed invocation",
            "cpu_wall": "median synchronized invocation",
            "memory": "peak allocator stats around one warmed invocation",
            "launches": "CUPTI activity events; foreign launches may be unavailable",
        },
    }
    try:
        compiled = operator.compile(
            role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True
        )
        plan = compiled.operator._plans[compiled.specialization]
        estimate = next(
            item for identity, item in plan.plan.candidate_estimates
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
            if contract == "continuation" else [inputs]
        )
        meta_call = benchmark._meta_call(compiled, parts, contract)
        actual, states = meta_call()
        expected, expected_states = benchmark._token_oracle(operator, inputs, sizes)
        result["meta_correctness"] = benchmark._correctness(
            actual, states, expected, expected_states, contract
        )
        del actual, states
        result["meta"] = benchmark._measure_call(meta_call, warmup, repetitions, timing)
        result["status"] = (
            "OK" if result["meta_correctness"]["status"] == "VALID" else "META_INVALID"
        )
    except Exception as error:
        result.update(status="META_COMPILE_REJECTED", error=str(error))
        return result

    try:
        adapter = benchmark._fla_call(case, inputs)
        if adapter is None:
            result.update(fla_status="UNAVAILABLE", fla_error="no FLA adapter")
            return result
        fla_contract = contract if contract == "output-only" or adapter.state_contract else "output-only"
        fla_parts = parts if fla_contract == contract else [inputs]
        baseline = benchmark._baseline_call(adapter, fla_parts, fla_contract)
        result.update(
            fla_status="OK",
            fla_contract=fla_contract,
            fla_contract_comparable=fla_contract == contract,
            fla_preprocessing=adapter.preprocessing,
            fla=benchmark._measure_call(baseline, warmup, repetitions, timing),
        )
        for metric in ("cpu_wall_ms", "public_cuda_event_ms", "graph_ms", "kernel_ms"):
            meta_value = result["meta"].get(metric)
            fla_value = result["fla"].get(metric)
            if isinstance(meta_value, (int, float)) and isinstance(fla_value, (int, float)) and meta_value > 0:
                result[f"speedup_{metric}"] = fla_value / meta_value
        if "speedup_public_cuda_event_ms" in result:
            result["winner"] = "MetaAttn DSL" if result["speedup_public_cuda_event_ms"] > 1 else "FLA"
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        result.update(fla_status="FAILED", fla_error=str(error))
    return result


def _csv_rows(records: list[dict]) -> list[dict]:
    fields = (
        "schema_version", "family", "batch", "heads", "seqlen", "contract", "status",
        "fla_status", "fla_contract_comparable", "meta_lowerer", "mathematical_representation",
        "meta_ms", "fla_ms", "speedup", "meta_graph_ms", "fla_graph_ms", "meta_kernel_ms",
        "fla_kernel_ms", "meta_launches", "fla_launches", "meta_peak_incremental_bytes",
        "fla_peak_incremental_bytes", "meta_peak_reserved_bytes", "fla_peak_reserved_bytes",
    )
    rows = []
    for record in records:
        meta = record.get("meta", {})
        fla = record.get("fla", {})
        row = {
            "schema_version": record.get("schema_version"), "family": record.get("family"),
            "batch": record.get("batch"), "heads": record.get("heads"), "seqlen": record.get("seqlen"),
            "contract": record.get("contract"), "status": record.get("status"),
            "fla_status": record.get("fla_status", "not-run"),
            "fla_contract_comparable": record.get("fla_contract_comparable"),
            "meta_lowerer": record.get("meta_lowerer"),
            "mathematical_representation": record.get("mathematical_representation"),
            "meta_ms": meta.get("public_cuda_event_ms"), "fla_ms": fla.get("public_cuda_event_ms"),
            "speedup": record.get("speedup_public_cuda_event_ms"),
            "meta_graph_ms": meta.get("graph_ms"), "fla_graph_ms": fla.get("graph_ms"),
            "meta_kernel_ms": meta.get("kernel_ms"), "fla_kernel_ms": fla.get("kernel_ms"),
            "meta_launches": meta.get("total_launches"), "fla_launches": fla.get("total_launches"),
            "meta_peak_incremental_bytes": meta.get("peak_incremental_bytes"),
            "fla_peak_incremental_bytes": fla.get("peak_incremental_bytes"),
            "meta_peak_reserved_bytes": meta.get("peak_reserved_bytes"),
            "fla_peak_reserved_bytes": fla.get("peak_reserved_bytes"),
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

    usable = [row for row in rows if row["meta_ms"] and row["fla_ms"]]
    if not usable:
        raise RuntimeError("no successful Meta/FLA pairs available for plotting")
    labels = [f"{row['family']}\nB{row['batch']} H{row['heads']} T{row['seqlen']}" for row in usable]
    positions = np.arange(len(usable))
    width = 0.38
    figure, (latency, speedup) = plt.subplots(2, 1, figsize=(min(48, max(12, len(labels) * 0.45)), 9), constrained_layout=True)
    latency.bar(positions - width / 2, [row["meta_ms"] for row in usable], width, label="MetaAttn DSL")
    latency.bar(positions + width / 2, [row["fla_ms"] for row in usable], width, label="FLA")
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
    import matplotlib.pyplot as plt
    import numpy as np

    usable = [row for row in rows if row["speedup"] is not None]
    for contract in sorted({row["contract"] for row in usable}):
        selected = [row for row in usable if row["contract"] == contract]
        labels = [f"{row['family']}\nB{row['batch']} H{row['heads']} T{row['seqlen']}" for row in selected]
        values = np.asarray([row["speedup"] for row in selected], dtype=float)[None, :]
        figure, axis = plt.subplots(figsize=(min(48, max(12, len(labels) * 0.36)), 2.8), constrained_layout=True)
        image = axis.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(2.0, float(values.max())))
        axis.set_title(f"{title}: FLA / MetaAttn ({contract})")
        axis.set_yticks([0], ["speedup"])
        axis.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        figure.colorbar(image, ax=axis, label=">1 means MetaAttn faster")
        figure.savefig(output_dir / f"speedup_{contract}.png", dpi=160)
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default="linear,simple-gla,gla,normalized-linear,retention,rwkv6,gsa,hgrn")
    parser.add_argument("--seqlens", type=int, nargs="+", default=list(P0_SEQLENS))
    parser.add_argument("--batches", type=int, nargs="+", default=list(P0_BATCHES))
    parser.add_argument("--heads", type=int, nargs="+", default=list(P0_HEADS))
    parser.add_argument("--contract", choices=(*P0_CONTRACTS, "all"), default="all")
    parser.add_argument("--timing", choices=("e2e", "graph", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=None, help="request a physical GPU, still subject to idle checks")
    parser.add_argument("--max-utilization", type=int, default=0)
    parser.add_argument("--max-used-memory-mib", type=int, default=1024)
    parser.add_argument("--min-free-memory-mib", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="show GPU and matrix without importing torch")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.repetitions < 1 or any(value < 1 for value in (*args.seqlens, *args.batches, *args.heads)):
        raise SystemExit("matrix axes must be positive; warmup nonnegative; repetitions positive")
    contracts = P0_CONTRACTS if args.contract == "all" else (args.contract,)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    try:
        selected = choose_idle_gpu(list_gpus(), requested=args.gpu, max_utilization=args.max_utilization, max_used_memory_mib=args.max_used_memory_mib, min_free_memory_mib=args.min_free_memory_mib, allowed_indices=_visible_physical_indices())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    matrix = {"families": families, "batches": args.batches, "heads": args.heads, "seqlens": args.seqlens, "contracts": contracts}
    print(json.dumps({"schema_version": RESULT_SCHEMA_VERSION, "selected_gpu": asdict(selected), "matrix": matrix}, sort_keys=True), flush=True)
    if args.dry_run:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.index)
    torch, benchmark = _load_benchmark_api()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA became unavailable after selecting the GPU")
    torch.set_num_threads(1)
    try:
        workloads = _workloads(benchmark, families, args.batches, args.heads, args.seqlens, contracts)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not workloads:
        raise SystemExit("no workloads selected")
    records = []
    for case, seqlen, contract in workloads:
        torch.manual_seed(42)
        try:
            record = _measure_pair(benchmark, torch, case, seqlen, torch.device("cuda"), args.warmup, args.repetitions, contract, args.timing)
        except Exception as error:
            record = {"schema_version": RESULT_SCHEMA_VERSION, "family": case.name, "batch": case.batch, "heads": case.heads, "seqlen": seqlen, "contract": contract, "status": "HARNESS_ERROR", "fla_correctness": "ignored", "error": repr(error)}
        records.append(record)
        print(json.dumps({"family": case.name, "B": case.batch, "H": case.heads, "T": seqlen, "contract": contract, "meta_ms": record.get("meta", {}).get("public_cuda_event_ms"), "fla_ms": record.get("fla", {}).get("public_cuda_event_ms"), "speedup": record.get("speedup_public_cuda_event_ms"), "status": record.get("status"), "fla_status": record.get("fla_status")}, sort_keys=True), flush=True)
    output_dir = args.output_dir or Path("benchmark_results") / f"stateful_fla_meta_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps({"schema_version": RESULT_SCHEMA_VERSION, "selected_gpu": asdict(selected), "matrix": matrix, "expected_records": len(workloads), "actual_records": len(records), "timing": args.timing, "warmup": args.warmup, "repetitions": args.repetitions, "fla_correctness": "not measured by design"}, indent=2), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = _csv_rows(records)
    _write_csv(output_dir / "summary.csv", rows)
    plot_status = "not attempted"
    try:
        _plot(output_dir / "comparison.png", rows, f"Stateful Operator on {selected.name}")
        _plot_matrix(output_dir, rows, f"Stateful Operator on {selected.name}")
        plot_status = str(output_dir / "comparison.png")
    except (ImportError, RuntimeError) as error:
        plot_status = f"plot unavailable: {error}"
    print(json.dumps({"records": str(output_dir / "records.json"), "manifest": str(output_dir / "manifest.json"), "csv": str(output_dir / "summary.csv"), "plot": plot_status}, sort_keys=True))


if __name__ == "__main__":
    main()
