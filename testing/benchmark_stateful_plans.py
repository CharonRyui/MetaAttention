"""Structural performance diagnostics for the Stateful Operator plans.

This is the measurement half of the plan work: it reports, per family, which
Lowered Candidate the compiler selected, how much workspace that candidate needs,
how many kernels it launches, and how its kernel and wall time scale with the
sequence length. Kernel numbers print as ``n/a`` when a launch happens through a
cached foreign runtime that the profiler cannot attribute; wall time is always
measured. Cross-library speedups stay in
``testing/benchmark_linear_attention.py``, which owns the FP32-oracle-validated
Flash Linear Attention comparison.

Usage::

    python testing/benchmark_stateful_plans.py --seqlen 2048
    python testing/benchmark_stateful_plans.py --families gla,based --json /tmp/plans.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attn_engine import (  # noqa: E402
    Batch,
    FeatureRole,
    HeadRole,
    One,
    ProductFeatureRole,
    Sequence,
    Target,
    StatefulCompilationError,
)
from examples.based_attention import based_attention  # noqa: E402
from examples.gated_delta_rule import gated_delta_rule  # noqa: E402
from examples.gated_slot_attention import gated_slot_attention  # noqa: E402
from examples.hgrn import hgrn  # noqa: E402
from examples.mamba2 import mamba2  # noqa: E402
from examples.linear_attention import (  # noqa: E402
    gla_attention,
    linear_attention,
    simple_gla_attention,
)
from examples.normalized_linear_attention import normalized_linear_attention  # noqa: E402
from examples.rebased_attention import rebased_attention  # noqa: E402
from examples.retnet_recurrent import retnet_recurrent  # noqa: E402
from examples.retention_attention import retention_attention  # noqa: E402
from examples.rwkv6_attention import rwkv6_attention  # noqa: E402
from examples.rwkv7_state import rwkv7_state  # noqa: E402


STANDARD = ("batch", "heads", "seqlen", "dim", "dim_value")


@dataclass(frozen=True)
class Case:
    """One Program, its workload, its parameter names, and its Feature sizes."""

    name: str
    factory: object
    signature: tuple = STANDARD
    dim: int = 64
    dim_value: int = 64
    dim_slot: int = 16
    heads: int = 8
    batch: int = 1
    feature_sizes: dict = field(default_factory=dict)

    def kwargs(self, seqlen: int) -> dict:
        by_name = {
            "batch": self.batch,
            "heads": self.heads,
            "seqlen": seqlen,
            "dim": self.dim,
            "dim_value": self.dim_value,
            "dim_slot": self.dim_slot,
        }
        aliases = {
            "B": "batch",
            "H": "heads",
            "HQ": "heads",
            "S": "seqlen",
            "D": "dim",
            "DV": "dim_value",
        }
        values = {}
        for parameter in self.signature:
            resolved = aliases.get(parameter, parameter)
            if resolved in by_name:
                values[parameter] = by_name[resolved]
        values["dtype"] = torch.bfloat16
        return values


CASES = (
    Case("linear", linear_attention),
    Case("simple-gla", simple_gla_attention),
    Case("gla", gla_attention),
    Case("normalized-linear", normalized_linear_attention),
    Case("retention", retention_attention),
    Case("rwkv7", rwkv7_state),
    Case("rwkv6", rwkv6_attention),
    Case("gsa", gated_slot_attention, signature=("batch", "heads", "seqlen", "dim", "dim_slot", "dim_value")),
    Case("hgrn", hgrn, signature=("batch", "heads", "seqlen", "dim_value"), heads=1),
    Case("based", based_attention, dim=16),
    Case("rebased", rebased_attention, dim=16),
    Case("gdn", gated_delta_rule, signature=("B", "H", "S", "D", "DV")),
    Case("mamba2", mamba2, signature=("B", "HQ", "S", "D", "DV")),
    Case("retnet", retnet_recurrent, signature=("B", "H", "S", "D", "DV")),
)


def role_sizes_for(operator, case: Case, seqlen: int) -> dict:
    """Resolve every declared role to a concrete axis size."""
    named = {
        "key_dim": case.dim,
        "value_dim": case.dim_value,
        "slot_dim": case.dim_slot,
        "unit_dim": 1,
    }
    named.update(case.feature_sizes)
    sizes: dict = {}
    for spec in (*operator.algorithm.inputs, *operator.algorithm.states):
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes[role] = case.heads
            elif isinstance(role, FeatureRole):
                if isinstance(role, ProductFeatureRole):
                    sizes[role] = sizes[role.left] * sizes[role.right]
                else:
                    sizes[role] = named.get(role.name, case.dim)
    return sizes


def inputs_for(operator, case: Case, seqlen: int, device: torch.device) -> dict:
    sizes = role_sizes_for(operator, case, seqlen)
    values = {}
    for spec in operator.algorithm.inputs:
        shape = tuple(
            {
                Batch: case.batch,
                Sequence: seqlen,
                One: 1,
            }.get(role, sizes.get(role))
            for role in spec.roles
        )
        tensor = torch.randn(shape, device=device, dtype=spec.dtype)
        if spec.name in ("gate", "decay"):
            tensor = torch.nn.functional.logsigmoid(tensor.float()).to(spec.dtype)
        values[spec.name] = tensor
    return values


def _median_time(call, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _kernel_profile(call, attempts: int = 2) -> tuple[float, int, int]:
    """Return (kernel milliseconds, kernel launches, other launches).

    The profiler occasionally captures nothing for a kernel launched through a
    cached foreign runtime, so an empty capture is retried once.
    """
    from torch.profiler import ProfilerActivity, profile

    events: list = []
    for attempt in range(attempts):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            call()
            torch.cuda.synchronize()
        events = [
            event for event in profiler.key_averages() if event.self_device_time_total > 0
        ]
        if events or attempt == attempts - 1:
            break
    kernel_ms = sum(event.self_device_time_total for event in events) / 1000.0
    launches = sum(event.count for event in events)
    tilelang = sum(
        event.count for event in events if event.key in ("kernel_kernel", "kernel")
    )
    return kernel_ms, tilelang, launches - tilelang


def measure(case: Case, seqlen: int, device: torch.device, warmup: int, repetitions: int) -> dict:
    operator = case.factory(**case.kwargs(seqlen))
    sizes = role_sizes_for(operator, case, seqlen)
    record: dict = {"family": case.name, "seqlen": seqlen, "heads": case.heads}
    try:
        compiled = operator.compile(
            role_sizes=sizes, target=Target(), mode="dense", optimized=True
        )
    except StatefulCompilationError as error:
        record["status"] = "COMPILE_REJECTED"
        record["error"] = f"{error.category} at {error.path}"
        return record
    plan = compiled.operator._plans[compiled.specialization]
    record["lowerer"] = plan.plan.candidate.lowerer
    record["representation"] = plan.plan.mathematical_plan.representation.value
    record["rewrites"] = list(plan.plan.mathematical_plan.rewrites)
    workspace = plan.plan.workspace
    record["workspace_bytes_per_token"] = workspace.bytes_per_token
    record["workspace_limit_bytes"] = workspace.maximum_bytes
    inputs = inputs_for(operator, case, seqlen, device)
    call = lambda: compiled(**inputs)  # noqa: E731
    try:
        record["workspace_bytes"] = plan.plan.workspace.required_bytes(seqlen)
    except RuntimeError as error:
        record["status"] = "RUNTIME_RESOURCE"
        record["error"] = str(error)
        return record
    try:
        call()
    except StatefulCompilationError as error:
        record["status"] = "RUNTIME_REJECTED"
        record["error"] = f"{error.category} at {error.path}"
        return record
    record["status"] = "OK"
    record["wall_ms"] = _median_time(call, warmup, repetitions)
    kernel_ms, kernel_launches, other_launches = _kernel_profile(call)
    if kernel_ms == 0.0 and record["wall_ms"] > 0.0:
        # the profiler can miss launches made through a foreign runtime
        record["kernel_ms"] = None
        record["kernel_launches"] = None
        record["other_launches"] = None
    else:
        record["kernel_ms"] = kernel_ms
        record["kernel_launches"] = kernel_launches
        record["other_launches"] = other_launches
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--families", help="comma separated family names")
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json", type=Path, help="write the records as JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    wanted = set(args.families.split(",")) if args.families else None
    records = []
    for seqlen in args.seqlen:
        for case in CASES:
            if wanted is not None and case.name not in wanted:
                continue
            case = (
                Case(**{**case.__dict__, "heads": args.heads}) if args.heads else case
            )
            record = measure(case, seqlen, device, args.warmup, args.repetitions)
            records.append(record)
            if record["status"] != "OK":
                print(
                    f"{record['family']:16s} T={seqlen:6d} {record['status']:17s} "
                    f"{record.get('lowerer', '-'):24s} {record.get('error', '')}",
                    flush=True,
                )
                continue
            kernel = record["kernel_ms"]
            launches = record["kernel_launches"]
            others = record["other_launches"]
            kernel_text = "n/a" if kernel is None else f"{kernel:9.3f}"
            launch_text = "n/a" if launches is None else f"{launches:3d}+{others:3d}"
            print(
                f"{record['family']:16s} T={seqlen:6d} {record['lowerer']:24s} "
                f"wall={record['wall_ms']:9.3f} ms kernel={kernel_text} ms "
                f"launches={launch_text} "
                f"workspace={record['workspace_bytes'] / 2**20:8.1f} MiB "
                f"({record['workspace_bytes_per_token'] / 2**10:.1f} KiB/token)",
                flush=True,
            )
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
