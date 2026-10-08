"""Structural performance diagnostics for the Stateful Operator plans.

This diagnostic reports structural plans and warmed measurements under explicit
output-only, final-State, or two-call continuation contracts. CPU wall time
includes synchronization; public-call CUDA events and Graph replay are separate
metrics. Native-layout timings exclude preparation, whose cost is also reported.
Meta correctness is checked against an independent authored-order token recurrence.

Usage::

    python testing/benchmark_stateful_plans.py --seqlen 2048
    python testing/benchmark_stateful_plans.py --families gla,based --json /tmp/plans.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
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
from examples.gated_delta_rule import gated_delta_rule, compositional_pressure_profile  # noqa: E402
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
    Case(
        "gsa",
        gated_slot_attention,
        signature=("batch", "heads", "seqlen", "dim", "dim_slot", "dim_value"),
    ),
    Case("hgrn", hgrn, signature=("batch", "heads", "seqlen", "dim_value"), heads=1),
    Case("based", based_attention, dim=16),
    Case("rebased", rebased_attention, dim=16),
    Case("gdn", gated_delta_rule, signature=("B", "H", "S", "D", "DV")),
    Case(
        "pressure",
        compositional_pressure_profile,
        signature=("B", "H", "S", "D", "DV", "HQ"),
    ),
    Case("mamba2", mamba2, signature=("B", "HQ", "S", "D", "DV")),
    Case("retnet", retnet_recurrent, signature=("B", "H", "S", "D", "DV")),
)
MATRIX_BATCHES = (1, 2)
MATRIX_HEADS = (1, 8, 16)
MATRIX_SEQLENS = (63, 64, 65, 130, 2048, 8192)


def expand_matrix(
    cases=CASES,
    *,
    batches=MATRIX_BATCHES,
    heads=MATRIX_HEADS,
    seqlens=MATRIX_SEQLENS,
):
    """Return the deterministic ``(Case, seqlen)`` matrix workload list.

    This helper only clones shape metadata; it does not construct an operator or
    allocate tensors, so callers can use it for planning and CPU-only tests.
    The case's feature dimensions are deliberately preserved, including
    non-square state dimensions such as GSA's key/slot/value axes.
    """
    batches = tuple(batches)
    heads = tuple(heads)
    seqlens = tuple(seqlens)
    if not batches or not heads or not seqlens:
        raise ValueError("matrix axes must not be empty")
    if any(value < 1 for value in (*batches, *heads, *seqlens)):
        raise ValueError("matrix axes must be positive")
    return tuple(
        (
            replace(case, batch=batch, heads=head),
            seqlen,
        )
        for case in cases
        for batch in batches
        for head in heads
        for seqlen in seqlens
    )


MATRIX_RECORD_FIELDS = (
    "family",
    "batch",
    "heads",
    "seqlen",
    "contract",
    "status",
    "device",
    "lowerer",
    "mathematical_representation",
    "rewrites",
    "static_estimate",
    "kernel_ms",
    "public_cuda_event_ms",
    "graph_ms",
    "total_launches",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "meta_correctness",
)


def matrix_record_schema(record: dict) -> dict:
    """Return the stable, JSON-safe matrix subset of a measurement record."""
    return {name: record.get(name) for name in MATRIX_RECORD_FIELDS}


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
        if spec.name in ("gate", "decay", "left_gate", "right_gate"):
            tensor = (torch.nn.functional.logsigmoid(tensor.float()) / 4).to(spec.dtype)
            tensor[:, :, 16 : min(32, seqlen)] = 0
            tensor[:, :, 256 : min(272, seqlen)] = 0
        elif spec.name == "beta":
            tensor = tensor.sigmoid()
        elif spec.name == "coefficient":
            tensor = -tensor.sigmoid() / 4
        elif spec.name == "dt":
            tensor = torch.nn.functional.softplus(tensor) + 0.01
        elif spec.name == "A":
            tensor = -(torch.nn.functional.softplus(tensor.float()) + 0.1).to(
                spec.dtype
            )
        if case.name in ("gdn", "rwkv7", "pressure") and spec.name in (
            "key",
            "removal",
            "left",
            "right",
        ):
            tensor = torch.nn.functional.normalize(tensor.float(), dim=-1).to(
                spec.dtype
            )
        if case.name == "normalized-linear" and spec.name in ("query", "key"):
            tensor = (torch.nn.functional.softplus(tensor.float()) + 0.1).to(spec.dtype)
        if case.name == "rwkv6" and spec.name == "bonus":
            tensor = tensor[:1, :, :1].expand_as(tensor).contiguous()
        if case.name in ("retention", "retnet") and spec.name in ("decay", "gate"):
            from examples.retention_attention import retention_decay

            tensor = retention_decay(case.heads).to(device)
            if case.name == "retnet":
                tensor = tensor.exp()
            tensor = tensor.expand(shape).contiguous()
        values[spec.name] = tensor.to(spec.dtype).contiguous()
    return values


CONTRACTS = ("output-only", "final-state", "continuation")


def _token_oracle(operator, inputs, sizes, *, initial_state=None):
    """CPU FP32 recurrence, independent of lowering/scan/rewrite implementation.

    Expressions carry named Feature axes; every propagation is applied in authored
    order, followed by injections. This also preserves prior-State readouts.
    """
    from attn_engine import stateful_operator as ir

    algorithm = operator.algorithm
    batch = next(value.shape[0] for value in inputs.values())
    length = next(inputs[spec.name].shape[spec.roles.index(Sequence)]
                  for spec in algorithm.inputs if Sequence in spec.roles)
    heads = sizes[algorithm.states[0].head_role]
    states = {
        spec.name: (torch.zeros(batch, heads, *(sizes[r] for r in spec.feature_roles))
                    if initial_state is None else initial_state[spec.name].cpu().float().clone())
        for spec in algorithm.states
    }
    specs = {spec.name: spec for spec in algorithm.states}
    bound = {name: value.detach().cpu().float() for name, value in inputs.items()}

    def align(value, roles, target):
        ordered = tuple(role for role in target if role in roles)
        value = value.permute(0, 1, *(2 + roles.index(role) for role in ordered))
        return value.reshape(*value.shape[:2], *(value.shape[2 + ordered.index(role)]
                             if role in ordered else 1 for role in target))

    def evaluate(node, token, prior):
        if isinstance(node, ir.Input):
            spec = next(spec for spec in algorithm.inputs if spec.name == node.name)
            value = bound[node.name]
            roles = list(spec.roles)
            if Sequence in roles:
                value = value.select(roles.index(Sequence), token)
                roles.remove(Sequence)
            features = tuple(r for r in roles if isinstance(r, FeatureRole))
            head = next((r for r in roles if isinstance(r, HeadRole)), None)
            leading = Batch if Batch in roles else One
            value = value.permute(roles.index(leading),
                                  *([roles.index(head)] if head else []),
                                  *(roles.index(r) for r in features))
            if head is None:
                value = value.unsqueeze(1)
            if heads % value.shape[1]:
                raise ValueError("oracle HeadMapping is not an integral replication")
            return value.repeat_interleave(heads // value.shape[1], dim=1), features
        if isinstance(node, ir.Constant):
            return torch.tensor(node.value, dtype=torch.float32).reshape(1, 1), ()
        if isinstance(node, ir.Shift):
            if token < node.lag:
                value, roles = evaluate(node.operand, 0, prior)
                return torch.zeros_like(value), roles
            return evaluate(node.operand, token - node.lag, prior)
        if isinstance(node, ir.StateTerm):
            value, roles = evaluate(node.operand, token, prior)
            spec = specs[node.state]
            state = (prior if node.prior else states)[node.state]
            product = state * align(value, roles, spec.feature_roles)
            if node.role is None:
                result, remaining = product, spec.feature_roles
            else:
                axis = spec.feature_roles.index(node.role)
                result = product.sum(2 + axis)
                remaining = tuple(r for r in spec.feature_roles if r != node.role)
            # Readouts treat singleton Feature axes as scalars (e.g. Based z0).
            for axis in reversed(range(len(remaining))):
                if result.shape[2 + axis] == 1:
                    result = result.squeeze(2 + axis)
                    remaining = remaining[:axis] + remaining[axis + 1:]
            return result, remaining
        if isinstance(node, (ir.Add, ir.Multiply, ir.Divide, ir.Outer, ir.Contract)):
            left, lr = evaluate(node.left, token, prior)
            right, rr = evaluate(node.right, token, prior)
            roles = tuple(dict.fromkeys((*lr, *rr)))
            if isinstance(node, ir.Outer):
                return (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(2), (node.role,)
            left, right = align(left, lr, roles), align(right, rr, roles)
            if isinstance(node, ir.Contract):
                return (left * right).sum(2 + roles.index(node.role)), tuple(r for r in roles if r != node.role)
            value = left + right if isinstance(node, ir.Add) else left * right if isinstance(node, ir.Multiply) else left / right
            return value, roles
        value, roles = evaluate(node.operand, token, prior)
        if isinstance(node, ir.Softmax):
            return value.softmax(2 + roles.index(node.role)), roles
        if isinstance(node, ir.Clamp):
            return value.clamp(node.minimum, node.maximum), roles
        unary = {ir.Negate: torch.neg, ir.Exp: torch.exp, ir.Log: torch.log,
                 ir.Sigmoid: torch.sigmoid, ir.Sqrt: torch.sqrt,
                 ir.Reciprocal: torch.reciprocal}
        return unary[type(node)](value), roles

    outputs = {readout.name: [] for readout in algorithm.readouts}
    for token in range(length):
        prior = dict(states)
        for transition in algorithm.transitions:
            spec = specs[transition.state]
            state = states[spec.name]
            for propagation in transition.propagations:
                if isinstance(propagation, ir.AxisScale):
                    factor, roles = evaluate(propagation.factor, token, prior)
                    state = state * align(factor, roles, spec.feature_roles)
                elif isinstance(propagation, ir.RankOnePropagation):
                    left, lr = evaluate(propagation.left, token, prior)
                    right, rr = evaluate(propagation.right, token, prior)
                    coefficient, cr = evaluate(propagation.coefficient, token, prior)
                    axis = 2 + spec.feature_roles.index(propagation.role)
                    contraction = (state * align(right, rr, spec.feature_roles)).sum(axis, keepdim=True)
                    state = state + align(coefficient, cr, spec.feature_roles) * align(left, lr, spec.feature_roles) * contraction
                else:
                    raise TypeError(f"unsupported oracle propagation {type(propagation).__name__}")
            for injection in transition.injections:
                product = torch.ones(1, 1, *([1] * spec.rank))
                for factor in injection.factors:
                    value, roles = evaluate(factor.expression, token, prior)
                    product = product * align(value, roles, spec.feature_roles)
                state = state + product
            states[spec.name] = state
        for readout in algorithm.readouts:
            expression = (ir.StateTerm(readout.state, readout.operand, readout.role)
                          if isinstance(readout, ir.StateContraction) else readout.expression)
            value, roles = evaluate(expression, token, prior)
            if not roles:
                value = value.unsqueeze(-1)
            outputs[readout.name].append(value)
    return {name: torch.stack(values, dim=2) for name, values in outputs.items()}, states


def _correctness(outputs, states, expected, expected_states, contract):
    errors = {}
    valid = True
    pairs = [(f"output:{name}", outputs.get(name), value) for name, value in expected.items()]
    if contract != "output-only":
        pairs.extend((f"state:{name}", None if states is None else states.get(name), value)
                     for name, value in expected_states.items())
    tolerances = {}
    for name, actual, reference in pairs:
        if actual is None or actual.shape != reference.shape:
            errors[name] = "missing or wrong shape"
            valid = False
            continue
        actual = actual.detach().cpu().float()
        errors[name] = (actual - reference).abs().max().item()
        # Same numerical contract as the linear recurrence functional oracle.
        # FP32 carry States are checked more tightly than BF16 readouts.
        rtol, atol = (0.01, 0.02) if name.startswith("state:") else (0.02, 0.05)
        tolerances[name] = {"rtol": rtol, "atol": atol}
        valid = valid and torch.allclose(actual, reference, rtol=rtol, atol=atol)
    return {"status": "VALID" if valid else "INVALID", "max_abs_errors": errors,
            "oracle": "independent authored-order FP32 CPU token recurrence", "tolerances": tolerances}


def _cpu_wall_time(call, warmup, repetitions):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


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


def _graph_time(call, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return _median_time(graph.replay, warmup, repetitions)




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
        events = [event for event in profiler.events()
                  if event.device_type == torch.autograd.DeviceType.CUDA]
        if events or attempt == attempts - 1:
            break
    kernel_ms = sum(event.device_time_total for event in events) / 1000.0
    launches = len(events)
    tilelang = sum(
        1 for event in events if event.name in ("kernel_kernel", "kernel")
    )
    return kernel_ms, tilelang, launches - tilelang


def _split_inputs(operator, inputs, split):
    parts = [{}, {}]
    for spec in operator.algorithm.inputs:
        value = inputs[spec.name]
        if Sequence in spec.roles:
            axis = spec.roles.index(Sequence)
            parts[0][spec.name] = value.narrow(axis, 0, split).contiguous()
            parts[1][spec.name] = value.narrow(axis, split, value.shape[axis] - split).contiguous()
        else:
            parts[0][spec.name] = parts[1][spec.name] = value
    return parts


def _meta_call(compiled, parts, contract):
    def run():
        prefix = compiled(**parts[0], return_final_state=contract != "output-only")
        result = prefix
        if contract == "continuation":
            result = compiled(**parts[1], initial_state=prefix.final_state, return_final_state=True)
        outputs = {name: result.outputs[name] for name in result.outputs.names}
        if contract == "continuation":
            outputs = {name: torch.cat((prefix.outputs[name], output), dim=2)
                       for name, output in outputs.items()}
        states = None if result.final_state is None else {name: result.final_state[name] for name in result.final_state.names}
        return outputs, states
    return run




def _measure_call(call, warmup, repetitions, timing="both"):
    # First invocation and warming precede every timed/profiler/memory interval.
    call()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    record = {
        "cpu_wall_ms": _cpu_wall_time(call, 0, repetitions),
        "public_cuda_event_ms": _median_time(call, 0, repetitions),
        "compilation_timed": False,
        "memory_scope": "PyTorch CUDA allocator, warmed single invocation; includes returned outputs/States",
    }
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = call()
    torch.cuda.synchronize()
    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    record["peak_incremental_bytes"] = record["peak_allocated_bytes"] - before
    record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    del result
    try:
        kernel_ms, kernels, others = _kernel_profile(call)
        record.update(kernel_ms=kernel_ms or None, kernel_launches=kernels if kernel_ms else None,
                      other_launches=others if kernel_ms else None,
                      total_launches=kernels + others if kernel_ms else None)
    except RuntimeError as error:
        record.update(kernel_ms=None, kernel_launches=None, other_launches=None,
                      total_launches=None, profiler_error=str(error))
    record["launch_source"] = "CUPTI CUDA activity events (kernels and device copies); unavailable when foreign launches are not captured"
    if timing in ("graph", "both"):
        try:
            record["graph_ms"] = _graph_time(call, 0, repetitions)
        except RuntimeError as error:
            record["graph_error"] = str(error)
    return record




def measure(
    case: Case, seqlen: int, device: torch.device, warmup: int, repetitions: int,
    *, contract: str = "final-state", timing: str = "both",
) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if contract not in CONTRACTS:
        raise ValueError(f"unknown contract: {contract}")
    if seqlen < (2 if contract == "continuation" else 1) or warmup < 0 or repetitions < 1:
        raise ValueError("positive sequence/repetitions and nonnegative warmup required; continuation needs >=2 tokens")
    operator = case.factory(**case.kwargs(seqlen))
    sizes = role_sizes_for(operator, case, seqlen)
    record = {"family": case.name, "seqlen": seqlen, "heads": case.heads,
              "batch": case.batch, "dim": case.dim, "dim_value": case.dim_value,
              "contract": contract, "continuation_split": seqlen // 2 if contract == "continuation" else None,
              "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
              "timing_units": "milliseconds",
              "input_layout": "B,H,T,features; preparation inside public-call intervals",
              "initial_state": "zero; continuation suffix receives actual prefix final-State"}
    record["contract_scope"] = (
        "one public call; output only" if contract == "output-only" else
        "one public call; all declared final States" if contract == "final-state" else
        "prefix + suffix public calls; actual prefix State passed to suffix; output concatenation timed"
    )
    try:
        compiled = operator.compile(role_sizes=sizes, target=Target(), mode="dense", optimized=True)
    except StatefulCompilationError as error:
        record.update(status="COMPILE_REJECTED", error=f"{error.category} at {error.path}")
        return record
    plan = compiled.operator._plans[compiled.specialization]
    selected_estimate = next(
        estimate for identity, estimate in plan.plan.candidate_estimates
        if identity == plan.plan.candidate.identity
    )
    record.update(
        lowerer=plan.plan.candidate.lowerer,
        representation=plan.plan.mathematical_plan.representation.value,
        mathematical_representation=plan.plan.mathematical_plan.representation.value,
        rewrites=list(plan.plan.mathematical_plan.rewrites),
        static_estimate=asdict(selected_estimate),
    )
    workspace = plan.plan.workspace
    record.update(workspace_bytes_per_token=workspace.bytes_per_token,
                  workspace_bytes_per_chunk=workspace.bytes_per_chunk,
                  workspace_limit_bytes=workspace.maximum_bytes)
    inputs = inputs_for(operator, case, seqlen, device)
    parts = _split_inputs(operator, inputs, seqlen // 2) if contract == "continuation" else [inputs]
    call = _meta_call(compiled, parts, contract)
    try:
        record["workspace_bytes"] = workspace.required_bytes(case.batch * seqlen, sequence_count=case.batch, sequence_length=seqlen)
        actual, states = call()
    except (StatefulCompilationError, RuntimeError) as error:
        record.update(status="RUNTIME_REJECTED", error=str(error))
        return record
    expected, expected_states = _token_oracle(operator, inputs, sizes)
    record["meta_correctness"] = _correctness(actual, states, expected, expected_states, contract)
    del actual, states
    record.update(_measure_call(call, warmup, repetitions, timing))
    record["status"] = "OK" if record["meta_correctness"]["status"] == "VALID" else "META_INVALID"
    record["meta_native_layout"] = {"layout": "public authored layout", "preparation_cpu_wall_ms": 0.0,
                                    "preparation_cuda_event_ms": 0.0,
                                    "note": "no separate native-layout public interface; internal normalization remains timed"}
    return record


def _print_record(record):
    """Print the entire measurement contract, including unsupported diagnostics."""
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--families", help="comma separated family names")
    parser.add_argument("--matrix-families", dest="matrix_families", help="matrix-only comma separated family names")
    parser.add_argument("--matrix-cases", dest="matrix_families", help=argparse.SUPPRESS)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json", type=Path, help="write the records as JSON")
    parser.add_argument("--contract", choices=(*CONTRACTS, "all"), default="final-state")
    parser.add_argument("--matrix-contract", choices=(*CONTRACTS, "all"), default=None)
    parser.add_argument("--timing", choices=("e2e", "graph", "both"), default="both")
    parser.add_argument("--matrix", action="store_true", help="run the deterministic B/H/T matrix (opt-in)")
    parser.add_argument("--matrix-batches", type=int, nargs="+", default=MATRIX_BATCHES)
    parser.add_argument("--matrix-heads", type=int, nargs="+", default=MATRIX_HEADS)
    parser.add_argument("--matrix-seqlens", type=int, nargs="+", default=MATRIX_SEQLENS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    family_arg = args.matrix_families if args.matrix and args.matrix_families else args.families
    wanted = set(family_arg.split(",")) if family_arg else None
    known = {case.name for case in CASES}
    if wanted is not None and not wanted <= known:
        parser.error(f"unknown families: {sorted(wanted - known)}")
    contract_arg = args.matrix_contract if args.matrix and args.matrix_contract else args.contract
    contracts = CONTRACTS if contract_arg == "all" else (contract_arg,)
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("positive repetitions and nonnegative warmup required")
    if args.heads is not None and args.heads < 1:
        parser.error("heads must be positive")
    if args.matrix:
        if args.heads is not None:
            parser.error("--heads cannot be combined with --matrix; use --matrix-heads")
        workloads = expand_matrix(
            tuple(case for case in CASES if wanted is None or case.name in wanted),
            batches=args.matrix_batches,
            heads=args.matrix_heads,
            seqlens=args.matrix_seqlens,
        )
    else:
        if any(length < (2 if "continuation" in contracts else 1) for length in args.seqlen):
            parser.error("positive lengths; continuation needs >=2 tokens")
        workloads = tuple(
            (replace(case, heads=args.heads) if args.heads else case, seqlen)
            for seqlen in args.seqlen
            for case in CASES
            if wanted is None or case.name in wanted
        )
    records = []
    for case, seqlen in workloads:
        for contract in contracts:
            if seqlen < (2 if contract == "continuation" else 1):
                parser.error("positive lengths; continuation needs >=2 tokens")
            torch.manual_seed(42)
            record = measure(
                case, seqlen, device, args.warmup, args.repetitions,
                contract=contract, timing=args.timing,
            )
            records.append(record)
            _print_record(matrix_record_schema(record) if args.matrix else record)
    if args.json is not None:
        args.json.write_text(json.dumps(
            [matrix_record_schema(record) for record in records] if args.matrix else records,
            indent=2,
        ))
    if any(record["status"] != "OK" for record in records):
        raise SystemExit("Meta measurement failed; see records")


if __name__ == "__main__":
    main()
