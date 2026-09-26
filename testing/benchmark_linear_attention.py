"""Compare MetaAttention linear recurrences with Flash Linear Attention.

The FP32 token recurrence in this file is the correctness oracle.  Stock FLA
is measured separately and is never used as MetaAttention's reference.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.linear_attention import gla_attention, linear_attention, simple_gla_attention
from fla.ops.gla import chunk_gla
from fla.ops.linear_attn import chunk_linear_attn
from fla.ops.simple_gla import chunk_simple_gla


@dataclass(frozen=True)
class Case:
    name: str
    factory: object
    batch: int
    heads: int
    seqlen: int
    dim: int
    dim_value: int


CASES = (
    Case("linear", linear_attention, 1, 8, 512, 64, 64),
    Case("simple-gla", simple_gla_attention, 1, 8, 512, 64, 64),
    Case("gla", gla_attention, 1, 8, 512, 64, 64),
)


def _time(call, warmup: int, repetitions: int) -> float:
    """End-to-end GPU timing, including the launches made by ``call``."""
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


def _time_cuda_graph(call, warmup: int, repetitions: int) -> float:
    """Time graph replay only, separating kernel work from launch overhead."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _compile(factory, case):
    operator = factory(
        case.batch,
        case.heads,
        case.seqlen,
        case.dim,
        case.dim_value,
        dtype=torch.bfloat16,
    )
    roles = {
        role.name: role
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if hasattr(role, "name")
    }
    return operator.compile(
        role_sizes={
            roles["query_heads"]: case.heads,
            roles["key_heads"]: case.heads,
            roles["value_heads"]: case.heads,
            roles["key_dim"]: case.dim,
            roles["value_dim"]: case.dim_value,
        },
        target=__import__("attn_engine").Target(),
        mode="dense",
        optimized=True,
    )


def _oracle(query, key, value, gate):
    """Return FP32 token-recurrence outputs and final FP32 state."""
    query_f, key_f, value_f = query.float(), key.float(), value.float()
    state = torch.zeros(
        query.shape[0], query.shape[1], query.shape[-1], value.shape[-1],
        device=query.device,
        dtype=torch.float32,
    )
    outputs = []
    for token in range(query.shape[2]):
        if gate is not None:
            if gate.ndim == 3:
                scale = gate[:, :, token].float()
                state = state * scale[:, :, None, None].exp()
            else:
                scale = gate[:, :, token, :].float()
                state = state * scale[:, :, :, None].exp()
        state = state + torch.einsum(
            "bhk,bhv->bhkv", key_f[..., token, :], value_f[..., token, :]
        )
        outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                query_f[..., token, :] * query.shape[-1] ** -0.5,
                state,
            )
        )
    return torch.stack(outputs, dim=2), state


def _fla_call(case, query, key, value, gate):
    if case.name == "linear":
        return chunk_linear_attn(query, key, value, normalize=False, head_first=True)
    if case.name == "simple-gla":
        return chunk_simple_gla(query, key, value, gate, head_first=True)
    return chunk_gla(query, key, value, gate, head_first=True)


def _split_fla_result(result):
    if isinstance(result, (tuple, list)):
        return result[0], result[1] if len(result) > 1 else None
    return result, None


def _max_errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().clamp_min(1e-6)
    return difference.max().item(), (difference / scale).max().item()


def _check(label, actual, expected, state=None, expected_state=None):
    output_error = _max_errors(actual, expected)
    state_error = None if state is None else _max_errors(state, expected_state)
    valid = torch.allclose(actual.float(), expected.float(), rtol=0.2, atol=0.2)
    if state is not None:
        valid = valid and torch.allclose(state.float(), expected_state, rtol=0.2, atol=0.2)
    suffix = "" if state_error is None else f" state_max_abs={state_error[0]:.5g}"
    status = "VALID" if valid else "INVALID"
    print(f"{label:12s} correctness={status} output_max_abs={output_error[0]:.5g}{suffix}")
    return valid


def _run_case(case: Case, warmup: int, repetitions: int, timing: str) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    query = torch.randn(
        case.batch, case.heads, case.seqlen, case.dim,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    key = torch.randn(
        case.batch, case.heads, case.seqlen, case.dim,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    value = torch.randn(
        case.batch, case.heads, case.seqlen, case.dim_value,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    gate = None
    if case.name == "simple-gla":
        gate = torch.randn(
            case.batch, case.heads, case.seqlen, device=device,
            dtype=torch.float32, generator=generator,
        ).clamp(-0.1, 0)
    elif case.name == "gla":
        gate = torch.randn(
            case.batch, case.heads, case.seqlen, case.dim, device=device,
            dtype=torch.float32, generator=generator,
        ).clamp(-0.1, 0)

    compiled = _compile(case.factory, case)
    plan = compiled.operator._plans[compiled.specialization]
    print(
        f"{case.name:12s} S={case.seqlen} selected={plan.plan.candidate.lowerer} "
        f"representation={plan.plan.mathematical_plan.representation}"
    )
    inputs = {"query": query, "key": key, "value": value}
    if gate is not None:
        inputs["gate"] = gate
    expected, expected_state = _oracle(query, key, value, gate)

    meta_result = compiled(**inputs, return_final_state=True)
    meta_state = None if meta_result.final_state is None else meta_result.final_state["memory"]
    meta_output = meta_result.outputs["output"]
    if not _check("MetaAttention", meta_output, expected, meta_state, expected_state):
        raise RuntimeError(f"MetaAttention failed the FP32 oracle for {case.name}")

    fla_result = _fla_call(case, query, key, value, gate)
    fla_output, fla_state = _split_fla_result(fla_result)
    # FLA state is a BF16 implementation detail, not an independent oracle.
    # It is displayed only as a diagnostic against the FP32 recurrence state.
    fla_valid = _check("stock FLA", fla_output, expected)
    if fla_state is not None:
        state_error = _max_errors(fla_state, expected_state)
        print(f"{'stock FLA':12s} final_state_diagnostic_max_abs={state_error[0]:.5g} (not oracle)")

    run_meta = lambda: compiled(**inputs).outputs["output"]
    run_fla = lambda: _fla_call(case, query, key, value, gate)
    meta_e2e = fla_e2e = None
    meta_graph = fla_graph = None
    for label, call in (("MetaAttention", run_meta), ("stock FLA", run_fla)):
        if timing in {"e2e", "both"}:
            value_ms = _time(call, warmup, repetitions)
            print(f"{case.name:12s} {label} end_to_end={value_ms:.3f} ms")
            if label == "MetaAttention":
                meta_e2e = value_ms
            else:
                fla_e2e = value_ms
        if timing in {"graph", "both"}:
            try:
                graph_ms = _time_cuda_graph(call, warmup, repetitions)
            except RuntimeError as error:
                print(f"{case.name:12s} {label} cuda_graph=UNAVAILABLE ({error})")
            else:
                print(f"{case.name:12s} {label} cuda_graph_replay={graph_ms:.3f} ms")
                if label == "MetaAttention":
                    meta_graph = graph_ms
                else:
                    fla_graph = graph_ms
    if fla_valid:
        if meta_e2e is not None and fla_e2e is not None:
            print(f"{case.name:12s} end_to_end_speedup={meta_e2e / fla_e2e:.3f}x")
        if meta_graph is not None and fla_graph is not None:
            print(f"{case.name:12s} cuda_graph_speedup={meta_graph / fla_graph:.3f}x")
    else:
        print(f"{case.name:12s} speedup=NOT REPORTED (stock FLA is invalid against oracle)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="use one case")
    parser.add_argument("--seqlen", type=int, help="override sequence length")
    parser.add_argument("--timing", choices=("e2e", "graph", "both"), default="both")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = CASES[:1] if args.quick else CASES
    if args.seqlen is not None:
        cases = tuple(
            Case(c.name, c.factory, c.batch, c.heads, args.seqlen, c.dim, c.dim_value)
            for c in cases
        )
    for case in cases:
        _run_case(case, args.warmup, args.repetitions, args.timing)


if __name__ == "__main__":
    main()
