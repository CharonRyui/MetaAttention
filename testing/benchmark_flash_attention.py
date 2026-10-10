"""Compare the softmax attention DSL against local FlashAttention CuTe.

Run on an otherwise idle supported CUDA GPU, for example::

    CUDA_VISIBLE_DEVICES=4 uv run python testing/benchmark_flash_attention.py

The first invocation includes kernel compilation; reported steady-state numbers
exclude compilation and use the same BSHD tensors for both implementations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.softmax_attention import softmax_attention


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    kv_heads: int
    seqlen: int
    dim: int
    causal: bool


CASES = (
    Case("mha-causal", 1, 16, 16, 512, 64, True),
    Case("mha-noncausal", 1, 16, 16, 512, 64, False),
    Case("gqa-causal", 1, 16, 4, 512, 64, True),
)


def _time(call, warmup: int, repetitions: int) -> float:
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


def _attention_flops(case: Case) -> float:
    pairs = case.seqlen * (case.seqlen + 1) // 2 if case.causal else case.seqlen**2
    # QK^T and attention-value each perform one multiply-add per pair/dimension.
    return float(4 * case.batch * case.heads * pairs * case.dim)


def _profile(call, attempts: int = 2) -> dict:
    from torch.profiler import ProfilerActivity, profile

    events = []
    error = None
    for attempt in range(attempts):
        try:
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CUDA], acc_events=True
            ) as profiler:
                call()
                torch.cuda.synchronize()
            events = [
                event
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
        except RuntimeError as caught:
            error = str(caught)
        if events or attempt == attempts - 1:
            break
    names = [event.name for event in events]
    non_kernel_tokens = ("memcpy", "memset", "alloc", "free")
    kernel_events = [
        name
        for name in names
        if not any(token in name.lower() for token in non_kernel_tokens)
    ]
    result = {
        "cuda_activity_events": len(events) if events else None,
        "cuda_activity_ms": sum(event.device_time_total for event in events) / 1000.0
        if events
        else None,
        "kernel_launches": len(kernel_events) if events else None,
        "launch_source": "PyTorch CUPTI CUDA activity events; empty capture is reported as unavailable",
    }
    if error is not None:
        result["profiler_error"] = error
    if not events:
        result["profiler_status"] = "unavailable"
    return result


def _measure(call, case: Case, warmup: int, repetitions: int) -> dict:
    latency_ms = _time(call, warmup, repetitions)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    result = call()
    torch.cuda.synchronize()
    metrics = {
        "cuda_event_ms": latency_ms,
        "attention_tflops": _attention_flops(case) / (latency_ms * 1e9),
        "peak_incremental_allocated_bytes": torch.cuda.max_memory_allocated()
        - before_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "PyTorch CUDA allocator around one warmed invocation",
    }
    del result
    metrics.update(_profile(call))
    return metrics


def _reference(q, k, v, causal: bool):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=k.shape[1] != q.shape[1]
    )
    return output.transpose(1, 2)


def _run_case(case: Case, warmup: int, repetitions: int) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        case.batch,
        case.seqlen,
        case.heads,
        case.dim,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    k = torch.randn(
        case.batch,
        case.seqlen,
        case.kv_heads,
        case.dim,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    v = torch.randn_like(k)

    meta = softmax_attention(
        case.batch,
        case.heads,
        case.seqlen,
        case.dim,
        kv_heads=case.kv_heads,
        causal=case.causal,
    )
    from flash_attn.cute.interface import flash_attn_func

    def run_meta():
        return meta(q, k, v)

    def run_flash():
        output = flash_attn_func(q, k, v, causal=case.causal)
        return output[0] if isinstance(output, tuple) else output

    # Trigger compilation before measuring either implementation.
    meta_output = run_meta()
    flash_output = run_flash()
    reference = _reference(q, k, v, case.causal)
    torch.testing.assert_close(meta_output, reference, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(flash_output, reference, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(meta_output, flash_output, rtol=3e-2, atol=3e-2)
    correctness = {
        "reference": "torch.nn.functional.scaled_dot_product_attention",
        "meta_vs_reference": "passed",
        "flash_vs_reference": "passed",
        "meta_vs_flash": "passed",
        "rtol": 3e-2,
        "atol": 3e-2,
    }
    meta_metrics = _measure(run_meta, case, warmup, repetitions)
    flash_metrics = _measure(run_flash, case, warmup, repetitions)
    record = {
        "name": case.name,
        "batch": case.batch,
        "heads": case.heads,
        "kv_heads": case.kv_heads,
        "seqlen": case.seqlen,
        "dim": case.dim,
        "causal": case.causal,
        "dtype": "float16",
        "correctness": correctness,
        "meta_attention": meta_metrics,
        "flash_attention": flash_metrics,
        "flash_over_meta_latency_ratio": flash_metrics["cuda_event_ms"]
        / meta_metrics["cuda_event_ms"],
        "meta_over_flash_latency_ratio": meta_metrics["cuda_event_ms"]
        / flash_metrics["cuda_event_ms"],
    }
    print(
        f"{case.name:14s} B={case.batch} H={case.heads} HKV={case.kv_heads} "
        f"S={case.seqlen} D={case.dim} causal={case.causal} "
        f"MetaAttention={meta_metrics['cuda_event_ms']:.3f} ms "
        f"FlashAttention={flash_metrics['cuda_event_ms']:.3f} ms "
        f"ratio={record['meta_over_flash_latency_ratio']:.3f}x",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="use one short case")
    parser.add_argument(
        "--output",
        type=Path,
        help="write structured results JSON; defaults to no artifact",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = CASES[:1] if args.quick else CASES
    records = [_run_case(case, args.warmup, args.repetitions) for case in cases]
    if args.output is not None:
        artifact = {
            "schema_version": 2,
            "benchmark": "MetaAttention softmax DSL vs FlashAttention-4 CuTe",
            "scope": "quick case" if args.quick else "complete declared case matrix",
            "device": {
                "visible_device": 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "name": torch.cuda.get_device_name(),
            },
            "software": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "tilelang": getattr(__import__("tilelang"), "__version__", "unknown"),
                "flash_attention_api": "flash_attn.cute.interface.flash_attn_func",
            },
            "timing": {
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "clock": "CUDA events",
                "statistic": "median",
                "compilation_excluded": True,
            },
            "throughput": {
                "metric": "attention TFLOP/s",
                "formula": "4 * batch * query_heads * attended_pairs * head_dim; one multiply-add counted as two FLOPs",
            },
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
