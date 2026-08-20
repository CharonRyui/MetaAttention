from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable

import torch

from attn_engine import LinearAttentionEngine
from core import CustomIO
from core.utils import meta_tensor
from examples.gated_retention import gated_retention


BATCHES = 10
REPETITIONS = 20
WARMUP = 10


def _legacy_gated_retention(B, H, S, D, DV, dtype=torch.bfloat16):
    scale = 1 / D**0.5

    def q_mod(query, custom_io):
        return query * scale

    return LinearAttentionEngine(
        (
            meta_tensor(B, H, S, D, dtype=dtype),
            meta_tensor(B, H, S, D, dtype=dtype),
            meta_tensor(B, H, S, DV, dtype=dtype),
        ),
        q_mod=q_mod,
        custom_io=CustomIO({}),
        tune=False,
    )


BATCHES = 10
REPETITIONS = 20
WARMUP = 10


def _median_runtime(call: Callable[[], object]) -> tuple[float, float]:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(REPETITIONS):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), max(samples)


def _paired_median_bootstrap(
    new_values: list[float], old_values: list[float], *, seed: int = 17
) -> tuple[float, float]:
    if len(new_values) != len(old_values) or not new_values:
        raise ValueError("paired samples must have equal nonzero length")
    generator = torch.Generator().manual_seed(seed)
    new = torch.tensor(new_values, dtype=torch.float64)
    old = torch.tensor(old_values, dtype=torch.float64)
    draws = torch.randint(0, len(new_values), (20_000, len(new_values)), generator=generator)
    ratios = new[draws].median(dim=1).values / old[draws].median(dim=1).values
    interval = torch.quantile(ratios, torch.tensor((0.025, 0.975), dtype=new.dtype))
    return interval[0].item(), interval[1].item()


def _profile(device: torch.device):
    generator = torch.Generator(device=device).manual_seed(11)
    query = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16, generator=generator)
    gate = -torch.rand(1, 2, 128, device=device, generator=generator)
    return query, key, value, gate


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    query, key, value, gate = _profile(device)
    compile_start = time.perf_counter()
    new = gated_retention(1, 2, 128, 64, 64)

    def new_call():
        return new(query=query, key=key, value=value, gate=gate).outputs["output"]

    new_call()
    torch.cuda.synchronize()
    first_compile_ms = (time.perf_counter() - compile_start) * 1_000
    old = _legacy_gated_retention(1, 2, 128, 64, 64)

    def old_call():
        return old(query, key, value, gate)
    new_samples, old_samples = [], []
    peak_new, peak_old = [], []
    for batch in range(BATCHES):
        order = (new_call, old_call) if batch % 2 == 0 else (old_call, new_call)
        for call in order:
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            median, _ = _median_runtime(call)
            peak = torch.cuda.max_memory_allocated(device) - baseline
            if call is new_call:
                new_samples.append(median)
                peak_new.append(peak)
            else:
                old_samples.append(median)
                peak_old.append(peak)
    latency_ratios = [
        new_value / old_value
        for new_value, old_value in zip(new_samples, old_samples, strict=True)
    ]
    memory_ratios = [
        new_value / old_value
        for new_value, old_value in zip(peak_new, peak_old, strict=True)
    ]
    print(
        json.dumps(
            {
                "batches": BATCHES,
                "repetitions": REPETITIONS,
                "warmup": WARMUP,
                "first_compile_ms": first_compile_ms,
                "latency_ratio_ci95": _paired_median_bootstrap(
                    new_samples, old_samples
                ),
                "memory_ratio_ci95": _paired_median_bootstrap(peak_new, peak_old),
                "latency_ratios": latency_ratios,
                "memory_ratios": memory_ratios,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
