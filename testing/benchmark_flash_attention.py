"""Compare the softmax attention DSL against local FlashAttention CuTe.

Run on an otherwise idle supported CUDA GPU, for example::

    CUDA_VISIBLE_DEVICES=4 uv run python testing/benchmark_flash_attention.py

The first invocation includes kernel compilation; reported steady-state numbers
exclude compilation and use the same BSHD tensors for both implementations.
"""

from __future__ import annotations

import argparse
import statistics
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


def _reference(q, k, v, causal: bool):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=k.shape[1] != q.shape[1]
    )
    return output.transpose(1, 2)


def _run_case(case: Case, warmup: int, repetitions: int) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        case.batch, case.seqlen, case.heads, case.dim,
        device=device, dtype=torch.float16, generator=generator,
    )
    k = torch.randn(
        case.batch, case.seqlen, case.kv_heads, case.dim,
        device=device, dtype=torch.float16, generator=generator,
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

    meta_ms = _time(run_meta, warmup, repetitions)
    flash_ms = _time(run_flash, warmup, repetitions)
    print(
        f"{case.name:14s} B={case.batch} H={case.heads} HKV={case.kv_heads} "
        f"S={case.seqlen} D={case.dim} causal={case.causal} "
        f"MetaAttention={meta_ms:.3f} ms FlashAttention={flash_ms:.3f} ms "
        f"ratio={meta_ms / flash_ms:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="use one short case")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = CASES[:1] if args.quick else CASES
    for case in cases:
        _run_case(case, args.warmup, args.repetitions)


if __name__ == "__main__":
    main()
