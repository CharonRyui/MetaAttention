from __future__ import annotations

import torch

from examples.gated_retention import gated_retention


def main() -> None:
    device = torch.device("cuda")
    query = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = -torch.rand(1, 2, 128, device=device)
    operator = gated_retention(1, 2, 128, 64, 64)
    operator(query=query, key=key, value=value, gate=gate)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        operator(query=query, key=key, value=value, gate=gate)
    torch.cuda.synchronize()
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_memory_usage", row_limit=20
        )
    )


if __name__ == "__main__":
    main()
