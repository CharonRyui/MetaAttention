"""Compare the public inference result with a direct ordered residual recurrence."""

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from testing import benchmark_stateful_plans as b

torch.set_num_threads(1)
torch.manual_seed(42)
case = replace(next(c for c in b.CASES if c.name == "delta-product"), batch=2, heads=16)
length = 8192
op = case.factory(**case.kwargs(length))
sizes = b.role_sizes_for(op, case, length)
co = op.compile(role_sizes=sizes, target=b.Target(), mode="dense")
inputs = b.inputs_for(op, case, length, torch.device("cuda"))
with torch.inference_mode():
    actual = co(**inputs, return_final_state=True)
    values = {n: v.cpu().float() for n, v in inputs.items()}
    state = torch.zeros(2, 16, 64, 64)
    outputs = []
    for t in range(length):
        state = state * values["gate"][:, :, t].exp()[..., None, None]
        for j in range(2):
            key = values[f"key_{j}"][:, :, t]
            value = values[f"value_{j}"][:, :, t]
            beta = values[f"beta_{j}"][:, :, t]
            prediction = (key[..., None] * state).sum(-2)
            state = (
                state
                + key[..., None]
                * (beta[..., None] * (value - prediction))[..., None, :]
            )
        outputs.append((values["query"][:, :, t, :, None] * (64**-0.5) * state).sum(-2))
    expected = torch.stack(outputs, 2)
    output = actual.outputs["output"].cpu().float()
    delta = (output - expected).abs()
    bad = delta > (0.05 + 0.02 * expected.abs())
    row = {
        "seed": 42,
        "batch": 2,
        "heads": 16,
        "seqlen": 8192,
        "dtype": "bfloat16",
        "oracle": "direct ordered residual recurrence, FP32 CPU",
        "output_max_abs": delta.max().item(),
        "output_failing_elements": int(bad.sum()),
        "output_valid": not bool(bad.any()),
        "first_failure_index": bad.nonzero()[0].tolist() if bad.any() else None,
        "state_max_abs": (actual.final_state["memory"].cpu() - state)
        .abs()
        .max()
        .item(),
        "state_valid": torch.allclose(
            actual.final_state["memory"].cpu(), state, atol=0.02, rtol=0.01
        ),
    }
    print(json.dumps(row), flush=True)
    with open("/tmp/delta_product_plan_probe.json", "w") as f:
        json.dump(row, f, indent=2)
