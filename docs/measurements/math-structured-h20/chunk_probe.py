"""Explore inference replay granularity without changing production dispatch."""

import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch
from attn_engine import Target
from attn_engine.stateful.lowering.lowrank import checkpoint
from testing import benchmark_stateful_plans as benchmark

original = checkpoint.forward
source = inspect.getsource(original)
needle = "        64\n        if not record"
assert source.count(needle) == 1
records = []
for family in ("delta-product", "gdn2", "dplr"):
    case = replace(next(c for c in benchmark.CASES if c.name == family), heads=8)
    operator = case.factory(**case.kwargs(2048))
    roles = benchmark.role_sizes_for(operator, case, 2048)
    torch.manual_seed(42)
    inputs = benchmark.inputs_for(operator, case, 2048, torch.device("cuda"))
    expected, states = benchmark._token_oracle(operator, inputs, roles)
    compiled = operator.compile(role_sizes=roles, target=Target(), mode="dense")
    call = benchmark._meta_call(compiled, [inputs], "final-state")
    for chunk in (16, 32, 64, 128):
        namespace = dict(checkpoint.__dict__)
        exec(
            compile(
                source.replace(needle, f"        {chunk}\n        if not record"),
                f"chunk-{chunk}",
                "exec",
            ),
            namespace,
        )
        checkpoint.forward = namespace["forward"]
        output, actual_states = call()
        correctness = benchmark._correctness(
            output, actual_states, expected, states, "final-state"
        )
        assert correctness["status"] == "VALID", correctness
        measurement = benchmark._measure_call(call, 5, 20, "both")
        record = {
            "family": family,
            "chunk_tokens": chunk,
            "batch": 1,
            "heads": 8,
            "length": 2048,
            "correctness": correctness,
            "measurement": measurement,
        }
        records.append(record)
        Path("/tmp/meta_math_chunk_probe.json").write_text(
            json.dumps(records, indent=2)
        )
        print(json.dumps(record), flush=True)
checkpoint.forward = original
