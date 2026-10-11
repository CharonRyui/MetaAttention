import json
import sys
import types
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from attn_engine import Target
from testing import benchmark_stateful_plans as bench
from attn_engine.stateful.lowering.lowrank import checkpoint

source = subprocess.check_output(
    [
        "git",
        "show",
        "43de339:attention_engine/attn_engine/stateful/lowering/lowrank/checkpoint.py",
    ],
    text=True,
)
name = checkpoint.__package__ + ".baseline_ordered"
old = types.ModuleType(name)
old.__package__ = checkpoint.__package__
old.__file__ = "/tmp/meta_ordered_baseline.py"
sys.modules[name] = old
Path(old.__file__).write_text(source)
exec(compile(source, old.__file__, "exec"), old.__dict__)
current = checkpoint.forward
records = []
base = next(c for c in bench.CASES if c.name == "delta-product")
for dims in ((64, 64), (64, 128), (128, 64)):
    for length in (128, 2048):
        case = replace(base, dim=dims[0], dim_value=dims[1])
        torch.manual_seed(92)
        op = case.factory(**case.kwargs(length))
        compiled = op.compile(
            role_sizes=bench.role_sizes_for(op, case, length),
            target=Target(),
            mode="dense",
        )
        inputs = bench.inputs_for(op, case, length, torch.device("cuda"))

        def call():
            return compiled(**inputs, return_final_state=True)

        with torch.no_grad():
            a = call()
            with patch.object(checkpoint, "forward", old.forward):
                b = call()
            torch.testing.assert_close(
                a.outputs["output"], b.outputs["output"], atol=2e-3, rtol=0.02
            )
            torch.testing.assert_close(
                a.final_state["memory"], b.final_state["memory"], atol=3e-6, rtol=3e-5
            )
            rec = {
                "dims": dims,
                "length": length,
                "max_output_error": (a.outputs["output"] - b.outputs["output"])
                .abs()
                .max()
                .item(),
                "max_final_error": (a.final_state["memory"] - b.final_state["memory"])
                .abs()
                .max()
                .item(),
            }
            with patch.object(checkpoint, "forward", old.forward):
                rec["before"] = bench._measure_call(call, 5, 20)
            rec["after"] = bench._measure_call(call, 5, 20)
            records.append(rec)
            print(json.dumps(rec), flush=True)
Path("/tmp/meta_ordered_measure_records.json").write_text(json.dumps(records, indent=2))
