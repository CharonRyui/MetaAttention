"""Alternate executor order across rounds; exclude compilation from timing."""

import gc
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
from attn_engine import Target
from attn_engine.stateful.lowering import chunked
from testing import benchmark_stateful_plans as bench

source = subprocess.check_output(
    [
        "git",
        "show",
        "d1289d6:attention_engine/attn_engine/stateful/lowering/chunked.py",
    ],
    text=True,
)
name = chunked.__package__ + ".baseline_binding"
old = types.ModuleType(name)
old.__package__ = chunked.__package__
old.__file__ = "/tmp/meta_binding_baseline.py"
sys.modules[name] = old
Path(old.__file__).write_text(source)
exec(compile(source, old.__file__, "exec"), old.__dict__)
records = []
torch.set_num_threads(1)
with torch.no_grad():
    for family in ("abc", "gsa", "based"):
        case = next(c for c in bench.CASES if c.name == family)
        for length in (128, 2048):
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

            a = call()
            with patch.object(
                chunked, "execute_chunked_states", old.execute_chunked_states
            ):
                b = call()
            for output in a.outputs.names:
                torch.testing.assert_close(
                    a.outputs[output], b.outputs[output], rtol=0, atol=0
                )
            for state in a.final_state.names:
                torch.testing.assert_close(
                    a.final_state[state], b.final_state[state], rtol=0, atol=0
                )
            del a, b
            record = {
                "family": family,
                "length": length,
                "paired_correctness": "bitwise equal",
                "rounds": [],
            }
            for round_index in range(5):
                measurements = {}
                for label in (
                    ("before", "after") if round_index % 2 == 0 else ("after", "before")
                ):
                    executor = (
                        old.execute_chunked_states
                        if label == "before"
                        else chunked.execute_chunked_states
                    )
                    with patch.object(chunked, "execute_chunked_states", executor):
                        measurements[label] = {
                            "cpu_wall_ms": bench._cpu_wall_time(call, 10, 100),
                            "public_cuda_event_ms": bench._median_time(call, 10, 100),
                        }
                record["rounds"].append(measurements)
            for label in ("before", "after"):
                gc.collect()
                torch.cuda.synchronize()
                executor = (
                    old.execute_chunked_states
                    if label == "before"
                    else chunked.execute_chunked_states
                )
                with patch.object(chunked, "execute_chunked_states", executor):
                    measurement = bench._measure_call(call, 10, 100)
                record[label] = measurement
                record[label]["round_median_cpu_wall_ms"] = statistics.median(
                    r[label]["cpu_wall_ms"] for r in record["rounds"]
                )
                record[label]["round_median_public_cuda_event_ms"] = statistics.median(
                    r[label]["public_cuda_event_ms"] for r in record["rounds"]
                )
            records.append(record)
            print(json.dumps(record), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
Path("/tmp/meta_binding_paired_records.json").write_text(json.dumps(records, indent=2))
