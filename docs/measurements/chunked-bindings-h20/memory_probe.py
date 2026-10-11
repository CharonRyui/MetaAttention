"""Fresh-process allocator measurement before any Graph capture/profiling."""

import json
import subprocess
import sys
import types
from unittest.mock import patch

import torch
from attn_engine import Target
from attn_engine.stateful.lowering import chunked
from testing import benchmark_stateful_plans as bench

label = sys.argv[1]
length = int(sys.argv[2])
if label == "before":
    source = subprocess.check_output(
        [
            "git",
            "show",
            "d1289d6:attention_engine/attn_engine/stateful/lowering/chunked.py",
        ],
        text=True,
    )
    name = chunked.__package__ + ".baseline_memory"
    old = types.ModuleType(name)
    old.__package__ = chunked.__package__
    sys.modules[name] = old
    exec(compile(source, "<baseline>", "exec"), old.__dict__)
    executor = old.execute_chunked_states
else:
    executor = chunked.execute_chunked_states
case = next(c for c in bench.CASES if c.name == "abc")
torch.manual_seed(92)
op = case.factory(**case.kwargs(length))
compiled = op.compile(
    role_sizes=bench.role_sizes_for(op, case, length), target=Target(), mode="dense"
)
inputs = bench.inputs_for(op, case, length, torch.device("cuda"))
with torch.no_grad(), patch.object(chunked, "execute_chunked_states", executor):
    for _ in range(10):
        compiled(**inputs, return_final_state=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = compiled(**inputs, return_final_state=True)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "label": label,
                "length": length,
                "peak_incremental_bytes": torch.cuda.max_memory_allocated() - before,
            }
        ),
        flush=True,
    )
