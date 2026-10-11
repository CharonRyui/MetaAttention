import json
import os

import torch
from dataclasses import replace
from attn_engine import AlgorithmIR, StateContraction, StatefulOperator, Target
from testing import benchmark_stateful_plans as b
from testing.benchmark_fla_stateful import choose_idle_gpu, list_gpus

selected = choose_idle_gpu(list_gpus(), requested=0)
assert selected.index == 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(1)
results = []
for name in ("gdn2", "iplr", "dplr"):
    case = replace(next(c for c in b.CASES if c.name == name), heads=8)
    original = case.factory(**case.kwargs(512))
    a = original.algorithm
    canonical = StatefulOperator(
        AlgorithmIR(
            inputs=a.inputs,
            states=a.states,
            transition=a.transitions,
            readouts=tuple(
                StateContraction(
                    r.name,
                    r.expression.state,
                    r.expression.operand,
                    r.expression.role,
                    r.output_dtype,
                )
                for r in a.readouts
            ),
            head_mapping=a.head_mapping,
        )
    )
    sizes = b.role_sizes_for(original, case, 512)
    torch.manual_seed(42)
    values = b.inputs_for(original, case, 512, torch.device("cuda"))
    expected, expected_states = b._token_oracle(original, values, sizes)
    for label, operator in (("authored", original), ("canonical-readout", canonical)):
        r = {
            "family": name,
            "variant": label,
            "batch": 1,
            "heads": 8,
            "length": 512,
            "factored_eligible": operator.algorithm._analysis.rank_one_factored
            is not None,
        }
        try:
            compiled = operator.compile(
                role_sizes=sizes, target=Target(workspace_bytes=8 << 30), mode="dense"
            )
            r["lowerer"] = compiled.operator._plans[
                compiled.specialization
            ].plan.candidate.lowerer
            call = b._meta_call(compiled, [values], "final-state")
            output, states = call()
            r["correctness"] = b._correctness(
                output, states, expected, expected_states, "final-state"
            )
            assert r["correctness"]["status"] == "VALID", r["correctness"]
            r["measurement"] = b._measure_call(call, 3, 10, "both")
            r["status"] = "OK"
        except Exception as e:
            r["status"] = "FAILED"
            r["error"] = str(e)
        results.append(r)
        print(json.dumps(r), flush=True)
with open("/tmp/meta_slow_structure_probe.json", "w") as f:
    json.dump(results, f, indent=2)
