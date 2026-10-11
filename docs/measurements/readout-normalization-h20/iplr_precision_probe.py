"""Verify IPLR affine-scan precision after retaining GEMM residuals.

Run on an idle H20 with the same PYTHONPATH as benchmark.py. Original failing
benchmark records are preserved; this writes a separate precision-fix record.
"""

import hashlib
import json
from pathlib import Path

import torch

from attn_engine import Target
from testing import benchmark_stateful_plans as bench

OUTPUT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    original = json.loads((OUTPUT / "records.json").read_text())
    records = []
    with torch.inference_mode():
        for length in (512, 2048):
            case = next(case for case in bench.CASES if case.name == "iplr")
            operator = case.factory(**case.kwargs(length))
            roles = bench.role_sizes_for(operator, case, length)
            torch.manual_seed(42)
            inputs = bench.inputs_for(operator, case, length, torch.device("cuda"))
            expected, expected_states = bench._token_oracle(operator, inputs, roles)
            for optimized in (False, True):
                compiled = operator.compile(
                    role_sizes=roles,
                    target=Target(workspace_bytes=8 << 30),
                    mode="dense",
                    optimized=optimized,
                )
                plan = operator._plans[compiled.specialization].plan
                expected_lowerer = (
                    "tilelang-factored-recurrence" if optimized else "tilelang-generic"
                )
                assert plan.candidate.lowerer == expected_lowerer, plan.candidate
                call = bench._meta_call(compiled, [inputs], "final-state")
                outputs, states = call()
                correctness = bench._correctness(
                    outputs, states, expected, expected_states, "final-state"
                )
                assert correctness["status"] == "VALID", correctness
                baseline = next(
                    r
                    for r in original
                    if r["family"] == "iplr"
                    and r["length"] == length
                    and r["variant"]
                    == ("authored-optimized" if optimized else "authored-unoptimized")
                )
                record = {
                    "family": "iplr",
                    "length": length,
                    "optimized": optimized,
                    "lowerer": plan.candidate.lowerer,
                    "correctness": correctness,
                    "before_correctness": baseline["correctness"],
                    "warmup": 5,
                    "repetitions": 20,
                    "public_cuda_event_ms": bench._median_time(call, 5, 20),
                    "graph_ms": bench._graph_time(call, 2, 20),
                }
                records.append(record)
                (OUTPUT / "iplr-precision-fix.json").write_text(
                    json.dumps(
                        {
                            "device": torch.cuda.get_device_name(),
                            "shape": "B=1, H=8, K=V=64; BF16 input/output, FP32 State; dense inference, zero initial State, output and final State",
                            "affine_source_sha256": hashlib.sha256(
                                Path(
                                    "attention_engine/attn_engine/stateful/lowering/scan/affine.py"
                                ).read_bytes()
                            ).hexdigest(),
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
