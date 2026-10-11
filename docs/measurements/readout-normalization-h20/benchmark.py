"""Small public-call inference benchmark for analyzed readout normalization.

Run with CUDA_VISIBLE_DEVICES selecting an idle H20 and PYTHONPATH containing
attention_engine and the repository root. Compilation and correctness checks
are excluded from warmed timing; CUDA events and Graph replay are separate.
"""

import gc
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from attn_engine import AlgorithmIR, StateContraction, StatefulOperator, Target
from testing import benchmark_stateful_plans as bench

OUTPUT = Path(__file__).resolve().parent
WARMUP, REPETITIONS = 5, 20
FAMILIES = ("gdn2", "iplr", "dplr", "delta-product")
LENGTHS = (512, 2048)


def contraction_operator(algorithm):
    return StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=tuple(
                StateContraction(
                    readout.name,
                    readout.expression.state,
                    readout.expression.operand,
                    readout.expression.role,
                    readout.output_dtype,
                )
                for readout in algorithm.readouts
            ),
            head_mapping=algorithm.head_mapping,
        )
    )


def measure(call, *, graph_supported):
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    cpu_ms = bench._cpu_wall_time(call, 0, REPETITIONS)
    event_ms = bench._median_time(call, 0, REPETITIONS)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = call()
    torch.cuda.synchronize()
    incremental = torch.cuda.max_memory_allocated() - before
    del result
    graph_ms = bench._graph_time(call, 2, REPETITIONS) if graph_supported else None
    return {
        "cpu_wall_ms": cpu_ms,
        "public_cuda_event_ms": event_ms,
        "graph_ms": graph_ms,
        "peak_incremental_bytes": incremental,
    }


def main():
    torch.set_num_threads(1)
    records_path = OUTPUT / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    completed = {(r["family"], r["length"], r["variant"]) for r in records}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_worktree_diff": subprocess.check_output(["git", "diff"], text=True),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "contract": "dense inference, zero initial State, output and final State",
        "shape": "B=1, H=8, K=V=64; BF16 inputs/output, FP32 State",
        "warmup": WARMUP,
        "repetitions": REPETITIONS,
        "compilation_timed": False,
        "memory_scope": "warmed single-call incremental PyTorch CUDA allocator peak, including output and final State",
    }
    if not (OUTPUT / "manifest.json").exists():
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with torch.inference_mode():
        for length in LENGTHS:
            for family in FAMILIES:
                case = next(case for case in bench.CASES if case.name == family)
                authored = case.factory(**case.kwargs(length))
                contraction = contraction_operator(authored.algorithm)
                roles = bench.role_sizes_for(authored, case, length)
                torch.manual_seed(42)
                inputs = bench.inputs_for(authored, case, length, torch.device("cuda"))
                expected, expected_states = bench._token_oracle(authored, inputs, roles)
                first = None
                optimized_plan = None
                for label, operator, optimized in (
                    ("authored-optimized", authored, True),
                    ("contraction-optimized", contraction, True),
                    ("authored-unoptimized", authored, False),
                ):
                    compiled = operator.compile(
                        role_sizes=roles,
                        target=Target(workspace_bytes=8 << 30),
                        mode="dense",
                        optimized=optimized,
                    )
                    executable = operator._plans[compiled.specialization]
                    plan = executable.plan
                    call = bench._meta_call(compiled, [inputs], "final-state")
                    invoked = []
                    original_factored = StatefulOperator._lower_factored_recurrence
                    original_reference = StatefulOperator._execute_reference

                    def factored(owner, *args, **kwargs):
                        invoked.append("factored-recurrence")
                        return original_factored(owner, *args, **kwargs)

                    def reference(owner, *args, **kwargs):
                        invoked.append("reference")
                        return original_reference(owner, *args, **kwargs)

                    with (
                        patch.object(
                            StatefulOperator, "_lower_factored_recurrence", factored
                        ),
                        patch.object(StatefulOperator, "_execute_reference", reference),
                    ):
                        output, states = call()
                    correctness = bench._correctness(
                        output, states, expected, expected_states, "final-state"
                    )
                    if optimized:
                        assert correctness["status"] == "VALID", (
                            family,
                            label,
                            correctness,
                        )
                        assert (
                            plan.candidate.lowerer == "tilelang-factored-recurrence"
                        ), plan.candidate
                        assert invoked == ["factored-recurrence"], invoked
                        if first is None:
                            first = (output, states)
                            optimized_plan = plan
                        else:
                            assert (
                                plan.binary_identity == optimized_plan.binary_identity
                            )
                            for name, value in output.items():
                                torch.testing.assert_close(
                                    value, first[0][name], rtol=0, atol=0
                                )
                            for name, value in states.items():
                                torch.testing.assert_close(
                                    value, first[1][name], rtol=0, atol=0
                                )
                    if (family, length, label) in completed:
                        continue
                    record = {
                        "family": family,
                        "length": length,
                        "variant": label,
                        "lowerer": plan.candidate.lowerer,
                        "runtime_family": executable.runtime_family,
                        "observed_execution": invoked or ["matrix-summary"],
                        "representation": plan.mathematical_plan.representation.value,
                        "binary_identity": plan.binary_identity,
                        "correctness": correctness,
                        "status": "OK"
                        if correctness["status"] == "VALID"
                        else "INVALID",
                        "graph_supported": executable.runtime_family
                        != "stateful-reference",
                        "measurement": measure(
                            call,
                            graph_supported=executable.runtime_family
                            != "stateful-reference",
                        )
                        if correctness["status"] == "VALID"
                        else None,
                    }
                    records.append(record)
                    (OUTPUT / "records.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(json.dumps(record), flush=True)
                    del output, states, call, compiled
                del first, inputs, authored, contraction
                gc.collect()
                torch.cuda.empty_cache()
    assert len(records) == len(FAMILIES) * len(LENGTHS) * 3


if __name__ == "__main__":
    main()
