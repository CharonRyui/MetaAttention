import json
from pathlib import Path
import torch
from dataclasses import replace
from attn_engine import (
    AlgorithmIR,
    Batch,
    Contract,
    Input,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StatefulOperator,
    Target,
    TensorInput,
)
from testing import benchmark_stateful_plans as b

torch.set_num_threads(1)
records = []
for length in (512, 2048):
    case = replace(next(c for c in b.CASES if c.name == "delta-product"), heads=8)
    authored = case.factory(**case.kwargs(length))
    a = authored.algorithm
    readouts = tuple(
        StateContraction(
            r.name,
            r.expression.state,
            r.expression.operand,
            r.expression.role,
            r.output_dtype,
        )
        for r in a.readouts
    )
    # Both engines see the same IR. Prebind input-only Contract to work around
    # the existing dense fallback's scalar-reduction broadcast bug.
    inputs = list(a.inputs)
    overlap_inputs = []
    injections = []
    for i, injection in enumerate(a.transition.injections):
        factors = []
        for j, factor in enumerate(injection.factors):
            if isinstance(factor.expression, Contract):
                name = f"overlap_{i}_{j}"
                head = a.states[0].head_role
                inputs.append(TensorInput(name, (Batch, head, Sequence), torch.float32))
                overlap_inputs.append((name, factor.expression))
                factors.append(ProductFactor(Input(name), ()))
            else:
                factors.append(factor)
        injections.append(ProductInjection(tuple(factors)))
    transition = replace(a.transition, injections=tuple(injections))
    op = StatefulOperator(
        AlgorithmIR(
            inputs=tuple(inputs),
            states=a.states,
            transition=transition,
            readouts=readouts,
            head_mapping=a.head_mapping,
        )
    )
    sizes = b.role_sizes_for(authored, case, length)
    torch.manual_seed(42)
    values = b.inputs_for(authored, case, length, torch.device("cuda"))
    expected, expected_states = b._token_oracle(authored, values, sizes)
    for name, expr in overlap_inputs:
        values[name] = (
            values[expr.left.name].float() * values[expr.right.name].float()
        ).sum(-1)
    for optimized in (False, True):
        compiled = op.compile(
            role_sizes=sizes,
            target=Target(workspace_bytes=8 << 30),
            mode="dense",
            optimized=optimized,
        )
        executable = op._plans[compiled.specialization]
        call = b._meta_call(compiled, [values], "final-state")
        output, states = call()
        correctness = b._correctness(
            output, states, expected, expected_states, "final-state"
        )
        assert correctness["status"] == "VALID", correctness
        measurement = b._measure_call(call, 3, 10, "both")
        record = dict(
            family="delta-product",
            probe_readout="explicit-StateContraction",
            prebound_overlap=True,
            optimized=optimized,
            length=length,
            heads=8,
            lowerer=executable.plan.candidate.lowerer,
            correctness=correctness,
            measurement=measurement,
        )
        print(json.dumps(record), flush=True)
        records.append(record)
    # Actual input-only Contract, through weighted executor, default 1GiB budget.
    actual = StatefulOperator(
        AlgorithmIR(
            inputs=a.inputs,
            states=a.states,
            transition=a.transitions,
            readouts=readouts,
            head_mapping=a.head_mapping,
        )
    )
    compiled = actual.compile(role_sizes=sizes, target=Target(), mode="dense")
    actual_values = {
        name: value
        for name, value in values.items()
        if name in {spec.name for spec in a.inputs}
    }
    call = b._meta_call(compiled, [actual_values], "final-state")
    output, states = call()
    correctness = b._correctness(
        output, states, expected, expected_states, "final-state"
    )
    assert correctness["status"] == "VALID", correctness
    record = dict(
        family="delta-product",
        probe_readout="explicit-StateContraction",
        prebound_overlap=False,
        optimized=True,
        length=length,
        heads=8,
        lowerer=actual._plans[compiled.specialization].plan.candidate.lowerer,
        correctness=correctness,
        measurement=b._measure_call(call, 3, 10, "both"),
    )
    print(json.dumps(record), flush=True)
    records.append(record)
Path(__file__).with_name("records.json").write_text(json.dumps(records, indent=2))
