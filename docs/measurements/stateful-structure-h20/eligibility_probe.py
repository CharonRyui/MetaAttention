import json
from attn_engine import AlgorithmIR, StateContraction, StateSpec
from attn_engine.stateful.lowering.reference import requires_reference_lowering
from attn_engine.stateful.lowering.lowrank.wy_kernels import eligible
from testing import benchmark_stateful_plans as b

results = []
for name in ("gdn2", "iplr", "dplr", "oja", "delta-product", "abc"):
    case = next(c for c in b.CASES if c.name == name)
    op = case.factory(**case.kwargs(512))
    a = op.algorithm
    variants = [("authored", a)]
    if name in ("gdn2", "iplr", "dplr", "delta-product"):
        variants.append(
            (
                "canonical-readout",
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
                ),
            )
        )
    if name == "oja":
        state = a.states[0]
        variants.append(
            (
                "transpose-state-view",
                AlgorithmIR(
                    inputs=a.inputs,
                    states=(
                        StateSpec(
                            state.name,
                            (*state.roles[:2], *reversed(state.feature_roles)),
                        ),
                    ),
                    transition=a.transitions,
                    readouts=a.readouts,
                    head_mapping=a.head_mapping,
                ),
            )
        )
    for label, alg in variants:
        factored = alg._analysis.rank_one_factored
        counts = [len(x.factors) for x in alg.transitions[0].injections]
        event_count = sum(
            type(n).__name__ == "RankOnePropagation"
            for n in alg.transitions[0].propagations
        ) + len(counts)
        r = {
            "family": name,
            "variant": label,
            "requires_reference": requires_reference_lowering(alg),
            "factored_eligible": factored is not None,
            "injection_factor_counts": counts,
            "event_count": event_count,
            "feature_roles": [x.name for x in alg.states[0].feature_roles],
            "propagation_roles": [
                getattr(n, "role", None).name
                if getattr(n, "role", None) is not None
                else None
                for n in alg.transitions[0].propagations
            ],
            "readout_role": getattr(alg.readouts[0], "role", None).name
            if getattr(alg.readouts[0], "role", None) is not None
            else None,
        }
        if factored:
            codes = tuple(x.kind for x in factored.operations)
            r.update(
                operation_codes=codes,
                wy_eligible=eligible(codes, factored.injection_count, 64, 64),
            )
        results.append(r)
        print(json.dumps(r), flush=True)
with open("/tmp/meta_structure_eligibility_probe.json", "w") as f:
    json.dump(results, f, indent=2)
