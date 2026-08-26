from __future__ import annotations

from .stateful_operator import (
    AlgorithmIR,
    AxisScale,
    Identity,
    RankOnePropagation,
    StatefulCompilationError,
    _reject_duplicate_names,
)


def validate_algorithm_structure(algorithm: AlgorithmIR) -> None:
    _reject_duplicate_names(algorithm.inputs, "inputs")
    _reject_duplicate_names(algorithm.states, "states")
    if len(algorithm.states) != 1:
        raise StatefulCompilationError("IR_SCHEMA", "states", {"expected": 1})
    state = algorithm.states[0]
    if algorithm.transition.state != state.name:
        raise StatefulCompilationError(
            "IR_TYPE", "transition.state", {"unknown_state": algorithm.transition.state}
        )
    if not algorithm.readouts:
        raise StatefulCompilationError(
            "IR_SCHEMA", "readouts", {"expected": "at least one"}
        )
    _reject_duplicate_names(algorithm.readouts, "readouts")


def validate_transition_structure(algorithm: AlgorithmIR) -> None:
    state = algorithm.states[0]
    features = set(state.feature_roles)
    for index, propagation in enumerate(algorithm.transition.propagations):
        path = f"transition.propagations[{index}]"
        if isinstance(propagation, AxisScale):
            if not set(propagation.roles) <= features:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.roles",
                    {"roles": [str(role) for role in propagation.roles]},
                )
        elif isinstance(propagation, RankOnePropagation):
            if propagation.role not in features:
                raise StatefulCompilationError(
                    "IR_TYPE", f"{path}.role", {"role": str(propagation.role)}
                )
        elif not isinstance(propagation, Identity):
            raise StatefulCompilationError(
                "IR_SCHEMA", path, {"node": type(propagation).__name__}
            )
    for index, injection in enumerate(algorithm.transition.injections):
        covered = {role for factor in injection.factors for role in factor.roles}
        if not features <= covered:
            raise StatefulCompilationError(
                "IR_TYPE",
                f"transition.injections[{index}]",
                {"missing_roles": sorted(str(role) for role in features - covered)},
            )
    for index, readout in enumerate(algorithm.readouts):
        if readout.state != state.name or readout.role not in features:
            raise StatefulCompilationError(
                "IR_TYPE",
                f"readouts[{index}]",
                {"state": readout.state, "role": str(readout.role)},
            )
