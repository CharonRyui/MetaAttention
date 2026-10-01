"""Compiler-owned workloads exercised through the ordinary compiled callable."""
from dataclasses import replace
import math

import torch


def invocation_factory(operator, compilation, base_executable, plans, device):
    from ._stateful_plan import KernelConfig, _digest
    from .stateful_operator import (
        Batch, Sequence, One, CompiledStatefulOperator, StatefulOperator, StateTuple,
    )

    roles = dict(compilation.role_sizes)
    states = tuple(operator.algorithm.states)
    state = states[0]

    def build(candidate, config, workload):
        from ._stateful_tuning import workload_lengths

        lengths = workload_lengths(workload)
        sizes = {**roles, Batch: len(lengths), Sequence: max(lengths), One: 1}
        shapes = {}
        for spec in operator.algorithm.inputs:
            if compilation.mode == "packed":
                shape = (sum(lengths), *(roles[role] if role != One else 1
                    for role in spec.roles if role not in (Batch, Sequence)))
            else:
                shape = tuple(sizes[role] for role in spec.roles)
            shapes[spec.name] = shape
        # Conservative live-buffer bound before allocating or launching a candidate.
        elements = sum(math.prod(shape) for shape in shapes.values())
        rows, columns = base_executable.plan.feature_sizes
        heads = roles[state.head_role]
        required = sum(lengths) * heads * (rows * rows + columns * columns + rows * columns) * 64
        required += elements * 8
        free, _ = torch.cuda.mem_get_info(device)
        if required > min(compilation.target.workspace_bytes, free // 2):
            raise RuntimeError("tuning workload exceeds safe resource envelope")
        inputs = {
            spec.name: torch.full(shapes[spec.name], 0.001, device=device, dtype=spec.dtype)
            for spec in operator.algorithm.inputs
        }
        initial_tensors = []
        for spec in states:
            feature_shape = tuple(roles[role] for role in spec.feature_roles)
            initial_tensors.append(
                torch.zeros(len(lengths), roles[spec.head_role], *feature_shape, device=device)
            )
        initial = StateTuple(
            tuple(spec.name for spec in states), tuple(initial_tensors)
        )
        kwargs = {"initial_state": initial, "return_final_state": True}
        if compilation.mode == "packed":
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            kwargs["sequence_offsets"] = torch.tensor(offsets, dtype=torch.int32, device=device)
        kernel_config = KernelConfig(_digest((candidate.identity, config)), tuple(config))
        selected = next(plan for plan in plans if plan.identity == candidate.plan_identity)
        rank_one = (
            base_executable.rank_one_factored
            if candidate.lowerer == "tilelang-factored-recurrence"
            else None
        )
        executable = replace(base_executable, plan=replace(
            base_executable.plan, candidate=candidate, mathematical_plan=selected,
            kernel_config=kernel_config,
        ), rank_one_factored=rank_one)
        # An isolated invocation owner avoids mutating a live user's specialization.
        owner = StatefulOperator(operator.algorithm)
        key = candidate.identity + kernel_config.identity
        owner._plans[key] = executable
        compiled = CompiledStatefulOperator(owner, compilation, key)

        def invoke():
            with torch.no_grad():
                return compiled(**inputs, **kwargs)

        return invoke

    return build
