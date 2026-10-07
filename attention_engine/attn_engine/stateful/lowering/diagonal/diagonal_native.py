"""Native diagonal recipe compiler and TileLang-backed execution dispatcher."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from ..expression.dense_expression import compile_dense_expression

@dataclass(frozen=True)
class NativeStateRecipe:
    name: str
    rows: Any
    columns: Any
    row_gate: Any
    column_gate: Any
    injections: tuple[tuple[Any, Any], ...]
    identity_rows: bool
    identity_columns: bool
@dataclass(frozen=True)
class NativeDiagonalRecipe:
    states: tuple[NativeStateRecipe, ...]
    readouts: tuple[Any, ...]
    bonus: Any
    softmax: Any
    fused_softmax: bool
    shared_scores: bool = False
@dataclass(frozen=True)
class NativeDiagonalResources:
    output_bytes: int
    chunk_bytes: int
    launches: int
    flops_per_token: int
    score_bytes: int
    score_launches: int
    cut_bytes: int
    live_bytes: int
    fused_softmax: bool

def compile_native_diagonal(algorithm, readouts, bonus, role_sizes):
    from ....stateful_operator import AxisScale, Constant, Identity, ProductFeatureRole, Softmax, StateTerm
    specs = {s.name: s for s in algorithm.inputs}
    transitions = {t.state: t for t in algorithm.transitions}
    states = []
    for state in algorithm.states:
        transition=transitions[state.name]
        if state.rank != 2 or any(isinstance(r, ProductFeatureRole) for r in state.feature_roles):
            return None
        rows, columns = state.feature_roles
        rgs = []
        cgs = []
        for node in transition.propagations:
            if isinstance(node, Identity):
                continue
            if not isinstance(node, AxisScale) or node.roles not in ((rows,), (columns,)):
                return None
            (cgs if node.roles == (columns,) else rgs).append(node.factor)
        injections=[]
        for injection in transition.injections:
            factors={f.roles:f.expression for f in injection.factors}
            if (rows,) not in factors or (columns,) not in factors:
                return None
            key=compile_dense_expression(factors[(rows,)],specs,rows)
            value=compile_dense_expression(factors[(columns,)],specs,columns,factors=(factors[()],) if () in factors else ())
            if key is None or value is None:
                return None
            injections.append((key,value))
        rg=compile_dense_expression(rgs[0] if rgs else Constant(1.0),specs,rows,factors=tuple(rgs[1:]))
        cg=compile_dense_expression(cgs[0] if cgs else Constant(1.0),specs,columns,factors=tuple(cgs[1:]))
        if not injections or rg is None or cg is None:
            return None
        states.append(NativeStateRecipe(state.name,rows,columns,rg,cg,tuple(injections),not rgs,not cgs))
    native = []
    softmax = None
    for readout, expression in readouts:
        if bonus is not None and readout.name == bonus[0].name:
            continue
        if isinstance(expression, StateTerm) and isinstance(expression.operand, Softmax):
            first=expression.operand.operand
            if not isinstance(first, StateTerm) or first.state == expression.state:
                return None
            fs=next(s for s in algorithm.states if s.name == first.state)
            ss=next(s for s in algorithm.states if s.name == expression.state)
            if (first.role != fs.feature_roles[0] or expression.role != ss.feature_roles[0]
                    or fs.feature_roles[1] != ss.feature_roles[0]
                    or expression.operand.role != fs.feature_roles[1]):
                return None
            query=compile_dense_expression(first.operand,specs,first.role)
            if query is None or softmax is not None or len(readouts) != 1:
                return None
            softmax=(readout,first,expression,query)
        elif isinstance(expression, StateTerm):
            state=next(s for s in algorithm.states if s.name == expression.state)
            if expression.role != state.feature_roles[0]:
                return None
            query=compile_dense_expression(expression.operand,specs,expression.role)
            if query is None:
                return None
            native.append((readout,expression,query))
        else:
            return None
    native_bonus=None
    if bonus is not None:
        readout,term,_,_,_,_=bonus
        from ....stateful_operator import Add, Contract, Multiply
        expression=next(expr for out,expr in readouts if out.name == readout.name)
        other=expression.right if expression.left == term else expression.left
        if not isinstance(expression,Add) or not isinstance(other,Multiply):
            return None
        contraction=other.left if isinstance(other.left,Contract) else other.right
        query=compile_dense_expression(term.operand,specs,term.role)
        bonus_value=compile_dense_expression(contraction.left,specs,term.role)
        if query is None or bonus_value is None:
            return None
        native_bonus=(readout,term,query,bonus_value)
    fused=False
    if softmax is not None:
        if role_sizes is None:
            return None
        from ..readout.readout_regions import readout_region_resources
        fs=next(s for s in algorithm.states if s.name == softmax[1].state)
        ss=next(s for s in algorithm.states if s.name == softmax[2].state)
        first_shape=(1,1,*(role_sizes[r] for r in fs.feature_roles))
        second_shape=(1,1,*(role_sizes[r] for r in ss.feature_roles))
        fused=readout_region_resources(first_shape,second_shape)["eligible"]
    # The chunked planner selects shared-score lowering explicitly; compilation
    # must retain the ordinary replay default for resource planning.
    return NativeDiagonalRecipe(tuple(states),tuple(native),native_bonus,softmax,fused)

def native_diagonal_resources(recipe, sizes, heads):
    states={s.name:s for s in recipe.states}
    reads=list(recipe.readouts)
    if recipe.bonus:
        reads.append(recipe.bonus)
    output_bytes=sum(heads*sizes[states[t.state].columns]*out.output_dtype.itemsize for out,t,*_ in reads)
    if recipe.softmax is not None:
        out, first, second, _ = recipe.softmax
        output_bytes += heads*sizes[states[second.state].columns]*out.output_dtype.itemsize
    # Summary, carry and replay retain three FP32 matrices plus axis vectors.
    chunk_bytes=sum(4*heads*(3*sizes[s.rows]*sizes[s.columns] + 2*(sizes[s.rows]+sizes[s.columns])) for s in recipe.states)
    flops=3*heads*sum(sizes[s.rows]*sizes[s.columns]*len(s.injections) for s in recipe.states)
    score_groups={(term.state, query.loader.cache_key, term.prior) for _,term,query in recipe.readouts}
    score_bytes=sum(4*heads*64*(64*len(states[name].injections)+sizes[states[name].rows])
                    for name, _, _ in score_groups)
    score_launches=len(score_groups)
    cut_bytes=live_bytes=0
    if recipe.softmax is not None:
        first = states[recipe.softmax[1].state]
        second = states[recipe.softmax[2].state]
        cut_bytes=8*heads*sizes[first.columns]
        from ..readout.readout_regions import readout_region_resources
        live_bytes=readout_region_resources(
            (1,1,sizes[first.rows],sizes[first.columns]),
            (1,1,sizes[second.rows],sizes[second.columns]),
        )["live_fp32_bytes"]
    launches=2*len(recipe.states)+len(reads)
    if recipe.softmax is not None:
        launches += 1 if recipe.fused_softmax else 3
    return NativeDiagonalResources(output_bytes,chunk_bytes,launches,flops,
                                   score_bytes,score_launches,cut_bytes,live_bytes,
                                   recipe.fused_softmax)
def execute_native_diagonal(recipe, inputs, initial_states, *, sequence_count, sequence_length, heads, return_final_state):
    from .native_tile_kernels import launch_native_summary, launch_native_carry, launch_native_replay
    prepared = {}
    finals = {}
    for item in recipe.states:
        initial = initial_states[item.name]
        batch_heads = sequence_count * heads
        rows, columns = initial.shape[-2:]
        chunks = (sequence_length + 63) // 64
        if chunks == 1 and not return_final_state:
            carry = initial.reshape(batch_heads, rows, columns).unsqueeze(1)
            prepared[item.name] = (item, carry, initial)
            continue
        scale = torch.empty((batch_heads, chunks, rows, columns), device=initial.device, dtype=torch.float32)
        bias = torch.empty_like(scale)
        launch_native_summary(item.row_gate, item.column_gate, item.injections, inputs, scale, bias,
                              sequence_length=sequence_length, heads=heads)
        carry = torch.empty_like(scale)
        final = torch.empty_like(initial) if return_final_state else initial
        launch_native_carry(scale, bias, initial, carry, final,
                            identity=False, write_final=return_final_state)
        prepared[item.name] = (item, carry, final)
        if return_final_state:
            finals[item.name] = final
    outputs = {}
    if recipe.shared_scores and recipe.readouts:
        from .native_tile_softmax import shared_score_readouts
        grouped = {}
        for output, term, query in recipe.readouts:
            grouped.setdefault(term.state, []).append((output, term, query))
        for state_name, readouts in grouped.items():
            item, carry, _ = prepared[state_name]
            outputs.update(shared_score_readouts(item, carry, readouts, inputs,
                                                 sequence_length=sequence_length, heads=heads))
    else:
        for readout, term, query in recipe.readouts:
            item, carry, _ = prepared[term.state]
            output = torch.empty((sequence_count * sequence_length, heads,
                                  initial_states[term.state].shape[-1]),
                                 device=carry.device, dtype=readout.output_dtype)
            launch_native_replay(carry, item.row_gate, item.column_gate, item.injections,
                                 query, inputs, output, sequence_length=sequence_length,
                                 heads=heads, prior=term.prior)
            outputs[readout.name] = output
    if recipe.bonus is not None:
        from .native_tile_softmax import bonus_readout
        readout, term, query, bonus_value = recipe.bonus
        item, carry, _ = prepared[term.state]
        outputs[readout.name] = bonus_readout(item, carry, query, bonus_value, inputs,
                                              sequence_length=sequence_length, heads=heads,
                                              prior=term.prior, output_dtype=readout.output_dtype)
    if recipe.softmax is not None:
        from .native_tile_softmax import softmax_readout
        readout, first, second, query = recipe.softmax
        first_item, first_carry, _ = prepared[first.state]
        second_item, second_carry, _ = prepared[second.state]
        outputs[readout.name] = softmax_readout(first_item, first_carry, second_item,
                                                second_carry, query, inputs,
                                                sequence_length=sequence_length, heads=heads,
                                                prior_first=first.prior, prior_second=second.prior,
                                                output_dtype=readout.output_dtype,
                                                fused=recipe.fused_softmax)
    return outputs, (finals if return_final_state else None)
