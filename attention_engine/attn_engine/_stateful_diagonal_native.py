"""Raw-binding diagonal summaries and demand-driven contraction regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from ._stateful_dense_expression import compile_dense_expression


@dataclass(frozen=True)
class NativeStateRecipe:
    name: str
    rows: Any
    columns: Any
    row_gate: Any
    column_gate: Any
    key: Any
    value: Any
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
    cut_query: Any = None


@dataclass(frozen=True)
class NativeDiagonalResources:
    output_bytes: int
    chunk_bytes: int
    launches: int
    score_bytes: int
    score_launches: int
    cut_bytes: int
    live_bytes: int
    fused_softmax: bool


def compile_native_diagonal(algorithm, readouts, bonus, role_sizes):
    """Prove native matrix factors and contractions without profile identities."""
    from .stateful_operator import AxisScale, Constant, Identity, ProductFeatureRole, Softmax, StateTerm

    specs = {spec.name: spec for spec in algorithm.inputs}
    states = []
    transitions = {transition.state: transition for transition in algorithm.transitions}
    for state in algorithm.states:
        transition = transitions[state.name]
        if state.rank != 2 or any(isinstance(role, ProductFeatureRole) for role in state.feature_roles):
            return None
        rows, columns = state.feature_roles
        row_gates, column_gates = [], []
        for node in transition.propagations:
            if isinstance(node, Identity):
                continue
            if not isinstance(node, AxisScale):
                return None
            if node.roles not in ((rows,), (columns,)):
                return None
            (column_gates if node.roles == (columns,) else row_gates).append(node.factor)
        factors = {factor.roles: factor.expression for factor in transition.injections[0].factors}
        key = compile_dense_expression(factors[(rows,)], specs, rows)
        value = compile_dense_expression(factors[(columns,)], specs, columns,
                                         factors=(factors[()],) if () in factors else ())
        rg = compile_dense_expression(row_gates[0] if row_gates else Constant(1.0), specs, rows,
                                     factors=tuple(row_gates[1:]))
        cg = compile_dense_expression(column_gates[0] if column_gates else Constant(1.0), specs, columns,
                                     factors=tuple(column_gates[1:]))
        if any(item is None for item in (key, value, rg, cg)):
            return None
        states.append(NativeStateRecipe(state.name, rows, columns, rg, cg, key, value,
                                        not row_gates, not column_gates))
    native_reads = []
    softmax = None
    for readout, expression in readouts:
        if bonus is not None and readout.name == bonus[0].name:
            continue
        if isinstance(expression, StateTerm) and isinstance(expression.operand, Softmax):
            first = expression.operand.operand
            if not isinstance(first, StateTerm) or first.state == expression.state:
                return None
            first_state = next(state for state in algorithm.states if state.name == first.state)
            second_state = next(state for state in algorithm.states if state.name == expression.state)
            if first.role != first_state.feature_roles[0] or expression.role != second_state.feature_roles[0]:
                return None
            if (first_state.feature_roles[1] != second_state.feature_roles[0]
                    or expression.operand.role != first_state.feature_roles[1]):
                return None
            query = compile_dense_expression(first.operand, specs, first.role)
            if query is None or softmax is not None or len(readouts) != 1:
                return None
            softmax = readout, first, expression, query
        elif isinstance(expression, StateTerm):
            state = next(state for state in algorithm.states if state.name == expression.state)
            if expression.role != state.feature_roles[0]:
                return None
            query = compile_dense_expression(expression.operand, specs, expression.role)
            if query is None:
                return None
            native_reads.append((readout, expression, query))
        else:
            return None
    native_bonus = None
    if bonus is not None:
        readout, term, _, _, _, _ = bonus
        from .stateful_operator import Add, Contract, Multiply
        expression = next(expr for out, expr in readouts if out.name == readout.name)
        other = expression.right if expression.left == term else expression.left
        if not isinstance(expression, Add) or not isinstance(other, Multiply):
            return None
        contraction = other.left if isinstance(other.left, Contract) else other.right
        query = compile_dense_expression(term.operand, specs, term.role)
        bonus_value = compile_dense_expression(contraction.left, specs, term.role)
        if query is None or bonus_value is None:
            return None
        native_bonus = readout, term, query, bonus_value
    fused_softmax = False
    if softmax is not None:
        if role_sizes is None:
            return None
        from ._stateful_readout_regions import readout_region_resources
        first_shape = (1, 1, *(role_sizes[role] for role in
                             next(state for state in algorithm.states if state.name == softmax[1].state).feature_roles))
        second_shape = (1, 1, *(role_sizes[role] for role in
                              next(state for state in algorithm.states if state.name == softmax[2].state).feature_roles))
        fused_softmax = readout_region_resources(first_shape, second_shape)["eligible"]
    cut_query = None
    if softmax is not None:
        from .stateful_operator import Batch, HeadRole, Sequence, TensorInput, Input

        feature = softmax[2].role
        spec = TensorInput("contracted", (Batch, Sequence, HeadRole("contracted_heads"), feature), torch.float32)
        cut_query = compile_dense_expression(Input("contracted"), {"contracted": spec}, feature)
    return NativeDiagonalRecipe(tuple(states), tuple(native_reads), native_bonus, softmax,
                                fused_softmax, cut_query=cut_query)


def _score_key(state, query, prior, bonus):
    def expression_key(expression):
        return None if expression is None else (
            expression.loader.cache_key, expression.sources, expression.feature,
        )

    return state, expression_key(query), prior, expression_key(bonus)


def native_diagonal_resources(recipe, sizes, heads):
    """Price all outputs and compact scratch from the same typed runtime recipe."""
    states = {state.name: state for state in recipe.states}
    reads = list(recipe.readouts)
    if recipe.bonus is not None:
        reads.append(recipe.bonus)
    output_bytes = sum(heads * sizes[states[term.state].columns]
                       * out.output_dtype.itemsize
                       for out, term, *_ in reads)
    chunk_bytes = sum(4 * heads * (3 * sizes[state.rows] * sizes[state.columns]
                                  + 2 * (sizes[state.rows] + sizes[state.columns]))
                      for state in recipe.states)
    score_groups = {}
    for _, term, query, *bonus in reads:
        state = states[term.state]
        if state.identity_columns:
            key = _score_key(state.name, query, term.prior, bonus[0] if bonus else None)
            score_groups[key] = 4 * heads * (64 * 64 + 64 * sizes[state.rows] + (64 if bonus else 0))
    cut_bytes = live_bytes = 0
    launches = 2 * len(recipe.states) + len(reads)
    if recipe.softmax is not None:
        out, first, second, _ = recipe.softmax
        output_bytes += heads * sizes[states[second.state].columns] * out.output_dtype.itemsize
        cut_bytes = 8 * heads * sizes[states[first.state].columns]
        from ._stateful_readout_regions import readout_region_resources

        resources = readout_region_resources(
            (1, 1, sizes[states[first.state].rows], sizes[states[first.state].columns]),
            (1, 1, sizes[states[second.state].rows], sizes[states[second.state].columns]),
        )
        live_bytes = resources["live_fp32_bytes"]
        launches += 1 if recipe.fused_softmax else 3
    return NativeDiagonalResources(output_bytes, chunk_bytes, launches,
                                   sum(score_groups.values()), len(score_groups), cut_bytes, live_bytes,
                                   recipe.fused_softmax)


@triton.jit
def _load(P, Strides: tl.constexpr, Groups: tl.constexpr, Loaders: tl.constexpr,
          INDEX: tl.constexpr, seq, head, t, x, VALID, SIZE: tl.constexpr):
    loader: tl.constexpr = Loaders[INDEX]
    strides: tl.constexpr = Strides[INDEX]
    groups: tl.constexpr = Groups[INDEX]
    return loader(P[INDEX], seq, head, t, x, VALID, SIZE, strides, groups)


@triton.jit
def _summary(P, Row, Column, Bias, Initial, Final,
             Strides: tl.constexpr, Groups: tl.constexpr, Loaders: tl.constexpr,
             LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
             COLS: tl.constexpr, SUMMARIES: tl.constexpr, FINAL_ONLY: tl.constexpr):
    sh, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r = tile // tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    c = tile % tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    local = tl.arange(0, 64)
    t = chunk * 64 + local
    rg = _load(P, Strides, Groups, Loaders, 0, seq, head, t[None, :] + 1, r[:, None],
               (local[None, :] < 63) & (t[None, :] + 1 < LENGTH) & (r[:, None] < ROWS), ROWS)
    cg = _load(P, Strides, Groups, Loaders, 1, seq, head, t[:, None] + 1, c[None, :],
               (local[:, None] < 63) & (t[:, None] + 1 < LENGTH) & (c[None, :] < COLS), COLS)
    rg = tl.where((local[None, :] < 63) & (t[None, :] + 1 < LENGTH), rg, 1.0)
    cg = tl.where((local[:, None] < 63) & (t[:, None] + 1 < LENGTH), cg, 1.0)
    rs = tl.associative_scan(rg, 1, _multiply, reverse=True)
    cs = tl.associative_scan(cg, 0, _multiply, reverse=True)
    key = _load(P, Strides, Groups, Loaders, 2, seq, head, t[None, :], r[:, None],
                (t[None, :] < LENGTH) & (r[:, None] < ROWS), ROWS)
    value = _load(P, Strides, Groups, Loaders, 3, seq, head, t[:, None], c[None, :],
                  (t[:, None] < LENGTH) & (c[None, :] < COLS), COLS)
    weighted_key = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=f,f",
                                            args=[key * rs], dtype=tl.float32, is_pure=False, pack=1)
    weighted_value = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=f,f",
                                              args=[value * cs], dtype=tl.float32, is_pure=False, pack=1)
    bias = tl.dot(weighted_key, weighted_value, input_precision="tf32x3")
    ar = tl.sum(tl.where(local[None, :] == 0, rs, 0.0), 1)
    ac = tl.sum(tl.where(local[:, None] == 0, cs, 0.0), 0)
    ar *= _load(P, Strides, Groups, Loaders, 0, seq, head, chunk * 64, r, r < ROWS, ROWS)
    ac *= _load(P, Strides, Groups, Loaders, 1, seq, head, chunk * 64, c, c < COLS, COLS)
    if FINAL_ONLY:
        initial = tl.load(Initial + (sh * ROWS + r[:, None]) * COLS + c[None, :],
                          (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
        tl.store(Final + (sh * ROWS + r[:, None]) * COLS + c[None, :],
                 initial * (ar[:, None] * ac[None, :]) + bias,
                 (r[:, None] < ROWS) & (c[None, :] < COLS))
    else:
        tl.store(Bias + ((sh * SUMMARIES + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
                 bias, (r[:, None] < ROWS) & (c[None, :] < COLS))
        tl.store(Row + (sh * SUMMARIES + chunk) * ROWS + r, ar,
                 (r < ROWS) & (tile % tl.cdiv(COLS, 32) == 0))
        tl.store(Column + (sh * SUMMARIES + chunk) * COLS + c, ac,
                 (c < COLS) & (tile // tl.cdiv(COLS, 32) == 0))


@triton.jit
def _multiply(a, b):
    return a * b


@triton.jit
def _scores(P, Q, Bonus, Scores, WeightedQuery, BonusScore,
            Strides: tl.constexpr, Groups: tl.constexpr, Loaders: tl.constexpr,
            QS: tl.constexpr, QG: tl.constexpr, QL: tl.constexpr,
            BS: tl.constexpr, BG: tl.constexpr, BL: tl.constexpr,
            LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
            CHUNKS: tl.constexpr, PRIOR: tl.constexpr, HAS_BONUS: tl.constexpr):
    sh, chunk, target_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    target = target_block * 16 + tl.arange(0, 16)
    end = target - 1 if PRIOR else target
    score = tl.full((16, 64), 0.0, tl.float32)
    local = tl.arange(0, 64)
    bonus_score = tl.full((16,), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + tl.arange(0, 32)
        query = QL(Q, seq, head, (chunk * 64 + target)[:, None], r[None, :],
                   ((chunk * 64 + target)[:, None] < LENGTH) & (r[None, :] < ROWS), ROWS, QS, QG)
        weighted = query
        first = _load(P, Strides, Groups, Loaders, 0, seq, head, chunk * 64, r, r < ROWS, ROWS)
        for step in range(63, -1, -1):
            key = _load(P, Strides, Groups, Loaders, 2, seq, head, chunk * 64 + step, r,
                        (chunk * 64 + step < LENGTH) & (r < ROWS), ROWS)
            value = tl.sum(weighted * key[None, :], 1)
            score += tl.where((local[None, :] == step) & (step <= end[:, None]), value[:, None], 0.0)
            gate = _load(P, Strides, Groups, Loaders, 0, seq, head, chunk * 64 + step, r,
                         (chunk * 64 + step < LENGTH) & (r < ROWS), ROWS)
            weighted *= tl.where((step <= end[:, None]) & (step > 0), gate[None, :], 1.0)
        weighted *= tl.where(end[:, None] >= 0, first[None, :], 1.0)
        tl.store(WeightedQuery + ((sh * CHUNKS + chunk) * 64 + target[:, None]) * ROWS + r[None, :],
                 weighted, r[None, :] < ROWS)
        if HAS_BONUS:
            b = BL(Bonus, seq, head, (chunk * 64 + target)[:, None], r[None, :],
                   ((chunk * 64 + target)[:, None] < LENGTH) & (r[None, :] < ROWS), ROWS, BS, BG)
            bonus_score += tl.sum(b, 1)
    tl.store(Scores + ((sh * CHUNKS + chunk) * 64 + target[:, None]) * 64 + local[None, :], score)
    if HAS_BONUS:
        tl.store(BonusScore + (sh * CHUNKS + chunk) * 64 + target, bonus_score)


@triton.jit
def _contract(P, Scores, WeightedQuery, BonusScore, Carry, Output,
              Strides: tl.constexpr, Groups: tl.constexpr, Loaders: tl.constexpr,
              LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
              COLS: tl.constexpr, CHUNKS: tl.constexpr, DIRECT: tl.constexpr,
              HAS_BONUS: tl.constexpr):
    sh, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    target = tile // tl.cdiv(COLS, 32) * 16 + tl.arange(0, 16)
    c = tile % tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    result = tl.full((16, 32), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + tl.arange(0, 32)
        query = tl.load(WeightedQuery + ((sh * CHUNKS + chunk) * 64 + target[:, None]) * ROWS + r[None, :],
                        r[None, :] < ROWS, 0.0)
        offset = sh * ROWS * COLS if DIRECT else (sh * CHUNKS + chunk) * ROWS * COLS
        state = tl.load(Carry + offset + r[:, None] * COLS + c[None, :],
                        (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
        result += tl.dot(query, state, input_precision="tf32x3")
    local = tl.arange(0, 64)
    score = tl.load(Scores + ((sh * CHUNKS + chunk) * 64 + target[:, None]) * 64 + local[None, :])
    value = _load(P, Strides, Groups, Loaders, 3, seq, head, (chunk * 64 + local)[:, None], c[None, :],
                  ((chunk * 64 + local)[:, None] < LENGTH) & (c[None, :] < COLS), COLS)
    result += tl.dot(score, value, input_precision="tf32x3")
    if HAS_BONUS:
        bonus = tl.load(BonusScore + (sh * CHUNKS + chunk) * 64 + target)
        current = _load(P, Strides, Groups, Loaders, 3, seq, head, (chunk * 64 + target)[:, None], c[None, :],
                        ((chunk * 64 + target)[:, None] < LENGTH) & (c[None, :] < COLS), COLS)
        result += bonus[:, None] * current
    t = chunk * 64 + target
    tl.store(Output + ((seq * LENGTH + t[:, None]) * HEADS + head) * COLS + c[None, :],
             result, (t[:, None] < LENGTH) & (c[None, :] < COLS))


@triton.jit
def _replay(P, Q, Bonus, Carry, Output, Final,
            Strides: tl.constexpr, Groups: tl.constexpr, Loaders: tl.constexpr,
            QS: tl.constexpr, QG: tl.constexpr, QL: tl.constexpr,
            BS: tl.constexpr, BG: tl.constexpr, BL: tl.constexpr,
            LENGTH: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
            COLS: tl.constexpr, CHUNKS: tl.constexpr, BR: tl.constexpr,
            DIRECT: tl.constexpr, PRIOR: tl.constexpr, HAS_BONUS: tl.constexpr,
            WRITE_FINAL: tl.constexpr):
    sh, chunk, cb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    r, c = tl.arange(0, BR), cb * 32 + tl.arange(0, 32)
    offset = sh * ROWS * COLS if DIRECT else (sh * CHUNKS + chunk) * ROWS * COLS
    state = tl.load(Carry + offset + r[:, None] * COLS + c[None, :],
                    (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
    for step in range(64):
        t = chunk * 64 + step
        query = QL(Q, seq, head, t, r, (t < LENGTH) & (r < ROWS), ROWS, QS, QG)
        if PRIOR:
            result = tl.sum(query[:, None] * state, 0)
        rg = _load(P, Strides, Groups, Loaders, 0, seq, head, t, r, (t < LENGTH) & (r < ROWS), ROWS)
        cg = _load(P, Strides, Groups, Loaders, 1, seq, head, t, c, (t < LENGTH) & (c < COLS), COLS)
        key = _load(P, Strides, Groups, Loaders, 2, seq, head, t, r, (t < LENGTH) & (r < ROWS), ROWS)
        value = _load(P, Strides, Groups, Loaders, 3, seq, head, t, c, (t < LENGTH) & (c < COLS), COLS)
        updated = state * rg[:, None] * cg[None, :] + key[:, None] * value[None, :]
        state = tl.where(t < LENGTH, updated, state)
        if not PRIOR:
            result = tl.sum(query[:, None] * state, 0)
        if HAS_BONUS:
            b = BL(Bonus, seq, head, t, r, (t < LENGTH) & (r < ROWS), ROWS, BS, BG)
            result += tl.sum(b, 0) * value
        tl.store(Output + ((seq * LENGTH + t) * HEADS + head) * COLS + c, result,
                 (t < LENGTH) & (c < COLS))
    if WRITE_FINAL:
        if chunk == CHUNKS - 1:
            tl.store(Final + (sh * ROWS + r[:, None]) * COLS + c[None, :], state,
                     (r[:, None] < ROWS) & (c[None, :] < COLS))


@triton.jit
def _advance_native(state, P, Strides: tl.constexpr, Groups: tl.constexpr,
                    Loaders: tl.constexpr, seq, head, t, r, c,
                    ROWS: tl.constexpr, COLS: tl.constexpr, VALID):
    rg = _load(P, Strides, Groups, Loaders, 0, seq, head, t, r, VALID & (r < ROWS), ROWS)
    cg = _load(P, Strides, Groups, Loaders, 1, seq, head, t, c, VALID & (c < COLS), COLS)
    key = _load(P, Strides, Groups, Loaders, 2, seq, head, t, r, VALID & (r < ROWS), ROWS)
    value = _load(P, Strides, Groups, Loaders, 3, seq, head, t, c, VALID & (c < COLS), COLS)
    updated = state * rg[:, None] * cg[None, :] + key[:, None] * value[None, :]
    return tl.where(VALID, updated, state)


@triton.jit
def _softmax_region(P1, P2, Q, C1, C2, Output, F1, F2,
                    S1: tl.constexpr, G1: tl.constexpr, L1: tl.constexpr,
                    S2: tl.constexpr, G2: tl.constexpr, L2: tl.constexpr,
                    QS: tl.constexpr, QG: tl.constexpr, QL: tl.constexpr,
                    LENGTH: tl.constexpr, HEADS: tl.constexpr, CHUNKS: tl.constexpr,
                    R1: tl.constexpr, SLOTS: tl.constexpr, COLS: tl.constexpr,
                    BR: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                    PRIOR1: tl.constexpr, PRIOR2: tl.constexpr,
                    DIRECT: tl.constexpr, WRITE_FINAL: tl.constexpr):
    sh, chunk = tl.program_id(0), tl.program_id(1)
    seq, head = sh // HEADS, sh % HEADS
    r, m, c = tl.arange(0, BR), tl.arange(0, BM), tl.arange(0, BC)
    offset1 = sh * R1 * SLOTS if DIRECT else (sh * CHUNKS + chunk) * R1 * SLOTS
    offset2 = sh * SLOTS * COLS if DIRECT else (sh * CHUNKS + chunk) * SLOTS * COLS
    first = tl.load(C1 + offset1 + r[:, None] * SLOTS + m[None, :],
                    (r[:, None] < R1) & (m[None, :] < SLOTS), 0.0)
    second = tl.load(C2 + offset2 + m[:, None] * COLS + c[None, :],
                     (m[:, None] < SLOTS) & (c[None, :] < COLS), 0.0)
    for step in range(64):
        t = chunk * 64 + step
        query = QL(Q, seq, head, t, r, (t < LENGTH) & (r < R1), R1, QS, QG)
        if PRIOR1:
            logits = tl.sum(query[:, None] * first, 0)
        first = _advance_native(first, P1, S1, G1, L1, seq, head, t, r, m, R1, SLOTS, t < LENGTH)
        if not PRIOR1:
            logits = tl.sum(query[:, None] * first, 0)
        logits = tl.where(m < SLOTS, logits, float("-inf"))
        weight = tl.exp(logits - tl.max(logits, 0))
        weight = weight / tl.sum(weight, 0)
        if PRIOR2:
            result = tl.sum(weight[:, None] * second, 0)
        second = _advance_native(second, P2, S2, G2, L2, seq, head, t, m, c, SLOTS, COLS, t < LENGTH)
        if not PRIOR2:
            result = tl.sum(weight[:, None] * second, 0)
        tl.store(Output + ((seq * LENGTH + t) * HEADS + head) * COLS + c,
                 result, (t < LENGTH) & (c < COLS))
    if WRITE_FINAL:
        if chunk == CHUNKS - 1:
            tl.store(F1 + (sh * R1 + r[:, None]) * SLOTS + m[None, :], first,
                     (r[:, None] < R1) & (m[None, :] < SLOTS))
            tl.store(F2 + (sh * SLOTS + m[:, None]) * COLS + c[None, :], second,
                     (m[:, None] < SLOTS) & (c[None, :] < COLS))


def _read_softmax(recipe, first, second, bindings, query, *, inputs, sizes, batch, length, heads):
    c1, f1, direct = first
    c2, f2, _ = second
    rows, slots, columns = c1.shape[-2], c1.shape[-1], c2.shape[-1]
    readout, term1, term2, _ = recipe.softmax
    ql, qp, qs, qg = query.bind(inputs, sizes, batch, length, heads)
    output = torch.empty(batch * length, heads, columns, device=c1.device, dtype=readout.output_dtype)
    (p1, md1), (p2, md2) = bindings
    _softmax_region[(batch * heads, triton.cdiv(length, 64))](
        p1, p2, qp, c1, c2, output, f1, f2,
        S1=md1["Strides"], G1=md1["Groups"], L1=md1["Loaders"],
        S2=md2["Strides"], G2=md2["Groups"], L2=md2["Loaders"],
        QS=qs, QG=qg, QL=ql, LENGTH=length, HEADS=heads, CHUNKS=triton.cdiv(length, 64),
        R1=rows, SLOTS=slots, COLS=columns,
        BR=triton.next_power_of_2(rows), BM=triton.next_power_of_2(slots), BC=triton.next_power_of_2(columns),
        PRIOR1=term1.prior, PRIOR2=term2.prior, DIRECT=direct,
        WRITE_FINAL=direct and f1 is not c1, num_warps=8, enable_fp_fusion=False,
    )
    return output


def _bind_state(recipe, inputs, sizes, batch, length, heads):
    bindings = tuple(expr.bind(inputs, sizes, batch, length, heads)
                     for expr in (recipe.row_gate, recipe.column_gate, recipe.key, recipe.value))
    return (tuple(binding[1] for binding in bindings),
            dict(Strides=tuple(binding[2] for binding in bindings),
                 Groups=tuple(binding[3] for binding in bindings),
                 Loaders=tuple(binding[0] for binding in bindings)))


def _prepare(recipe, bound, initial, *, length, heads, final_requested):
    batch, _, rows, columns = initial.shape
    chunks = triton.cdiv(length, 64)
    final = torch.empty_like(initial) if final_requested else initial
    if chunks == 1:
        return initial, final, True
    summaries = chunks if final_requested else chunks - 1
    row = torch.empty((batch * heads, summaries, rows), device=initial.device, dtype=torch.float32)
    column = torch.empty((batch * heads, summaries, columns), device=initial.device, dtype=torch.float32)
    bias = torch.empty((batch * heads, summaries, rows, columns), device=initial.device, dtype=torch.float32)
    carry = torch.empty((batch * heads, chunks, rows, columns), device=initial.device, dtype=torch.float32)
    pointers, metadata = bound
    _summary[(batch * heads, summaries, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))](
        pointers, row, column, bias, initial, final, **metadata,
        LENGTH=length, HEADS=heads, ROWS=rows, COLS=columns, SUMMARIES=summaries,
        FINAL_ONLY=False, num_warps=4, enable_fp_fusion=False,
    )
    from ._stateful_diagonal_prefix import compact_carry
    compact_carry(row, column, bias, initial, carry, final, write_final=final_requested)
    return carry, final, False


def _read(recipe, bound, prepared, query, bonus, *, length, heads, output_dtype, prior,
          inputs, sizes, final_requested, shared_scores=False, score_cache=None):
    carry, final, direct = prepared
    batch_heads = carry.shape[0] * heads if direct else carry.shape[0]
    rows, columns = carry.shape[-2:]
    chunks = triton.cdiv(length, 64)
    batch = batch_heads // heads
    ql, qp, qs, qg = query.bind(inputs, sizes, batch, length, heads)
    bl, bp, bs, bg = (bonus or query).bind(inputs, sizes, batch, length, heads)
    output = torch.empty(batch * length, heads, columns, device=carry.device, dtype=output_dtype)
    pointers, metadata = bound
    # Independent chunks are the only replay domain; never serialize a sequence.
    if direct or not recipe.identity_columns or not shared_scores:
        _replay[(batch_heads, chunks, triton.cdiv(columns, 32))](
            pointers, qp, bp, carry, output, final, **metadata,
            QS=qs, QG=qg, QL=ql, BS=bs, BG=bg, BL=bl,
            LENGTH=length, HEADS=heads, ROWS=rows, COLS=columns, CHUNKS=chunks,
            BR=triton.next_power_of_2(rows), DIRECT=direct, PRIOR=prior,
            HAS_BONUS=bonus is not None, WRITE_FINAL=direct and final_requested,
            num_warps=4, enable_fp_fusion=False,
        )
    else:
        cache_key = _score_key(recipe.name, query, prior, bonus)
        cached = None if score_cache is None else score_cache.get(cache_key)
        if cached is None:
            scores = torch.empty((batch_heads, chunks, 64, 64), device=carry.device, dtype=torch.float32)
            weighted = torch.empty((batch_heads, chunks, 64, rows), device=carry.device, dtype=torch.float32)
            bonus_scores = torch.empty((batch_heads, chunks, 64), device=carry.device, dtype=torch.float32) if bonus else output
            _scores[(batch_heads, chunks, 4)](
                pointers, qp, bp, scores, weighted, bonus_scores, **metadata,
                QS=qs, QG=qg, QL=ql, BS=bs, BG=bg, BL=bl,
                LENGTH=length, HEADS=heads, ROWS=rows, CHUNKS=chunks,
                PRIOR=prior, HAS_BONUS=bonus is not None, num_warps=4, enable_fp_fusion=False,
            )
            if score_cache is not None:
                score_cache[cache_key] = scores, weighted, bonus_scores
        else:
            scores, weighted, bonus_scores = cached
        _contract[(batch_heads, chunks, 4 * triton.cdiv(columns, 32))](
            pointers, scores, weighted, bonus_scores, carry, output, **metadata,
            LENGTH=length, HEADS=heads, ROWS=rows, COLS=columns, CHUNKS=chunks,
            DIRECT=False, HAS_BONUS=bonus is not None, num_warps=4, enable_fp_fusion=False,
        )
    return output


def execute_native_diagonal(recipe, inputs, initial_states, *, role_sizes,
                            sequence_count, sequence_length, heads, return_final_state):
    states = {state.name: state for state in recipe.states}
    bound, prepared, outputs = {}, {}, {}
    score_cache = {}

    def prepare(name):
        if name not in prepared:
            state = states[name]
            bound[name] = _bind_state(state, inputs, role_sizes, sequence_count, sequence_length, heads)
            prepared[name] = _prepare(state, bound[name], initial_states[name], length=sequence_length,
                                      heads=heads, final_requested=return_final_state)
        return prepared[name]

    consumed = set()
    if recipe.softmax is not None:
        readout, term1, term2, query = recipe.softmax
        first, second = prepare(term1.state), prepare(term2.state)
        if recipe.fused_softmax:
            outputs[readout.name] = _read_softmax(
                recipe, first, second, (bound[term1.state], bound[term2.state]), query,
                inputs=inputs, sizes=role_sizes, batch=sequence_count,
                length=sequence_length, heads=heads,
            )
        else:
            logits = _read(states[term1.state], bound[term1.state], first, query, None,
                           length=sequence_length, heads=heads, output_dtype=torch.float32,
                           prior=term1.prior, inputs=inputs, sizes=role_sizes,
                           final_requested=return_final_state)
            weight = logits.softmax(-1).reshape(sequence_count, sequence_length, heads, -1)
            weight_query = recipe.cut_query
            outputs[readout.name] = _read(
                states[term2.state], bound[term2.state], second, weight_query, None,
                length=sequence_length, heads=heads, output_dtype=readout.output_dtype,
                prior=term2.prior, inputs={"contracted": weight}, sizes=role_sizes,
                final_requested=return_final_state,
            )
        consumed.update((term1.state, term2.state))
    reads = list(recipe.readouts)
    if recipe.bonus is not None:
        readout, term, query, bonus = recipe.bonus
        reads.append((readout, term, query, bonus))
    for item in reads:
        readout, term, query = item[:3]
        bonus = item[3] if len(item) == 4 else None
        state = prepare(term.state)
        outputs[readout.name] = _read(states[term.state], bound[term.state], state, query, bonus,
                                      length=sequence_length, heads=heads, output_dtype=readout.output_dtype,
                                      prior=term.prior, inputs=inputs, sizes=role_sizes,
                                      final_requested=return_final_state,
                                      shared_scores=recipe.shared_scores, score_cache=score_cache)
    finals = None
    if return_final_state:
        finals = {}
        consumed.update(item[1].state for item in reads)
        for name, state_recipe in states.items():
            carry, final, direct = prepare(name)
            if direct and name not in consumed:
                pointers, metadata = bound[name]
                rows, columns = initial_states[name].shape[-2:]
                _summary[(sequence_count * heads, 1, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))](
                    pointers, final, final, final, initial_states[name], final, **metadata,
                    LENGTH=sequence_length, HEADS=heads, ROWS=rows, COLS=columns,
                    SUMMARIES=1, FINAL_ONLY=True, num_warps=4, enable_fp_fusion=False,
                )
            finals[name] = final
    return outputs, finals
