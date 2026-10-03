"""Raw-input vector affine scans; no promoted matrices or token State scratch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ._stateful_bindings import build_binding


@dataclass(frozen=True)
class RankOneRecipe:
    state: str
    gate: object | None
    value: object
    log_gate: bool
    feature: object
    outputs: tuple[tuple[str, torch.dtype], ...]


def compile_rank_one(algorithm):
    """Prove a vector gate/injection and identity readout, without model names."""
    from .stateful_operator import AxisScale, Batch, Constant, Exp, Identity, Input, Sequence, StateTerm

    if len(algorithm.states) != 1:
        return None
    state = algorithm.states[0]
    if state.rank != 1:
        return None
    feature = state.feature_roles[0]
    transition = algorithm.transitions[0]
    gates = [node for node in transition.propagations if not isinstance(node, Identity)]
    gate = None
    log_gate = False
    specs = {spec.name: spec for spec in algorithm.inputs}
    if gates:
        if len(gates) != 1 or not isinstance(gates[0], AxisScale):
            return None
        if gates[0].roles != (feature,):
            return None
        expression = gates[0].factor
        log_gate = isinstance(expression, Exp)
        expression = expression.operand if log_gate else expression
        if not isinstance(expression, Input):
            return None
        gate = specs[expression.name]
    if len(transition.injections) != 1:
        return None
    factors = transition.injections[0].factors
    if len(factors) != 1 or factors[0].roles != (feature,):
        return None
    if not isinstance(factors[0].expression, Input):
        return None
    value = specs[factors[0].expression.name]
    if any(Batch not in spec.roles or Sequence not in spec.roles
           for spec in (value, *((gate,) if gate is not None else ()))):
        return None
    # Independent named outputs retain their existing storage ownership.
    outputs = []
    for readout in algorithm.readouts:
        term = getattr(readout, 'expression', None)
        if not isinstance(term, StateTerm) or term.state != state.name or term.prior:
            return None
        if term.role is not None or not isinstance(term.operand, Constant):
            return None
        if term.operand.value != 1.0:
            return None
        outputs.append((readout.name, readout.output_dtype))
    if len(outputs) != 1:
        return None
    return RankOneRecipe(state.name, gate, value, log_gate, feature, tuple(outputs))


@triton.jit
def _compose(a1, b1, a2, b2):
    return a2 * a1, a2 * b1 + b2


@triton.jit
def _scan(Gate, Value, Initial, Scale, Bias, Carry, Output, Final,
          LENGTH: tl.constexpr, HEADS: tl.constexpr, FEATURES: tl.constexpr,
          CHUNKS: tl.constexpr, GS: tl.constexpr, VS: tl.constexpr,
          GG: tl.constexpr, VG: tl.constexpr, LOG: tl.constexpr,
          IDENTITY: tl.constexpr, SUMMARY: tl.constexpr, DIRECT: tl.constexpr,
          WRITE_FINAL: tl.constexpr, BLOCK: tl.constexpr):
    sh, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    t = chunk * 64 + tl.arange(0, 64)
    f = tile * BLOCK + tl.arange(0, BLOCK)
    valid = (t[:, None] < LENGTH) & (f[None, :] < FEATURES)
    x = tl.load(Value + seq * VS[0] + (head // VG) * VS[1]
                + t[:, None] * VS[2] + f[None, :] * VS[3], valid, 0).to(tl.float32)
    a = tl.full((64, BLOCK), 1.0, tl.float32)
    if not IDENTITY:
        a = tl.load(Gate + seq * GS[0] + (head // GG) * GS[1]
                    + t[:, None] * GS[2] + f[None, :] * GS[3], valid, 0 if LOG else 1).to(tl.float32)
        if LOG:
            a = tl.exp(a)
    a, b = tl.associative_scan((a, x), 0, _compose)
    boundary = (sh * CHUNKS + chunk) * FEATURES + f
    if SUMMARY:
        last_a = tl.sum(tl.where(tl.arange(0, 64)[:, None] == 63, a, 0), 0)
        last_b = tl.sum(tl.where(tl.arange(0, 64)[:, None] == 63, b, 0), 0)
        tl.store(Scale + boundary, last_a, f < FEATURES)
        tl.store(Bias + boundary, last_b, f < FEATURES)
    else:
        if DIRECT:
            initial = tl.load(Initial + sh * FEATURES + f, f < FEATURES, 0)
        else:
            initial = tl.load(Carry + boundary, f < FEATURES, 0)
        result = a * initial[None, :] + b
        address = ((sh * LENGTH + t[:, None]) * FEATURES + f[None, :])
        tl.store(Output + address, result, valid)
        if WRITE_FINAL:
            if chunk == CHUNKS - 1:
                last = tl.sum(tl.where(t[:, None] == LENGTH - 1, result, 0), 0)
                tl.store(Final + sh * FEATURES + f, last, f < FEATURES)


@triton.jit
def _carry(Scale, Bias, Initial, Carry, FEATURES: tl.constexpr,
           CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    sh, tile = tl.program_id(0), tl.program_id(1)
    f = tile * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(Initial + sh * FEATURES + f, f < FEATURES, 0)
    for chunk in range(CHUNKS):
        offset = (sh * CHUNKS + chunk) * FEATURES + f
        tl.store(Carry + offset, state, f < FEATURES)
        scale = tl.load(Scale + offset, f < FEATURES, 1)
        bias = tl.load(Bias + offset, f < FEATURES, 0)
        state = scale * state + bias


def execute_rank_one(recipe, inputs, initial, *, role_sizes, sequence_count,
                     sequence_length, heads, return_final_state):
    from .stateful_operator import Batch, HeadRole, Sequence

    def bind(spec):
        descriptor = build_binding(
            spec, inputs[spec.name], role_sizes=role_sizes,
            sequence_count=sequence_count, uniform_length=sequence_length,
            state_heads=heads, feature_roles=(recipe.feature,),
        )
        roles = descriptor.source_roles
        head = next((role for role in roles if isinstance(role, HeadRole)), None)
        strides = tuple(descriptor.source_strides[roles.index(role)]
                        if role in roles else 0
                        for role in (Batch, head, Sequence, recipe.feature))
        return descriptor.raw, strides, descriptor.head_group_size

    value, vs, vg = bind(recipe.value)
    gate, gs, gg = bind(recipe.gate) if recipe.gate is not None else (initial, (0, 0, 0, 0), 1)
    features = initial.shape[-1]
    chunks = triton.cdiv(sequence_length, 64)
    output = torch.empty(sequence_count, heads, sequence_length, features,
                         device=initial.device, dtype=recipe.outputs[0][1])
    final = torch.empty_like(initial) if return_final_state else initial
    common = dict(LENGTH=sequence_length, HEADS=heads, FEATURES=features,
                  CHUNKS=chunks, GS=gs, VS=vs, GG=gg, VG=vg,
                  LOG=recipe.log_gate, IDENTITY=recipe.gate is None,
                  BLOCK=32, num_warps=4, enable_fp_fusion=False)
    grid = (sequence_count * heads, chunks, triton.cdiv(features, 32))
    if chunks == 1:
        _scan[grid](gate, value, initial, initial, initial, initial, output, final,
                    SUMMARY=False, DIRECT=True, WRITE_FINAL=return_final_state, **common)
    else:
        shape = (sequence_count * heads, chunks, features)
        scale = torch.empty(shape, device=initial.device, dtype=torch.float32)
        bias = torch.empty_like(scale)
        carry = torch.empty_like(scale)
        _scan[grid](gate, value, initial, scale, bias, carry, output, final,
                    SUMMARY=True, DIRECT=False, WRITE_FINAL=False, **common)
        # Keep the direct carry for small scans; long scans use grouped local
        # summaries so the serial prefix depth stays bounded.  These reshapes
        # are views over vector buffers (one row), not matrix state storage.
        if chunks <= 64:
            _carry[(sequence_count * heads, triton.cdiv(features, 128))](
                scale, bias, initial, carry, features, chunks, 128,
                num_warps=4, enable_fp_fusion=False)
        else:
            from ._stateful_diagonal_scan import hierarchical_carry

            hierarchical_carry(
                scale.reshape(sequence_count * heads, chunks, 1, features),
                bias.reshape(sequence_count * heads, chunks, 1, features),
                initial.reshape(sequence_count * heads, 1, features),
                carry.reshape(sequence_count * heads, chunks, 1, features),
                final.reshape(sequence_count * heads, 1, features),
                identity=False,
                write_final=False,
            )
        _scan[grid](gate, value, initial, scale, bias, carry, output, final,
                    SUMMARY=False, DIRECT=False,
                    WRITE_FINAL=return_final_state, **common)
    return output, final if return_final_state else None
