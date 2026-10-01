"""Work-efficient hierarchical affine prefix over diagonal chunk summaries."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _local(Scale, Bias, GroupScale, GroupBias,
           ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
           IDENTITY: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    sh, group, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    x = block * BLOCK + tl.arange(0, BLOCK)
    a = tl.full((BLOCK,), 1.0, tl.float32)
    b = tl.full((BLOCK,), 0.0, tl.float32)
    for step in range(GROUP):
        chunk = group * GROUP + step
        address = (sh * CHUNKS + chunk) * ELEMENTS + x
        scale = tl.full((BLOCK,), 1.0, tl.float32)
        if not IDENTITY:
            scale = tl.load(Scale + address, (chunk < CHUNKS) & (x < ELEMENTS), 1.0)
        bias = tl.load(Bias + address, (chunk < CHUNKS) & (x < ELEMENTS), 0.0)
        b = scale * b + bias
        a = scale * a
    address = (sh * GROUPS + group) * ELEMENTS + x
    tl.store(GroupScale + address, a, x < ELEMENTS)
    tl.store(GroupBias + address, b, x < ELEMENTS)


@triton.jit
def _apply(Scale, LocalBias, GroupCarry, Carry, Final,
           ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
           IDENTITY: tl.constexpr, WRITE_FINAL: tl.constexpr,
           GROUP: tl.constexpr, BLOCK: tl.constexpr):
    sh, group, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    x = block * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(GroupCarry + (sh * GROUPS + group) * ELEMENTS + x,
                    x < ELEMENTS, 0.0)
    for step in range(GROUP):
        chunk = group * GROUP + step
        address = (sh * CHUNKS + chunk) * ELEMENTS + x
        tl.store(Carry + address, state, (chunk < CHUNKS) & (x < ELEMENTS))
        scale = tl.full((BLOCK,), 1.0, tl.float32)
        if not IDENTITY:
            scale = tl.load(Scale + address, (chunk < CHUNKS) & (x < ELEMENTS), 1.0)
        bias = tl.load(LocalBias + address, (chunk < CHUNKS) & (x < ELEMENTS), 0.0)
        state = scale * state + bias
    if WRITE_FINAL:
        if group == GROUPS - 1:
            tl.store(Final + sh * ELEMENTS + x, state, x < ELEMENTS)


def hierarchical_carry(scale, bias, initial, carry, final, *, identity, write_final):
    """Prefix chunks in groups; recursive summaries bound the serial critical path.

    No division or inversion is used, including zero diagonal factors. The
    original summaries are immutable, so recursion needs only group buffers.
    """
    from ._stateful_diagonal_kernels import _carry

    bh, chunks, rows, columns = bias.shape
    elements = rows * columns
    if chunks <= 64:
        _carry[(bh, triton.cdiv(elements, 256))](
            scale, bias, initial, carry, final, rows, columns, chunks,
            identity, write_final,
        )
        return
    groups = triton.cdiv(chunks, 32)
    shape = (bh, groups, rows, columns)
    group_scale = torch.empty(shape, device=bias.device, dtype=torch.float32)
    group_bias = torch.empty_like(group_scale)
    group_carry = torch.empty_like(group_scale)
    # Only group summaries are materialized; _apply writes exclusive carries.
    _local[(bh, groups, triton.cdiv(elements, 256))](
        scale, bias, group_scale, group_bias,
        elements, chunks, groups, identity, 32, 256,
    )
    hierarchical_carry(group_scale, group_bias, initial, group_carry, final,
                       identity=identity, write_final=False)
    _apply[(bh, groups, triton.cdiv(elements, 256))](
        scale, bias, group_carry, carry, final, elements, chunks, groups,
        identity, write_final, 32, 256,
    )
