"""Work-efficient hierarchical affine prefix over diagonal chunk summaries."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _local_body(Scale, Bias, GroupScale, GroupBias, sh, group, block,
                ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
                IDENTITY: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
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
    if GroupScale is not None:
        tl.store(GroupScale + address, a, x < ELEMENTS)
    tl.store(GroupBias + address, b, x < ELEMENTS)


@triton.jit
def _local(Scale, Bias, GroupScale, GroupBias,
           ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
           IDENTITY: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    _local_body(Scale, Bias, GroupScale, GroupBias,
                tl.program_id(0), tl.program_id(1), tl.program_id(2),
                ELEMENTS, CHUNKS, GROUPS, IDENTITY, GROUP, BLOCK)


@triton.jit
def _apply_body(Scale, LocalBias, GroupCarry, Carry, Final, sh, group, block,
                ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
                IDENTITY: tl.constexpr, WRITE_FINAL: tl.constexpr,
                GROUP: tl.constexpr, BLOCK: tl.constexpr):
    x = block * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(GroupCarry + (sh * GROUPS + group) * ELEMENTS + x,
                    x < ELEMENTS, 0.0)
    for step in range(GROUP):
        chunk = group * GROUP + step
        address = (sh * CHUNKS + chunk) * ELEMENTS + x
        scale = tl.full((BLOCK,), 1.0, tl.float32)
        if not IDENTITY:
            scale = tl.load(Scale + address, (chunk < CHUNKS) & (x < ELEMENTS), 1.0)
        bias = tl.load(LocalBias + address, (chunk < CHUNKS) & (x < ELEMENTS), 0.0)
        tl.store(Carry + address, state, (chunk < CHUNKS) & (x < ELEMENTS))
        state = scale * state + bias
    if WRITE_FINAL:
        if group == GROUPS - 1:
            tl.store(Final + sh * ELEMENTS + x, state, x < ELEMENTS)


@triton.jit
def _apply(Scale, LocalBias, GroupCarry, Carry, Final,
           ELEMENTS: tl.constexpr, CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
           IDENTITY: tl.constexpr, WRITE_FINAL: tl.constexpr,
           GROUP: tl.constexpr, BLOCK: tl.constexpr):
    _apply_body(Scale, LocalBias, GroupCarry, Carry, Final,
                tl.program_id(0), tl.program_id(1), tl.program_id(2),
                ELEMENTS, CHUNKS, GROUPS, IDENTITY, WRITE_FINAL, GROUP, BLOCK)


@triton.jit
def _identity_local_pair(Bias, DenBias, GroupBias, DenGroupBias,
                         ELEMENTS: tl.constexpr, DEN_ELEMENTS: tl.constexpr,
                         CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
                         BLOCKS: tl.constexpr):
    sh, group, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if block < BLOCKS:
        _local_body(None, Bias, None, GroupBias, sh, group, block,
                    ELEMENTS, CHUNKS, GROUPS, True, 32, 256)
    else:
        _local_body(None, DenBias, None, DenGroupBias, sh, group, block - BLOCKS,
                    DEN_ELEMENTS, CHUNKS, GROUPS, True, 32, 256)


@triton.jit
def _identity_apply_pair(Bias, DenBias, GroupCarry, DenGroupCarry,
                         Carry, DenCarry, Final, DenFinal,
                         ELEMENTS: tl.constexpr, DEN_ELEMENTS: tl.constexpr,
                         CHUNKS: tl.constexpr, GROUPS: tl.constexpr,
                         BLOCKS: tl.constexpr, WRITE_FINAL: tl.constexpr):
    sh, group, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if block < BLOCKS:
        _apply_body(None, Bias, GroupCarry, Carry, Final, sh, group, block,
                    ELEMENTS, CHUNKS, GROUPS, True, WRITE_FINAL, 32, 256)
    else:
        _apply_body(None, DenBias, DenGroupCarry, DenCarry, DenFinal,
                    sh, group, block - BLOCKS,
                    DEN_ELEMENTS, CHUNKS, GROUPS, True, WRITE_FINAL, 32, 256)


def hierarchical_carry(scale, bias, initial, carry, final, *, identity, write_final):
    """Prefix chunks in groups; recursive summaries bound the serial critical path.

    No division or inversion is used, including zero diagonal factors. The
    original summaries are immutable, so recursion needs only group buffers.
    """
    from .diagonal_kernels import _carry

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


def _hierarchical_carry_pair(bias, den_bias, initial, den_initial, final, den_final,
                             *, write_final):
    """Pair each scan level; direct prefixes reuse summaries, long apply does not."""
    from .diagonal_kernels import _identity_carry_pair

    bh, chunks, rows, columns = bias.shape
    elements, den_elements = rows * columns, rows
    blocks = triton.cdiv(elements, 256)
    grid_blocks = blocks + triton.cdiv(den_elements, 256)
    if chunks <= 64:
        _identity_carry_pair[(bh, grid_blocks)](
            bias, den_bias, initial, den_initial, final, den_final,
            rows, columns, chunks, blocks, write_final,
        )
        return bias, den_bias
    groups = triton.cdiv(chunks, 32)
    group_bias = torch.empty((bh, groups, rows, columns),
                            device=bias.device, dtype=torch.float32)
    den_group_bias = torch.empty((bh, groups, rows, 1),
                                device=den_bias.device, dtype=torch.float32)
    # Group summaries consume original biases before direct recursion reuses
    # them. Long apply uses separate destinations to avoid serial alias stalls.
    _identity_local_pair[(bh, groups, grid_blocks)](
        bias, den_bias, group_bias, den_group_bias,
        elements, den_elements, chunks, groups, blocks,
    )
    group_carry, den_group_carry = _hierarchical_carry_pair(
        group_bias, den_group_bias, initial, den_initial,
        final, den_final, write_final=False,
    )
    carry = torch.empty_like(bias)
    den_carry = torch.empty_like(den_bias)
    _identity_apply_pair[(bh, groups, grid_blocks)](
        bias, den_bias, group_carry, den_group_carry,
        carry, den_carry, final, den_final,
        elements, den_elements, chunks, groups, blocks, write_final,
    )
    return carry, den_carry
