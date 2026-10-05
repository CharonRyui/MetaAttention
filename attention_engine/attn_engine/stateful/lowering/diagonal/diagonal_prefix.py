"""Inverse-free hierarchical carries for separated row/column summaries."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scale(Row, Column, sh, chunk, r, c, ROWS: tl.constexpr, COLS: tl.constexpr,
           SUMMARIES: tl.constexpr):
    row = tl.load(Row + (sh * SUMMARIES + chunk) * ROWS + r, r < ROWS, 1.0)
    column = tl.load(Column + (sh * SUMMARIES + chunk) * COLS + c, c < COLS, 1.0)
    return row * column


@triton.jit
def _direct(Row, Column, Bias, Initial, Carry, Final,
            ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
            SUMMARIES: tl.constexpr, WRITE_FINAL: tl.constexpr):
    sh, tile = tl.program_id(0), tl.program_id(1)
    x = tile * 256 + tl.arange(0, 256)
    r, c = x // COLS, x % COLS
    state = tl.load(Initial + sh * ROWS * COLS + x, x < ROWS * COLS, 0.0)
    for chunk in range(CHUNKS):
        tl.store(Carry + (sh * CHUNKS + chunk) * ROWS * COLS + x, state, x < ROWS * COLS)
        if chunk < SUMMARIES:
            scale = _scale(Row, Column, sh, chunk, r, c, ROWS, COLS, SUMMARIES)
            bias = tl.load(Bias + (sh * SUMMARIES + chunk) * ROWS * COLS + x,
                           x < ROWS * COLS, 0.0)
            state = state * scale + bias
    if WRITE_FINAL:
        tl.store(Final + sh * ROWS * COLS + x, state, x < ROWS * COLS)


@triton.jit
def _groups(Row, Column, Bias, GroupRow, GroupColumn, GroupBias,
            ROWS: tl.constexpr, COLS: tl.constexpr, SUMMARIES: tl.constexpr,
            GROUPS: tl.constexpr):
    sh, group, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    x = tile * 256 + tl.arange(0, 256)
    r, c = x // COLS, x % COLS
    ar = tl.full((256,), 1.0, tl.float32)
    ac = tl.full((256,), 1.0, tl.float32)
    b = tl.full((256,), 0.0, tl.float32)
    for step in range(32):
        chunk = group * 32 + step
        rg = tl.load(Row + (sh * SUMMARIES + chunk) * ROWS + r,
                     (chunk < SUMMARIES) & (r < ROWS), 1.0)
        cg = tl.load(Column + (sh * SUMMARIES + chunk) * COLS + c,
                     (chunk < SUMMARIES) & (c < COLS), 1.0)
        bias = tl.load(Bias + (sh * SUMMARIES + chunk) * ROWS * COLS + x,
                       (chunk < SUMMARIES) & (x < ROWS * COLS), 0.0)
        b = b * (rg * cg) + bias
        ar *= rg
        ac *= cg
    tl.store(GroupBias + (sh * GROUPS + group) * ROWS * COLS + x, b, x < ROWS * COLS)
    tl.store(GroupRow + (sh * GROUPS + group) * ROWS + r, ar, (r < ROWS) & (c == 0))
    tl.store(GroupColumn + (sh * GROUPS + group) * COLS + c, ac, (r == 0) & (c < COLS))


@triton.jit
def _apply(Row, Column, Bias, GroupCarry, Carry, Final,
           ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
           SUMMARIES: tl.constexpr, GROUPS: tl.constexpr, WRITE_FINAL: tl.constexpr):
    sh, group, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    x = tile * 256 + tl.arange(0, 256)
    r, c = x // COLS, x % COLS
    state = tl.load(GroupCarry + (sh * GROUPS + group) * ROWS * COLS + x,
                    x < ROWS * COLS, 0.0)
    for step in range(32):
        chunk = group * 32 + step
        tl.store(Carry + (sh * CHUNKS + chunk) * ROWS * COLS + x, state,
                 (chunk < CHUNKS) & (x < ROWS * COLS))
        if chunk < SUMMARIES:
            scale = _scale(Row, Column, sh, chunk, r, c, ROWS, COLS, SUMMARIES)
            bias = tl.load(Bias + (sh * SUMMARIES + chunk) * ROWS * COLS + x,
                           x < ROWS * COLS, 0.0)
            state = state * scale + bias
    if WRITE_FINAL:
        if group == GROUPS - 1:
            tl.store(Final + sh * ROWS * COLS + x, state, x < ROWS * COLS)


def compact_carry(row, column, bias, initial, carry, final, *, write_final):
    bh, chunks, rows, columns = carry.shape
    summaries = bias.shape[1]
    if chunks <= 64:
        _direct[(bh, triton.cdiv(rows * columns, 256))](
            row, column, bias, initial, carry, final, rows, columns, chunks,
            summaries, write_final, num_warps=4, enable_fp_fusion=False,
        )
        return
    groups = triton.cdiv(chunks, 32)
    group_row = torch.empty((bh, groups, rows), device=initial.device, dtype=torch.float32)
    group_column = torch.empty((bh, groups, columns), device=initial.device, dtype=torch.float32)
    group_bias = torch.empty((bh, groups, rows, columns), device=initial.device, dtype=torch.float32)
    group_carry = torch.empty_like(group_bias)
    _groups[(bh, groups, triton.cdiv(rows * columns, 256))](
        row, column, bias, group_row, group_column, group_bias,
        rows, columns, summaries, groups, num_warps=4, enable_fp_fusion=False,
    )
    compact_carry(group_row, group_column, group_bias, initial, group_carry, final,
                  write_final=False)
    _apply[(bh, groups, triton.cdiv(rows * columns, 256))](
        row, column, bias, group_carry, carry, final, rows, columns, chunks,
        summaries, groups, write_final, num_warps=4, enable_fp_fusion=False,
    )
