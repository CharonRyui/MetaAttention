"""Dense diagonal recurrence kernels; only chunk-boundary States reach memory.

Identity propagation uses chunk GEMMs. General diagonal propagation uses
inverse-free parallel forward products and weighted block GEMM contractions.
All tensor layouts are token-major; sequence length is a static dense contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..readout.native import _load_dense

CHUNK = tl.constexpr(64)


@triton.jit
def _identity_summary_body(
    K, K2, V, Bias, sh, chunk, tile,
    LENGTH: tl.constexpr, HEADS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    KR: tl.constexpr, KL: tl.constexpr,
    NATIVE: tl.constexpr, KS: tl.constexpr, KG: tl.constexpr,
    VS: tl.constexpr, VG: tl.constexpr, UNIT_VALUE: tl.constexpr,
    Initial=None,
):
    seq, head = sh // HEADS, sh % HEADS
    r = tile // tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    c = tile % tl.cdiv(COLS, 32) * 32 + tl.arange(0, 32)
    t = chunk * CHUNK + tl.arange(0, CHUNK)
    if NATIVE:
        k = _load_dense(K, seq, head, t[None, :], r[:, None], ROWS, KS, KG,
                        t[None, :] < LENGTH)
    elif KR:
        k = tl.load(
            K + ((seq * LENGTH + t[None, :]) * HEADS + head) * KL + r[:, None] // KR,
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
        k *= tl.load(
            K2 + ((seq * LENGTH + t[None, :]) * HEADS + head) * KR + r[:, None] % KR,
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
    else:
        k = tl.load(
            K + ((seq * LENGTH + t[None, :]) * HEADS + head) * ROWS + r[:, None],
            (t[None, :] < LENGTH) & (r[:, None] < ROWS),
            0,
        )
    if NATIVE:
        if UNIT_VALUE:
            v = tl.where((t[:, None] < LENGTH) & (c[None, :] < COLS), 1.0, 0.0)
        else:
            v = _load_dense(V, seq, head, t[:, None], c[None, :], COLS, VS, VG,
                            t[:, None] < LENGTH)
    else:
        v = tl.load(
            V + ((seq * LENGTH + t[:, None]) * HEADS + head) * COLS + c[None, :],
            (t[:, None] < LENGTH) & (c[None, :] < COLS),
            0,
        )
    bias = tl.dot(k, v, input_precision="tf32x3")
    if Initial is not None:
        tl.static_assert(CHUNKS == 1)
        # Match the direct prefix's separate FP32 initial + summary addition.
        initial = tl.load(
            Initial + (sh * ROWS + r[:, None]) * COLS + c[None, :],
            (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0,
        )
        bias = initial + bias
    tl.store(
        Bias + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
        bias,
        (r[:, None] < ROWS) & (c[None, :] < COLS),
    )


@triton.jit
def _identity_pairs(
    K, K2, V, Bias,
    LENGTH: tl.constexpr, HEADS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    KR: tl.constexpr, KL: tl.constexpr,
):
    _identity_summary_body(
        K, K2, V, Bias, tl.program_id(0), tl.program_id(1), tl.program_id(2),
        LENGTH, HEADS, ROWS, COLS, CHUNKS, KR, KL,
        False, (), 1, (), 1, False,
    )


@triton.jit
def _identity_pairs_native_pair(
    K, DenKey, V, Bias, DenBias,
    LENGTH: tl.constexpr, HEADS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    TILES: tl.constexpr, KS: tl.constexpr, KG: tl.constexpr,
    DKS: tl.constexpr, DKG: tl.constexpr, VS: tl.constexpr, VG: tl.constexpr,
):
    # Each CTA holds only one State's original 32x32 summary tile.
    sh, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if tile < TILES:
        _identity_summary_body(
            K, K, V, Bias, sh, chunk, tile,
            LENGTH, HEADS, ROWS, COLS, CHUNKS, 0, ROWS,
            True, KS, KG, VS, VG, False,
        )
    else:
        _identity_summary_body(
            DenKey, DenKey, V, DenBias, sh, chunk, tile - TILES,
            LENGTH, HEADS, ROWS, 1, CHUNKS, 0, ROWS,
            True, DKS, DKG, VS, VG, True,
        )


@triton.jit
def _identity_final_native_pair(
    K, DenKey, V, Initial, DenInitial, Final, DenFinal,
    LENGTH: tl.constexpr, HEADS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, TILES: tl.constexpr,
    KS: tl.constexpr, KG: tl.constexpr, DKS: tl.constexpr, DKG: tl.constexpr,
    VS: tl.constexpr, VG: tl.constexpr,
):
    sh, tile = tl.program_id(0), tl.program_id(1)
    if tile < TILES:
        _identity_summary_body(
            K, K, V, Final, sh, 0, tile,
            LENGTH, HEADS, ROWS, COLS, 1, 0, ROWS,
            True, KS, KG, VS, VG, False, Initial,
        )
    else:
        _identity_summary_body(
            DenKey, DenKey, V, DenFinal, sh, 0, tile - TILES,
            LENGTH, HEADS, ROWS, 1, 1, 0, ROWS,
            True, DKS, DKG, VS, VG, True, DenInitial,
        )


@triton.jit
def _carry_body(
    Scale, Bias, Initial, Carry, Final, sh, block,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr, WRITE_FINAL: tl.constexpr, INPLACE: tl.constexpr = False,
):
    x = block * 256 + tl.arange(0, 256)
    state = tl.load(Initial + sh * ROWS * COLS + x, x < ROWS * COLS, 0)
    # Paired direct prefixes have at most 64 chunks. Expose every disjoint
    # address so in-place stores cannot serialize later summary loads.
    for chunk in tl.range(CHUNKS, loop_unroll_factor=CHUNKS if INPLACE else 1):
        base = (sh * CHUNKS + chunk) * ROWS * COLS + x
        # Load the current summary before replacing it with its exclusive carry.
        bias = tl.load(Bias + base, x < ROWS * COLS, 0)
        tl.store(Carry + base, state, x < ROWS * COLS)
        if IDENTITY:
            state += bias
        else:
            scale = tl.load(Scale + base, x < ROWS * COLS, 1)
            state = state * scale + bias
    if WRITE_FINAL:
        tl.store(Final + sh * ROWS * COLS + x, state, x < ROWS * COLS)


@triton.jit
def _carry(
    Scale, Bias, Initial, Carry, Final,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    IDENTITY: tl.constexpr, WRITE_FINAL: tl.constexpr,
):
    _carry_body(
        Scale, Bias, Initial, Carry, Final, tl.program_id(0), tl.program_id(1),
        ROWS, COLS, CHUNKS, IDENTITY, WRITE_FINAL,
    )


@triton.jit
def _identity_carry_pair(
    Bias, DenBias, Initial, DenInitial, Final, DenFinal,
    ROWS: tl.constexpr, COLS: tl.constexpr, CHUNKS: tl.constexpr,
    BLOCKS: tl.constexpr, WRITE_FINAL: tl.constexpr,
):
    sh, block = tl.program_id(0), tl.program_id(1)
    if block < BLOCKS:
        _carry_body(None, Bias, Initial, Bias, Final, sh, block,
                    ROWS, COLS, CHUNKS, True, WRITE_FINAL, True)
    else:
        _carry_body(None, DenBias, DenInitial, DenBias, DenFinal, sh, block - BLOCKS,
                    ROWS, 1, CHUNKS, True, WRITE_FINAL, True)


@triton.jit
def _identity_readout(
    K,
    K2,
    V,
    Q,
    Q2,
    Carry,
    Output,
    LENGTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    PRIOR: tl.constexpr,
    KR: tl.constexpr,
    KL: tl.constexpr,
    QR: tl.constexpr,
    QL: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    t = chunk * CHUNK + tl.arange(0, CHUNK)
    src = chunk * CHUNK + tl.arange(0, CHUNK)
    c = block * 32 + tl.arange(0, 32)
    rr = tl.arange(0, 32)
    scores = tl.full((CHUNK, CHUNK), 0.0, tl.float32)
    out = tl.full((CHUNK, 32), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + rr
        if QR:
            q = tl.load(
                Q
                + ((seq * LENGTH + t[:, None]) * HEADS + head) * QL
                + r[None, :] // QR,
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
            q *= tl.load(
                Q2
                + ((seq * LENGTH + t[:, None]) * HEADS + head) * QR
                + r[None, :] % QR,
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
        else:
            q = tl.load(
                Q + ((seq * LENGTH + t[:, None]) * HEADS + head) * ROWS + r[None, :],
                (t[:, None] < LENGTH) & (r[None, :] < ROWS),
                0,
            )
        if KR:
            k = tl.load(
                K
                + ((seq * LENGTH + src[None, :]) * HEADS + head) * KL
                + r[:, None] // KR,
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
            k *= tl.load(
                K2
                + ((seq * LENGTH + src[None, :]) * HEADS + head) * KR
                + r[:, None] % KR,
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
        else:
            k = tl.load(
                K + ((seq * LENGTH + src[None, :]) * HEADS + head) * ROWS + r[:, None],
                (src[None, :] < LENGTH) & (r[:, None] < ROWS),
                0,
            )
        state = tl.load(
            Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
            (r[:, None] < ROWS) & (c[None, :] < COLS),
            0,
        )
        scores += tl.dot(q, k, input_precision="tf32x3")
        out += tl.dot(q, state, input_precision="tf32x3")
    causal = src[None, :] < t[:, None] if PRIOR else src[None, :] <= t[:, None]
    scores = tl.where(causal, scores, 0.0)
    v = tl.load(
        V + ((seq * LENGTH + src[:, None]) * HEADS + head) * COLS + c[None, :],
        (src[:, None] < LENGTH) & (c[None, :] < COLS),
        0,
    )
    out += tl.dot(scores, v, input_precision="tf32x3")
    tl.store(
        Output + ((seq * LENGTH + t[:, None]) * HEADS + head) * COLS + c[None, :],
        out,
        (t[:, None] < LENGTH) & (c[None, :] < COLS),
    )



def prepare_state(
    *,
    rows_gate,
    columns_gate,
    key,
    value,
    initial,
    sequence_length,
    return_final_state,
    identity=False,
):
    """Return reusable FP32 carries, final State, and normalized factor bindings."""
    batch, heads, rows, columns = initial.shape
    initial = initial.contiguous()
    chunks = triton.cdiv(sequence_length, CHUNK)
    if isinstance(key, tuple):
        key, key2 = (part.float().contiguous() for part in key)
        kl, kr = key.shape[-1], key2.shape[-1]
    else:
        key = key.reshape(batch * sequence_length, heads, rows).float().contiguous()
        key2, kl, kr = key, rows, 0
    value = value.reshape(batch * sequence_length, heads, columns).float().contiguous()
    tokens = batch * sequence_length
    rows_gate = (
        None if rows_gate is None
        else rows_gate.reshape(tokens, heads, -1).expand(tokens, heads, rows).float().contiguous()
    )
    columns_gate = (
        None if columns_gate is None
        else columns_gate.reshape(tokens, heads, -1).expand(tokens, heads, columns).float().contiguous()
    )
    bias = torch.empty(
        batch * heads, chunks, rows, columns, device=initial.device, dtype=torch.float32
    )
    scale = None if identity else torch.empty_like(bias)
    carry = torch.empty_like(bias)
    final = torch.empty_like(initial) if return_final_state else initial
    if identity:
        _identity_pairs[
            (batch * heads, chunks, triton.cdiv(rows, 32) * triton.cdiv(columns, 32))
        ](key, key2, value, bias, sequence_length, heads, rows, columns, chunks, kr, kl)
    else:
        from .diagonal_block import diagonal_block_summary

        diagonal_block_summary(
            rows_gate, columns_gate, key, key2, value, scale, bias,
            sequence_length=sequence_length, heads=heads, rows=rows,
            columns=columns, chunks=chunks, kl=kl, kr=kr,
        )
    from .diagonal_scan import hierarchical_carry

    hierarchical_carry(
        scale, bias, initial, carry, final,
        identity=identity, write_final=return_final_state,
    )
    return carry, final, (rows_gate, columns_gate, key, key2, value, kl, kr)


def read_state(carry, factors, query, *, sequence_length, prior=False, identity=False, product_scores=None):
    batch_heads, chunks, rows, columns = carry.shape
    rows_gate, columns_gate, key, key2, value, kl, kr = factors
    heads = key.shape[1]
    if isinstance(query, tuple):
        query, query2 = (part.float().contiguous() for part in query)
        ql, qr = query.shape[-1], query2.shape[-1]
    else:
        query = query.reshape(key.shape[0], heads, rows).float().contiguous()
        query2, ql, qr = query, rows, 0
    if identity and kr and qr and kl == ql and kr == qr:
        from ..readout.product_kernels import product_chunk_scores, read_product_state

        scores = None
        if product_scores is not None:
            cache_key = (
                key.data_ptr(), key2.data_ptr(), query.data_ptr(), query2.data_ptr(),
                kl, kr, sequence_length, prior,
            )
            scores = product_scores.get(cache_key)
            if scores is None:
                scores = product_chunk_scores(
                    (key, key2), (query, query2),
                    sequence_length=sequence_length, prior=prior,
                )
                product_scores[cache_key] = scores
        return read_product_state(
            carry, (key, key2), (query, query2), value,
            sequence_length=sequence_length, prior=prior, scores=scores,
        )
    output = torch.empty(
        key.shape[0], heads, columns, device=key.device, dtype=torch.float32
    )
    if identity:
        _identity_readout[(batch_heads, chunks, triton.cdiv(columns, 32))](
            key,
            key2,
            value,
            query,
            query2,
            carry,
            output,
            sequence_length,
            heads,
            rows,
            columns,
            chunks,
            prior,
            kr,
            kl,
            qr,
            ql,
        )
    else:
        from .diagonal_block import diagonal_block_readout

        diagonal_block_readout(
            carry, factors, query, query2, output,
            sequence_length=sequence_length, prior=prior, ql=ql, qr=qr,
        )
    return output
