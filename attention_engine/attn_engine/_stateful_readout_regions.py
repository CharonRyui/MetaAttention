"""Resource-bounded fused readout kernels over chunk-boundary States.

Only chunk carries and final outputs are global tensors. Softmax reduces first
State columns (second State rows), never tokens. Unsupported resources return
None for the typed DAG executor's existing materialization cut.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector(P, P2, flat, x, SIZE: tl.constexpr, LEFT: tl.constexpr,
            RIGHT: tl.constexpr, VALID):
    if RIGHT:
        result = tl.load(P + flat * LEFT + x // RIGHT, VALID & (x < SIZE), 0.0)
        result *= tl.load(P2 + flat * RIGHT + x % RIGHT, VALID & (x < SIZE), 0.0)
    else:
        result = tl.load(P + flat * SIZE + x, VALID & (x < SIZE), 0.0)
    return result


@triton.jit
def _advance(S, GR, GC, K, K2, V, flat, r, c,
             ROWS: tl.constexpr, COLS: tl.constexpr,
             KL: tl.constexpr, KR: tl.constexpr,
             UNIT_ROWS: tl.constexpr, UNIT_COLS: tl.constexpr, VALID):
    rg = tl.full(r.shape, 1.0, tl.float32)
    cg = tl.full(c.shape, 1.0, tl.float32)
    if not UNIT_ROWS:
        rg = tl.load(GR + flat * ROWS + r, VALID & (r < ROWS), 1.0)
    if not UNIT_COLS:
        cg = tl.load(GC + flat * COLS + c, VALID & (c < COLS), 1.0)
    k = _vector(K, K2, flat, r, ROWS, KL, KR, VALID)
    v = tl.load(V + flat * COLS + c, VALID & (c < COLS), 0.0)
    return S * rg[:, None] * cg[None, :] + k[:, None] * v[None, :]


@triton.jit
def _two_state_region(
    C1, GR1, GC1, K1, K12, V1, C2, GR2, GC2, K2, K22, V2,
    Q, Q2, Output,
    LENGTH: tl.constexpr, HEADS: tl.constexpr, CHUNKS: tl.constexpr,
    R1: tl.constexpr, D1: tl.constexpr, R2: tl.constexpr, D2: tl.constexpr,
    BR1: tl.constexpr, BD1: tl.constexpr, BR2: tl.constexpr, BD2: tl.constexpr,
    KL1: tl.constexpr, KR1: tl.constexpr, KL2: tl.constexpr, KR2: tl.constexpr,
    QL: tl.constexpr, QR: tl.constexpr,
    UR1: tl.constexpr, UC1: tl.constexpr, UR2: tl.constexpr, UC2: tl.constexpr,
    PRIOR1: tl.constexpr, PRIOR2: tl.constexpr, SOFTMAX: tl.constexpr,
    EPSILON: tl.constexpr,
):
    sh, chunk = tl.program_id(0), tl.program_id(1)
    seq, head = sh // HEADS, sh % HEADS
    r1, d1 = tl.arange(0, BR1), tl.arange(0, BD1)
    r2, d2 = tl.arange(0, BR2), tl.arange(0, BD2)
    s1 = tl.load(C1 + ((sh * CHUNKS + chunk) * R1 + r1[:, None]) * D1 + d1[None, :],
                 (r1[:, None] < R1) & (d1[None, :] < D1), 0.0)
    s2 = tl.load(C2 + ((sh * CHUNKS + chunk) * R2 + r2[:, None]) * D2 + d2[None, :],
                 (r2[:, None] < R2) & (d2[None, :] < D2), 0.0)
    for step in range(64):
        t = chunk * 64 + step
        flat = (seq * LENGTH + t) * HEADS + head
        q = _vector(Q, Q2, flat, r1, R1, QL, QR, t < LENGTH)
        if PRIOR1:
            first = tl.sum(s1 * q[:, None], 0)
        s1 = _advance(s1, GR1, GC1, K1, K12, V1, flat, r1, d1,
                      R1, D1, KL1, KR1, UR1, UC1, t < LENGTH)
        if not PRIOR1:
            first = tl.sum(s1 * q[:, None], 0)
        if SOFTMAX:
            logits = tl.where(d1 < D1, first, float("-inf"))
            weight = tl.exp(logits - tl.max(logits, 0))
            weight = weight / tl.sum(weight, 0)
            if PRIOR2:
                out = tl.sum(s2 * weight[:, None], 0)
            s2 = _advance(s2, GR2, GC2, K2, K22, V2, flat, r2, d2,
                          R2, D2, KL2, KR2, UR2, UC2, t < LENGTH)
            if not PRIOR2:
                out = tl.sum(s2 * weight[:, None], 0)
        else:
            if PRIOR2:
                denominator = tl.sum(s2 * q[:, None], 0)
            s2 = _advance(s2, GR2, GC2, K2, K22, V2, flat, r2, d2,
                          R2, D2, KL2, KR2, UR2, UC2, t < LENGTH)
            if not PRIOR2:
                denominator = tl.sum(s2 * q[:, None], 0)
            out = first / (tl.sum(denominator, 0) + EPSILON)
        if SOFTMAX:
            tl.store(Output + flat * D2 + d2, out, (t < LENGTH) & (d2 < D2))
        else:
            tl.store(Output + flat * D1 + d1, out, (t < LENGTH) & (d1 < D1))


def _query_parts(query, tokens, heads, rows):
    if isinstance(query, tuple):
        q, q2 = (part.reshape(tokens, heads, -1).float().contiguous() for part in query)
        if q.shape[-1] * q2.shape[-1] != rows:
            return None
        return q, q2, q.shape[-1], q2.shape[-1]
    q = query.reshape(tokens, heads, rows).float().contiguous()
    return q, q, rows, 0


def readout_region_resources(first_shape, second_shape, *, softmax=True):
    """Conservative live FP32 elements/CTA, capped at 16384 with eight warps.

    Counts both States, one live update temporary, gate/key/value/query vectors,
    contraction and softmax scratch. Padded dimensions must be at most 256.
    This is a structural bound, not a measured compiler register count.
    """
    _, _, r1, d1 = first_shape
    _, _, r2, d2 = second_shape
    a, b, c, d = map(triton.next_power_of_2, (r1, d1, r2, d2))
    live = a * b + c * d + max(a * b, c * d) + 4 * (a + b + c + d) + (3 * b if softmax else b)
    return {"live_fp32_elements": live, "live_fp32_bytes": 4 * live,
            "eligible": max(a, b, c, d) <= 256 and live <= 16384}


@triton.jit
def _identity_normalized_region(
    C1, K1, K12, V1, C2, K2, K22, V2, Q, Q2, Output,
    LENGTH: tl.constexpr, HEADS: tl.constexpr, CHUNKS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    KL1: tl.constexpr, KR1: tl.constexpr, KL2: tl.constexpr, KR2: tl.constexpr,
    QL: tl.constexpr, QR: tl.constexpr,
    PRIOR1: tl.constexpr, PRIOR2: tl.constexpr,
    SHARED_KEY: tl.constexpr, EPSILON: tl.constexpr,
):
    sh, chunk, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, head = sh // HEADS, sh % HEADS
    t = chunk * 64 + tl.arange(0, 64)
    src = chunk * 64 + tl.arange(0, 64)
    flat = (seq * LENGTH + t) * HEADS + head
    src_flat = (seq * LENGTH + src) * HEADS + head
    c = block * 32 + tl.arange(0, 32)
    rr = tl.arange(0, 32)
    scores = tl.full((64, 64), 0.0, tl.float32)
    if not SHARED_KEY:
        den_scores = tl.full((64, 64), 0.0, tl.float32)
    numerator = tl.full((64, 32), 0.0, tl.float32)
    denominator = tl.full((64,), 0.0, tl.float32)
    for rb in range(tl.cdiv(ROWS, 32)):
        r = rb * 32 + rr
        q = _vector(Q, Q2, flat[:, None], r[None, :], ROWS, QL, QR,
                    t[:, None] < LENGTH)
        k = _vector(K1, K12, src_flat[None, :], r[:, None], ROWS, KL1, KR1,
                    src[None, :] < LENGTH)
        memory = tl.load(
            C1 + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
            (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0,
        )
        key_sum = tl.load(C2 + (sh * CHUNKS + chunk) * ROWS + r, r < ROWS, 0.0)
        scores += tl.dot(q, k, input_precision="tf32x3")
        numerator += tl.dot(q, memory, input_precision="tf32x3")
        denominator += tl.sum(q * key_sum[None, :], 1)
        if not SHARED_KEY:
            k2 = _vector(K2, K22, src_flat[None, :], r[:, None], ROWS, KL2, KR2,
                         src[None, :] < LENGTH)
            den_scores += tl.dot(q, k2, input_precision="tf32x3")
    if SHARED_KEY:
        den_scores = scores
    causal1 = src[None, :] < t[:, None] if PRIOR1 else src[None, :] <= t[:, None]
    causal2 = src[None, :] < t[:, None] if PRIOR2 else src[None, :] <= t[:, None]
    scores = tl.where(causal1, scores, 0.0)
    den_scores = tl.where(causal2, den_scores, 0.0)
    value = tl.load(V1 + src_flat[:, None] * COLS + c[None, :],
                    (src[:, None] < LENGTH) & (c[None, :] < COLS), 0.0)
    den_value = tl.load(V2 + src_flat, src < LENGTH, 0.0)
    numerator += tl.dot(scores, value, input_precision="tf32x3")
    denominator += tl.sum(den_scores * den_value[None, :], 1)
    output = numerator / (denominator[:, None] + EPSILON)
    tl.store(Output + flat[:, None] * COLS + c[None, :], output,
             (t[:, None] < LENGTH) & (c[None, :] < COLS))


def _region(first, second, query, sequence_length, prior_first, prior_second,
            output_dtype, *, softmax, epsilon=0.0):
    c1, f1, _ = first
    c2, f2, _ = second
    if sequence_length <= 0:
        return None
    if c1.shape[:2] != c2.shape[:2]:
        return None
    bh, chunks, r1, d1 = c1.shape
    _, _, r2, d2 = c2.shape
    if (d1 != r2 if softmax else r1 != r2 or d2 != 1):
        return None
    if not readout_region_resources(c1.shape, c2.shape, softmax=softmax)["eligible"]:
        return None
    gr1, gc1, k1, k12, v1, kl1, kr1 = f1
    gr2, gc2, k2, k22, v2, kl2, kr2 = f2
    tokens, heads = k1.shape[:2]
    if k2.shape[:2] != (tokens, heads) or bh != tokens // sequence_length * heads:
        return None
    if tokens % sequence_length or chunks != triton.cdiv(sequence_length, 64):
        return None
    parts = _query_parts(query, tokens, heads, r1)
    if parts is None:
        return None
    q, q2, ql, qr = parts
    output = torch.empty(tokens, heads, d2 if softmax else d1, device=c1.device, dtype=output_dtype)
    if not softmax and first[2] and second[2]:
        shared_key = (k1.data_ptr() == k2.data_ptr() and k12.data_ptr() == k22.data_ptr()
                      and (kl1, kr1) == (kl2, kr2))
        _identity_normalized_region[(bh, chunks, triton.cdiv(d1, 32))](
            c1, k1, k12, v1, c2, k2, k22, v2, q, q2, output,
            sequence_length, heads, chunks, r1, d1,
            kl1, kr1, kl2, kr2, ql, qr,
            prior_first, prior_second, shared_key, epsilon,
            num_warps=4, enable_fp_fusion=False,
        )
        return output
    _two_state_region[(bh, chunks)](
        c1, gr1, gc1, k1, k12, v1, c2, gr2, gc2, k2, k22, v2,
        q, q2, output, sequence_length, heads, chunks, r1, d1, r2, d2,
        triton.next_power_of_2(r1), triton.next_power_of_2(d1),
        triton.next_power_of_2(r2), triton.next_power_of_2(d2),
        kl1, kr1, kl2, kr2, ql, qr,
        gr1 is None, gc1 is None, gr2 is None, gc2 is None,
        prior_first, prior_second, softmax, epsilon, num_warps=8, enable_fp_fusion=False,
    )
    return output


def fused_readout_region(first, second, query, *, sequence_length,
                         prior_first=False, prior_second=False, output_dtype=torch.float32):
    """Fuse ``(query @ first).softmax(columns) @ second`` or return None.

    Prepared inputs are ``(carry, factors, identity)``. A successful call
    launches one kernel and allocates only tokens*heads*second.columns*dtype
    bytes, plus FP32 query conversion if needed. No softmax/State sequence is
    stored. Prior flags select each State independently before its update.
    """
    return _region(first, second, query, sequence_length, prior_first, prior_second,
                   output_dtype, softmax=True)


def fused_normalized_region(first, second, query, *, sequence_length,
                            prior_first=False, prior_second=False,
                            output_dtype=torch.float32, epsilon=0.0):
    """Fuse numerator/denominator with a scalar-column denominator.

    ``epsilon`` is an authored FP32 epilogue constant.  The default preserves
    the historical bare-division helper; normalized-linear supplies its
    expression's ``EPS`` explicitly.
    """
    return _region(first, second, query, sequence_length, prior_first, prior_second,
                   output_dtype, softmax=False, epsilon=epsilon)


@triton.jit
def _shared_region(Carry, GR, GC, K, K2, V, Queries, Queries2, Outputs,
                   LENGTH: tl.constexpr, HEADS: tl.constexpr, CHUNKS: tl.constexpr,
                   ROWS: tl.constexpr, COLS: tl.constexpr,
                   BR: tl.constexpr, BC: tl.constexpr,
                   KL: tl.constexpr, KR: tl.constexpr,
                   QLS: tl.constexpr, QRS: tl.constexpr, PRIORS: tl.constexpr,
                   UNIT_ROWS: tl.constexpr, UNIT_COLS: tl.constexpr,
                   COUNT: tl.constexpr, EPILOGUE: tl.constexpr):
    sh, chunk = tl.program_id(0), tl.program_id(1)
    seq, head = sh // HEADS, sh % HEADS
    r, c = tl.arange(0, BR), tl.arange(0, BC)
    state = tl.load(Carry + ((sh * CHUNKS + chunk) * ROWS + r[:, None]) * COLS + c[None, :],
                    (r[:, None] < ROWS) & (c[None, :] < COLS), 0.0)
    for step in range(64):
        t = chunk * 64 + step
        flat = (seq * LENGTH + t) * HEADS + head
        prior_values = ()
        for i in tl.static_range(COUNT):
            q = _vector(Queries[i], Queries2[i], flat, r, ROWS, QLS[i], QRS[i], t < LENGTH)
            if PRIORS[i]:
                value = tl.sum(state * q[:, None], 0)
            else:
                value = tl.full((BC,), 0.0, tl.float32)
            prior_values += (value,)
        state = _advance(state, GR, GC, K, K2, V, flat, r, c,
                         ROWS, COLS, KL, KR, UNIT_ROWS, UNIT_COLS, t < LENGTH)
        results = ()
        for i in tl.static_range(COUNT):
            if PRIORS[i]:
                value = prior_values[i]
            else:
                q = _vector(Queries[i], Queries2[i], flat, r, ROWS, QLS[i], QRS[i], t < LENGTH)
                value = tl.sum(state * q[:, None], 0)
            results += (value,)
        if EPILOGUE == "add":
            value = results[0] + results[1]
        elif EPILOGUE == "multiply":
            value = results[0] * results[1]
        elif EPILOGUE == "divide":
            value = results[0] / results[1]
        if EPILOGUE != "none":
            tl.store(Outputs[0] + flat * COLS + c, value, (t < LENGTH) & (c < COLS))
        else:
            for i in tl.static_range(COUNT):
                tl.store(Outputs[i] + flat * COLS + c, results[i], (t < LENGTH) & (c < COLS))


def fused_shared_readouts(prepared, queries, *, sequence_length, priors=None,
                          output_dtypes=None, epilogue=None):
    """Share one replay across 2--4 contractions, optionally fuse binary arithmetic.

    Return a tuple of outputs, or None for the existing executor fallback.
    ``epilogue`` is None, "add", "multiply", or "divide"; a binary epilogue
    takes two queries and emits one output. Output casts happen only after
    FP32 contraction/arithmetic. Independent outputs may have different dtypes.
    """
    carry, factors, _ = prepared
    bh, chunks, rows, columns = carry.shape
    count = len(queries)
    if count < 2 or count > 4:
        return None
    if epilogue not in (None, "add", "multiply", "divide") or (epilogue and count != 2):
        return None
    br, bc = triton.next_power_of_2(rows), triton.next_power_of_2(columns)
    live = 2 * br * bc + 4 * (br + bc) + count * (br + 2 * bc)
    if max(br, bc) > 256 or live > 16384:
        return None
    gr, gc, key, key2, value, kl, kr = factors
    tokens, heads = key.shape[:2]
    if sequence_length <= 0 or tokens % sequence_length:
        return None
    if bh != tokens // sequence_length * heads or chunks != triton.cdiv(sequence_length, 64):
        return None
    priors = (False,) * count if priors is None else tuple(priors)
    outputs_count = 1 if epilogue else count
    output_dtypes = ((torch.float32,) * outputs_count if output_dtypes is None
                     else tuple(output_dtypes))
    if len(priors) != count or len(output_dtypes) != outputs_count:
        return None
    parts = tuple(_query_parts(query, tokens, heads, rows) for query in queries)
    if any(part is None for part in parts):
        return None
    outputs = tuple(torch.empty(tokens, heads, columns, device=carry.device, dtype=dtype)
                    for dtype in output_dtypes)
    _shared_region[(bh, chunks)](
        carry, gr, gc, key, key2, value,
        tuple(part[0] for part in parts), tuple(part[1] for part in parts), outputs,
        sequence_length, heads, chunks, rows, columns, br, bc, kl, kr,
        tuple(part[2] for part in parts), tuple(part[3] for part in parts), priors,
        gr is None, gc is None, count, epilogue or "none",
        num_warps=8, enable_fp_fusion=False,
    )
    return outputs
