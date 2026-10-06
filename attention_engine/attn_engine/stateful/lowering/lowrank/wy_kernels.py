"""TileLang compact WY event solves for scalar/rank-one plans."""
from functools import lru_cache
import torch

_BLOCK = 32


def eligible(operation_codes, injection_count, rows, columns):
    events = operation_codes.count("rank-one") + injection_count
    return ("rank-one" in operation_codes and
            all(x in ("scale", "rank-one") for x in operation_codes) and
            1 <= events <= 4 and rows <= 128 and columns <= 128)


def chunk_tokens(event_count):
    return 32 if event_count <= 2 else 16


def _meta(x):
    return tuple(x.shape), tuple(x.stride()), str(x.dtype).removeprefix("torch.")


@lru_cache(maxsize=None)
def _compile(left_meta, right_meta, coeff_meta, inj_l_meta, inj_r_meta,
            query_meta, initial_meta, output_meta, final_meta, *, batch, heads,
            rows, columns, length, rank_count, scale_count, injection_count,
            readouts, chunk, events, event_total, device_index):
    import tilelang
    import tilelang.language as T

    chunks = (length + chunk - 1) // chunk

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Left: T.StridedTensor(*left_meta), Right: T.StridedTensor(*right_meta),
            Coeff: T.StridedTensor(*coeff_meta), InjectionLeft: T.StridedTensor(*inj_l_meta),
            InjectionRight: T.StridedTensor(*inj_r_meta), Query: T.StridedTensor(*query_meta),
            Initial: T.StridedTensor(*initial_meta), Output: T.StridedTensor(*output_meta),
            Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(batch * heads, (columns + _BLOCK - 1) // _BLOCK, threads=128) as (sh, cb):
                sequence = sh // heads
                head = sh % heads
                state = T.alloc_shared((rows, _BLOCK), "float32")
                entry = T.alloc_shared((rows, _BLOCK), "float32")
                u = T.alloc_shared((event_total, rows), "float32")
                rvec = T.alloc_shared((event_total, rows), "float32")
                inj = T.alloc_shared((event_total, _BLOCK), "float32")
                response = T.alloc_shared((event_total, _BLOCK), "float32")
                prefix = T.alloc_shared((chunk,), "float32")
                gate = T.alloc_shared((chunk,), "float32")
                intervals = T.alloc_shared((event_total, event_total), "float32")
                for r, c in T.Parallel(rows, _BLOCK):
                    state[r, c] = T.if_then_else(c + cb * _BLOCK < columns, T.cast(Initial[sequence, head, r, c + cb * _BLOCK], "float32"), 0.0)
                T.sync_threads()
                for chunk_index in T.serial(chunks):
                    for r, c in T.Parallel(rows, _BLOCK):
                        entry[r, c] = state[r, c]
                    T.sync_threads()
                    for t in T.Parallel(chunk):
                        prefix[t] = 1.0
                        gate[t] = 1.0
                    T.sync_threads()
                    for t in T.serial(chunk):
                        token = chunk_index * chunk + t
                        if token < length:
                            for s in T.serial(scale_count):
                                gate[t] = gate[t] * T.cast(Coeff[s, sequence * length + token, head], "float32")
                            prefix[t] = gate[t] if t == 0 else prefix[t - 1] * gate[t]
                        T.sync_threads()
                    for e in T.serial(event_total):
                        et = e // events
                        kind = e % events
                        token = chunk_index * chunk + et
                        valid = token < length
                        if kind < rank_count:
                            for r in T.Parallel(rows):
                                u[e, r] = T.if_then_else(valid, T.cast(Coeff[scale_count + kind, sequence * length + token, head], "float32") * T.cast(Left[kind, sequence * length + token, head, r], "float32"), 0.0)
                                rvec[e, r] = T.if_then_else(valid, T.cast(Right[kind, sequence * length + token, head, r], "float32"), 0.0)
                            for c in T.Parallel(_BLOCK):
                                inj[e, c] = 0.0
                        else:
                            j = kind - rank_count
                            for r in T.Parallel(rows):
                                u[e, r] = T.if_then_else(valid, T.cast(InjectionLeft[j, sequence * length + token, head, r], "float32"), 0.0)
                                rvec[e, r] = 0.0
                            for c in T.Parallel(_BLOCK):
                                inj[e, c] = T.if_then_else(valid & (c + cb * _BLOCK < columns), T.cast(InjectionRight[j, sequence * length + token, head, c + cb * _BLOCK], "float32"), 0.0)
                    T.sync_threads()
                    for e in T.serial(event_total):
                        et = e // events
                        kind = e % events
                        for p in T.serial(e):
                            pt = p // events
                            interval = T.alloc_var("float32", init=1.0)
                            for q in T.serial(chunk):
                                if q > pt and q <= et:
                                    interval = interval * gate[q]
                            dot = T.alloc_var("float32", init=0.0)
                            for r in T.serial(rows):
                                dot += rvec[e, r] * u[p, r]
                            intervals[e, p] = interval * dot
                        T.sync_threads()
                        for c in T.Parallel(_BLOCK):
                            value = T.alloc_var("float32", init=inj[e, c])
                            if kind < rank_count:
                                seed = T.alloc_var("float32", init=0.0)
                                for r in T.serial(rows):
                                    seed += rvec[e, r] * entry[r, c]
                                value = value + prefix[et] * seed
                            for p in T.serial(e):
                                value = value + intervals[e, p] * response[p, c]
                            response[e, c] = value
                        T.sync_threads()
                    for t in T.serial(chunk):
                        token = chunk_index * chunk + t
                        if token < length:
                            for r, c in T.Parallel(rows, _BLOCK):
                                value = T.alloc_var("float32", init=prefix[t] * entry[r, c])
                                for e in T.serial((t + 1) * events):
                                    et = e // events
                                    interval = T.alloc_var("float32", init=1.0)
                                    for q in T.serial(chunk):
                                        if q > et and q <= t:
                                            interval = interval * gate[q]
                                    value = value + u[e, r] * interval * response[e, c]
                                state[r, c] = value
                            T.sync_threads()
                            for q in T.unroll(readouts):
                                for c in T.Parallel(_BLOCK):
                                    out = T.alloc_var("float32", init=0.0)
                                    for r in T.serial(rows):
                                        out += T.cast(Query[q, sequence * length + token, head, r], "float32") * state[r, c]
                                    if c + cb * _BLOCK < columns:
                                        Output[q, sequence * length + token, head, c + cb * _BLOCK] = out
                            T.sync_threads()
                    T.sync_threads()
                for r, c in T.Parallel(rows, _BLOCK):
                    if c + cb * _BLOCK < columns:
                        Final[sequence, head, r, c + cb * _BLOCK] = state[r, c]
        return kernel
    return build()


def execute(*, codes, left, right, coefficient, injection_left, injection_right,
            query, initial_state, length, output_dtypes):
    batch, heads, rows, columns = initial_state.shape
    rank_ops = tuple(i for i, code in enumerate(codes) if code == 2)
    scale_ops = tuple(i for i, code in enumerate(codes) if code == 0)
    injection_count, readout_count = len(injection_left), len(query)
    events = len(rank_ops) + injection_count
    chunk = chunk_tokens(events)
    event_total = chunk * events
    device = initial_state.device

    def bind(x, size):
        if x.numel() == 1:
            return x.reshape(1, 1, 1).expand(batch * length, heads, size)
        return x.reshape(batch * length, heads, -1).expand(batch * length, heads, size)
    L = torch.stack([bind(left[i].float(), rows) for i in range(len(rank_ops))]).contiguous()
    R = torch.stack([bind(right[i].float(), rows) for i in range(len(rank_ops))]).contiguous()
    # The kernel stores scalar gates first, then rank coefficients.  Both tuples
    # retain authored order within their operation kind while rank factors stay
    # compact, so original operation indices must not index the compact tensors.
    C = torch.stack([bind(coefficient[i].float(), 1).squeeze(-1)
                     for i in scale_ops + rank_ops]).contiguous()
    IL = (torch.stack([bind(x.float(), rows) for x in injection_left])
          if injection_count else torch.empty((1, batch * length, heads, rows), device=device, dtype=torch.float32))
    IR = (torch.stack([bind(x.float(), columns) for x in injection_right])
          if injection_count else torch.empty((1, batch * length, heads, columns), device=device, dtype=torch.float32))
    Q = torch.stack([bind(x.float(), rows) for x in query]).contiguous()
    output = torch.empty((readout_count, batch * length, heads, columns), device=device, dtype=torch.float32)
    final = torch.empty((batch, heads, rows, columns), device=device, dtype=torch.float32)
    kernel = _compile(_meta(L), _meta(R), _meta(C), _meta(IL), _meta(IR), _meta(Q), _meta(initial_state), _meta(output), _meta(final), batch=batch, heads=heads, rows=rows, columns=columns, length=length, rank_count=len(rank_ops), scale_count=len(scale_ops), injection_count=injection_count, readouts=readout_count, chunk=chunk, events=events, event_total=event_total, device_index=device.index)
    kernel(L, R, C, IL, IR, Q, initial_state, output, final)
    return tuple(output[i].reshape(batch, length, heads, columns).permute(0, 2, 1, 3).to(dtype) for i, dtype in enumerate(output_dtypes)), final.to(initial_state.dtype)
