"""TileLang lowering for ordered low-rank affine recurrences.

The kernel intentionally uses a simple one-program-per-(sequence,head,column
block) schedule.  All state updates and contractions are FP32 and the Python
ABI remains the historical token-major ABI.
"""

from functools import lru_cache
import torch

_BLOCK = 32


def _meta(x):
    return tuple(x.shape), tuple(x.stride()), str(x.dtype).removeprefix("torch.")


@lru_cache(maxsize=None)
def _compile(
    left_meta, right_meta, coeff_meta, gate_meta, inj_l_meta, inj_r_meta,
    query_meta, initial_meta, output_meta, final_meta,
    *, batch, heads, rows, columns, tokens, ops, injections, readouts,
    device_index,
):
    import tilelang
    import tilelang.language as T

    col_blocks = (columns + _BLOCK - 1) // _BLOCK
    length = tokens // batch
    opcodes = tuple(ops)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Left: T.StridedTensor(*left_meta), Right: T.StridedTensor(*right_meta),
            Coeff: T.StridedTensor(*coeff_meta), Gate: T.StridedTensor(*gate_meta),
            InjectionLeft: T.StridedTensor(*inj_l_meta), InjectionRight: T.StridedTensor(*inj_r_meta),
            Query: T.StridedTensor(*query_meta), Initial: T.StridedTensor(*initial_meta),
            Output: T.StridedTensor(*output_meta), Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(batch * heads, col_blocks, threads=128) as (sh, cb):
                sequence = sh // heads
                head = sh % heads
                state = T.alloc_shared((rows, _BLOCK), "float32")
                for r, c in T.Parallel(rows, _BLOCK):
                    state[r, c] = T.if_then_else(
                        c + cb * _BLOCK < columns,
                        T.cast(Initial[sequence, head, r, c + cb * _BLOCK], "float32"), 0.0,
                    )
                T.sync_threads()
                for token in T.serial(length):
                    flat = sequence * length + token
                    # Authored operation order is retained exactly.
                    for op, code in enumerate(opcodes):
                        if code == 0:
                            factor = T.cast(Coeff[op, flat, head], "float32")
                            for r, c in T.Parallel(rows, _BLOCK):
                                state[r, c] = state[r, c] * factor
                        elif code == 1:
                            for r, c in T.Parallel(rows, _BLOCK):
                                factor = T.if_then_else(r < rows, T.cast(Left[op, flat, head, r], "float32"), 1.0)
                                state[r, c] = state[r, c] * factor
                        elif code == 3:
                            for r, c in T.Parallel(rows, _BLOCK):
                                factor = T.if_then_else(c + cb * _BLOCK < columns, T.cast(Gate[op, flat, head, c + cb * _BLOCK], "float32"), 1.0)
                                state[r, c] = state[r, c] * factor
                        else:
                            projection = T.alloc_shared((_BLOCK,), "float32")
                            for c in T.Parallel(_BLOCK):
                                projection[c] = 0.0
                            T.sync_threads()
                            for r in T.serial(rows):
                                for c in T.Parallel(_BLOCK):
                                    projection[c] += state[r, c] * T.cast(Right[op, flat, head, r], "float32")
                            T.sync_threads()
                            coefficient = T.cast(Coeff[op, flat, head], "float32")
                            for r, c in T.Parallel(rows, _BLOCK):
                                state[r, c] = state[r, c] + coefficient * T.cast(Left[op, flat, head, r], "float32") * projection[c]
                            T.sync_threads()
                    for inj in range(injections):
                        for r, c in T.Parallel(rows, _BLOCK):
                            if c + cb * _BLOCK < columns:
                                state[r, c] = state[r, c] + T.cast(InjectionLeft[inj, flat, head, r], "float32") * T.cast(InjectionRight[inj, flat, head, c + cb * _BLOCK], "float32")
                        T.sync_threads()
                    for readout in range(readouts):
                        projection = T.alloc_shared((_BLOCK,), "float32")
                        for c in T.Parallel(_BLOCK):
                            projection[c] = 0.0
                        T.sync_threads()
                        for r in T.serial(rows):
                            query = T.cast(Query[readout, flat, head, r], "float32")
                            for c in T.Parallel(_BLOCK):
                                projection[c] += query * state[r, c]
                        T.sync_threads()
                        for c in T.Parallel(_BLOCK):
                            if c + cb * _BLOCK < columns:
                                Output[readout, flat, head, c + cb * _BLOCK] = projection[c]
                for r, c in T.Parallel(rows, _BLOCK):
                    if c + cb * _BLOCK < columns:
                        Final[sequence, head, r, c + cb * _BLOCK] = state[r, c]
        return kernel
    return build()


def execute(*, operation_codes, left_factors, right_factors, coefficients,
            injection_left, injection_right, readouts, initial_state, output_dtypes):
    batch, heads, rows, columns = initial_state.shape
    tokens = readouts[0].shape[0]
    if tokens % batch:
        raise ValueError("low-rank token count must divide the State batch")
    length = tokens // batch
    device = initial_state.device
    codes = tuple({"scale": 0, "row-scale": 1, "rank-one": 2, "column-scale": 3}[x] for x in operation_codes)

    def binding(part, size):
        if part.numel() == 1:
            return part.reshape(1, 1, 1).expand(tokens, heads, size)
        return part.reshape(tokens, heads, -1).expand(tokens, heads, size)

    from .wy_kernels import eligible, execute as execute_wy

    if eligible(tuple(operation_codes), len(injection_left), rows, columns):
        rank_ops = tuple(i for i, code in enumerate(codes) if code == 2)
        return execute_wy(
            codes=codes,
            left=tuple(left_factors[i] for i in rank_ops),
            right=tuple(right_factors[i] for i in rank_ops),
            coefficient=coefficients,
            injection_left=injection_left,
            injection_right=injection_right,
            query=readouts,
            initial_state=initial_state,
            length=length,
            output_dtypes=output_dtypes,
        )

    # Static branches only read slots populated for their operation kind.
    left = torch.empty((len(codes), tokens, heads, rows), device=device, dtype=torch.float32)
    right = torch.empty((len(codes), tokens, heads, rows), device=device, dtype=torch.float32)
    coeff = torch.empty((len(codes), tokens, heads), device=device, dtype=torch.float32)
    gate = torch.empty((len(codes), tokens, heads, columns), device=device, dtype=torch.float32)
    for i, code in enumerate(codes):
        if code in (1, 2):
            left[i].copy_(binding(left_factors[i], rows))
        if code == 3:
            gate[i].copy_(binding(left_factors[i], columns))
        if code == 2:
            right[i].copy_(binding(right_factors[i], rows))
        if code in (0, 2):
            coeff[i].copy_(binding(coefficients[i], 1).squeeze(-1))

    def stack(parts, size):
        if not parts:
            return torch.zeros((1, tokens, heads, size), device=device, dtype=torch.float32)
        return torch.stack([binding(x, size) for x in parts]).float().contiguous()
    il = stack(injection_left, rows)
    ir = stack(injection_right, columns)
    query = stack(readouts, rows)
    output = torch.empty((len(readouts), tokens, heads, columns), device=device, dtype=torch.float32)
    final = torch.empty_like(initial_state, dtype=torch.float32)
    kernel = _compile(
        _meta(left), _meta(right), _meta(coeff), _meta(gate), _meta(il), _meta(ir),
        _meta(query), _meta(initial_state), _meta(output), _meta(final),
        batch=batch, heads=heads, rows=rows, columns=columns, tokens=tokens,
        ops=codes, injections=len(injection_left), readouts=len(readouts),
        device_index=device.index,
    )
    kernel(left, right, coeff, gate, il, ir, query, initial_state, output, final)
    return tuple(output[i].reshape(batch, length, heads, columns).permute(0, 2, 1, 3).to(dtype) for i, dtype in enumerate(output_dtypes)), final.to(initial_state.dtype)
