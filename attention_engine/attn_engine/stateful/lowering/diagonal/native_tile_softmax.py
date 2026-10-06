"""Native diagonal readout regions lowered entirely to TileLang.

Shared scores are a chunk-local cut, not a materialized input expansion. The
softmax cut is FP32 and uses a maximum subtraction. Authored pointwise loaders
are supplied by native_tile_kernels, including their BF16 node boundaries.
"""

import hashlib
import linecache

import torch

_KERNELS = {}
_CHUNK = 64


def _dtype(dtype):
    return str(dtype).removeprefix("torch.")


def _meta(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), _dtype(tensor.dtype)


def _compile(annotations, body, grid, threads=128):
    """Compile a specialization with concrete raw-buffer annotations."""
    import tilelang
    import tilelang.language as T

    text = ("@tilelang.jit(compile_flags=['-fmad=false'])\n"
            "def build():\n"
            "    @T.prim_func\n"
            f"    def kernel({', '.join(annotations)}):\n"
            f"        with T.Kernel({grid}, threads={threads}) as block:\n"
            + "\n".join("            " + line for line in body)
            + "\n    return kernel\n")
    key = hashlib.sha256(text.encode()).hexdigest()
    if key not in _KERNELS:
        filename = f"<native-diagonal-region-{key}>"
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        namespace = {"tilelang": tilelang, "T": T}
        exec(compile(text, filename, "exec", dont_inherit=True), namespace)
        _KERNELS[key] = namespace["build"]()
    return _KERNELS[key]


def _annotation(name, tensor):
    return f"{name}: T.StridedTensor(*{_meta(tensor)!r})"


def _state_expressions(state):
    return (state.row_gate, state.column_gate,
            *(expression for pair in state.injections for expression in pair))


def _append(body, indent, lines):
    body.extend(indent + line for line in lines)


def _load(context, expression, token, feature, prefix):
    value = context.emit_value(expression, token, "head", feature)
    return (f"{prefix} = {value}",), prefix


def _advance(body, indent, context, state, variable, token, row, column, prefix):
    lines, gr = _load(context, state.row_gate, token, row, prefix + "gr")
    _append(body, indent, lines)
    lines, gc = _load(context, state.column_gate, token, column, prefix + "gc")
    _append(body, indent, lines)
    body.append(f"{indent}{variable} = {variable} * {gr} * {gc}")
    terms = []
    for injection, (key, value) in enumerate(state.injections):
        lines, k = _load(context, key, token, row, f"{prefix}k{injection}")
        _append(body, indent, lines)
        lines, v = _load(context, value, token, column, f"{prefix}v{injection}")
        _append(body, indent, lines)
        terms.append(f"({k} * {v})")
    bias = " + ".join(terms)
    body.append(f"{indent}{variable} = {variable} + ({bias})")


def _context(expressions, inputs, heads, length):
    from .native_tile_kernels import RawLoaderContext
    return RawLoaderContext(expressions, inputs, heads, length)


def _replay(state, carry, query, inputs, *, sequence_length, heads, prior,
            output_dtype, bonus=None):
    """Read a state using an authored loader or an explicit FP32 query cut."""
    length = sequence_length
    batch_heads, chunks, rows, columns = carry.shape
    batch = batch_heads // heads
    dense_query = isinstance(query, torch.Tensor)
    expressions = _state_expressions(state)
    if not dense_query:
        expressions += (query,)
    if bonus is not None:
        expressions += (bonus,)
    context = _context(expressions, inputs, heads, length)
    output = torch.empty((batch * length, heads, columns), device=carry.device,
                         dtype=output_dtype)
    annotations = list(context.annotations) + [_annotation("Carry", carry),
                                              _annotation("Output", output)]
    arguments = list(context.pointers) + [carry, output]
    if dense_query:
        annotations.append(_annotation("Query", query))
        arguments.append(query)
    tiles = (columns + 127) // 128
    body = [f"tile = block % {tiles}", f"target = (block // {tiles}) % {length}",
            f"sh = block // {tiles * length}", f"head = sh % {heads}",
            f"sequence = sh // {heads}", f"chunk = target // {_CHUNK}",
            f"token = sequence * {length} + target",
            "for lane in T.Parallel(128):", "    c = tile * 128 + lane",
            f"    if c < {columns}:",
            "        result = T.alloc_local((1,), 'float32')",
            "        element = T.alloc_local((1,), 'float32')",
            "        result[0] = 0.0", f"        for r in T.serial({rows}):",
            "            element[0] = T.cast(Carry[sh, chunk, r, c], 'float32')",
            f"            for step in T.serial({_CHUNK}):",
            f"                local_t = chunk * {_CHUNK} + step",
            f"                if local_t < target + {0 if prior else 1}:",
            f"                    source = sequence * {length} + local_t"]
    _advance(body, "                    ", context, state, "element[0]", "source", "r", "c", "replay")
    if dense_query:
        q = "T.cast(Query[token, head, r], 'float32')"
    else:
        lines, q = _load(context, query, "token", "r", "query")
        _append(body, "            ", lines)
    body.append(f"            result[0] = result[0] + {q} * element[0]")
    if bonus is not None:
        body += ["        extra = T.alloc_local((1,), 'float32')", "        extra[0] = 0.0",
                 f"        for r in T.serial({rows}):"]
        lines, b = _load(context, bonus, "token", "r", "bonus")
        _append(body, "            ", lines)
        body.append(f"            extra[0] = extra[0] + {b}")
        lines, v = _load(context, state.injections[0][1], "token", "c", "bonusvalue")
        _append(body, "        ", lines)
        body.append(f"        result[0] = result[0] + extra[0] * {v}")
    body.append("        Output[token, head, c] = result[0]")
    _compile(annotations, body, batch_heads * length * tiles)(*arguments)
    return output


def bonus_readout(state, carry, query, bonus, inputs, *, sequence_length, heads,
                  prior=True, output_dtype=torch.float32):
    """Prior-state contraction plus the authored current-token bonus.

    ``bonus`` is the complete pointwise left operand of Contract(..., 1), not
    just the raw bonus input. Its loader retains query/key/bonus association and
    rounding. The compiler's bonus proof identifies the first injection value.
    """
    return _replay(state, carry, query, inputs, sequence_length=sequence_length,
                   heads=heads, prior=prior, output_dtype=output_dtype, bonus=bonus)


def softmax_cut(scores):
    """Produce a stable FP32 softmax without host-side CUDA arithmetic."""
    tokens, heads, features = scores.shape
    padded = 1 << (features - 1).bit_length()
    output = torch.empty(scores.shape, device=scores.device, dtype=torch.float32)
    body = [f"token = block // {heads}", f"head = block % {heads}",
            f"values = T.alloc_fragment(({padded},), 'float32')",
            "maximum = T.alloc_fragment((1,), 'float32')",
            "denominator = T.alloc_fragment((1,), 'float32')",
            f"for r in T.Parallel({padded}):",
            f"    values[r] = T.if_then_else(r < {features}, T.cast(Scores[token, head, r], 'float32'), -T.infinity('float32'))",
            "T.reduce_max(values, maximum, dim=0)",
            f"for r in T.Parallel({padded}):",
            "    values[r] = T.exp(values[r] - maximum[0])",
            "T.reduce_sum(values, denominator, dim=0)",
            f"for r in T.Parallel({padded}):", f"    if r < {features}:",
            "        Output[token, head, r] = values[r] / denominator[0]"]
    _compile([_annotation("Scores", scores), _annotation("Output", output)],
             body, tokens * heads)(scores, output)
    return output


def softmax_readout(first_state, first_carry, second_state, second_carry,
                    query, inputs, *, sequence_length, heads, prior_first=False,
                    prior_second=False, output_dtype=torch.float32, fused=False):
    """Execute q·state1 → softmax → state2, respecting both prior flags."""
    if first_carry.shape[-1] != second_carry.shape[-2]:
        raise ValueError("softmax state dimensions do not compose")
    if fused:
        return _fused_softmax(first_state, first_carry, second_state, second_carry,
                              query, inputs, sequence_length, heads,
                              prior_first, prior_second, output_dtype)
    scores = _replay(first_state, first_carry, query, inputs,
                     sequence_length=sequence_length, heads=heads,
                     prior=prior_first, output_dtype=torch.float32)
    weights = softmax_cut(scores)
    return _replay(second_state, second_carry, weights, inputs,
                   sequence_length=sequence_length, heads=heads,
                   prior=prior_second, output_dtype=output_dtype)


def _fused_softmax(first, carry1, second, carry2, query, inputs, length, heads,
                   prior1, prior2, output_dtype):
    batch_heads, _, rows, features = carry1.shape
    columns = carry2.shape[-1]
    padded = 1 << (features - 1).bit_length()
    context = _context(_state_expressions(first) + _state_expressions(second) + (query,),
                       inputs, heads, length)
    output = torch.empty((batch_heads // heads * length, heads, columns),
                         device=carry1.device, dtype=output_dtype)
    body = [f"target = block % {length}", f"sh = block // {length}",
            f"head = sh % {heads}", f"sequence = sh // {heads}",
            f"chunk = target // {_CHUNK}", f"token = sequence * {length} + target",
            f"scores = T.alloc_fragment(({padded},), 'float32')",
            f"weights = T.alloc_shared(({padded},), 'float32')",
            "maximum = T.alloc_fragment((1,), 'float32')",
            "denominator = T.alloc_fragment((1,), 'float32')",
            f"for c in T.Parallel({padded}):",
            "    total = T.alloc_local((1,), 'float32')",
            "    element = T.alloc_local((1,), 'float32')",
            "    total[0] = 0.0", f"    if c < {features}:",
            f"        for r in T.serial({rows}):",
            "            element[0] = T.cast(Carry1[sh, chunk, r, c], 'float32')",
            f"            for step in T.serial({_CHUNK}):",
            f"                local_t = chunk * {_CHUNK} + step",
            f"                if local_t < target + {0 if prior1 else 1}:",
            f"                    source = sequence * {length} + local_t"]
    _advance(body, "                    ", context, first, "element[0]", "source", "r", "c", "first")
    lines, q = _load(context, query, "token", "r", "query")
    _append(body, "            ", lines)
    body += [f"            total[0] = total[0] + {q} * element[0]",
             f"    scores[c] = T.if_then_else(c < {features}, total[0], -T.infinity('float32'))",
             "T.reduce_max(scores, maximum, dim=0)",
             f"for c in T.Parallel({padded}):",
             "    scores[c] = T.exp(scores[c] - maximum[0])",
             "T.reduce_sum(scores, denominator, dim=0)",
             f"for c in T.Parallel({padded}):",
             "    weights[c] = scores[c] / denominator[0]", "T.sync_threads()",
             f"for c in T.Parallel({columns}):",
             "    total2 = T.alloc_local((1,), 'float32')",
             "    element2 = T.alloc_local((1,), 'float32')",
             "    total2[0] = 0.0", f"    for r in T.serial({features}):",
             "        element2[0] = T.cast(Carry2[sh, chunk, r, c], 'float32')",
             f"        for step in T.serial({_CHUNK}):",
             f"            local_t = chunk * {_CHUNK} + step",
             f"            if local_t < target + {0 if prior2 else 1}:",
             f"                source = sequence * {length} + local_t"]
    _advance(body, "                ", context, second, "element2[0]", "source", "r", "c", "second")
    body += ["        total2[0] = total2[0] + weights[r] * element2[0]",
             "    Output[token, head, c] = total2[0]"]
    annotations = list(context.annotations) + [_annotation("Carry1", carry1),
                                              _annotation("Carry2", carry2),
                                              _annotation("Output", output)]
    _compile(annotations, body, batch_heads * length)(*context.pointers, carry1, carry2, output)
    return output


def _shared_scores(state, carry, query, inputs, length, heads, prior):
    batch_heads, _, rows, _ = carry.shape
    count = len(state.injections)
    context = _context((state.row_gate, query, *(pair[0] for pair in state.injections)),
                       inputs, heads, length)
    scores = torch.empty((batch_heads, length, _CHUNK, count),
                         device=carry.device, dtype=torch.float32)
    body = [f"target = block % {length}", f"sh = block // {length}",
            f"head = sh % {heads}", f"sequence = sh // {heads}",
            f"start = (target // {_CHUNK}) * {_CHUNK}",
            f"token = sequence * {length} + target",
            f"for slot in T.Parallel({_CHUNK}):", "    local_j = start + slot",
            f"    source_j = sequence * {length} + local_j"]
    for injection, (key, _) in enumerate(state.injections):
        body += [f"    total{injection} = T.alloc_local((1,), 'float32')",
                 f"    factor{injection} = T.alloc_local((1,), 'float32')",
                 f"    total{injection}[0] = 0.0",
                 f"    if local_j < target + {0 if prior else 1}:",
                 f"        for r in T.serial({rows}):",
                 f"            factor{injection}[0] = 1.0",
                 f"            for step in T.serial({_CHUNK}):",
                 "                local_t = start + step",
                 f"                if (local_t > local_j) & (local_t < target + {0 if prior else 1}):",
                 f"                    source = sequence * {length} + local_t"]
        lines, gr = _load(context, state.row_gate, "source", "r", f"sg{injection}")
        _append(body, "                    ", lines)
        body.append(f"                    factor{injection}[0] = factor{injection}[0] * {gr}")
        lines, q = _load(context, query, "token", "r", f"sq{injection}")
        _append(body, "            ", lines)
        lines, k = _load(context, key, "source_j", "r", f"sk{injection}")
        _append(body, "            ", lines)
        body += [f"            total{injection}[0] = total{injection}[0] + {q} * {k} * factor{injection}[0]",
                 f"    Scores[sh, target, slot, {injection}] = total{injection}[0]"]
    _compile(list(context.annotations) + [_annotation("Scores", scores)],
             body, batch_heads * length)(*context.pointers, scores)
    return scores


def _apply_shared_scores(state, carry, query, scores, inputs, length, heads,
                         prior, output_dtype):
    batch_heads, _, rows, columns = carry.shape
    context = _context(_state_expressions(state) + (query,), inputs, heads, length)
    output = torch.empty((batch_heads // heads * length, heads, columns),
                         device=carry.device, dtype=output_dtype)
    tiles = (columns + 127) // 128
    body = [f"tile = block % {tiles}", f"target = (block // {tiles}) % {length}",
            f"sh = block // {tiles * length}", f"head = sh % {heads}",
            f"sequence = sh // {heads}", f"chunk = target // {_CHUNK}",
            f"start = chunk * {_CHUNK}", f"token = sequence * {length} + target",
            "for lane in T.Parallel(128):", "    c = tile * 128 + lane",
            f"    if c < {columns}:",
            "        result = T.alloc_local((1,), 'float32')",
            "        element = T.alloc_local((1,), 'float32')",
            "        column_factor = T.alloc_local((1,), 'float32')",
            "        result[0] = 0.0", f"        for r in T.serial({rows}):",
            "            element[0] = T.cast(Carry[sh, chunk, r, c], 'float32')",
            f"            for step in T.serial({_CHUNK}):",
            "                local_t = start + step",
            f"                if local_t < target + {0 if prior else 1}:",
            f"                    source = sequence * {length} + local_t"]
    lines, gr = _load(context, state.row_gate, "source", "r", "carryrg")
    _append(body, "                    ", lines)
    lines, gc = _load(context, state.column_gate, "source", "c", "carrycg")
    _append(body, "                    ", lines)
    body.append(f"                    element[0] = element[0] * {gr} * {gc}")
    lines, q = _load(context, query, "token", "r", "carryq")
    _append(body, "            ", lines)
    body += [f"            result[0] = result[0] + {q} * element[0]",
             f"        for slot in T.serial({_CHUNK}):", "            local_j = start + slot",
             f"            if local_j < target + {0 if prior else 1}:",
             f"                source_j = sequence * {length} + local_j",
             "                column_factor[0] = 1.0",
             f"                for step in T.serial({_CHUNK}):",
             "                    local_t = start + step",
             f"                    if (local_t > local_j) & (local_t < target + {0 if prior else 1}):",
             f"                        source = sequence * {length} + local_t"]
    lines, gc = _load(context, state.column_gate, "source", "c", "scorecg")
    _append(body, "                        ", lines)
    body.append(f"                        column_factor[0] = column_factor[0] * {gc}")
    for injection, (_, value) in enumerate(state.injections):
        lines, v = _load(context, value, "source_j", "c", f"scorev{injection}")
        _append(body, "                ", lines)
        body.append(f"                result[0] = result[0] + Scores[sh, target, slot, {injection}] * {v} * column_factor[0]")
    body.append("        Output[token, head, c] = result[0]")
    annotations = list(context.annotations) + [_annotation("Carry", carry),
                                              _annotation("Scores", scores),
                                              _annotation("Output", output)]
    _compile(annotations, body, batch_heads * length * tiles)(*context.pointers, carry, scores, output)
    return output


def shared_score_readouts(state, carry, readouts, inputs, *, sequence_length, heads):
    """Reuse row scores for every readout with the same authored query and time.

    Each readout is ``(output_spec, state_term, DenseExpression)``. All injection
    terms participate; arbitrary column gates are applied in the score consumer.
    """
    score_cache = {}
    outputs = {}
    for output, term, query in readouts:
        key = (query.loader.cache_key, term.prior)
        if key not in score_cache:
            score_cache[key] = _shared_scores(state, carry, query, inputs,
                                               sequence_length, heads, term.prior)
        outputs[output.name] = _apply_shared_scores(
            state, carry, query, score_cache[key], inputs, sequence_length,
            heads, term.prior, output.output_dtype)
    return outputs
