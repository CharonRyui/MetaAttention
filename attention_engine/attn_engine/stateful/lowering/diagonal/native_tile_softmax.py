"""Native diagonal readout regions lowered entirely to TileLang.

Replay CTAs retain one independent chunk's State in cooperative row/column
fragments, advancing each element once per token. Shared scores use inverse-free
reverse gate recurrences and a weighted-query cut reused by column consumers.
The softmax cut is FP32 and uses maximum subtraction. Authored pointwise loaders
retain their BF16 node boundaries through native_tile_kernels.
"""

import hashlib
import linecache

import torch

from .native_tile_kernels import _tensor_metadata

_KERNELS = {}
_REGION_KERNELS = {}
_CHUNK = 64


def _dtype(dtype):
    return str(dtype).removeprefix("torch.")


def _meta(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), _dtype(tensor.dtype)


def _compile(annotations, body, grid, threads=128, *, device):
    """Compile a specialization with concrete raw-buffer annotations."""
    import tilelang
    import tilelang.language as T

    text = (
        "@tilelang.jit(compile_flags=['-fmad=false'])\n"
        "def build():\n"
        "    @T.prim_func\n"
        f"    def kernel({', '.join(annotations)}):\n"
        f"        with T.Kernel({grid}, threads={threads}) as block:\n"
        + "\n".join("            " + line for line in body)
        + "\n    return kernel\n"
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    key = (device, digest)
    if key not in _KERNELS:
        filename = f"<native-diagonal-region-{digest}>"
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        namespace = {"tilelang": tilelang, "T": T}
        exec(compile(text, filename, "exec", dont_inherit=True), namespace)
        _KERNELS[key] = namespace["build"]()
    return _KERNELS[key]


def _annotation(name, tensor):
    return f"{name}: T.StridedTensor(*{_meta(tensor)!r})"


def _state_expressions(state):
    return (
        state.row_gate,
        state.column_gate,
        *(expression for pair in state.injections for expression in pair),
    )


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


def _replay(
    state,
    carry,
    query,
    inputs,
    *,
    sequence_length,
    heads,
    prior,
    output_dtype,
    bonus=None,
    final=None,
):
    """Read a state using an authored loader or an explicit FP32 query cut."""
    length = sequence_length
    batch_heads, chunks, rows, columns = carry.shape
    batch = batch_heads // heads
    if final is not None and (
        chunks != 1 or final.shape != (batch, heads, rows, columns)
    ):
        raise ValueError(
            "direct replay final requires one chunk and a matching rank-two State"
        )
    dense_query = isinstance(query, torch.Tensor)
    expressions = _state_expressions(state)
    if not dense_query:
        expressions += (query,)
    if bonus is not None:
        expressions += (bonus,)
    context = _context(expressions, inputs, heads, length)
    output = torch.empty(
        (batch * length, heads, columns), device=carry.device, dtype=output_dtype
    )
    key = (
        "replay",
        context.specialization_key(expressions),
        batch,
        heads,
        length,
        _CHUNK,
        bool(prior),
        dense_query,
        bonus is not None,
        _tensor_metadata(carry),
        _tensor_metadata(output),
        _tensor_metadata(query) if dense_query else None,
        _tensor_metadata(final) if final is not None else None,
    )
    kernel = _REGION_KERNELS.get(key)
    if kernel is not None:
        arguments = (
            (*context.pointers, carry, output, query)
            if dense_query
            else (*context.pointers, carry, output)
        )
        if final is not None:
            arguments += (final,)
        kernel(*arguments)
        return output
    annotations = list(context.annotations) + [
        _annotation("Carry", carry),
        _annotation("Output", output),
    ]
    arguments = list(context.pointers) + [carry, output]
    if dense_query:
        annotations.append(_annotation("Query", query))
        arguments.append(query)
    if final is not None:
        annotations.append(_annotation("Final", final))
        arguments.append(final)
    # Match the original Triton replay domain: a CTA owns a chunk and a
    # column tile, while rows cooperate in the contraction. Each State element
    # advances once per token, not once per future target in the same chunk.
    width = 32
    padded_rows = 1 << (rows - 1).bit_length()
    tiles = (columns + width - 1) // width
    body = [
        f"tile = block % {tiles}",
        f"chunk = (block // {tiles}) % {chunks}",
        f"sh = block // {tiles * chunks}",
        f"head = sh % {heads}",
        f"sequence = sh // {heads}",
        f"memory = T.alloc_fragment(({padded_rows}, {width}), 'float32')",
        f"products = T.alloc_fragment(({padded_rows}, {width}), 'float32')",
        f"result = T.alloc_fragment(({width},), 'float32')",
        f"for r, lane in T.Parallel({padded_rows}, {width}):",
        f"    c = tile * {width} + lane",
        f"    memory[r, lane] = T.if_then_else((r < {rows}) & (c < {columns}), Carry[sh, chunk, r, c], 0.0)",
        f"for step in T.serial({_CHUNK}):",
        f"    target = chunk * {_CHUNK} + step",
        f"    if target < {length}:",
        f"        token = sequence * {length} + target",
    ]

    def advance():
        body.extend(
            [
                f"        for r, lane in T.Parallel({padded_rows}, {width}):",
                f"            c = tile * {width} + lane",
                f"            if (r < {rows}) & (c < {columns}):",
            ]
        )
        _advance(
            body,
            "                ",
            context,
            state,
            "memory[r, lane]",
            "token",
            "r",
            "c",
            "replay",
        )

    if not prior:
        advance()
    body.extend(
        [
            f"        for r, lane in T.Parallel({padded_rows}, {width}):",
            f"            c = tile * {width} + lane",
            "            products[r, lane] = 0.0",
            f"            if (r < {rows}) & (c < {columns}):",
        ]
    )
    if dense_query:
        q = "T.cast(Query[token, head, r], 'float32')"
    else:
        lines, q = _load(context, query, "token", "r", "query")
        _append(body, "                ", lines)
    body += [
        f"                products[r, lane] = {q} * memory[r, lane]",
        "        T.reduce_sum(products, result, dim=0)",
    ]
    if bonus is not None:
        body += [
            f"        extra = T.alloc_fragment(({padded_rows},), 'float32')",
            "        bonus_sum = T.alloc_fragment((1,), 'float32')",
            f"        for r in T.Parallel({padded_rows}):",
            "            extra[r] = 0.0",
            f"            if r < {rows}:",
        ]
        lines, b = _load(context, bonus, "token", "r", "bonus")
        _append(body, "                ", lines)
        body += [
            f"                extra[r] = {b}",
            "        T.reduce_sum(extra, bonus_sum, dim=0)",
        ]
    body += [
        f"        for lane in T.Parallel({width}):",
        f"            c = tile * {width} + lane",
        f"            if c < {columns}:",
    ]
    if bonus is not None:
        lines, v = _load(context, state.injections[0][1], "token", "c", "bonusvalue")
        _append(body, "                ", lines)
        body.append(f"                result[lane] = result[lane] + bonus_sum[0] * {v}")
    body.append("                Output[token, head, c] = result[lane]")
    if prior:
        advance()
    if final is not None:
        body.extend(
            [
                f"for r, lane in T.Parallel({padded_rows}, {width}):",
                f"    c = tile * {width} + lane",
                f"    if (r < {rows}) & (c < {columns}):",
                "        Final[sequence, head, r, c] = memory[r, lane]",
            ]
        )
    kernel = _compile(
        annotations, body, batch_heads * chunks * tiles, device=carry.device
    )
    _REGION_KERNELS[key] = kernel
    kernel(*arguments)
    return output


def bonus_readout(
    state,
    carry,
    query,
    bonus,
    inputs,
    *,
    sequence_length,
    heads,
    prior=True,
    output_dtype=torch.float32,
    final=None,
):
    """Prior-state contraction plus the authored current-token bonus.

    ``bonus`` is the complete pointwise left operand of Contract(..., 1), not
    just the raw bonus input. Its loader retains query/key/bonus association and
    rounding. The compiler's bonus proof identifies the first injection value.
    """
    return _replay(
        state,
        carry,
        query,
        inputs,
        sequence_length=sequence_length,
        heads=heads,
        prior=prior,
        output_dtype=output_dtype,
        bonus=bonus,
        final=final,
    )


def softmax_cut(scores):
    """Produce a stable FP32 softmax without host-side CUDA arithmetic."""
    tokens, heads, features = scores.shape
    padded = 1 << (features - 1).bit_length()
    output = torch.empty(scores.shape, device=scores.device, dtype=torch.float32)
    body = [
        f"token = block // {heads}",
        f"head = block % {heads}",
        f"values = T.alloc_fragment(({padded},), 'float32')",
        "maximum = T.alloc_fragment((1,), 'float32')",
        "denominator = T.alloc_fragment((1,), 'float32')",
        f"for r in T.Parallel({padded}):",
        f"    values[r] = T.if_then_else(r < {features}, T.cast(Scores[token, head, r], 'float32'), -T.infinity('float32'))",
        "T.reduce_max(values, maximum, dim=0)",
        f"for r in T.Parallel({padded}):",
        "    values[r] = T.exp(values[r] - maximum[0])",
        "T.reduce_sum(values, denominator, dim=0)",
        f"for r in T.Parallel({padded}):",
        f"    if r < {features}:",
        "        Output[token, head, r] = values[r] / denominator[0]",
    ]
    _compile(
        [_annotation("Scores", scores), _annotation("Output", output)],
        body,
        tokens * heads,
        device=scores.device,
    )(scores, output)
    return output


def softmax_readout(
    first_state,
    first_carry,
    second_state,
    second_carry,
    query,
    inputs,
    *,
    sequence_length,
    heads,
    prior_first=False,
    prior_second=False,
    output_dtype=torch.float32,
    fused=False,
):
    """Execute q·state1 → softmax → state2, respecting both prior flags."""
    if first_carry.shape[-1] != second_carry.shape[-2]:
        raise ValueError("softmax state dimensions do not compose")
    if fused:
        return _fused_softmax(
            first_state,
            first_carry,
            second_state,
            second_carry,
            query,
            inputs,
            sequence_length,
            heads,
            prior_first,
            prior_second,
            output_dtype,
        )
    scores = _replay(
        first_state,
        first_carry,
        query,
        inputs,
        sequence_length=sequence_length,
        heads=heads,
        prior=prior_first,
        output_dtype=torch.float32,
    )
    weights = softmax_cut(scores)
    return _replay(
        second_state,
        second_carry,
        weights,
        inputs,
        sequence_length=sequence_length,
        heads=heads,
        prior=prior_second,
        output_dtype=output_dtype,
    )


def _fused_softmax(
    first,
    carry1,
    second,
    carry2,
    query,
    inputs,
    length,
    heads,
    prior1,
    prior2,
    output_dtype,
):
    # Retain both State matrices for one independent chunk. Parallel feature
    # reductions replace the migrated target x row x replay nesting.
    batch_heads, chunks, rows, features = carry1.shape
    columns = carry2.shape[-1]
    pr, pm, pc = (1 << (size - 1).bit_length() for size in (rows, features, columns))
    expressions = _state_expressions(first) + _state_expressions(second) + (query,)
    context = _context(expressions, inputs, heads, length)
    output = torch.empty(
        (batch_heads // heads * length, heads, columns),
        device=carry1.device,
        dtype=output_dtype,
    )
    key = (
        "fused-softmax",
        context.specialization_key(expressions),
        len(first.injections),
        len(second.injections),
        batch_heads // heads,
        heads,
        length,
        _CHUNK,
        bool(prior1),
        bool(prior2),
        _tensor_metadata(carry1),
        _tensor_metadata(carry2),
        _tensor_metadata(output),
    )
    kernel = _REGION_KERNELS.get(key)
    if kernel is not None:
        kernel(*context.pointers, carry1, carry2, output)
        return output
    body = [
        f"chunk = block % {chunks}",
        f"sh = block // {chunks}",
        f"head = sh % {heads}",
        f"sequence = sh // {heads}",
        f"memory1 = T.alloc_fragment(({pr}, {pm}), 'float32')",
        f"memory2 = T.alloc_fragment(({pm}, {pc}), 'float32')",
        f"products1 = T.alloc_fragment(({pr}, {pm}), 'float32')",
        f"products2 = T.alloc_fragment(({pm}, {pc}), 'float32')",
        f"scores = T.alloc_fragment(({pm},), 'float32')",
        f"result = T.alloc_fragment(({pc},), 'float32')",
        f"weights = T.alloc_shared(({pm},), 'float32')",
        "maximum = T.alloc_fragment((1,), 'float32')",
        "denominator = T.alloc_fragment((1,), 'float32')",
        f"for r, c in T.Parallel({pr}, {pm}):",
        f"    memory1[r, c] = T.if_then_else((r < {rows}) & (c < {features}), Carry1[sh, chunk, r, c], 0.0)",
        f"for r, c in T.Parallel({pm}, {pc}):",
        f"    memory2[r, c] = T.if_then_else((r < {features}) & (c < {columns}), Carry2[sh, chunk, r, c], 0.0)",
        f"for step in T.serial({_CHUNK}):",
        f"    target = chunk * {_CHUNK} + step",
        f"    if target < {length}:",
        f"        token = sequence * {length} + target",
    ]

    def advance(state, variable, nrows, ncolumns, padded_r, padded_c, prefix):
        body.extend(
            [
                f"        for r, c in T.Parallel({padded_r}, {padded_c}):",
                f"            if (r < {nrows}) & (c < {ncolumns}):",
            ]
        )
        _advance(
            body,
            "                ",
            context,
            state,
            f"{variable}[r, c]",
            "token",
            "r",
            "c",
            prefix,
        )

    if not prior1:
        advance(first, "memory1", rows, features, pr, pm, "first")
    body += [
        f"        for r, c in T.Parallel({pr}, {pm}):",
        "            products1[r, c] = 0.0",
        f"            if (r < {rows}) & (c < {features}):",
    ]
    lines, q = _load(context, query, "token", "r", "query")
    _append(body, "                ", lines)
    body += [
        f"                products1[r, c] = {q} * memory1[r, c]",
        "        T.reduce_sum(products1, scores, dim=0)",
        f"        for c in T.Parallel({pm}):",
        f"            scores[c] = T.if_then_else(c < {features}, scores[c], -T.infinity('float32'))",
        "        T.reduce_max(scores, maximum, dim=0)",
        f"        for c in T.Parallel({pm}):",
        "            scores[c] = T.exp(scores[c] - maximum[0])",
        "        T.reduce_sum(scores, denominator, dim=0)",
        f"        for c in T.Parallel({pm}):",
        "            weights[c] = scores[c] / denominator[0]",
        "        T.sync_threads()",
    ]
    if not prior2:
        advance(second, "memory2", features, columns, pm, pc, "second")
    body += [
        f"        for r, c in T.Parallel({pm}, {pc}):",
        "            products2[r, c] = weights[r] * memory2[r, c]",
        "        T.reduce_sum(products2, result, dim=0)",
        f"        for c in T.Parallel({pc}):",
        f"            if c < {columns}:",
        "                Output[token, head, c] = result[c]",
    ]
    if prior1:
        advance(first, "memory1", rows, features, pr, pm, "first")
    if prior2:
        advance(second, "memory2", features, columns, pm, pc, "second")
    body.append("        T.sync_threads()")
    annotations = list(context.annotations) + [
        _annotation("Carry1", carry1),
        _annotation("Carry2", carry2),
        _annotation("Output", output),
    ]
    kernel = _compile(
        annotations, body, batch_heads * chunks, threads=256, device=carry1.device
    )
    _REGION_KERNELS[key] = kernel
    kernel(*context.pointers, carry1, carry2, output)
    return output


def _shared_scores(state, carry, query, inputs, length, heads, prior):
    """Reverse multiply recurrence; no prefix division at zero/signed gates."""
    batch_heads, _, rows, _ = carry.shape
    count = len(state.injections)
    padded = 1 << (rows - 1).bit_length()
    context = _context(
        (state.row_gate, query, *(pair[0] for pair in state.injections)),
        inputs,
        heads,
        length,
    )
    scores = torch.empty(
        (batch_heads, length, _CHUNK, count), device=carry.device, dtype=torch.float32
    )
    weighted = torch.empty(
        (batch_heads, length, rows), device=carry.device, dtype=torch.float32
    )
    body = [
        f"target = block % {length}",
        f"sh = block // {length}",
        f"head = sh % {heads}",
        f"sequence = sh // {heads}",
        f"start = (target // {_CHUNK}) * {_CHUNK}",
        f"token = sequence * {length} + target",
        f"weight = T.alloc_fragment(({padded},), 'float32')",
        f"products = T.alloc_fragment(({padded},), 'float32')",
        "total = T.alloc_fragment((1,), 'float32')",
        f"for r in T.Parallel({padded}):",
        "    weight[r] = 0.0",
        f"    if r < {rows}:",
    ]
    lines, q = _load(context, query, "token", "r", "query")
    _append(body, "        ", lines)
    body += [
        f"        weight[r] = {q}",
        f"for reverse in T.serial({_CHUNK}):",
        f"    slot = {_CHUNK - 1} - reverse",
        "    local_j = start + slot",
        f"    if local_j < target + {0 if prior else 1}:",
        f"        source_j = sequence * {length} + local_j",
    ]
    for injection, (key, _) in enumerate(state.injections):
        body += [
            f"        for r in T.Parallel({padded}):",
            "            products[r] = 0.0",
            f"            if r < {rows}:",
        ]
        lines, k = _load(context, key, "source_j", "r", f"key{injection}")
        _append(body, "                ", lines)
        body += [
            f"                products[r] = weight[r] * {k}",
            "        T.reduce_sum(products, total, dim=0)",
            f"        Scores[sh, target, slot, {injection}] = total[0]",
        ]
    body += [f"        for r in T.Parallel({padded}):", f"            if r < {rows}:"]
    lines, gr = _load(context, state.row_gate, "source_j", "r", "gate")
    _append(body, "                ", lines)
    body += [f"                weight[r] = weight[r] * {gr}", "    else:"]
    for injection in range(count):
        body.append(f"        Scores[sh, target, slot, {injection}] = 0.0")
    body += [
        f"for r in T.Parallel({padded}):",
        f"    if r < {rows}:",
        "        Weighted[sh, target, r] = weight[r]",
    ]
    _compile(
        list(context.annotations)
        + [_annotation("Scores", scores), _annotation("Weighted", weighted)],
        body,
        batch_heads * length,
        device=carry.device,
    )(*context.pointers, scores, weighted)
    return scores, weighted


def _apply_shared_scores(
    state, carry, query, score_cut, inputs, length, heads, prior, output_dtype
):
    batch_heads, _, rows, columns = carry.shape
    scores, weighted = score_cut
    context = _context(
        (state.column_gate, *(pair[1] for pair in state.injections)),
        inputs,
        heads,
        length,
    )
    output = torch.empty(
        (batch_heads // heads * length, heads, columns),
        device=carry.device,
        dtype=output_dtype,
    )
    width = 32
    padded = 1 << (rows - 1).bit_length()
    tiles = (columns + width - 1) // width
    body = [
        f"tile = block % {tiles}",
        f"target = (block // {tiles}) % {length}",
        f"sh = block // {tiles * length}",
        f"head = sh % {heads}",
        f"sequence = sh // {heads}",
        f"chunk = target // {_CHUNK}",
        f"start = chunk * {_CHUNK}",
        f"token = sequence * {length} + target",
        f"products = T.alloc_fragment(({padded}, {width}), 'float32')",
        f"result = T.alloc_fragment(({width},), 'float32')",
        f"column_factor = T.alloc_fragment(({width},), 'float32')",
        f"injected = T.alloc_fragment(({width},), 'float32')",
        f"for r, lane in T.Parallel({padded}, {width}):",
        f"    c = tile * {width} + lane",
        "    products[r, lane] = 0.0",
        f"    if (r < {rows}) & (c < {columns}):",
        "        products[r, lane] = Weighted[sh, target, r] * Carry[sh, chunk, r, c]",
        "T.reduce_sum(products, result, dim=0)",
        f"for lane in T.Parallel({width}):",
        "    column_factor[lane] = 1.0",
        "    injected[lane] = 0.0",
        f"for reverse in T.serial({_CHUNK}):",
        f"    slot = {_CHUNK - 1} - reverse",
        "    local_j = start + slot",
        f"    if local_j < target + {0 if prior else 1}:",
        f"        source_j = sequence * {length} + local_j",
        f"        for lane in T.Parallel({width}):",
        f"            c = tile * {width} + lane",
        f"            if c < {columns}:",
    ]
    for injection, (_, value) in enumerate(state.injections):
        lines, v = _load(context, value, "source_j", "c", f"value{injection}")
        _append(body, "                ", lines)
        body.append(
            f"                injected[lane] = injected[lane] + Scores[sh, target, slot, {injection}] * {v} * column_factor[lane]"
        )
    lines, gc = _load(context, state.column_gate, "source_j", "c", "gate")
    _append(body, "                ", lines)
    body += [
        f"                column_factor[lane] = column_factor[lane] * {gc}",
        f"for lane in T.Parallel({width}):",
        f"    c = tile * {width} + lane",
        f"    if c < {columns}:",
        "        Output[token, head, c] = result[lane] * column_factor[lane] + injected[lane]",
    ]
    annotations = list(context.annotations) + [
        _annotation("Carry", carry),
        _annotation("Scores", scores),
        _annotation("Weighted", weighted),
        _annotation("Output", output),
    ]
    _compile(annotations, body, batch_heads * length * tiles, device=carry.device)(
        *context.pointers, carry, scores, weighted, output
    )
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
            score_cache[key] = _shared_scores(
                state, carry, query, inputs, sequence_length, heads, term.prior
            )
        outputs[output.name] = _apply_shared_scores(
            state,
            carry,
            query,
            score_cache[key],
            inputs,
            sequence_length,
            heads,
            term.prior,
            output.output_dtype,
        )
    return outputs
