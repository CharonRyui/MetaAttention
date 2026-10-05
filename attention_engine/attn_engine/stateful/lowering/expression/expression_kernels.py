"""Private Triton lowering for pointwise Stateful expression DAGs.

The generated kernel keeps the expression association authored by the IR and loads
BindingDescriptor storage directly.  Metadata is passed as scalar launch arguments;
no expanded/head-repeated input is created.  This module is intentionally private:
its stable seam is ``launch_expression`` used by the sibling ``expression`` module.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence

import hashlib
import linecache

import torch
from ....stateful_operator import Batch, HeadRole, Sequence as SequenceRole
from ...runtime.bindings import _PACKED_TOKEN

try:  # Keep CPU/import-only environments usable.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None


_CACHE: dict[tuple[Any, ...], Any] = {}


def _descriptor_cache_key(source: Any) -> tuple[Any, ...]:
    """Return metadata that changes a descriptor's generated address formula.

    Runtime strides remain launch arguments, so they deliberately do not belong
    in the key.  The role order, extents, grouping, and other values below are
    embedded in the generated source and therefore must distinguish kernels.
    """
    if not _descriptor(source):
        return ("tensor",)
    return (
        "descriptor",
        tuple(source.source_roles),
        tuple(source.feature_roles),
        tuple(int(extent) for extent in source.source_shape),
        int(source.target_heads),
        int(source.head_group_size),
        source.mode,
        None if source.sequence_length is None else int(source.sequence_length),
        int(source.storage_offset),
    )


def _kernel_cache_key(
    compiled: Any,
    sources: Sequence[Any],
    outputs: Sequence[torch.Tensor],
    block: int,
) -> tuple[Any, ...]:
    """Identify generated source without retaining tensor identities.

    Node expressions are frozen/hashable IR values.  Keeping the expressions,
    node edges, and roots in the key distinguishes operators, constants, and
    output ordering even when their node counts and shapes happen to match.
    """
    expression, roots = compiled._kernel_identity
    return (
        expression,
        roots,
        tuple(t.dtype for t in (s.tensor if _descriptor(s) else s for s in sources)),
        tuple(o.dtype for o in outputs),
        tuple(_logical_shape(s) for s in sources),
        tuple(_descriptor_cache_key(s) for s in sources),
        int(block),
    )


def _descriptor(source: Any) -> bool:
    return hasattr(source, "source_roles") and hasattr(source, "source_strides")


def _logical_shape(source: Any) -> tuple[int, ...]:
    if _descriptor(source):
        roles = tuple(source.feature_roles)
        extents = []
        for role in roles:
            if role in source.source_roles:
                extents.append(int(source.source_shape[source.source_roles.index(role)]))
            else:
                extents.append(1)
        return (int(source.token_count), int(source.target_heads), *extents)
    return tuple(source.shape)


def _op_source(nodes: Sequence[Any], roots: Sequence[int], inputs: Sequence[str], constants: Sequence[str]) -> str:
    """Emit a scalar-free vector program; all values are tl vectors."""
    lines: list[str] = []
    for i, node in enumerate(nodes):
        kind = node.kind
        if kind == "input":
            lines.append(f"    v{i} = load_{inputs.index(node.expression.name)}()")
        elif kind == "constant":
            dtype = "tl.bfloat16" if node.expression.dtype == torch.bfloat16 else "tl.float32"
            lines.append(f"    v{i} = tl.full((BLOCK,), {node.expression.value!r}, {dtype})")
        elif kind == "state":
            lines.append(f"    v{i} = load_state_{i}()")
        elif kind == "add":
            lines.append(f"    v{i} = v{node.children[0]} + v{node.children[1]}")
        elif kind == "multiply":
            lines.append(f"    v{i} = v{node.children[0]} * v{node.children[1]}")
        elif kind == "divide":
            lines.append(f"    v{i} = v{node.children[0]} / v{node.children[1]}")
        elif kind == "outer":
            raise NotImplementedError("Outer requires a feature-axis tile and is lowered by chunked readout")
        elif kind == "contract":
            raise NotImplementedError("Contract is a reduction and is a materialization cut")
        elif kind == "shift":
            raise NotImplementedError("Shift requires sequence metadata and is lowered by chunked readout")
        elif kind == "softmax":
            raise NotImplementedError("Softmax is a reduction and is a materialization cut")
        elif kind == "negate":
            lines.append(f"    v{i} = -v{node.children[0]}")
        elif kind == "exp":
            lines.append(f"    v{i} = tl.exp(v{node.children[0]}.to(tl.float32))")
        elif kind == "log":
            lines.append(f"    v{i} = tl.log(v{node.children[0]}.to(tl.float32))")
        elif kind == "sigmoid":
            lines.append(f"    x{i} = v{node.children[0]}.to(tl.float32); v{i} = 1.0 / (1.0 + tl.exp(-x{i}))")
        elif kind == "sqrt":
            lines.append(f"    v{i} = tl.sqrt(v{node.children[0]}.to(tl.float32))")
        elif kind == "reciprocal":
            lines.append(f"    v{i} = 1.0 / v{node.children[0]}.to(tl.float32)")
        elif kind == "clamp":
            lines.append(f"    v{i} = tl.maximum(tl.minimum(v{node.children[0]}.to(tl.float32), {node.expression.maximum!r}), {node.expression.minimum!r})")
        else:
            raise NotImplementedError(f"unsupported fused node {kind}")
    for out, root in enumerate(roots):
        lines.append(f"    tl.store(O{out} + p, v{root}, mask=mask)")
    return "\n".join(lines)


def _make_kernel(compiled: Any, sources: Sequence[Any], states: Sequence[Any], roots: Sequence[int], out_dtypes: Sequence[torch.dtype], block: int):
    if triton is None:
        return None
    input_nodes = [n for n in compiled._nodes if n.kind == "input"]
    args = ["O" + str(i) for i in range(len(roots))]
    args += ["TOTAL", "BLOCK: tl.constexpr"]
    defs: list[str] = []
    body = _op_source(compiled._nodes, roots, [n.expression.name for n in input_nodes], [])
    for i, source in enumerate(sources):
        args.append(f"X{i}")
        if _descriptor(source):
            for axis in range(len(source.source_strides)):
                args.append(f"X{i}_s{axis}")
            roles = tuple(source.source_roles)
            feature_extents = [
                int(source.source_shape[roles.index(role)]) if role in roles else 1
                for role in source.feature_roles
            ]
            q = "p"
            terms: list[str] = []
            for feature, extent in reversed(tuple(zip(source.feature_roles, feature_extents))):
                idx = f"(({q}) % {extent})"
                if feature in roles:
                    terms.append(f"({idx}) * X{i}_s{roles.index(feature)}")
                q = f"(({q}) // {extent})"
            head = f"(({q}) % {int(source.target_heads)})"
            q = f"(({q}) // {int(source.target_heads)})"
            head_role = next((r for r in roles if isinstance(r, HeadRole)), None)
            if head_role is not None:
                axis = roles.index(head_role)
                terms.append(f"(({head}) // {int(source.head_group_size)}) * X{i}_s{axis}")
            if source.mode == "dense":
                length = int(source.sequence_length or 1)
                local, batch = f"(({q}) % {length})", f"(({q}) // {length})"
                if SequenceRole in roles:
                    terms.append(f"{local} * X{i}_s{roles.index(SequenceRole)}")
                if Batch in roles:
                    terms.append(f"{batch} * X{i}_s{roles.index(Batch)}")
            else:
                token_axis = roles.index(_PACKED_TOKEN)
                terms.append(f"p * X{i}_s{token_axis}")
            offset = str(int(source.storage_offset))
            if terms:
                offset += " + " + " + ".join(terms)
            defs.append(f"    vload{i} = tl.load(X{i} + {offset}, mask=mask, other=0)")
        else:
            defs.append(f"    vload{i} = tl.load(X{i} + p, mask=mask, other=0)")
        body = body.replace(f"load_{i}()", f"vload{i}")
    signature = ", ".join(args)
    source_text = f"""import triton
import triton.language as tl
@triton.jit
def fused({signature}):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < TOTAL
{chr(10).join(defs)}
{body}
"""
    namespace: dict[str, Any] = {}
    filename = f"<stateful-expression-{hashlib.sha1(source_text.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source_text), None, source_text.splitlines(True), filename)
    code = compile(source_text, filename, "exec")
    exec(code, namespace)
    return namespace["fused"]


def launch_expression(compiled: Any, bindings: Mapping[str, Any], states: Mapping[Any, Any], output_dtype: torch.dtype | None = None) -> Any:
    """Launch one generated Triton kernel for eligible descriptor pointwise DAGs.

    Returns ``None`` when the invocation is not CUDA or includes a reduction/cut;
    callers must then use their explicit CPU/reference path.  CUDA callers are
    never silently routed through Torch: unsupported CUDA graphs raise.
    """
    sources = []
    for node in compiled._nodes:
        if node.kind == "input":
            source = bindings[node.expression.name]
            if not isinstance(source, torch.Tensor) and not _descriptor(source):
                raise TypeError("unsupported expression binding")
            sources.append(source)
    if not sources:
        return None
    tensors = [s if isinstance(s, torch.Tensor) else s.tensor for s in sources]
    if not any(t.is_cuda for t in tensors):
        return None
    if any(node.kind in ("outer", "contract", "shift", "softmax") for node in compiled._nodes):
        raise RuntimeError("CUDA expression DAG contains a materialization cut; use the lowering that owns the cut")
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("CUDA expression bindings must share a device")
    shape = _logical_shape(sources[0])
    for source in sources[1:]:
        source_tensor = source.tensor if _descriptor(source) else source
        if _logical_shape(source) != shape and source_tensor.numel() != 1:
            raise ValueError("fused expression inputs must have matching logical shapes")
    total = 1
    for n in shape:
        total *= int(n)
    device = tensors[0].device
    result_dtype = output_dtype or (sources[0].dtype if hasattr(sources[0], "dtype") else tensors[0].dtype)
    outputs = [torch.empty(shape, device=device, dtype=result_dtype) for _ in compiled._roots]
    key = _kernel_cache_key(compiled, sources, outputs, 128)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _make_kernel(compiled, sources, (), compiled._roots, [o.dtype for o in outputs], 128)
        _CACHE[key] = kernel
    if kernel is None:
        raise RuntimeError("Triton is unavailable for CUDA expression lowering")
    kwargs = {f"O{i}": output for i, output in enumerate(outputs)}
    kwargs.update({f"X{i}": source.tensor if _descriptor(source) else source for i, source in enumerate(sources)})
    for i, source in enumerate(sources):
        if _descriptor(source):
            kwargs.update({f"X{i}_s{axis}": int(stride) for axis, stride in enumerate(source.source_strides)})
    kernel[(triton.cdiv(total, 128),)](**kwargs, TOTAL=total, BLOCK=128)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


__all__ = ["launch_expression"]
