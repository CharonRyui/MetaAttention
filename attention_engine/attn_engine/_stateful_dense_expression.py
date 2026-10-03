"""Authored pointwise expressions loaded directly from dense input bindings."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import linecache
from typing import Any

import torch

from ._stateful_bindings import DenseRawBindingRecipe, compile_dense_binding
from ._stateful_native import _load_dense


@dataclass(frozen=True)
class DenseExpression:
    loader: Any
    sources: tuple[Any, ...]
    feature: Any
    bindings: tuple[DenseRawBindingRecipe, ...]

    def bind(self, inputs, heads):
        pointers, strides, groups = [], [], []
        for binding in self.bindings:
            raw, source_strides, group = binding.bind(inputs[binding.name], heads)
            pointers.append(raw)
            strides.extend(source_strides)
            groups.append(group)
        return self.loader, tuple(pointers), tuple(strides), tuple(groups)


def compile_dense_expression(expression, specs, feature, *, factors=()):
    """Return a raw loader, or None for an expression requiring a reduction cut.

    Arithmetic rounds at every authored node. Extra factors are multiplied at
    the FP32 transition seam, not reassociated into the authored expression.
    """
    from .stateful_operator import (
        Add, Clamp, Constant, Divide, Exp, Input, Log, Multiply, Negate,
        Reciprocal, Sigmoid, Sqrt,
    )

    sources, lines, nodes = [], [], {}

    def visit(node):
        if node in nodes:
            return nodes[node]
        index = len(nodes)
        # Reserve before visiting children so the emitted name is unique.
        nodes[node] = None
        if isinstance(node, Input):
            spec = specs[node.name]
            from .stateful_operator import FeatureRole
            if any(isinstance(role, FeatureRole) and role != feature for role in spec.roles):
                raise ValueError("expression carries a different Feature")
            if spec not in sources:
                sources.append(spec)
            source = sources.index(spec)
            dtype, scalar = spec.dtype, False
            stride = ", ".join(f"STRIDES[{source * 4 + axis}]" for axis in range(4))
            lines.append(f"    s{source}: tl.constexpr = ({stride})")
            lines.append(f"    g{source}: tl.constexpr = GROUPS[{source}]")
            code = (f"_load_dense(P[{source}], seq, head, t, x, SIZE, "
                    f"s{source}, g{source}, VALID)")
        elif isinstance(node, Constant):
            dtype, scalar = node.dtype, True
            code = f"tl.full((), {node.value!r}, tl.float32)"
            if dtype == torch.bfloat16:
                code += ".to(tl.bfloat16).to(tl.float32)"
        elif isinstance(node, (Add, Multiply, Divide)):
            left, ld, ls = visit(node.left)
            right, rd, rs = visit(node.right)
            dtype = (rd if ls and not rs else ld if rs and not ls
                     else torch.promote_types(ld, rd))
            scalar = ls and rs
            symbol = "+" if isinstance(node, Add) else "*" if isinstance(node, Multiply) else "/"
            code = f"({left} {symbol} {right})"
            if dtype == torch.bfloat16:
                code += ".to(tl.bfloat16).to(tl.float32)"
        elif isinstance(node, (Negate, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp)):
            operand, dtype, scalar = visit(node.operand)
            if isinstance(node, Negate):
                code = f"(-{operand})"
            else:
                dtype = torch.float32
                if isinstance(node, Exp):
                    code = f"tl.exp({operand})"
                elif isinstance(node, Log):
                    code = f"tl.log({operand})"
                elif isinstance(node, Sigmoid):
                    code = f"(1.0 / (1.0 + tl.exp(-{operand})))"
                elif isinstance(node, Sqrt):
                    code = f"tl.sqrt({operand})"
                elif isinstance(node, Reciprocal):
                    code = f"(1.0 / {operand})"
                else:
                    code = f"tl.maximum(tl.minimum({operand}, {node.maximum!r}), {node.minimum!r})"
        else:
            raise ValueError("expression requires a materialization cut")
        name = f"v{index}"
        lines.append(f"    {name} = {code}")
        nodes[node] = name, dtype, scalar
        return nodes[node]

    try:
        root, _, _ = visit(expression)
        for factor in factors:
            other, _, _ = visit(factor)
            name = f"f{len(lines)}"
            lines.append(f"    {name} = {root} * {other}")
            root = name
    except ValueError:
        return None
    source = ("import triton\nimport triton.language as tl\n"
              "@triton.jit\n"
              "def load(P, seq, head, t, x, VALID, SIZE: tl.constexpr, "
              "STRIDES: tl.constexpr, GROUPS: tl.constexpr):\n"
              + "\n".join(lines) + f"\n    return tl.where(VALID & (x < SIZE), {root}, 0.0)\n")
    name = "load_" + hashlib.sha256(source.encode()).hexdigest()
    source = source.replace("def load(", f"def {name}(")
    filename = f"<stateful-dense-load-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"_load_dense": _load_dense}
    exec(compile(source, filename, "exec"), namespace)
    return DenseExpression(namespace[name], tuple(sources), feature,
                           tuple(compile_dense_binding(spec, feature) for spec in sources))
