"""Native-layout identity contractions with authored input arithmetic at load."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ._stateful_bindings import DenseRawBindingRecipe, compile_dense_binding


@dataclass(frozen=True)
class NativeNormalizedRecipe:
    output: str
    output_dtype: torch.dtype
    memory: str
    normalizer: str
    query: DenseRawBindingRecipe
    key: DenseRawBindingRecipe
    denominator_key: DenseRawBindingRecipe
    value: DenseRawBindingRecipe
    query_scale: float | None
    epsilon: float
    prior_memory: bool
    prior_normalizer: bool


def compile_native_normalized(algorithm, normalized):
    """Prove native vectors, identity updates and a shared scalar-scaled query."""
    from .stateful_operator import Batch, Constant, FeatureRole, Identity, Input, Multiply, Sequence

    if normalized is None or len(algorithm.states) != 2:
        return None
    readout, numerator, denominator, epsilon = normalized
    specs = {spec.name: spec for spec in algorithm.inputs}
    states = {state.name: state for state in algorithm.states}
    memory, normalizer = states[numerator.state], states[denominator.state]
    query = numerator.operand
    scale = None
    if isinstance(query, Multiply):
        if isinstance(query.left, Constant) and isinstance(query.right, Input):
            constant, query = query.left, query.right
        elif isinstance(query.right, Constant) and isinstance(query.left, Input):
            constant, query = query.right, query.left
        else:
            return None
        scale = constant.value
    if not isinstance(query, Input):
        return None
    keys = []
    value = None
    for name in (memory.name, normalizer.name):
        transition = next(item for item in algorithm.transitions if item.state == name)
        if len(transition.injections) != 1:
            return None
        if any(not isinstance(node, Identity) for node in transition.propagations):
            return None
        key = None
        for factor in transition.injections[0].factors:
            if not isinstance(factor.expression, Input):
                return None
            if factor.roles == (memory.feature_roles[0],):
                key = specs[factor.expression.name]
            elif name == memory.name and factor.roles == (memory.feature_roles[1],):
                value = specs[factor.expression.name]
            else:
                return None
        if key is None:
            return None
        keys.append(key)
    if value is None:
        return None
    query = specs[query.name]
    for spec, feature in ((query, memory.feature_roles[0]), (keys[0], memory.feature_roles[0]),
                          (keys[1], memory.feature_roles[0]), (value, memory.feature_roles[1])):
        features = tuple(role for role in spec.roles if isinstance(role, FeatureRole))
        if features != (feature,) or Batch not in spec.roles or Sequence not in spec.roles:
            return None
    return NativeNormalizedRecipe(
        readout.name, readout.output_dtype, memory.name, normalizer.name,
        compile_dense_binding(query, memory.feature_roles[0]),
        compile_dense_binding(keys[0], memory.feature_roles[0]),
        compile_dense_binding(keys[1], memory.feature_roles[0]),
        compile_dense_binding(value, memory.feature_roles[1]),
        scale, epsilon, numerator.prior, denominator.prior,
    )


@triton.jit
def _load_dense(P, seq, head, t, x, SIZE: tl.constexpr, STRIDES: tl.constexpr,
                GROUP: tl.constexpr, VALID, SCALE: tl.constexpr = None):
    dtype: tl.constexpr = P.dtype.element_ty
    result = tl.load(P + seq * STRIDES[0] + (head // GROUP) * STRIDES[1]
                     + t * STRIDES[2] + x * STRIDES[3], VALID & (x < SIZE), 0).to(tl.float32)
    if SCALE is not None:
        # CUDA TensorIterator converts scalar to the vector computation dtype,
        # then rounds the authored multiply before the FP32 contraction seam.
        scalar = tl.full((), SCALE, tl.float32).to(dtype).to(tl.float32)
        result = (result * scalar).to(dtype).to(tl.float32)
    if dtype == tl.bfloat16:
        # Stop Triton's source-bitwidth analysis at the FP32 contraction seam.
        # Otherwise a BF16 load selects kWidth=2 instead of the staged FP32
        # kWidth=1 and reorders the summary accumulation. This bitwise move
        # is non-pure only to keep that analysis from walking through it.
        result = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=f,f",
                                         args=[result], dtype=tl.float32,
                                         is_pure=False, pack=1)
    return tl.where(VALID & (x < SIZE), result, 0.0)


def _prepare_normalized_pair(key, den_key, value, initial, den_initial,
                             *, sequence_length, heads, ks, kg, dks, dkg, vs, vg,
                             return_final_state):
    """Own paired summaries/carries and release recursive scratch before readout."""
    from ._stateful_diagonal_kernels import _identity_pairs_native_pair
    from ._stateful_diagonal_scan import _hierarchical_carry_pair

    rows, columns = initial.shape[-2:]
    bh = initial.shape[0] * heads
    chunks = triton.cdiv(sequence_length, 64)
    if chunks == 1:
        # The only exclusive carry is the initial State; no scratch is consumed.
        if not return_final_state:
            return initial, den_initial, None, None
        from ._stateful_diagonal_kernels import _identity_final_native_pair

        final, den_final = torch.empty_like(initial), torch.empty_like(den_initial)
        tiles = triton.cdiv(rows, 32) * triton.cdiv(columns, 32)
        _identity_final_native_pair[(bh, tiles + triton.cdiv(rows, 32))](
            key, den_key, value, initial, den_initial, final, den_final,
            sequence_length, heads, rows, columns, tiles,
            ks, kg, dks, dkg, vs, vg, enable_fp_fusion=False,
        )
        return initial, den_initial, final, den_final
    carry = torch.empty((bh, chunks, rows, columns),
                        device=initial.device, dtype=torch.float32)
    den_carry = torch.empty((bh, chunks, rows, 1),
                            device=den_initial.device, dtype=torch.float32)
    final = torch.empty_like(initial) if return_final_state else initial
    den_final = torch.empty_like(den_initial) if return_final_state else den_initial
    tiles = triton.cdiv(rows, 32) * triton.cdiv(columns, 32)
    _identity_pairs_native_pair[(bh, chunks, tiles + triton.cdiv(rows, 32))](
        key, den_key, value, carry, den_carry,
        sequence_length, heads, rows, columns, chunks, tiles,
        ks, kg, dks, dkg, vs, vg,
    )
    carry, den_carry = _hierarchical_carry_pair(
        carry, den_carry, initial, den_initial, final, den_final,
        write_final=return_final_state,
    )
    return carry, den_carry, final, den_final


def execute_native_normalized(recipe, inputs, initial_states, *,
                              sequence_count, sequence_length, heads, return_final_state):
    from ._stateful_readout_regions import _identity_normalized_region

    bindings = {}

    def bind(binding):
        if binding.name not in bindings:
            bindings[binding.name] = binding.bind(inputs[binding.name], heads)
        return bindings[binding.name]

    query, qs, qg = bind(recipe.query)
    key, ks, kg = bind(recipe.key)
    den_key, dks, dkg = bind(recipe.denominator_key)
    value, vs, vg = bind(recipe.value)
    initial_memory = initial_states[recipe.memory]
    initial_normalizer = initial_states[recipe.normalizer].unsqueeze(-1)
    rows, columns = initial_memory.shape[-2:]
    chunks = triton.cdiv(sequence_length, 64)

    memory, normalizer, final_memory, final_normalizer = _prepare_normalized_pair(
        key, den_key, value, initial_memory, initial_normalizer,
        sequence_length=sequence_length, heads=heads,
        ks=ks, kg=kg, dks=dks, dkg=dkg, vs=vs, vg=vg,
        return_final_state=return_final_state,
    )
    output = torch.empty(sequence_count * sequence_length, heads, columns,
                         device=query.device, dtype=recipe.output_dtype)
    _identity_normalized_region[(sequence_count * heads, chunks, triton.cdiv(columns, 32))](
        memory, key, key, value, normalizer, den_key, den_key, value, query, query, output,
        sequence_length, heads, chunks, rows, columns, rows, 0, rows, 0, rows, 0,
        recipe.prior_memory, recipe.prior_normalizer,
        recipe.key.name == recipe.denominator_key.name, recipe.epsilon,
        NATIVE=True, KS=ks, KG=kg, DKS=dks, DKG=dkg, VS=vs, VG=vg, QS=qs, QG=qg,
        QUERY_SCALE=recipe.query_scale, num_warps=4, enable_fp_fusion=False,
    )
    finals = ({recipe.memory: final_memory, recipe.normalizer: final_normalizer.squeeze(-1)}
              if return_final_state else None)
    return {recipe.output: output}, finals
