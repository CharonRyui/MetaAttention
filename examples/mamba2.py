from attn_engine import (
    AlgorithmIR,
    CompileOptions,
    ElementwiseScale,
    Exp,
    HeadMapping,
    Input,
    MatrixReadout,
    OuterProduct,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)
from autotuner.arch import get_attn_device
import torch

"""Illustrative selective-state profile authored from generic Stateful Operator IR.

The explicit equation and compiler contract are defined by
``docs/stateful-operator-ir-spec.md``; this name is an acceptance profile, not
a lowering dispatch key.
"""


def mamba2(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, tune=False):
    if HK is None:
        HK = HQ
    if HV is None:
        HV = HQ
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput(
                "query", ("batch", "query_heads", "sequence", "key_dim"), dtype
            ),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), dtype),
            TensorInput(
                "value", ("batch", "value_heads", "sequence", "value_dim"), dtype
            ),
            TensorInput("delta", ("batch", "state_heads", "sequence"), torch.float32),
            TensorInput("A", ("one", "state_heads"), dtype),
            TensorInput("dt", ("batch", "state_heads", "sequence"), dtype),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            ElementwiseScale(Exp(Input("delta") * Input("A"))),
            OuterProduct(Input("key"), Input("value") * Input("dt")),
        ),
        readout=MatrixReadout("memory", Input("query")),
        head_mapping=HeadMapping(
            "query_heads", "key_heads", "value_heads", "state_heads"
        ),
    )
    return StatefulOperator(
        algorithm,
        compile_options=CompileOptions(
            tune=tune,
            tune_filename=f"tuned_config/{get_attn_device().name}/mamba2",
            tune_backward=tune,
        ),
    )
