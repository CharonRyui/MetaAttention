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

"""
Example of simple gla/gated retnet

fwd:
input:
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, DV]
decay: [B, H, T]
...custom_inputs

output:
O: [B, H, T, DV]
"""


def gated_retention(B, H, S, D, DV, dtype=torch.bfloat16, tune=False):
    scale = 1 / D**0.5
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), dtype),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), dtype),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), dtype),
            TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory", ElementwiseScale(Exp(Input("gate"))), OuterProduct(Input("key"), Input("value"))
        ),
        readout=MatrixReadout("memory", Input("query") * scale),
        head_mapping=HeadMapping("query_heads", "key_heads", "value_heads", "state_heads"),
    )
    return StatefulOperator(
        algorithm,
        compile_options=CompileOptions(
            tune=tune,
            tune_filename=f"tuned_config/{get_attn_device().name}/simple_gla",
            tune_backward=tune,
        ),
    )
