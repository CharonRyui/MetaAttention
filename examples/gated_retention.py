import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    CompileOptions,
    Exp,
    HeadMapping,
    Input,
    ProductFactor,
    ProductInjection,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)
from autotuner.arch import get_attn_device


def gated_retention(B, H, S, D, DV, dtype=torch.bfloat16, tune=False):
    """GLA acceptance profile expressed only with generic IR nodes."""
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), dtype),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), dtype),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), dtype),
            TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("gate"))),),
            injections=(ProductInjection((ProductFactor(Input("key"), ("key_dim",)), ProductFactor(Input("value"), ("value_dim",)))),),
        ),
        readouts=(StateContraction("output", "memory", Input("query") * (D**-0.5), "key_dim", dtype),),
        head_mapping=HeadMapping({"query_heads": "state_heads", "key_heads": "state_heads", "value_heads": "state_heads"}),
    )
    return StatefulOperator(algorithm, compile_options=CompileOptions(tune=tune, tune_filename=f"tuned_config/{get_attn_device().name}/simple_gla", tune_backward=tune))
