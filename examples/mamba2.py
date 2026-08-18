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


def mamba2(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, tune=False):
    """Selective-state acceptance profile expressed only with generic IR nodes."""
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), dtype),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), dtype),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), dtype),
            TensorInput("A", ("one", "state_heads", "key_dim"), dtype),
            TensorInput("dt", ("batch", "state_heads", "sequence"), dtype),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("dt") * Input("A")), ("key_dim",)),),
            injections=(ProductInjection((ProductFactor(Input("dt") * Input("key"), ("key_dim",)), ProductFactor(Input("value"), ("value_dim",)))),),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), "key_dim", dtype),),
        head_mapping=HeadMapping({"query_heads": "state_heads", "key_heads": "state_heads", "value_heads": "state_heads"}),
    )
    return StatefulOperator(algorithm, compile_options=CompileOptions(tune=tune, tune_filename=f"tuned_config/{get_attn_device().name}/mamba2", tune_backward=tune))
