import torch

from attn_engine import (
    Batch,
    FeatureRole,
    HeadRole,
    One,
    AlgorithmIR,
    Sequence,
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

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def mamba2(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, tune=False):
    """Selective-state acceptance profile expressed only with generic IR nodes."""
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("A", (One, STATE_HEADS, KEY), dtype),
            TensorInput("dt", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("dt") * Input("A")), (KEY,)),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("dt") * Input("key"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(StateContraction("output", "memory", Input("query"), KEY, dtype),),
        head_mapping=HeadMapping(
            {QUERY_HEADS: STATE_HEADS, KEY_HEADS: STATE_HEADS, VALUE_HEADS: STATE_HEADS}
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
