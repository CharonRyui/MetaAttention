from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Exp,
    HeadMapping,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    StateTuple,
    TensorInput,
)


def gated_delta_rule(device: str | torch.device = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Gated Delta Rule expressed only with generic Stateful Operator nodes."""
    device = torch.device(device)
    batch, query_heads, state_heads, length, dim = 1, 1, 2, 64, 128
    query = torch.randn(batch, query_heads, length, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(batch, state_heads, length, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    gate = (-torch.rand(batch, state_heads, length, device=device)).requires_grad_()
    beta = torch.rand(batch, state_heads, length, device=device, requires_grad=True)
    initial = torch.zeros(batch, state_heads, dim, dim, device=device, dtype=torch.float32, requires_grad=True)

    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", ("batch", "query_heads", "sequence", "key_dim"), torch.bfloat16),
            TensorInput("key", ("batch", "key_heads", "sequence", "key_dim"), torch.bfloat16),
            TensorInput("value", ("batch", "value_heads", "sequence", "value_dim"), torch.bfloat16),
            TensorInput("gate", ("batch", "state_heads", "sequence"), torch.float32),
            TensorInput("beta", ("batch", "state_heads", "sequence"), torch.float32),
        ),
        states=(StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("gate"))),
                RankOnePropagation("key_dim", Input("key"), Input("key"), -Input("beta")),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), ("key_dim",)),
                        ProductFactor(Input("value") * Input("beta"), ("value_dim",)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction("output", "memory", Input("query") * (dim**-0.5), "key_dim", torch.bfloat16),
        ),
        head_mapping=HeadMapping(
            mappings={
                "query_heads": "state_heads",
                "key_heads": "state_heads",
                "value_heads": "state_heads",
            }
        ),
    )
    result = StatefulOperator(algorithm)(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    assert result.final_state is not None
    output, final = result.outputs["output"], result.final_state["memory"]
    (output.float().square().mean() + final.square().mean()).backward()
    return output, final


if __name__ == "__main__":
    result, state = gated_delta_rule()
    print(f"output={tuple(result.shape)} final_state={tuple(state.shape)}")
