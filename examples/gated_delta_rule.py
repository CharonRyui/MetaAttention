from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    ElementwiseScale,
    Exp,
    HeadMapping,
    Input,
    MatrixReadout,
    OuterProduct,
    PropagationComposition,
    RankOneDelta,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)


def gated_delta_rule(
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Illustrative generic-IR GDN profile; implementation follows the SPEC."""
    device = torch.device(device)
    batch, query_heads, value_heads, length, dim = 1, 1, 2, 64, 128
    query = torch.randn(
        batch,
        query_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(
        batch,
        value_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = (-torch.rand(batch, value_heads, length, device=device)).requires_grad_()
    beta = torch.rand(batch, value_heads, length, device=device, requires_grad=True)
    initial_state = torch.zeros(
        batch,
        value_heads,
        dim,
        dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput(
                    "query",
                    ("batch", "query_heads", "sequence", "key_dim"),
                    torch.bfloat16,
                ),
                TensorInput(
                    "key", ("batch", "key_heads", "sequence", "key_dim"), torch.bfloat16
                ),
                TensorInput(
                    "value",
                    ("batch", "value_heads", "sequence", "value_dim"),
                    torch.bfloat16,
                ),
                TensorInput(
                    "gate", ("batch", "state_heads", "sequence"), torch.float32
                ),
                TensorInput(
                    "beta", ("batch", "state_heads", "sequence"), torch.float32
                ),
            ),
            states=(
                StateSpec("memory", ("batch", "state_heads", "key_dim", "value_dim")),
            ),
            transition=StateTransition(
                "memory",
                PropagationComposition(
                    (
                        ElementwiseScale(Exp(Input("gate"))),
                        RankOneDelta(Input("key"), Input("beta")),
                    )
                ),
                OuterProduct(Input("key"), Input("value") * Input("beta")),
            ),
            readout=MatrixReadout("memory", Input("query") * (128**-0.5)),
            head_mapping=HeadMapping(
                "query_heads", "key_heads", "value_heads", "state_heads"
            ),
        )
    )
    output, final_state = operator(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        initial_state={"memory": initial_state},
        return_final_state=True,
    )
    (output.float().square().mean() + final_state["memory"].square().mean()).backward()
    return output, final_state["memory"]


if __name__ == "__main__":
    result, state = gated_delta_rule()
    print(f"output={tuple(result.shape)} final_state={tuple(state.shape)}")
