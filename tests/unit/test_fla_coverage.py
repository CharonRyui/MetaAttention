"""CPU expressibility and numeric checks for the FLA coverage Programs.

Every factory in ``examples/fla_coverage.py`` must construct, type-check, compile,
and execute; the two Programs that cannot be checked against a library operator
directly are compared with independent token recurrences.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from attn_engine import Target
from examples import fla_coverage as coverage

pytestmark = pytest.mark.unit

BATCH, HEADS, LENGTH, DIM, DIM_VALUE = 1, 2, 4, 3, 2


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _role_sizes(operator, heads=HEADS):
    from attn_engine import FeatureRole, HeadRole

    return {
        role: (DIM_VALUE if role.name == "value_dim" else DIM)
        if isinstance(role, FeatureRole)
        else heads
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if isinstance(role, (HeadRole, FeatureRole))
    }


CASES = {
    "normalized_linear_attention": dict(
        heads=HEADS,
        kwargs=dict(
            batch=BATCH,
            heads=HEADS,
            seqlen=LENGTH,
            dim=DIM,
            dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
        },
    ),
    "hgrn": dict(
        heads=1,
        kwargs=dict(
            batch=BATCH,
            heads=1,
            seqlen=LENGTH,
            dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "value": torch.randn(BATCH, 1, LENGTH, DIM_VALUE, generator=generator),
            "gate": torch.randn(BATCH, 1, LENGTH, DIM_VALUE, generator=generator).clamp(
                -1, 0
            ),
        },
    ),
    "rwkv6_recurrence": dict(
        kwargs=dict(
            batch=BATCH,
            heads=HEADS,
            seqlen=LENGTH,
            dim=DIM,
            dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "decay": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator).clamp(
                -1, 0
            ),
            "mix": torch.rand(BATCH, HEADS, LENGTH, DIM, generator=generator),
        },
    ),
    "retention_attention": dict(
        kwargs=dict(
            batch=BATCH,
            heads=HEADS,
            seqlen=LENGTH,
            dim=DIM,
            dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "decay": coverage.retention_decay(HEADS)
            .expand(BATCH, HEADS, LENGTH)
            .contiguous(),
        },
    ),
    "rwkv7_state": dict(
        kwargs=dict(
            batch=BATCH,
            heads=HEADS,
            seqlen=LENGTH,
            dim=DIM,
            dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "removal": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "decay": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator).clamp(
                -1, 0
            ),
            "beta": torch.rand(BATCH, HEADS, LENGTH, generator=generator),
        },
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_coverage_factory_compiles_and_runs_on_cpu(name):
    case = CASES[name]
    operator = coverage.FACTORIES[name](**case["kwargs"])
    compiled = operator.compile(
        role_sizes=_role_sizes(operator, case.get("heads", HEADS)),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    result = compiled(**case["inputs"](torch.Generator().manual_seed(3)))
    output = result.outputs[0]
    assert output.shape == (BATCH, case.get("heads", HEADS), LENGTH, DIM_VALUE)
    assert torch.isfinite(output).all()


def test_rwkv6_recurrence_matches_token_recurrence():
    inputs = CASES["rwkv6_recurrence"]["inputs"](torch.Generator().manual_seed(4))
    operator = coverage.rwkv6_recurrence(
        BATCH, HEADS, LENGTH, DIM, DIM_VALUE, dtype=torch.float32
    )
    compiled = operator.compile(
        role_sizes=_role_sizes(operator), target=Target(), mode="dense", optimized=True
    )
    actual = compiled(**inputs).outputs["output"].float()

    memory = torch.zeros(BATCH, HEADS, DIM, DIM_VALUE)
    outputs = []
    previous = torch.zeros(BATCH, HEADS, DIM)
    for token in range(LENGTH):
        shifted = previous * inputs["mix"][:, :, token] + inputs["key"][:, :, token] * (
            1 - inputs["mix"][:, :, token]
        )
        memory = memory * (-inputs["decay"][:, :, token]).exp().unsqueeze(-1)
        removal = torch.einsum("bhf,bhfg->bhg", shifted, memory)
        memory = memory - torch.einsum("bhf,bhg->bhfg", shifted, removal)
        memory = memory + torch.einsum(
            "bhf,bhg->bhfg", shifted, inputs["value"][:, :, token]
        )
        outputs.append(
            torch.einsum("bhf,bhfg->bhg", inputs["query"][:, :, token], memory)
        )
        previous = inputs["key"][:, :, token]
    expected = torch.stack(outputs, dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rwkv7_state_matches_dplr_recurrence():
    inputs = CASES["rwkv7_state"]["inputs"](torch.Generator().manual_seed(5))
    operator = coverage.rwkv7_state(
        BATCH, HEADS, LENGTH, DIM, DIM_VALUE, dtype=torch.float32
    )
    compiled = operator.compile(
        role_sizes=_role_sizes(operator), target=Target(), mode="dense", optimized=True
    )
    actual = compiled(**inputs).outputs["output"].float()

    memory = torch.zeros(BATCH, HEADS, DIM, DIM_VALUE)
    outputs = []
    for token in range(LENGTH):
        beta = inputs["beta"][:, :, token]
        memory = memory * inputs["decay"][:, :, token].exp().unsqueeze(-1)
        contraction = torch.einsum(
            "bhf,bhfg->bhg", inputs["removal"][:, :, token], memory
        )
        memory = memory - beta.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], contraction
        )
        memory = memory + beta.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], inputs["value"][:, :, token]
        )
        outputs.append(
            torch.einsum("bhf,bhfg->bhg", inputs["query"][:, :, token], memory)
        )
    expected = torch.stack(outputs, dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fla_coverage_table_covers_every_installed_fla_op_module():
    import fla.ops

    root = pathlib.Path(fla.ops.__file__).parent
    modules = {
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "__init__.py").exists()
        and not path.name.startswith("_")
    } - {"common", "utils"}
    covered = {
        entry[0].split(".")[2]
        for entry in coverage.FLA_COVERAGE
        if len(entry[0].split(".")) > 2
    }
    assert not modules - covered, sorted(modules - covered)
