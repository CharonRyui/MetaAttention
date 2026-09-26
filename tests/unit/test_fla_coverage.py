"""CPU expressibility and numeric checks for the FLA coverage Programs.

Every Program the coverage table points at must construct, type-check, compile, and
execute; the Programs whose library operator is not directly callable here are
compared with independent token recurrences written from the mathematics.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from attn_engine import FeatureRole, HeadRole, Target
from examples import fla_coverage
from examples.based_attention import based_attention
from examples.gated_slot_attention import gated_slot_attention
from examples.hgrn import hgrn
from examples.normalized_linear_attention import normalized_linear_attention
from examples.rebased_attention import rebased_attention
from examples.retention_attention import retention_attention, retention_decay
from examples.rwkv6_attention import rwkv6_attention
from examples.rwkv7_state import rwkv7_state

# every Program the coverage table points at, by factory name
FACTORIES = {
    "based_attention": based_attention,
    "gated_slot_attention": gated_slot_attention,
    "hgrn": hgrn,
    "normalized_linear_attention": normalized_linear_attention,
    "rebased_attention": rebased_attention,
    "retention_attention": retention_attention,
    "rwkv6_attention": rwkv6_attention,
    "rwkv7_state": rwkv7_state,
}

pytestmark = pytest.mark.unit

BATCH, HEADS, LENGTH, DIM, DIM_VALUE, DIM_SLOT = 1, 2, 4, 3, 2, 2


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _role_sizes(operator, heads=HEADS):
    sizes: dict[object, int | None] = {}
    for spec in (*operator.algorithm.inputs, *operator.algorithm.states):
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes[role] = heads
            elif isinstance(role, FeatureRole):
                if "_x_" in role.name:
                    sizes[role] = None
                elif role.name == "value_dim":
                    sizes[role] = DIM_VALUE
                elif role.name == "slot_dim":
                    sizes[role] = DIM_SLOT
                elif role.name == "unit_dim":
                    sizes[role] = 1
                else:
                    sizes[role] = DIM
    for role, size in list(sizes.items()):
        if size is None:
            sizes[role] = sizes[role.left] * sizes[role.right]
    return sizes


def _run(factory, inputs, kwargs, heads=HEADS):
    operator = factory(**kwargs)
    compiled = operator.compile(
        role_sizes=_role_sizes(operator, heads),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    return compiled(**inputs).outputs["output"].float()


def _generator():
    return torch.Generator().manual_seed(0)


def _tensors():
    generator = _generator()
    return (
        torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
        torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
        torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
    )


CASES = {
    "normalized_linear_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
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
            batch=BATCH, heads=1, seqlen=LENGTH, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "value": torch.randn(BATCH, 1, LENGTH, DIM_VALUE, generator=generator),
            "gate": torch.randn(BATCH, 1, LENGTH, DIM_VALUE, generator=generator).clamp(-1, 0),
        },
    ),
    "rwkv6_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "decay": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator).clamp(-1, 0),
            "bonus": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
        },
    ),
    "retention_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "decay": retention_decay(HEADS)
            .expand(BATCH, HEADS, LENGTH)
            .contiguous(),
        },
    ),
    "rwkv7_state": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "removal": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "decay": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator).clamp(-1, 0),
            "beta": torch.rand(BATCH, HEADS, LENGTH, generator=generator),
        },
    ),
    "based_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
        },
    ),
    "rebased_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_value=DIM_VALUE,
            dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
        },
    ),
    "gated_slot_attention": dict(
        kwargs=dict(
            batch=BATCH, heads=HEADS, seqlen=LENGTH, dim=DIM, dim_slot=DIM_SLOT,
            dim_value=DIM_VALUE, dtype=torch.float32,
        ),
        inputs=lambda generator: {
            "query": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "key": torch.randn(BATCH, HEADS, LENGTH, DIM, generator=generator),
            "slot": torch.randn(BATCH, HEADS, LENGTH, DIM_SLOT, generator=generator),
            "value": torch.randn(BATCH, HEADS, LENGTH, DIM_VALUE, generator=generator),
            "gate": torch.randn(BATCH, HEADS, LENGTH, DIM_SLOT, generator=generator).clamp(-1, 0),
        },
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_coverage_factory_compiles_and_runs_on_cpu(name):
    case = CASES[name]
    output = _run(
        FACTORIES[name],
        case["inputs"](_generator()),
        case["kwargs"],
        case.get("heads", HEADS),
    )
    assert output.shape == (BATCH, case.get("heads", HEADS), LENGTH, DIM_VALUE)
    assert torch.isfinite(output).all()


def test_rwkv6_attention_matches_prior_state_readout_recurrence():
    inputs = CASES["rwkv6_attention"]["inputs"](_generator())
    actual = _run(
        rwkv6_attention,
        inputs,
        CASES["rwkv6_attention"]["kwargs"],
    )
    scale = DIM**-0.5
    state = torch.zeros(BATCH, HEADS, DIM, DIM_VALUE)
    outputs = []
    for token in range(LENGTH):
        outputs.append(
            torch.einsum(
                "bhf,bhfg->bhg", inputs["query"][:, :, token] * scale, state
            )
            + (
                (inputs["query"][:, :, token]
                 * scale
                 * inputs["bonus"][:, :, token]
                 * inputs["key"][:, :, token])
                .sum(-1)
                .unsqueeze(-1)
            )
            * inputs["value"][:, :, token]
        )
        state = state * inputs["decay"][:, :, token].exp().unsqueeze(-1) + torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], inputs["value"][:, :, token]
        )
    torch.testing.assert_close(actual, torch.stack(outputs, dim=2), rtol=1e-5, atol=1e-6)


def test_rebased_attention_matches_squared_attention_recurrence():
    inputs = CASES["rebased_attention"]["inputs"](_generator())
    actual = _run(
        rebased_attention, inputs, CASES["rebased_attention"]["kwargs"]
    )
    scale = DIM**-0.5
    numerator = torch.zeros(BATCH, HEADS, DIM * DIM, DIM_VALUE)
    denominator = torch.zeros(BATCH, HEADS, DIM * DIM)
    outputs = []
    for token in range(LENGTH):
        pair = torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], inputs["key"][:, :, token]
        ).reshape(BATCH, HEADS, DIM * DIM)
        numerator = numerator + pair.unsqueeze(-1) * inputs["value"][:, :, token].unsqueeze(-2)
        denominator = denominator + pair
        query_pair = torch.einsum(
            "bhf,bhg->bhfg",
            inputs["query"][:, :, token] * scale,
            inputs["query"][:, :, token] * scale,
        ).reshape(BATCH, HEADS, DIM * DIM)
        outputs.append(
            (query_pair.unsqueeze(-1) * numerator).sum(-2)
            / ((query_pair * denominator).sum(-1) + 1e-5).unsqueeze(-1)
        )
    torch.testing.assert_close(actual, torch.stack(outputs, dim=2), rtol=1e-5, atol=1e-6)


def test_based_attention_matches_second_order_taylor_recurrence():
    inputs = CASES["based_attention"]["inputs"](_generator())
    actual = _run(
        based_attention, inputs, CASES["based_attention"]["kwargs"]
    )
    scale = DIM**-0.5
    zero = torch.zeros(BATCH, HEADS, DIM_VALUE)
    first = torch.zeros(BATCH, HEADS, DIM, DIM_VALUE)
    second = torch.zeros(BATCH, HEADS, DIM * DIM, DIM_VALUE)
    count = torch.zeros(BATCH, HEADS)
    key_first = torch.zeros(BATCH, HEADS, DIM)
    key_second = torch.zeros(BATCH, HEADS, DIM * DIM)
    outputs = []
    for token in range(LENGTH):
        pair = torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], inputs["key"][:, :, token]
        ).reshape(BATCH, HEADS, DIM * DIM)
        zero = zero + inputs["value"][:, :, token]
        first = first + torch.einsum(
            "bhf,bhg->bhfg", inputs["key"][:, :, token], inputs["value"][:, :, token]
        )
        second = second + pair.unsqueeze(-1) * inputs["value"][:, :, token].unsqueeze(-2)
        count = count + 1
        key_first = key_first + inputs["key"][:, :, token]
        key_second = key_second + pair
        scaled = inputs["query"][:, :, token] * scale
        query_pair = torch.einsum("bhf,bhg->bhfg", scaled, scaled).reshape(
            BATCH, HEADS, DIM * DIM
        )
        numerator = (
            zero
            + (first * scaled.unsqueeze(-1)).sum(-2)
            + 0.5 * (second * query_pair.unsqueeze(-1)).sum(-2)
        )
        denominator = (
            count
            + (key_first * scaled).sum(-1)
            + 0.5 * (key_second * query_pair).sum(-1)
        )
        outputs.append(numerator / (denominator.unsqueeze(-1) + 1e-6))
    torch.testing.assert_close(actual, torch.stack(outputs, dim=2), rtol=1e-5, atol=1e-6)


def test_gated_slot_attention_matches_two_state_softmax_recurrence():
    inputs = CASES["gated_slot_attention"]["inputs"](_generator())
    actual = _run(
        gated_slot_attention,
        inputs,
        CASES["gated_slot_attention"]["kwargs"],
    )
    scale = DIM**-0.5
    key_slot = torch.zeros(BATCH, HEADS, DIM, DIM_SLOT)
    slot_scores = torch.zeros(BATCH, HEADS, LENGTH, DIM_SLOT)
    for token in range(LENGTH):
        gate = inputs["gate"][:, :, token].exp()
        key_slot = key_slot * gate[:, :, None, :] + torch.einsum(
            "bhf,bhm->bhfm", inputs["key"][:, :, token], inputs["slot"][:, :, token]
        )
        slot_scores[:, :, token] = torch.einsum(
            "bhf,bhfm->bhm", inputs["query"][:, :, token] * scale, key_slot
        )
    slot_weights = torch.softmax(slot_scores, dim=-1)
    slot_value = torch.zeros(BATCH, HEADS, DIM_SLOT, DIM_VALUE)
    outputs = []
    for token in range(LENGTH):
        gate = inputs["gate"][:, :, token].exp()
        slot_value = slot_value * gate[:, :, :, None] + torch.einsum(
            "bhm,bhg->bhmg", inputs["slot"][:, :, token], inputs["value"][:, :, token]
        )
        outputs.append(
            torch.einsum("bhm,bhmg->bhg", slot_weights[:, :, token], slot_value)
        )
    torch.testing.assert_close(actual, torch.stack(outputs, dim=2), rtol=1e-5, atol=1e-6)


def test_rwkv7_state_matches_dplr_recurrence():
    inputs = CASES["rwkv7_state"]["inputs"](_generator())
    actual = _run(rwkv7_state, inputs, CASES["rwkv7_state"]["kwargs"])
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
    torch.testing.assert_close(actual, torch.stack(outputs, dim=2), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["based_attention", "rebased_attention"])
def test_product_feature_roles_must_match_their_factor_sizes(name):
    from attn_engine import StatefulCompilationError

    case = CASES[name]
    operator = FACTORIES[name](**case["kwargs"])
    sizes = _role_sizes(operator)
    product = next(role for role in sizes if "_x_" in role.name)
    sizes[product] = sizes[product] + 1
    with pytest.raises(StatefulCompilationError) as captured:
        operator.compile(role_sizes=sizes, target=Target(), mode="dense")
    assert captured.value.category in ("COMPILE_ROLE_SIZE", "RUNTIME_SHAPE")


def test_every_coverage_program_pointer_resolves():
    import importlib

    programs = {
        row[1] for row in fla_coverage.FLA_COVERAGE if row[1].startswith("examples/")
    }
    assert programs
    for program in programs:
        module_path, _, factory = program.partition(":")
        assert module_path.endswith(".py") and factory, program
        module = importlib.import_module(module_path[:-3].replace("/", "."))
        assert callable(getattr(module, factory)), program


def test_fla_coverage_table_covers_every_installed_fla_op_module():
    import fla.ops

    root = pathlib.Path(fla.ops.__file__).parent
    modules = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").exists() and not path.name.startswith("_")
    } - {"common", "utils"}
    covered = {
        entry[0].split(".")[2]
        for entry in fla_coverage.FLA_COVERAGE
        if len(entry[0].split(".")) > 2
    }
    assert not modules - covered, sorted(modules - covered)
