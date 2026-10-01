"""Resource contracts for the Stateful Operator plans.

`WorkspacePolicy` is the compiler's gate on sequence-scaled buffers, so the test
pins what each plan is allowed to charge per token and where the H20 envelope
stops accepting a workload. A plan that silently starts scaling with the token
count, or one that stops being reachable at long sequences, fails here.
"""

from __future__ import annotations

import pytest

from attn_engine import HeadRole, ProductFeatureRole, Target
from examples.gated_delta_rule import gated_delta_rule
from examples.linear_attention import gla_attention, simple_gla_attention

pytestmark = pytest.mark.unit

ENVELOPE = Target().workspace_bytes
HEADS = 8
ROWS = COLUMNS = 128


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _compile(factory, *, seqlen: int, backward: bool, size: int = ROWS):
    kwargs = {"HQ": 1} if "gated_delta_rule" in factory.__module__ else {}
    operator = factory(1, HEADS, seqlen, size, size, **kwargs)
    sizes: dict = {operator.algorithm.states[0].head_role: HEADS}
    for role in operator.algorithm.states[0].feature_roles:
        sizes[role] = size
    for spec in operator.algorithm.inputs:
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes.setdefault(role, HEADS)
            elif isinstance(role, ProductFeatureRole):
                sizes[role] = sizes[role.left] * sizes[role.right]
    return operator.compile(
        role_sizes=sizes,
        target=Target(),
        mode="dense",
        optimized=True,
        backward=backward,
    )


def _workspace(compiled):
    plan = compiled.operator._plans[compiled.specialization]
    assert plan.plan.candidate.lowerer is not None
    return plan.plan.candidate.lowerer, plan.plan.workspace


def test_factored_recurrence_charges_nothing_without_a_backward():
    """Inference keeps only the output and the final State."""
    lowerer, workspace = _workspace(
        _compile(gated_delta_rule, seqlen=2048, backward=False)
    )
    assert lowerer == "tilelang-factored-recurrence"
    assert workspace.bytes_per_token == 0
    # so a long sequence is accepted where the summary plans are not
    assert workspace.required_bytes(32768) == 0


def test_factored_recurrence_charges_recorded_states_with_a_backward():
    """A compiled backward records the per-token State it replays from."""
    lowerer, workspace = _workspace(
        _compile(gated_delta_rule, seqlen=2048, backward=True)
    )
    assert lowerer == "tilelang-factored-recurrence"
    # one State per token per head, recorded in bf16
    assert workspace.bytes_per_token == HEADS * ROWS * COLUMNS * 2
    assert workspace.required_bytes(2048) < ENVELOPE
    with pytest.raises(RuntimeError):
        workspace.required_bytes(32768)


def test_vector_plan_uses_no_token_state_workspace():
    """The vector kernel keeps each State column in registers."""
    lowerer, workspace = _workspace(
        _compile(gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-vector-propagation"
    assert workspace.bytes_per_token == 0
    assert workspace.alignment_tokens == 0
    assert workspace.required_bytes(2048) == 0
    assert workspace.required_bytes(32768) == 0


def test_fused_scalar_plan_charges_nothing_and_is_accepted_at_long_sequences():
    """The fused scalar kernel moves only its factors, so it has no cliff."""
    lowerer, workspace = _workspace(
        _compile(simple_gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-scalar-factorized"
    assert workspace.bytes_per_token == 0
    assert workspace.required_bytes(32768) == 0
