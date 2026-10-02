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


def test_factored_inference_accepts_chunks_but_rejects_excessive_scratch():
    """Chunk summaries avoid token-State storage but remain envelope-bounded."""
    lowerer, workspace = _workspace(
        _compile(gated_delta_rule, seqlen=2048, backward=False)
    )
    assert lowerer == "tilelang-factored-recurrence"
    assert 0 < workspace.required_bytes(2048) < ENVELOPE
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(32768)


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


def test_vector_plan_bounds_chunk_scratch_and_long_unaligned_sequences():
    """Aligned and unaligned lengths both charge bounded chunk scratch."""
    lowerer, workspace = _workspace(
        _compile(gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-vector-propagation"
    assert 0 < workspace.required_bytes(2048) < ENVELOPE
    assert 0 < workspace.required_bytes(32768) < ENVELOPE
    assert 0 < workspace.required_bytes(32769) < ENVELOPE


def test_fused_scalar_plan_charges_chunk_scratch_at_long_sequences():
    """The scalar schedule charges its bounded chunk carries and tail."""
    lowerer, workspace = _workspace(
        _compile(simple_gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-scalar-factorized"
    assert workspace.bytes_per_token == 0
    assert 0 < workspace.required_bytes(32768) < ENVELOPE


def test_chunk_workspace_counts_tail_and_batch_before_rejecting():
    from attn_engine._stateful_plan import WorkspacePolicy

    workspace = WorkspacePolicy(
        bytes_per_token=2, maximum_bytes=500, bytes_per_chunk=100
    )
    # Two sequences of 32 tokens: each needs a carry and final-State allowance.
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(64, sequence_count=2, sequence_length=32)
    # One sequence uses fewer boundaries for the same token count.
    assert workspace.required_bytes(64, sequence_length=64) == 328
    # A tail adds a chunk, but the exact envelope edge is still accepted.
    workspace = WorkspacePolicy(
        bytes_per_token=2, maximum_bytes=430, bytes_per_chunk=100
    )
    assert workspace.required_bytes(65, sequence_length=65) == 430
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(66, sequence_length=66)
    assert workspace.required_bytes(0, sequence_count=2, sequence_length=0) == 0


def test_native_normalized_bounds_long_sequences_without_input_staging():
    from examples.normalized_linear_attention import normalized_linear_attention

    _, workspace = _workspace(
        _compile(normalized_linear_attention, seqlen=8192, backward=False, size=64)
    )
    # Two sequences and recursive carries fit without raising the production cap.
    assert 0 < workspace.required_bytes(
        2 * 8192, sequence_count=2, sequence_length=8192
    ) < ENVELOPE
    # Batch still scales scratch; eliminating staging cannot bypass the gate.
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(
            32 * 32769, sequence_count=32, sequence_length=32769
        )
