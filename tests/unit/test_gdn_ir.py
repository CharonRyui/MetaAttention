from __future__ import annotations

import pytest
import torch
from attn_engine.gdn_engine import DEFAULT_SCALE, GDNEngine, _build_gdn_operator

pytestmark = pytest.mark.unit


def test_gdn_factory_builds_explicit_rank_one_ir():
    operator = _build_gdn_operator()
    algorithm = operator.algorithm

    assert algorithm.transition.state == "memory"
    assert algorithm.readout.state == "memory"
    assert algorithm.structural_identity
    assert algorithm.inputs[-2].name == "gate"
    assert algorithm.inputs[-2].dtype == torch.float32
    assert algorithm.inputs[-1].name == "beta"


def test_gdn_factory_scale_changes_structural_identity():
    assert _build_gdn_operator().structural_identity != _build_gdn_operator(scale=0.1).structural_identity
    assert _build_gdn_operator(scale=DEFAULT_SCALE).structural_identity == _build_gdn_operator().structural_identity


def test_gdn_factory_rejects_non_python_scale():
    with pytest.raises(TypeError, match="scale must be a Python float"):
        _build_gdn_operator(scale=torch.tensor(DEFAULT_SCALE))


def test_gdn_engine_warns_before_device_validation():
    with pytest.warns(DeprecationWarning, match="StatefulOperator"):
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA H20"):
            GDNEngine("cpu")
