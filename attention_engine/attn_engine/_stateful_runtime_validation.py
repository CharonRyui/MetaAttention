from __future__ import annotations

import os
from typing import Mapping

import torch

from .stateful_operator import StatefulCompilationError


# Successful entries only: unsupported hardware and constraint failures must be
# rechecked and continue to raise their existing structured errors.
_VALIDATED_TARGETS: set[
    tuple[int, str | None, str | None, tuple[int, int] | None]
] = set()


def _validated_target_key(index: int, target):
    """Return a hashable key without changing malformed-target behavior."""
    if target is None:
        return (index, None, None, None)
    device_type = getattr(target, "device_type", None)
    product_family = getattr(target, "product_family", None)
    capability = getattr(target, "capability", None)
    if not (
        isinstance(device_type, str)
        and isinstance(product_family, str)
        and isinstance(capability, tuple)
        and len(capability) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in capability
        )
    ):
        return None
    return (
        index,
        device_type,
        product_family.upper(),
        capability,
    )


def validate_input_names(
    expected_names: set[str], inputs: Mapping[str, torch.Tensor]
) -> None:
    missing = expected_names - inputs.keys()
    unknown = inputs.keys() - expected_names
    if missing:
        raise StatefulCompilationError(
            "RUNTIME_BINDING", "inputs", {"missing": sorted(missing)}
        )
    if unknown:
        raise StatefulCompilationError(
            "RUNTIME_BINDING", "inputs", {"unknown": sorted(unknown)}
        )


def validate_target(device: torch.device, target) -> None:
    if device.type == "cpu" and os.environ.get("STATEFUL_OPERATOR_TEST_CPU") == "1":
        return
    if device.type != "cuda" or torch.version.hip is not None:
        raise StatefulCompilationError(
            "TARGET", "target", {"expected": "CUDA H20", "actual": device.type}
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    target_key = _validated_target_key(index, target)
    if target_key is not None and target_key in _VALIDATED_TARGETS:
        return
    product = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    if "MIG" in product.upper() or "H20" not in product.upper() or capability != (9, 0):
        raise StatefulCompilationError(
            "TARGET",
            "target",
            {
                "expected_product": "H20",
                "actual_product": product,
                "capability": capability,
            },
        )
    if target is not None and (
        target.device_type != "cuda"
        or target.product_family.upper() not in product.upper()
        or target.capability != capability
    ):
        from .stateful_operator import _canonical

        raise StatefulCompilationError(
            "TARGET_CONSTRAINT", "target", {"constraint": _canonical(target)}
        )
    if target_key is not None:
        _VALIDATED_TARGETS.add(target_key)


def validate_aliases(inputs: Mapping[str, torch.Tensor]) -> None:
    tensors = list(inputs.items())
    for index, (left_name, left) in enumerate(tensors):
        for right_name, right in tensors[index + 1 :]:
            if (
                left is right
                or left.untyped_storage().data_ptr()
                != right.untyped_storage().data_ptr()
            ):
                continue
            left_start = left.storage_offset() * left.element_size()
            right_start = right.storage_offset() * right.element_size()
            left_end = left_start + left.numel() * left.element_size()
            right_end = right_start + right.numel() * right.element_size()
            if max(left_start, right_start) < min(left_end, right_end):
                raise StatefulCompilationError(
                    "RUNTIME_ALIAS", "inputs", {"overlapping": [left_name, right_name]}
                )

