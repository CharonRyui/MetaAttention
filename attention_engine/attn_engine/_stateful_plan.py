from dataclasses import asdict, dataclass
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

import torch


@dataclass(frozen=True)
class MathematicalPlan:
    identity: str
    representation: str
    rewrites: tuple[str, ...]
    semantic_regions: tuple[str, ...]


@dataclass(frozen=True)
class LoweredCandidate:
    identity: str
    plan_identity: str
    lowerer: str
    direction: str


@dataclass(frozen=True)
class KernelConfig:
    identity: str
    values: tuple[tuple[str, int], ...]

@dataclass(frozen=True)
class ContextValue:
    name: str
    dtype: str
    roles: tuple[str, ...]
    shape_rule: str


@dataclass(frozen=True)
class ContextContract:
    produced: tuple[ContextValue, ...]
    required: tuple[ContextValue, ...]
    recomputable: tuple[ContextValue, ...]

    def validate(self) -> None:
        available = {value.name: value for value in (*self.produced, *self.recomputable)}
        for required in self.required:
            if available.get(required.name) != required:
                raise ValueError(f"incompatible context value: {required.name}")


@dataclass(frozen=True)
class WorkspacePolicy:
    bytes_per_token: int
    maximum_bytes: int

    def required_bytes(self, token_count: int) -> int:
        required = token_count * self.bytes_per_token
        if required > self.maximum_bytes:
            raise RuntimeError("runtime workload exceeds compiled resource envelope")
        return required



@dataclass(frozen=True)
class TileLangPlan:
    """Portable identity and environment descriptor for one specialization."""

    semantic_identity: str
    mathematical_plan: MathematicalPlan
    candidate: LoweredCandidate
    kernel_config: KernelConfig
    binary_identity: str
    tuning_identity: str
    mode: str
    feature_sizes: tuple[int, int]
    context_contract: ContextContract
    workspace: WorkspacePolicy
    uses_parallel_summary: bool = True

    @property
    def backend(self) -> str:
        return "tilelang"


@dataclass(frozen=True)
class TileLangExecutable:
    """Compiler-owned forward/backward dispatch for one analyzed specialization."""

    plan: TileLangPlan
    backward_ir: Any | None
    dense_summary: bool
    scalar_factorized: Any | None

    def prefix(self, transform, bias, segment, initial_state):
        if self.dense_summary:
            from ._stateful_affine import tilelang_dense_affine_prefix

            assert isinstance(transform, tuple)
            return tilelang_dense_affine_prefix(
                transform[0], transform[1], bias, segment, initial_state
            )
        from ._stateful_elementwise import tilelang_elementwise_prefix

        assert isinstance(transform, torch.Tensor)
        return tilelang_elementwise_prefix(transform, bias, segment, initial_state)

    def scalar_factorized_dense_raw(self, raw_inputs, initial_state, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_dense_raw

        if self.scalar_factorized is None or self.scalar_factorized.dense is None:
            raise ValueError("plan has no raw dense scalar lowering")
        return tilelang_scalar_factorized_dense_raw(
            raw_inputs, initial_state, self.scalar_factorized.dense, **kwargs
        )

    def scalar_factorized_dense(self, *args, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_dense

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_dense(*args, **kwargs)

    def scalar_factorized_packed(self, *args, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_forward

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_forward(*args, **kwargs)



def compile_plan(
    *,
    source_identity: str,
    mode: str,
    feature_sizes: tuple[int, int],
    target: Any,
    backward_ir: Any | None,
    dense_summary: bool,
    scalar_factorized: Any | None,
    structural_features: Any,
    optimized: bool,
    autotune: bool,
) -> TileLangExecutable:
    """Compile an executable and atomically publish its portable descriptor."""
    rewrites = _structured_rewrites(structural_features) if optimized else ()
    representation = rewrites[0] if rewrites else "generic-dense-affine"
    mathematical_plan = MathematicalPlan(
        _digest((source_identity, representation, rewrites)),
        representation,
        rewrites,
        ("state-transition", "state-readouts"),
    )
    candidate = LoweredCandidate(
        _digest((mathematical_plan.identity, "tilelang", "forward")),
        mathematical_plan.identity,
        "tilelang-generic" if not rewrites else f"tilelang-{representation}",
        "forward",
    )
    kernel_config = KernelConfig(
        _digest((candidate.identity, "stable-default", autotune)), ()
    )
    toolchain = _toolchain_fingerprint()
    environment = _target_fingerprint(target)
    binary_identity = _digest(
        (candidate.identity, kernel_config.identity, mode, feature_sizes, toolchain, environment)
    )
    tuning_identity = _digest(
        (candidate.identity, environment, "workload-policy/pending-h20-v1")
    )
    context_value = ContextValue(
        "inputs", "algorithm-declared", ("named-inputs",), "runtime-workload"
    )
    context_contract = ContextContract(
        produced=(context_value,) if backward_ir is not None else (),
        required=(context_value,) if backward_ir is not None else (),
        recomputable=(),
    )
    context_contract.validate()
    workspace = WorkspacePolicy(
        bytes_per_token=feature_sizes[0] * feature_sizes[1] * 4,
        maximum_bytes=target.capability[0] * 1024**3,
    )
    plan = TileLangPlan(
        source_identity,
        mathematical_plan,
        candidate,
        kernel_config,
        binary_identity,
        tuning_identity,
        mode,
        feature_sizes,
        context_contract,
        workspace,
    )
    return TileLangExecutable(plan, backward_ir, dense_summary, scalar_factorized)


def _structured_rewrites(features: Any) -> tuple[str, ...]:
    rewrites = []
    if features.axis_scale:
        rewrites.append("axis-scale")
    if features.product_injection:
        rewrites.append("product-injection-gemm")
    if features.rank_one:
        rewrites.append("rank-one-factorized")
    if features.shared_state_contraction:
        rewrites.append("shared-state-contraction")
    return tuple(rewrites)


def _target_fingerprint(target: Any) -> tuple[Any, ...]:
    return (target.device_type, target.product_family, target.capability)


def _toolchain_fingerprint() -> tuple[str, ...]:
    try:
        tilelang_version = version("tilelang")
    except PackageNotFoundError:
        tilelang_version = "unavailable"
    return (
        platform.python_implementation(),
        platform.python_version(),
        torch.__version__,
        str(torch.version.cuda),
        tilelang_version,
    )


def _environment_fingerprint(device: torch.device) -> tuple[Any, ...]:
    if device.type != "cuda":
        return (device.type,)
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return (
        properties.name,
        tuple(torch.cuda.get_device_capability(index)),
        properties.total_memory,
        properties.multi_processor_count,
        _nvidia_smi_fingerprint(index, "driver_version"),
        _toolchain_fingerprint(),
    )


def _clock_fingerprint(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _nvidia_smi_fingerprint(
        index,
        "clocks.applications.graphics,clocks.applications.memory,persistence_mode,power.limit",
    )


def _nvidia_smi_fingerprint(index: int, fields: str) -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "unable to fingerprint the CUDA driver and clock policy"
        ) from error


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _publish_manifest(plan: TileLangPlan) -> None:
    root = Path(
        os.environ.get(
            "STATEFUL_OPERATOR_CACHE_DIR",
            Path.home() / ".cache" / "meta-attention" / "stateful-operator",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{plan.binary_identity}.json"
    lock_path = root / f"{plan.binary_identity}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".plan-", dir=root)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(asdict(plan), stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def is_parallel_plan(plan: Any) -> bool:
    return isinstance(plan, TileLangExecutable) and plan.plan.uses_parallel_summary
