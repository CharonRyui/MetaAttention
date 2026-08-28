from dataclasses import asdict, dataclass
from enum import Enum
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


class PlanRepresentation(str, Enum):
    GENERIC_DENSE_AFFINE = "generic-dense-affine"
    AXIS_SCALE = "axis-scale"
    PRODUCT_INJECTION_GEMM = "product-injection-gemm"
    RANK_ONE_FACTORIZED = "rank-one-factorized"
    SHARED_STATE_CONTRACTION = "shared-state-contraction"
    SCALAR_FACTORIZED = "scalar-factorized"


@dataclass(frozen=True)
class MathematicalPlan:
    identity: str
    representation: PlanRepresentation
    rewrites: tuple[str, ...]
    semantic_regions: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    vjp_equivalence: str = "compiler-derived-vjp"
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanRule:
    representation: PlanRepresentation
    preconditions: tuple[str, ...]
    conflicts: tuple[str, ...] = ()


PLAN_RULES = {
    PlanRepresentation.GENERIC_DENSE_AFFINE: PlanRule(
        PlanRepresentation.GENERIC_DENSE_AFFINE,
        ("validated-scanable-affine-program",),
    ),
    PlanRepresentation.AXIS_SCALE: PlanRule(
        PlanRepresentation.AXIS_SCALE,
        ("typed-axis-scale-propagation", "authored-propagation-order"),
        (PlanRepresentation.RANK_ONE_FACTORIZED.value,),
    ),
    PlanRepresentation.PRODUCT_INJECTION_GEMM: PlanRule(
        PlanRepresentation.PRODUCT_INJECTION_GEMM,
        ("typed-role-compatible-product-injection",),
    ),
    PlanRepresentation.RANK_ONE_FACTORIZED: PlanRule(
        PlanRepresentation.RANK_ONE_FACTORIZED,
        ("typed-ordered-rank-one-propagation",),
    ),
    PlanRepresentation.SHARED_STATE_CONTRACTION: PlanRule(
        PlanRepresentation.SHARED_STATE_CONTRACTION,
        ("multiple-independent-readouts-of-shared-state",),
    ),
    PlanRepresentation.SCALAR_FACTORIZED: PlanRule(
        PlanRepresentation.SCALAR_FACTORIZED,
        ("scalar-propagation", "two-disjoint-single-feature-factors", "one-readout"),
        (
            PlanRepresentation.RANK_ONE_FACTORIZED.value,
            PlanRepresentation.SHARED_STATE_CONTRACTION.value,
        ),
    ),
}


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
class CandidateEstimate:
    feasible: bool
    static_cost: int
    workspace_bytes_per_token: int
    reason: str | None = None


@dataclass(frozen=True)
class CandidateLowerer:
    name: str
    representations: tuple[PlanRepresentation, ...]
    static_cost: int
    runtime_family: str = "generic"

    def lower(
        self, plan: MathematicalPlan, feature_sizes: tuple[int, int]
    ) -> LoweredCandidate | None:
        try:
            representation = PlanRepresentation(plan.representation)
        except ValueError:
            return None
        if representation not in self.representations:
            return None
        estimate = self.estimate(plan, feature_sizes)
        if not estimate.feasible:
            return None
        return LoweredCandidate(
            _digest((plan.identity, self.name, "forward")),
            plan.identity,
            self.name,
            "forward",
        )

    def estimate(
        self, plan: MathematicalPlan, feature_sizes: tuple[int, int]
    ) -> CandidateEstimate:
        rows, columns = feature_sizes
        workspace = rows * columns * 4
        return CandidateEstimate(True, self.static_cost, workspace)


LOWERERS = (
    CandidateLowerer(
        "tilelang-generic", (PlanRepresentation.GENERIC_DENSE_AFFINE,), 100
    ),
    CandidateLowerer(
        "tilelang-elementwise", (PlanRepresentation.AXIS_SCALE,), 30, "elementwise"
    ),
    CandidateLowerer(
        "tilelang-scalar-factorized",
        (PlanRepresentation.SCALAR_FACTORIZED,),
        20,
        "scalar-factorized",
    ),
)


def choose_kernel_config(
    candidate: LoweredCandidate,
    measurements: dict[str, tuple[float, ...]] | None,
    *,
    autotune: bool,
) -> KernelConfig:
    """Tune only the selected candidate; disabled mode uses its stable default."""
    if not autotune or not measurements:
        values: tuple[tuple[str, int], ...] = ()
    else:
        scores = normalized_geometric_mean(measurements)
        if candidate.identity not in scores:
            raise ValueError("selected candidate has no successful measurements")
        values = (("probe-score-millionths", int(scores[candidate.identity] * 1_000_000)),)
    return KernelConfig(_digest((candidate.identity, values)), values)


def select_candidate(candidates: tuple[LoweredCandidate, ...]) -> LoweredCandidate:
    """Select by lowerer estimate with a deterministic identity tie-break."""
    if not candidates:
        raise ValueError("no feasible lowered candidates")
    costs = {lowerer.name: lowerer.static_cost for lowerer in LOWERERS}
    return min(
        candidates,
        key=lambda item: (costs[item.lowerer], item.lowerer, item.identity),
    )


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

    @property
    def identity(self) -> str:
        return _digest(asdict(self))

    def validate(self) -> None:
        available = {value.name: value for value in (*self.produced, *self.recomputable)}
        if len(available) != len(self.produced) + len(self.recomputable):
            raise ValueError("duplicate context value name")
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


class WorkspaceAllocator:
    """Compiler-owned per-device workspace cache for dynamic workloads."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[torch.device, int], torch.Tensor] = {}

    def acquire(self, device: torch.device, size: int) -> torch.Tensor | None:
        if size == 0:
            return None
        key = (device, size)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(size, dtype=torch.uint8, device=device)
            self._buffers[key] = buffer
        return buffer


@dataclass(frozen=True)
class Composition:
    forward_candidate: LoweredCandidate
    backward_candidate: LoweredCandidate | None
    context_contract: ContextContract

    def validate(self) -> None:
        self.context_contract.validate()
        if self.backward_candidate is not None and self.backward_candidate.direction != "backward":
            raise ValueError("composition backward candidate must lower backward direction")


@dataclass(frozen=True)
class WorkloadPoint:
    sequence_count: int
    token_count: int
    distribution: str


@dataclass(frozen=True)
class WorkloadPolicy:
    version: str
    dense: tuple[WorkloadPoint, ...]
    packed: tuple[WorkloadPoint, ...]

    def points(self, mode: str) -> tuple[WorkloadPoint, ...]:
        return self.dense if mode == "dense" else self.packed


WORKLOAD_POLICY = WorkloadPolicy(
    "stateful-workloads/provisional-h20-v1",
    dense=(
        WorkloadPoint(1, 64, "uniform"),
        WorkloadPoint(2, 512, "uniform"),
        WorkloadPoint(4, 4096, "uniform"),
    ),
    packed=(
        WorkloadPoint(4, 256, "uniform"),
        WorkloadPoint(8, 1024, "long-tail"),
        WorkloadPoint(8, 512, "empty-containing"),
    ),
)


def normalized_geometric_mean(
    measurements: dict[str, tuple[float, ...]],
) -> dict[str, float]:
    import math

    if not measurements:
        return {}
    widths = {len(values) for values in measurements.values()}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("candidate measurements must cover one common workload policy")
    width = next(iter(widths))
    best = tuple(min(values[index] for values in measurements.values()) for index in range(width))
    return {
        candidate: math.exp(
            sum(math.log(value / best[index]) for index, value in enumerate(values)) / width
        )
        for candidate, values in measurements.items()
    }


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
    mathematical_plans: tuple[MathematicalPlan, ...] = ()
    lowered_candidates: tuple[LoweredCandidate, ...] = ()
    backward_candidate: LoweredCandidate | None = None
    backward_kernel_config: KernelConfig | None = None
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
    workspace_allocator: WorkspaceAllocator
    composition: Composition | None = None

    @property
    def runtime_family(self) -> str:
        return next(
            lowerer.runtime_family
            for lowerer in LOWERERS
            if lowerer.name == self.plan.candidate.lowerer
        )

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


def publish_successful_artifact(identity: str, payload: Any, *, layer: str) -> Path:
    """Atomically publish one successful compiler artifact under its layer identity."""
    root = Path(
        os.environ.get(
            "STATEFUL_OPERATOR_CACHE_DIR",
            Path.home() / ".cache" / "meta-attention" / "stateful-operator",
        )
    ) / layer
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{identity}.json"
    lock_path = root / f"{identity}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return destination
        descriptor, temporary = tempfile.mkstemp(prefix=".artifact-", dir=root)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination






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
    """Enumerate mathematical plans, then bind one stable executable candidate."""
    rewrites = _structured_rewrites(structural_features) if optimized else ()
    representations = (PlanRepresentation.GENERIC_DENSE_AFFINE, *rewrites)
    if optimized and scalar_factorized is not None:
        representations += (PlanRepresentation.SCALAR_FACTORIZED,)
    mathematical_plans = tuple(
        MathematicalPlan(
            _digest((source_identity, representation.value)),
            representation,
            tuple(item.value for item in rewrites)
            if representation is not PlanRepresentation.GENERIC_DENSE_AFFINE
            else (),
            ("state-transition", "state-readouts"),
            PLAN_RULES[representation].preconditions,
            "compiler-derived-vjp",
            PLAN_RULES[representation].conflicts,
        )
        for representation in representations
    )
    active_rewrites = {item.value for item in rewrites}
    candidates = tuple(
        candidate
        for item in mathematical_plans
        if not active_rewrites.intersection(item.conflicts)
        for lowerer in LOWERERS
        if not (backward_ir is not None and lowerer.runtime_family == "scalar-factorized")
        if (candidate := lowerer.lower(item, feature_sizes)) is not None
    )
    candidate = select_candidate(candidates)
    selected_plan = next(
        item for item in mathematical_plans if item.identity == candidate.plan_identity
    )
    kernel_config = choose_kernel_config(candidate, None, autotune=autotune)
    toolchain = _toolchain_fingerprint()
    environment = _target_fingerprint(target)
    context_value = ContextValue(
        "inputs", "algorithm-declared", ("named-inputs",), "runtime-workload"
    )
    context_contract = ContextContract(
        produced=(context_value,) if backward_ir is not None else (),
        required=(context_value,) if backward_ir is not None else (),
        recomputable=(),
    )
    context_contract.validate()
    binary_identity = _digest(
        (
            candidate.identity,
            kernel_config.identity,
            context_contract.identity,
            mode,
            feature_sizes,
            toolchain,
            environment,
        )
    )
    tuning_identity = _digest(
        (candidate.identity, environment, WORKLOAD_POLICY.version, mode)
    )
    workspace = WorkspacePolicy(
        bytes_per_token=feature_sizes[0] * feature_sizes[1] * 4,
        maximum_bytes=target.workspace_bytes,
    )
    backward_candidate = None
    backward_kernel_config = None
    if backward_ir is not None:
        backward_candidate = LoweredCandidate(
            _digest((source_identity, "tilelang-generic-backward", "backward")),
            source_identity,
            "tilelang-generic-backward",
            "backward",
        )
        backward_kernel_config = KernelConfig(
            _digest((backward_candidate.identity, "stable-default", autotune)), ()
        )
    plan = TileLangPlan(
        source_identity,
        selected_plan,
        candidate,
        kernel_config,
        binary_identity,
        tuning_identity,
        mode,
        feature_sizes,
        context_contract,
        workspace,
        mathematical_plans,
        candidates,
        backward_candidate,
        backward_kernel_config,
    )
    composition = Composition(candidate, backward_candidate, context_contract)
    composition.validate()
    if os.environ.get("STATEFUL_OPERATOR_CACHE_DIR"):
        publish_successful_artifact(
            selected_plan.identity, asdict(selected_plan), layer="plans"
        )
        publish_successful_artifact(
            kernel_config.identity, asdict(kernel_config), layer="configs"
        )
        publish_successful_artifact(binary_identity, asdict(plan), layer="binaries")
        publish_successful_artifact(
            context_contract.identity, asdict(context_contract), layer="contexts"
        )
        if backward_candidate is not None:
            publish_successful_artifact(
                backward_candidate.identity,
                asdict(backward_candidate),
                layer="backward",
            )
    return TileLangExecutable(
        plan,
        backward_ir,
        dense_summary,
        scalar_factorized,
        WorkspaceAllocator(),
        composition,
    )


def _structured_rewrites(features: Any) -> tuple[PlanRepresentation, ...]:
    rewrites = []
    if features.axis_scale:
        rewrites.append(PlanRepresentation.AXIS_SCALE)
    if features.product_injection:
        rewrites.append(PlanRepresentation.PRODUCT_INJECTION_GEMM)
    if features.rank_one:
        rewrites.append(PlanRepresentation.RANK_ONE_FACTORIZED)
    if features.shared_state_contraction:
        rewrites.append(PlanRepresentation.SHARED_STATE_CONTRACTION)
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


@dataclass(frozen=True)
class EvidenceStatus:
    target: str
    workload_policy: str
    status: str
    note: str


H20_EVIDENCE = EvidenceStatus(
    "NVIDIA H20",
    WORKLOAD_POLICY.version,
    "pending",
    "Freeze policy and record measurements on an otherwise idle H20.",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



def is_parallel_plan(plan: Any) -> bool:
    return isinstance(plan, TileLangExecutable) and plan.plan.uses_parallel_summary
