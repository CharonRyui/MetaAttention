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
    STATEFUL_REFERENCE = "stateful-reference"


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
    PlanRepresentation.STATEFUL_REFERENCE: PlanRule(
        PlanRepresentation.STATEFUL_REFERENCE,
        (
            "independent-affine-states",
            "post-transition-readout-expressions",
            "authored-propagation-order",
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
        self,
        plan: MathematicalPlan,
        feature_sizes: tuple[int, int],
        *,
        maximum_workspace_bytes: int | None = None,
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
        if (
            maximum_workspace_bytes is not None
            and estimate.workspace_bytes_per_token > maximum_workspace_bytes
        ):
            return None
        return LoweredCandidate(
            _digest((plan.identity, self.name, "forward")),
            plan.identity,
            self.name,
            "forward",
        )

    def estimate(
        self, plan: MathematicalPlan, feature_sizes: tuple[int, ...]
    ) -> CandidateEstimate:
        elements = 1
        for size in feature_sizes:
            elements *= size
        return CandidateEstimate(True, self.static_cost, elements * 4)


LOWERERS = (
    CandidateLowerer(
        "tilelang-generic", (PlanRepresentation.GENERIC_DENSE_AFFINE,), 100
    ),
    CandidateLowerer(
        "tilelang-elementwise", (PlanRepresentation.AXIS_SCALE,), 30, "axis-scale"
    ),
    CandidateLowerer(
        "tilelang-product-gemm",
        (PlanRepresentation.PRODUCT_INJECTION_GEMM,),
        40,
        "product-injection-gemm",
    ),
    CandidateLowerer(
        "tilelang-rank-one", (PlanRepresentation.RANK_ONE_FACTORIZED,), 35,
        "rank-one-factorized",
    ),
    CandidateLowerer(
        "tilelang-shared-contraction",
        (PlanRepresentation.SHARED_STATE_CONTRACTION,),
        45,
        "shared-state-contraction",
    ),
    CandidateLowerer(
        "tilelang-scalar-factorized",
        (PlanRepresentation.SCALAR_FACTORIZED,),
        20,
        "scalar-factorized",
    ),
    CandidateLowerer(
        "reference-stateful",
        (PlanRepresentation.STATEFUL_REFERENCE,),
        200,
        "stateful-reference",
    ),
)


def backward_context_family(
    algorithm: Any, candidate: LoweredCandidate, dense_summary: bool
) -> str:
    """Select the backward context that matches the forward runtime branch.

    The forward pass picks its elementwise scan whenever the per-token summary is
    a scalar pair rather than dense ``(left, right)`` factors, so the generated
    backward must be bound to the same context or it would ask for values the
    forward never saved.
    """
    family = runtime_family_for(candidate)
    if family == "axis-scale":
        return "dense" if dense_summary else "elementwise"
    if family in {"generic", "product-injection-gemm", "shared-state-contraction"}:
        identity_only = not algorithm.transitions[0].propagations
        return "elementwise" if identity_only and not dense_summary else "dense"
    return "dense"


def runtime_family_for(candidate: LoweredCandidate) -> str:
    return next(
        lowerer.runtime_family for lowerer in LOWERERS if lowerer.name == candidate.lowerer
    )

def kernel_configurations(candidate: LoweredCandidate) -> tuple[tuple[tuple[str, int], ...], ...]:
    if candidate.lowerer == "tilelang-elementwise":
        return tuple((("block_tokens", size),) for size in (128, 64, 256))
    return tuple((("threads", size),) for size in (128, 256))


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
    backward_binding: Any | None = None

    @property
    def runtime_family(self) -> str:
        return next(
            lowerer.runtime_family
            for lowerer in LOWERERS
            if lowerer.name == self.plan.candidate.lowerer
        )

    def _validate_gradient_dispatch(self, *tensors: torch.Tensor) -> None:
        if not torch.is_grad_enabled() or not any(tensor.requires_grad for tensor in tensors):
            return
        if (
            self.backward_ir is None
            or self.backward_binding is None
            or self.plan.backward_candidate is None
            or self.composition is None
            or self.composition.backward_candidate != self.plan.backward_candidate
        ):
            raise RuntimeError("gradient execution has no compiler-selected backward candidate")

    def prefix(self, transform, bias, segment, initial_state):
        transform_inputs = transform if isinstance(transform, tuple) else (transform,)
        self._validate_gradient_dispatch(*transform_inputs, bias, initial_state)
        if isinstance(transform, tuple):
            from ._stateful_affine import tilelang_dense_affine_prefix

            return tilelang_dense_affine_prefix(
                transform[0], transform[1], bias, segment, initial_state,
                backward=self.backward_binding,
                config=self.plan.backward_kernel_config.values
                if self.plan.backward_kernel_config else (),
            )
        from ._stateful_elementwise import tilelang_elementwise_prefix

        return tilelang_elementwise_prefix(
            transform, bias, segment, initial_state,
            backward=self.backward_binding,
            config=self.plan.backward_kernel_config.values
            if self.plan.backward_kernel_config else (),
        )

    def scalar_factorized_dense_raw(self, raw_inputs, initial_state, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_dense_raw

        if self.scalar_factorized is None or self.scalar_factorized.dense is None:
            raise ValueError("plan has no raw dense scalar lowering")
        return tilelang_scalar_factorized_dense_raw(
            raw_inputs, initial_state, self.scalar_factorized.dense,
            threads=dict(self.plan.kernel_config.values).get("threads", 128), **kwargs
        )

    def scalar_factorized_dense(self, *args, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_dense

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_dense(
            *args, threads=dict(self.plan.kernel_config.values).get("threads", 128), **kwargs
        )

    def scalar_factorized_packed(self, *args, **kwargs):
        from ._stateful_scalar import tilelang_scalar_factorized_forward

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_forward(
            *args, threads=dict(self.plan.kernel_config.values).get("threads", 128), **kwargs
        )


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
    operator: Any,
    compilation: Any,
) -> TileLangExecutable:
    """Enumerate mathematical plans, then bind one stable executable candidate."""
    from ._stateful_reference import requires_reference_lowering

    if requires_reference_lowering(operator.algorithm):
        representations = (PlanRepresentation.STATEFUL_REFERENCE,)
        rewrites = ()
    else:
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
    from ._stateful_structured import RUNTIME_FAMILY_ELIGIBILITY
    active_rewrites = {item.value for item in rewrites}
    candidates = tuple(
        candidate
        for item in mathematical_plans
        if not active_rewrites.intersection(item.conflicts)
        for lowerer in LOWERERS
        if not (backward_ir is not None and lowerer.runtime_family == "scalar-factorized")
        if item.representation is PlanRepresentation.GENERIC_DENSE_AFFINE
        or RUNTIME_FAMILY_ELIGIBILITY.get(lowerer.runtime_family, lambda _: (True, None))(operator.algorithm)[0]
        if (candidate := lowerer.lower(
            item, feature_sizes, maximum_workspace_bytes=target.workspace_bytes
        )) is not None
    )
    if not candidates:
        from .stateful_operator import StatefulCompilationError

        raise StatefulCompilationError(
            "COMPILE_RESOURCE", "target.workspace_bytes", {"reason": "no feasible lowered candidate"}
        )
    candidate = select_candidate(candidates)
    selected_plan = next(
        item for item in mathematical_plans if item.identity == candidate.plan_identity
    )
    values = kernel_configurations(candidate)[0]
    tuning_result = None
    if autotune:
        from ._stateful_benchmark import invocation_factory
        from ._stateful_tuning import MeasurementCache, tune_candidates

        provisional_plan = TileLangPlan(
            source_identity,
            selected_plan,
            candidate,
            KernelConfig(_digest((candidate.identity, "probe")), ()),
            "probe-binary",
            "probe-tuning",
            mode,
            feature_sizes,
            ContextContract((), (), ()),
            WorkspacePolicy(feature_sizes[0] * feature_sizes[1] * 4, target.workspace_bytes),
            mathematical_plans,
            candidates,
        )
        provisional = TileLangExecutable(
            provisional_plan, backward_ir, dense_summary, scalar_factorized,
            WorkspaceAllocator()
        )
        device = torch.device(target.device_type)
        cache = None
        cache_root = os.environ.get("STATEFUL_OPERATOR_CACHE_DIR")
        if cache_root:
            cache = MeasurementCache(str(Path(cache_root) / "tuning" / "measurements.json"))
        factory = invocation_factory(operator, compilation, provisional, mathematical_plans, device)
        costs = {lowerer.name: lowerer.static_cost for lowerer in LOWERERS}
        generic = next(item for item in candidates if item.lowerer == "tilelang-generic")
        tuning_result = tune_candidates(
            candidates,
            {item: kernel_configurations(item) for item in candidates},
            WORKLOAD_POLICY,
            factory,
            autotune=True,
            mode=mode,
            environment=_target_fingerprint(target),
            cache=cache,
            generic_candidate=generic,
            static_costs={item: costs[item.lowerer] for item in candidates},
        )
        candidate = tuning_result.winner
        selected_plan = next(
            item for item in mathematical_plans if item.identity == candidate.plan_identity
        )
        values = tuning_result.winner_config
    kernel_config = KernelConfig(_digest((candidate.identity, values)), values)
    toolchain = _toolchain_fingerprint()
    environment = _target_fingerprint(target)
    fixed_roles = tuple((str(role), size) for role, size in compilation.role_sizes)
    backward_binding = None
    backward_values: tuple[tuple[str, int], ...] = ()
    if backward_ir is not None and not requires_reference_lowering(operator.algorithm):
        from ._stateful_backward import bind_backward

        family = backward_context_family(operator.algorithm, candidate, dense_summary)
        backward_values = (("threads", 128),) if family == "dense" else (("block_tokens", 128),)
        backward_binding = bind_backward(backward_ir, family, backward_values)
    context_values = (
        tuple(ContextValue(*value) for value in backward_binding.context_values)
        if backward_binding is not None else ()
    )
    context_contract = ContextContract(context_values, context_values, ())
    context_contract.validate()
    binary_identity = _digest(
        (
            candidate.identity,
            kernel_config.identity,
            context_contract.identity,
            mode,
            feature_sizes,
            fixed_roles,
            toolchain,
            environment,
        )
    )
    tuning_identity = _digest(
        (source_identity, fixed_roles, environment, toolchain, WORKLOAD_POLICY.version, mode)
    )
    fixed_sizes = dict(compilation.role_sizes)
    state_heads = fixed_sizes[operator.algorithm.states[0].head_role]
    per_token_elements = 0
    for spec in operator.algorithm.states:
        sizes = [fixed_sizes[role] for role in spec.feature_roles]
        if spec.rank == 2:
            rows, columns = sizes
            per_token_elements += rows * rows + columns * columns + rows * columns
        else:
            per_token_elements += sizes[0] * sizes[0] + sizes[0]
    workspace = WorkspacePolicy(
        bytes_per_token=state_heads * per_token_elements * 4,
        maximum_bytes=target.workspace_bytes,
    )
    backward_candidate = None
    backward_kernel_config = None
    if backward_ir is not None and not requires_reference_lowering(operator.algorithm):
        backward_candidate = LoweredCandidate(
            _digest((source_identity, "tilelang-generic-backward", "backward")),
            source_identity,
            "tilelang-generic-backward",
            "backward",
        )
        backward_kernel_config = KernelConfig(
            _digest((backward_candidate.identity, backward_values, autotune)),
            backward_values,
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
        backward_binding,
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
    return (target.device_type, target.product_family, target.capability, target.workspace_bytes)


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
                "--id=" + str(torch.cuda.get_device_properties(index).uuid),
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
