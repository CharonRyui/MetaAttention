# Public scan-stage benchmarks against FLA

`testing/benchmark_fla_stages.py` registers three **attention operators**, not
model layers:

| CLI family | Public Meta factory | FLA prefill | FLA recurrent |
| --- | --- | --- | --- |
| `preconditioned-kda` | `examples.preconditioned_delta.preconditioned_delta(kind="kda")` | `chunk_precond_kda` | `fused_recurrent_precond_kda` |
| `preconditioned-gdn` | same, `kind="gdn"` | `chunk_precond_gated_delta_rule` | `fused_recurrent_precond_gated_delta_rule` |
| `lightnet` | `examples.lightnet_stable.lightnet_stable` | `chunk_lightnet` | `fused_recurrent_lightnet` |

The runner invokes the **public compiled stage callable**, not private kernels.
It records the selected lowerer and runtime family for each child, separately
for forward and backward. LogAddExp is a dedicated child runtime and has no
ordinary candidate lowerer; its `selected_lowerer` is null, not a missing
benchmark. A selected lowerer describes compilation, not proof that the
same native executor runs during autograd.

## Run

From the repository root, with the CUDA/TileLang environment installed:

```sh
.venv/bin/python testing/benchmark_fla_stages.py \
  --gpu 1 --family preconditioned-kda,preconditioned-gdn,lightnet \
  --seqlens 1 64 2048 8192 --heads 1 --dim 64 --dtype bfloat16 \
  --workspace 8589934592 --warmup 3 --reps 20 \
  --fla-path ../flash-linear-attention \
  --output /tmp/fla-stages.json
```

`--families` is an alias for `--family`. `--workspace` is **bytes**, not MiB.
`--heads` and `--seqlens` accept lists. Batch is deliberately fixed to one;
K/V dimensions and K/V head counts are equal. T1 is labeled decode with
nonzero supplied caches; larger T is labeled prefill with supplied caches.
This is a single invocation workload, not a preceding prefill followed by
an entire decode loop. `--no-backward` requests forward-only measurements.
Supported input dtypes are FP32, BF16 and FP16; unsupported FLA kernel dtypes
are recorded as failures, not coerced into another dtype.

The physical GPU is checked for idle utilization, memory and compute PIDs
**before importing torch**. An explicit busy `--gpu` is rejected; the runner
never silently uses another card. It respects an existing numeric
`CUDA_VISIBLE_DEVICES` mask. Run long jobs in the background with stdout and
stderr redirected to a log; each backend announces its start. The JSON file
is rewritten after every completed workload. Each workload runs in a fresh
child process (`--case-timeout`, default 900 seconds); completed backend
checkpoints survive a child timeout or fatal kernel/OS exit. Unfinished
backends are marked `CHILD_TIMEOUT` or `CHILD_FAILED`, not omitted. The
controller checks the chosen GPU again before each child; a newly occupied
GPU produces `GPU_BUSY_REJECTED` records rather than switching GPUs. GPU
snapshots at both boundaries flag observed external contention; absence of
a boundary PID is not a guarantee that no interference occurred in between.

## Numerical contract

All three implementations receive the same quantized q/k/v inputs, FP32
gates and FP32 nonzero initial states. In the inspected FLA checkout, public chunk
KDA unconditionally L2-normalizes q/k while recurrent KDA and the authored
operator use original q/k. This baseline semantic difference is explicitly
recorded, not corrected by changing inputs or replacing the public API. Preconditioned ATK state is positive;
center is fixed to -0.2, x to 1.5, epsilon to 1e-6, and query scale to
K**-0.5. GDN normalizes q/k; KDA does **not**. LightNet consumes raw logits,
not centered keys, and its initial log-normalizer is finite. Layer projections,
SiLU, convolutions, output norm and gating are outside both intervals.

Primary validation is an independent FP32 **CPU token recurrence**, not
agreement with a FLA optimized kernel. Every available forward backend is
checked against output **and both final states**. The JSON reports max absolute
and RMS error per tensor, with `rtol=0.02, atol=0.005`. Available FLA outputs
also get an explicit Meta pair-difference record. A numerical disagreement
is `NUMERICAL_MISMATCH`, and its measured latency is retained as diagnostic
evidence, **not** an accuracy-qualified speedup. No speedup/winner is emitted.
This tolerance is intended for mixed precision exploratory workloads, not a
proof of equivalence across arbitrary adversarial inputs.

Backward uses unit cotangents on output and both final states. An independent
full VJP oracle checks T<=64 with `rtol=0.04, atol=0.02`. Longer backward
measurements explicitly report `gradient_oracle.status=NOT_CHECKED` and
`status=MEASURED_UNVALIDATED` unless their forward contract already mismatches;
forward correctness is not represented as gradient correctness. Meta exposes
center and per-token scale derivatives that the FLA APIs do not expose;
these constants are excluded from **both** timed VJPs for an equal gradient
input set. Each backend records `excluded_gradients` and its actual gradient
inputs. The CPU factory tests additionally verify Meta's full exposed VJP. Remaining input and initial-state gradients are
compared against the oracle in canonical head-first layout. FLA recurrent
backward is never substituted for chunk backward.

## Timing and memory contract

- External tensors are prebound: contiguous **B,H,T,D** for Meta and
  **B,T,H,D** for FLA. External transpose/materialization is excluded on
  **both** sides. FLA backward leaves are already in native layout, so the
  VJP interval does not backpropagate through an external layout-copy graph.
- Internal preprocessing **is included**: ATK/LightNet gate construction,
  normalization, public input/state validation, dispatch, stage bindings,
  intermediate contiguous copies, allocations and kernel launches.
- Forward is compiled with backward disabled and invoked entirely in
  `torch.inference_mode()` to avoid grad-enabled differentiable execution.
  The CPU-test fallback environment is explicitly rejected for GPU runs.
- Compilation/JIT and correctness are outside timing. Backward forward-graph
  construction and cotangent allocation are also outside timing. Backward
  is **VJP-only**, repeatedly applying `autograd.grad(..., retain_graph=True)`
  to the same graph; it is not full training-step or forward+backward latency.
- CUDA-event and synchronized CPU-wall medians surround the same warmed
  invocation. There is at least one untimed priming call even with warmup=0
  to exclude lazy backward JIT. Both raw sample arrays are retained. No CUDA Graph capture,
  CUPTI kernel-only timing or subtraction of Python/public overhead is used.
- A separate warmed call precedes timing for allocator measurement. JSON
  includes resident allocated bytes, absolute peak allocated bytes and peak
  incremental allocated bytes. Backward graph storage is resident. CUDA
  context/Triton/driver allocations outside PyTorch are not included.
- The staged workspace estimate is the public compiled `workspace_bytes`
  bound, not measured peak memory. Forward and backward estimates are
  recorded separately. FLA has no equivalent public estimator here;
  its allocator peak must not be called a staged workspace estimate.

Each family/T/H record independently attempts Meta forward, FLA chunk forward,
FLA recurrent forward, Meta backward and FLA chunk backward. Compilation,
workspace/resource rejection, unavailable APIs and kernel exceptions appear
with status, exception type, message and available structured details. One
failed backend does not prevent attempts of other backends or later
workloads. There is no fallback to another family, dtype, schedule, CPU or
private kernel. A CUDA context fault may make later backends in the same child fail; those
failures remain visible. A fresh child prevents that context from poisoning
subsequent workloads.

## Validation and exploratory H20 evidence

Run the CPU contract tests on the checkout containing the new examples:

```sh
.venv/bin/python -m pytest tests/unit/test_benchmark_fla_stages.py -q
```

Tests cover registry/matrix types, compiled typed factories, output/final-state
and VJP recurrence checks, FLA layout/state arguments, exclusion of external
layout graphs, VJP-only timing, allocation accounting and continued baseline
attempts after a Meta resource rejection.

### Actual H20 smoke (2026-10-12, remote allinai2plus)

Final source validation: **34 passed**, including six GPU inference tests,
with `CUDA_VISIBLE_DEVICES=0 ... pytest --run-gpu`; Ruff check/format passed.
The CPU-only invocation reports 28 passed and six GPU skips. No git index
changes were made by this implementation scope.

B1/H1/K=V64, BF16, nonzero initial states, workspace cap **8 GiB**,
warmup=1/reps=2. These are **exploratory** public-call CUDA-event medians
in milliseconds, not stable performance claims. Both backend gradient input
sets exclude center/scale. KDA chunk numbers below are intentionally omitted
from comparison because its public API has the normalization mismatch.

| Family | T | Meta fwd | FLA chunk fwd | FLA recurrent fwd | Meta VJP | FLA chunk VJP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDA | 1 | 1.333 | mismatch | 0.126 | 1.451 | mismatch |
| KDA | 64 | 1.355 | mismatch | 0.168 | 3.198 | mismatch |
| KDA | 2048 | 2.190 | mismatch | 2.255 | 7.081* | mismatch |
| KDA | 8192 | 2.235 | mismatch | 8.701 | 8.863* | mismatch |
| GDN | 1 | 1.502 | 0.799 | 0.116 | 2.385 | 1.613 |
| GDN | 64 | 1.611 | 0.822 | 0.147 | 4.166 | 1.466 |
| GDN | 2048 | 2.593 | 0.871 | 1.779 | 5.649* | 1.524* |
| GDN | 8192 | 5.060 | 0.847 | 6.770 | 9.097* | 1.581* |
| LightNet | 1 | 0.975 | 0.512 | 0.239 | 1.364 | 0.806 |
| LightNet | 64 | 1.359 | 0.501 | 0.262 | 2.580 | 0.834 |
| LightNet (retry) | 2048 | 2.258 | 0.495 | 2.923 | 3.847* | 0.868* |
| LightNet (retry) | 8192 | 1.967 | 1.318 | 11.469 | 10.018* | 1.276* |

`*`: full gradient oracle not checked at long T (`MEASURED_UNVALIDATED`),
not an accuracy-qualified backward comparison. All twelve Meta forward
output/final-state checks passed the CPU oracle. GDN and LightNet FLA chunk
and recurrent forward also passed; KDA recurrent passed at all lengths,
whereas KDA chunk mismatched at all lengths. T1/64 VJP checks passed for
Meta and for FLA GDN/LightNet. No speedup claim is drawn from these two-sample
medians; notably CUDA-event and CPU-wall values can diverge for short FLA
calls, so the raw synchronized wall samples must also be consulted.

Inference-selected runtime children were `chunked-stateful` +
`factored-recurrence` for KDA/GDN and `segmented-logaddexp` +
`vector-propagation` for LightNet. Backward selects a reference preconditioner
for KDA/GDN and an axis-scale child for LightNet; these are recorded explicitly,
not presented as the same native forward schedule.

At T8192, Meta absolute allocator peaks / staged forward estimates were:
KDA **87.6 / 268.9 MiB**, GDN **99.7 / 316.3 MiB**, LightNet **20.2 / 54.4 MiB**.
Corresponding conservative backward estimates were approximately
**4.94 / 4.99 / 5.97 GiB**, explaining use of the 8 GiB budget. Allocator
peaks and estimates describe different quantities; all per-backend peaks,
resident bytes and samples are retained in JSON.

Evidence on the SSH host (outside the tracked source files):

- `/tmp/fla_stages_final_smoke.json` and `.log`: isolated GPU0 run,
  twelve requested records. LightNet T2048 completed all backend checkpoints
  but the child subsequently exited with signal 6; the post-exit PID observation
  is conservatively labeled `CONTENDED_EXTERNAL_PROCESS` (could include a
  not-yet-reaped worker). The first T8192 attempt was `GPU_BUSY_REJECTED`.
  **Those original records remain unchanged and are not used in the table.**
- `/tmp/fla_stages_lightnet_retry.json` and `.log`: clean, successful fresh
  GPU0 children for LightNet T2048/8192, no external PID observed at boundaries.
  These supply the two explicitly labeled retry rows above.
- `/tmp/fla_stages_smoke.json` and `.log`: earlier non-isolated GPU1 trial,
  nine completed records before an external roughly 81 GiB GPU1 job appeared
  and the process aborted at LightNet T64. Retained for diagnosis only; not
  used for final timing or memory conclusions.
- `/tmp/fla_stages_tests.log`: final 34-pass CPU/GPU contract suite.

All benchmark/waiting processes launched for this task have finished. The
GPU0 retry satisfied the requested four lengths for all three families;
original contention/fatal-exit evidence was retained rather than overwritten.

## Thirty-repetition H20 measurement (2026-10-12)

Source commit: `741c32fa53b93a8337dc747f9105eb8fa670f013`; FLA commit:
`f3bec720f7847d4f176872b34f002cc580e2cc6e`. GPU0, B1/H1/K=V64, BF16,
nonzero initial states, 8 GiB workspace cap, **warmup=5/reps=30**.
The same numerical and timing contracts above apply. All twelve workloads
completed; GPU snapshots before and after every isolated workload observed
no external compute PID. This is one measured run of these fixed shapes.

[Tracked JSON](benchmarks/stages-741c32f.json) retains all CUDA-event and
synchronized wall samples, per-tensor oracle errors, lowerers, allocator peaks,
workspace estimates, GPU snapshots and source hashes. The raw remote log is
`/tmp/fla_stages_perf_741c32f.log`.

Public CUDA-event medians in milliseconds:

| Family | T | Meta fwd | FLA chunk fwd | FLA recurrent fwd | Meta VJP | FLA chunk VJP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDA | 1 | 1.234 | mismatch | 0.091 | 1.376 | mismatch |
| KDA | 64 | 1.277 | mismatch | 0.141 | 3.147 | mismatch |
| KDA | 2048 | 1.411 | mismatch | 2.238 | 4.526* | mismatch |
| KDA | 8192 | 1.867 | mismatch | 8.683 | 8.525* | mismatch |
| GDN | 1 | 1.429 | 0.777 | 0.091 | 2.271 | 1.358 |
| GDN | 64 | 1.532 | 0.766 | 0.168 | 5.964 | 2.088 |
| GDN | 2048 | 2.279 | 0.801 | 1.752 | 5.535* | 1.440* |
| GDN | 8192 | 4.774 | 0.827 | 6.738 | 9.144* | 1.630* |
| LightNet | 1 | 1.215 | 0.435 | 0.179 | 1.269 | 0.765 |
| LightNet | 64 | 1.403 | 0.413 | 0.220 | 2.656 | 0.739 |
| LightNet | 2048 | 1.626 | 0.452 | 2.918 | 3.642* | 0.847* |
| LightNet | 8192 | 1.773 | 1.309 | 11.428 | 10.041* | 1.234* |

`*`: long-sequence full VJP oracle remains unchecked. All twelve Meta forward
output/final-state checks passed. Meta T1/64 VJPs passed. GDN/LightNet FLA
forward and short VJP checks passed. KDA chunk remains a semantic mismatch
because its public API normalizes q/k; its timing samples remain in the JSON
but are excluded from this table. KDA recurrent forward passed at all lengths.

At T8192, synchronized CPU-wall medians for Meta/FLA chunk were
**4.797/0.848 ms** for GDN and **1.803/1.327 ms** for LightNet. Meta KDA
was **1.888 ms**, compared with **8.705 ms** for the matching FLA recurrent
baseline. The latter is a recurrent schedule comparison, not a matched chunk
prefill comparison. LightNet T1 varied appreciably within this run: the sorted
sample interval from index 2 to index 26 was **0.845–1.531 ms**. Raw samples
should be consulted before extrapolating short-call results.

T8192 allocator peaks and conservative public workspace bounds (MiB):

| Family | Meta forward peak | FLA chunk forward peak | Meta VJP peak | FLA chunk VJP peak | Meta forward bound | Meta backward bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDA | 87.6 | 18.3 | 201.2 | 46.6 | 268.9 | 5058.6 |
| GDN | 99.7 | 12.3 | 225.6 | 25.8 | 316.3 | 5105.1 |
| LightNet | 20.2 | 11.0 | 1452.8 | 97.1 | 54.4 | 6112.3 |

KDA chunk memory figures describe the mismatched public API and are diagnostic
only. Allocator peaks include resident tensors; bounds are conservative
resource estimates and must not be interpreted as measured peak memory.

The selected child runtimes agree with the earlier smoke: chunked preconditioner
plus factored memory for KDA/GDN inference, segmented LogAddExp plus vector
propagation for LightNet inference. KDA/GDN backward uses the reference
preconditioner; LightNet backward uses axis-scale memory. The current public
stage calls cost about 1–1.5 ms even at T1/64. GDN T8192 is about 5.8 times
the matched FLA chunk latency, and LightNet about 1.35 times. Stage execution
overhead, auxiliary scan execution, and the differentiable child runtimes are
priorities for subsequent optimization. These measurements identify the
overall cost; they do not isolate each child or establish a causal breakdown.

The same source commit also received a 30-repetition ABC readout measurement;
see [ABC results](abc-fast-path-assessment.md#三十次重复的复测2026-10-12).
Abs/Cast are included in the preconditioned expressions and their full public
timing; this run does not attribute a separate latency to each expression node.
