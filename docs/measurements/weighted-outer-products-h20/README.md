# Weighted outer-product writes on H20

The factored recurrence now recognizes a Product Injection containing exactly
one vector for each State Feature role, plus any number of feature-free scalar
factors. The internal `WeightedOuterProductIR` records vector positions, scalar
positions, and authored multiplication order. Recognition depends on structure,
not an attention family name.

The execution path keeps each factor separate through input-expression
binding, forward summaries, replay, and checkpointed reverse execution. It does
not materialize the per-token matrix write. Scalar derivatives multiply the
remaining factors directly, so zero coefficients do not require division.
Scalar factor storage and partial gradients are included in dense and packed
workspace estimates. Weighted writes use the ordered checkpoint schedule;
compact WY remains available to the existing unweighted case.

Readout recognition is unchanged. In particular, a `Readout(StateTerm(...))`
does not become eligible for the factored recurrence through this change. This
probe explicitly constructs equivalent `StateContraction` nodes to isolate the
weighted-write capability. The authored Delta Product example still encounters
the existing readout restriction. Oja axis support and ABC multi-State readout
optimization are outside this implementation.

## Measurement

Environment: NVIDIA H20, driver 580.126.20, PyTorch 2.11.0+cu128, CUDA 12.8,
TileLang 0.1.15. The probe uses GPU 0, batch 1, 8 heads, 64 key features, 64 value
features, BF16 inputs, FP32 State, and output plus final-State execution.
Compilation is excluded. Measurements use three warmup calls and ten timed
calls through `testing.benchmark_stateful_plans._measure_call`.

| Tokens | Dense public CUDA time | Factored public CUDA time | Ratio | Dense incremental allocation | Factored incremental allocation |
|---|---:|---:|---:|---:|---:|
| 512 | 17.786 ms | 1.199 ms | 14.83× | 1088.02 MiB | 28.29 MiB |
| 2048 | 84.920 ms | 2.243 ms | 37.86× | 4352.09 MiB | 112.77 MiB |

Both columns execute the same IR with explicit `StateContraction` readouts.
The input-only `Contract` for the overlap coefficient is precomputed in FP32
for both columns: the existing dense fallback cannot broadcast the original
scalar reduction correctly. This preprocessing is outside the timed calls.
Both comparisons use an 8 GiB compiler workspace budget. Incremental allocation
is the peak PyTorch CUDA allocated memory minus allocated memory immediately
before one warmed invocation; it includes returned outputs and States. It is
not reserved memory or a direct count of matrix-write allocations.

The probe also executes the original input-only `Contract` inside the weighted
binding, using the default 1 GiB compiler workspace budget. That version takes
1.217 ms / 2.257 ms and allocates 28.30 MiB / 112.83 MiB incrementally at
512 / 2048 tokens. All six variants pass the independent authored-order FP32
CPU token oracle. For the factored variants, maximum output error is below
0.0039 and maximum final-State error is below 2.1e-7 against that oracle.
The benchmark oracle evaluates input expressions in FP32; separate functional
tests check BF16 expression rounding and gradients.

These are single-workload measurements, with no confidence interval or
performance gate. They demonstrate the benefit of avoiding the dense fallback
for this structural probe; they do not establish a speedup for the unchanged
public Delta Product example.

Reproduce from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python docs/measurements/weighted-outer-products-h20/probe.py
```

The script overwrites `records.json` next to itself. The committed records
include public timing, CUDA Graph timing, profiler launches, incremental
allocation, chosen lowerer, and correctness evidence.

## Validation

Final regression results: 80 GPU tests passed across weighted writes, checkpoint
execution, packed checkpoint execution, and compact WY; 131 CPU tests passed
with 18 GPU-dependent tests skipped. Ruff lint, formatting checks for the
changed formatted files, and `git diff --check` passed.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_weighted_lowrank.py tests/functional/test_stateful_checkpoint.py tests/functional/test_stateful_packed_checkpoint.py tests/functional/test_lowrank_wy.py -q --run-gpu -x
.venv/bin/pytest tests/unit/test_weighted_lowrank.py tests/unit/test_delta_family_examples.py tests/unit/test_stateful_operator.py tests/unit/test_stateful_workspace.py tests/unit/test_stateful_bindings.py tests/unit/test_stateful_selection.py -q
```

`tests/unit/test_weighted_lowrank.py` checks factor positions and order,
unsupported matrix-write exclusion, unchanged `StateTerm` recognition, dense
workspace accounting for inference/training, and packed scalar-gradient
accounting.

`tests/functional/test_weighted_lowrank.py` checks public inference,
continuation, nonzero initial State, grouped heads, input-only `Contract`,
BF16 expression rounding, FP32/BF16 gradients for output-only, final-State-only,
and joint losses, zero coefficients, multiple weighted terms, long sequences,
and packed execution with an empty segment. An extreme scalar-first case would
overflow if execution formed the outer product first, so it makes authored
multiplication association observable in both inference and recorded execution.

The BF16 gradient oracle binds each input expression once before token replay.
Casting separately inside each token changes BF16 gradient accumulation
rounding. It also keeps independently authored expressions separate rather than
sharing casts across them. This matches expression-binding boundaries without
reusing compiler recurrence or gradient kernels.
