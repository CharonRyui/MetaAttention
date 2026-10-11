# Compiled expression bindings on H20

The dense diagonal executor now classifies input-only and State-dependent
expressions at compilation and assigns integer cache slots. Structurally equal
expressions share a slot, including separately constructed expression objects.
Runtime lookup uses expression identity and integer slots rather than recursive
IR hashing and repeated dependency walks. The recipe retains the expression
trees through its transition and readout bindings. Tensor results remain local
to each invocation; a second call binds its own inputs and initial States.

Authored arithmetic, dtype boundaries, readout dependence, prior-State semantics,
GPU kernels and output/final-State layouts are unchanged. Eligibility follows
the existing mathematical IR proof, with no model-name dispatch. Packed and
differentiable calls retain their existing execution paths.

## Paired public-call measurement

Workspace: `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl`, NVIDIA H20 GPU 3.
B=1, H=8, K=V=64, BF16 inputs/output and FP32 accumulators, output plus final
State. The baseline is `execute_chunked_states` from `d1289d6`, using the same
compiled operator, recipe and input tensors as the new executor. Compilation is
excluded. Five rounds alternate executor order, with ten warmups and one hundred
repetitions per measurement. The table uses the median of the five per-round
medians. Every paired output and final State is bitwise equal.

| Construction | Tokens | Public before / after ms | CPU wall before / after ms |
| --- | ---: | ---: | ---: |
| ABC | 128 | 0.8232 / 0.7541 | 0.8177 / 0.7349 |
| ABC | 2048 | 0.9044 / 0.8428 | 0.8869 / 0.8300 |
| GSA | 128 | 0.3848 / 0.3851 | 0.3849 / 0.3894 |
| GSA | 2048 | 0.4409 / 0.4399 | 0.4388 / 0.4391 |
| Based | 128 | 0.3830 / 0.3810 | 0.3758 / 0.3758 |
| Based | 2048 | 0.6813 / 0.6816 | 0.6762 / 0.6826 |

ABC public-call reduction is 8.4% at 128 and 6.8% at 2048. It improves in all
five paired rounds at both lengths. GSA and Based use native recipes whose
runtime does not visit this expression cache; their variations are submission
noise, and no speedup is claimed for them. These are single-device measurements,
not confidence intervals or a hardware-wide performance guarantee.

GPU work remains unchanged: ABC has 26 CUDA activity submissions, including
kernels and copies. Graph times are approximately 0.076 ms at 128 and 0.200 ms
at 2048. The paired run's 2048 after-Graph sample is 0.206 ms; the independent
FLA comparison below measures 0.200 ms. No device speedup is claimed.

Fresh-process allocator measurements, taken before any Graph capture or
profiling, are identical before/after: 1,775,616 bytes (1.69 MiB) at 128 and
27,411,456 bytes (26.14 MiB) at 2048. Use `memory-records.json` for allocation
comparisons. The raw paired/FLA records also contain warmed allocator deltas
that can be reduced by storage already retained in Graph/profiler pools;
those deltas do not establish a memory reduction.

## Independent ABC/FLA comparison

The output-only comparison uses H20 GPU 3, ten warmups and fifty repetitions.
Both 128- and 2048-token points pass the independent authored-order FP32 oracle,
FLA output correctness and pairwise output comparison. FLA is installed 0.5.2.

| Tokens | Meta / FLA public ms | Meta / FLA Graph ms |
| ---: | ---: | ---: |
| 128 | 0.8112 / 0.6485 | 0.0760 / 0.0651 |
| 2048 | 0.8218 / 0.4612 | 0.1997 / 0.2298 |

At 2048, Meta device execution is competitive, but its public call remains
approximately 1.78 times FLA's time. This change removes one source of CPU
binding overhead; it does not close the public-call gap. ABC continues to carry
three raw accumulator States, unlike FLA's normalized key/value State contract.
The comparison is output-only and does not claim comparable continuation States.

## Rejected pointwise-fusion candidate

Compiling pointwise expressions into separate expression launches reduced ABC
submissions from 26 to 22. At 128, Graph time improved from 0.07650 to 0.06723 ms,
but public time regressed from 0.82726 to 0.94243 ms because runtime expression
binding added Python work. At 2048 public time also regressed, from 0.95917 to
0.99550 ms. The fusion candidate was removed. Its negative measurements are
preserved in `rejected-fusion-records.json`; this directory's code optimizes
compile-time classification/cache binding only.

## Validation and reproduction

The new public-call tests cover 64x128 and 128x64, BF16/FP32, shared query/key
heads, two readouts, nonzero initial States, a 67-token chunk tail, continuation,
input changes between invocations, dense/packed gradients for every input and
initial State, and empty packed segments. They also require that dense inference
uses the chunked executor without runtime dependency walks, and that distinct
equal expressions share a compiled slot.

The shared chunked oracles now include the authored `key + 0.1` expression and
its BF16 rounding before FP32 promotion. The whole-State decay test uses
supported 64x128 feature sizes instead of unsupported 3x5. These oracle/config
mismatches predate the optimization; production example formulas are unchanged.
The larger shared readout suite contains existing public-call cases with
unsupported feature sizes, including 13x35. The first such case fails the
{64,128} compile-time guard before the changed executor runs. Setting
`STATEFUL_OPERATOR_TEST_CPU=1` bypasses only the compile-time guard; runtime
validation still rejects the dimensions. These cases are excluded from the
supported shared regression and retained unchanged.

Validation: 12 new GPU tests passed; the standalone chunked suite passed all
7 tests; shared GPU regression passed 186 tests with 100 cases deselected from
the three existing unsupported-dimension test groups listed below. CPU regression
passed 100 tests with 65 GPU cases skipped. Ruff checks, formatting and
`git diff --check` passed.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 .venv/bin/python docs/measurements/chunked-bindings-h20/paired_probe.py
for length in 128 2048; do
  for label in before after; do
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 .venv/bin/python docs/measurements/chunked-bindings-h20/memory_probe.py "$label" "$length"
  done
done
.venv/bin/python testing/benchmark_fla_stateful.py --families abc \
  --seqlens 128 2048 --batches 1 --heads 8 --timing both \
  --warmup 10 --repetitions 50 --gpu 3 --output-dir /tmp/meta_binding_fla
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_chunked_expression_bindings.py -q --run-gpu -x
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_chunked_elementwise.py -q --run-gpu -x
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/functional/test_stateful_readout_regions.py tests/functional/test_chunked_elementwise.py tests/unit/test_delta_family_examples.py tests/unit/test_stateful_readout_normalization.py tests/unit/test_stateful_expression.py -q --run-gpu -x -k 'not native_diagonal_authored_bindings_and_continuation and not native_softmax_typed_cut_and_nonzero_initial and not native_bonus_parallel_rows_and_chunk_tail'
.venv/bin/pytest tests/unit/test_delta_family_examples.py tests/unit/test_stateful_readout_normalization.py tests/unit/test_stateful_expression.py tests/unit/test_stateful_workspace.py tests/unit/test_stateful_factored_recipes.py -q
```

Raw timing, provenance, correctness and allocator records are checked in beside
this document. Measurement provenance records the base commit plus the source
hash because measurements precede the optimization commit.
