# Delta-product ordered oracle correction on H20

The benchmark FP32 token oracle previously applied all propagations before all injections, ignoring the interleaved steps authored by delta-product. It now iterates `StateTransition.ordered_steps`, so each rank-one action consumes the State after the preceding write. Legacy grouped transitions preserve their order and prior-State readouts retain their token-start snapshot. Unsupported steps raise TypeError rather than being skipped; this change does not add coupled StateInjection support to the benchmark oracle.

GPU kernels, State representation, BF16 expression evaluation, tolerances and workspace limits are unchanged. The previous delta-product META_INVALID records were oracle errors, not evidence of a kernel precision failure.

## Reproduction and tests

The regression exercises the shared oracle called by both Stateful benchmark runners. Zero initial State, one token, K=V=2, gate=0, beta=1:

- First write: key=[1,0], value=[1,2].
- Second write: key=[0.5,0.5], value=[0,0].
- Correct ordered State=[[0.75,1.5],[-0.25,-0.5]]. Old grouped oracle gives [[1,2],[0,0]], max_abs=0.5.

Before the fix, the thirteen new precision checks yielded nine failures and four passes (one-step controls). After the fix they all pass. Unit coverage includes 1/2/3 writes, FP32/BF16, zero/nonzero initial State, grouped heads, unequal Feature sizes, prior/current readout timing, legacy grouped order and continuation.

Original B2/H16/T8192 seed 42 now has an independent residual-recurrence precision assertion in `test_factored_memory.py`, alongside the existing <512 MiB incremental allocator bound. A new functional test compares the compiled public call with a direct FP32 ordered residual recurrence at T=1/16/17/128/129/1023/1024/1025, FP32/BF16, three writes, grouped heads, K64/V128, nonzero initial State. T1025 also checks non-aligned continuation at token 1001, CUDA Graph replay and initial State immutability.

Validation:

- CPU full unit suite: **551 passed, 96 skipped**.
- Related factored/staged/ordered GPU inference suite: **23 passed**, 24 gradient tests deselected (backward code is unchanged).
- New independent boundary GPU suite: **16 passed**.
- Ruff and git diff --check pass.

Logs accompany this report. The pre-existing BF16 ordered-backward and two-axis GLA failures documented in the memory report are outside this correction; neither tolerance nor implementation was changed for them.

## Formal runner

H20, B=2, H=16, T=8192, K=V=64, BF16 inputs, FP32 State, seed 42, 1 GiB budget, ten warmups, thirty repetitions, public and CUDA Graph timing. Original FLA 0.5.2 installation. Manifests identify `ea78f3e` plus the modified working-tree runner digest; that commit alone does not contain the oracle fix.

Delta-product six records (recurrent/prefill × output-only/final-state/continuation): **all six Meta correctness checks VALID**. FLA prefill passes all three contracts, pair checks also pass. FLA has no public recurrent API for this family, so its three records are NOT_COMPARABLE.

| Contract | Meta output max_abs | Meta State max_abs | FLA prefill output max_abs | FLA prefill State max_abs |
|---|---:|---:|---:|---:|
| output-only | 0.0039052963 | — | 0.0103394985 | — |
| final-state | 0.0039052963 | 2.9802322e-7 | 0.0103394985 | 0.0052198768 |
| continuation | 0.0039052963 | 2.9802322e-7 | 0.0103394985 | 0.0052198768 |

Unchanged runner tolerances: output rtol=0.02/atol=0.05; State rtol=0.01/atol=0.02. Meta incremental peak is 342.4 MiB for full calls and 342.9 MiB for continuation. Public Meta latency is 52.5–52.9 ms; FLA prefill is 1.96 ms for full calls and 2.27 ms for continuation. These are standalone measurements, not an A/B performance claim.

The original 25-family, 50-record output-only matrix also completed:

| Status | Before oracle fix | After |
|---|---:|---:|
| OK | 32 | 33 |
| META_INVALID | 2 | **0** |
| META_RUNTIME_REJECTED | 0 | **0** |
| NOT_COMPARABLE | 9 | 10 |
| FLA_INVALID | 7 | 7 |

Meta is VALID in **all 50 records**. The only status changes are delta-product recurrent → NOT_COMPARABLE and prefill → OK. FLA prefill still fails its oracle for linear, retention, gsa, retnet, comba, iplr and oja; corresponding Meta checks pass.

The tracked six-record JSON includes timings, memory and correctness. Full matrix correctness, summary CSV, manifests and status comparison are also tracked here. Full raw records remain in `benchmark_results/stateful_fla_meta_oracle_fix_20261012` on the remote project.

Reproduce from the project root, selecting an idle GPU:

```bash
PYTHONPATH=attention_engine .venv/bin/python -m pytest tests/unit -q
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=attention_engine .venv/bin/python -m pytest \
  tests/functional/test_delta_product_precision.py \
  tests/functional/test_factored_memory.py tests/functional/test_staged_memory.py \
  tests/functional/test_ordered_lowrank.py --run-gpu -q -k 'not gradients'
PYTHONPATH=attention_engine .venv/bin/python testing/benchmark_fla_stateful.py \
  --families delta-product --batches 2 --heads 16 --seqlens 8192 \
  --contract all --fla-baseline both --timing both --gpu 4 \
  --output-dir benchmark_results/delta_product_oracle_fix_repeat
```
