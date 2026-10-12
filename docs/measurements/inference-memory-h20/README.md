# Dense inference memory on H20

Baseline: `65d33bfac3837ca851f6974e3a363367f8c57638`. Measurements were collected with the memory changes in the working tree; the final runner source digest and dependency versions are in `manifest.json`. The manifest commit identifies the baseline, not a clean optimized checkout.

Workload: B=2, H=16, T=8192, K=V=64, BF16 inputs, FP32 State. Peak incremental memory is the warmed PyTorch allocator peak minus pre-call allocations; it includes returned outputs and requested States. It excludes inputs already allocated before the call.

## Factored recurrence

Dense inference prepares factors and replays one 1024-token window at a time, carrying the mathematical State and retaining complete outputs. WY carry overwrites each consumed additive buffer with the response used by readout. Workspace charges window scratch and full outputs. Backward and packed paths retain their existing schedule.

Fixed seed 123, final-State contract, two warmups and median of five CUDA-event calls. Baseline uses an 8 GiB cap to execute; optimized uses the default 1 GiB. Raw measurements are `factored-before.json` and `factored-after.json`.

| Family | Before MiB | After MiB | Reduction | Before ms | After ms |
|---|---:|---:|---:|---:|---:|
| delta-rule | 1572.6 | 209.5 | 86.7% | 9.20 | 10.38 |
| kda | 1253.6 | 189.6 | 84.9% | 10.23 | 10.87 |
| gdn2 | 1379.6 | 205.4 | 85.1% | 10.40 | 11.14 |
| comba | 1671.6 | 221.9 | 86.7% | 9.47 | 10.68 |
| iplr | 1698.6 | 225.3 | 86.7% | 9.39 | 10.62 |
| dplr | 1315.6 | 197.4 | 85.0% | 10.41 | 11.20 |
| oja | 1254.1 | 190.1 | 84.8% | 9.92 | 12.34 |
| delta-product | 2763.6 | 342.4 | 87.6% | 49.85 | 52.45 |

Outputs/finals passed differential checks against the baseline. This is a memory improvement with a measured 5–24% latency cost, not a speedup claim. Delta-product remains invalid against the independent FP32 oracle on the runner's seed 42; the baseline also fails. Its memory measurement does not imply numerical acceptance.

## Scan Stage DAG

The remaining six workspace rejects in `stateful_fla_meta_20261012T072127Z` came from full-length upstream trajectories and child execution. Dense inference now completes the entire DAG per window and carries every stage's State. Windows shrink from 1024 by halves until the conservative stage budget plus retained outputs fits. Programs using Shift retain full-sequence execution because their input history is not carried by mathematical States. Backward and packed schedules remain unchanged.

The chunked readout evaluator also had mutually recursive closures retaining per-call tensor caches until cyclic GC. Breaking their reference cycle after evaluation releases scratch immediately, including on evaluation exceptions.

Isolated measurements, seed 123, final-State contract, three warmups and ten event samples:

| Family | Window | Estimated MiB | Peak incremental MiB | Event ms | Retained bytes after releasing result |
|---|---:|---:|---:|---:|---:|
| preconditioned-kda | 512 | 608.7 | 126.2 | 21.41 | 0 |
| preconditioned-gdn | 512 | 701.7 | 136.4 | 25.19 | 0 |
| lightnet | 1024 | 283.3 | 129.5 | 12.31 | 0 |

Raw measurements are in `isolated_memory.json`. Timings above are independent final-State measurements and must not be mixed with the formal output-only matrix as speedup ratios.

## Full runner and validation

Final full matrix: 25 families × recurrent/prefill, B2/H16/T8192, output-only, 1 GiB, ten warmups, thirty repetitions, public and Graph timing. The six runtime rejects become NOT_COMPARABLE: Meta executes and passes its oracle, but installed FLA 0.5.2 lacks `fla.ops.precond_kda`, `fla.ops.precond_gated_delta_rule`, and `fla.ops.lightnet`.

Status distribution: 32 OK, 0 META_RUNTIME_REJECTED, 7 FLA_INVALID, 9 NOT_COMPARABLE, 2 META_INVALID. Meta is VALID in 48 records. `summary.csv` exposes comparison timings only for valid comparable pairs; staged Meta standalone timings remain in the copied `status_changes.json`. `correctness.json` records all fifty correctness verdicts and tolerances.

- CPU full unit suite: 537 passed, 96 skipped.
- Related GPU suite: 97 passed, 4 skipped, 2 failed. Both failures are existing two-axis GLA dense/packed errors reproduced at the baseline commit; no tolerances changed.
- Added GPU tests cover long-window peak and immediate release, independent oracle, tails, non-square State, continuation, CUDA Graph replay, and unchanged initial State. Stage unit checks cover prior readouts, multiple exports, logical axis permutations, State order, insufficient output budget, and Shift boundaries.
- Earlier related factored/ordered suite: 77 passed and six BF16 backward failures; all six were also reproduced on the baseline. They remain outside this memory change.
- Ruff and `git diff --check` pass.

Remaining numerical failures: delta-product Meta recurrent/prefill; FLA prefill for linear, retention, retnet, gsa, comba, iplr, and oja. FLA is a comparison implementation, not the independent correctness oracle.

Full logs and raw runner records remain in the ignored remote artifact directory `benchmark_results/stateful_fla_meta_staged_fix_final_20261012`; earlier factored logs are in `benchmark_results/factored_memory_optimization`.
