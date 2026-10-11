# Reusable execution paths for the slow stateful examples

The largest gaps in the H20 example benchmark come from lost structural proofs
and dense intermediate representations. GDN2, IPLR and DPLR can already reuse
an existing low-rank executor after recognizing their simple readout. Delta
Product needs weighted outer-product recognition as well. Oja needs a recurrence
whose update axis and readout axis differ. ABC already has competitive device
execution; its public-call overhead should be measured separately.

This is a diagnosis and implementation proposal against `1dfac2d`, not an
engine change. Example and benchmark commits are `9e71bae` and `1dfac2d`.
The workspace is `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl`.

## Measured bottlenecks

The [benchmark report](stateful-example-benchmarks.md) and its
[records](measurements/stateful-examples-h20/records.json) establish correctness
for these forward comparisons. At B=1, H=8, T=2048, K=V=64:

| Family | Meta ms | FLA ms | Meta / FLA | Meta incremental allocation MiB | Selected lowerer |
| --- | ---: | ---: | ---: | ---: | --- |
| GDN2 | 46.050 | 2.902 | 15.87x | 3328.0 | `reference-stateful` |
| IPLR | 45.406 | 2.271 | 19.99x | 3328.0 | `reference-stateful` |
| DPLR | 46.035 | 1.782 | 25.83x | 3328.0 | `reference-stateful` |
| Gated Oja | 81.474 | 3.858 | 21.12x | 4100.1 | `tilelang-product-gemm` |
| Delta Product | 49.251 | 0.750 | 65.71x | 3328.0 | `reference-stateful` |
| ABC | 0.891 | 0.470 | 1.90x | 26.1 | `tilelang-chunked-stateful` |

Times are public-call CUDA-event medians. FLA uses public fused recurrent APIs
for the first four rows and chunk APIs for the last two. These ratios do not
compare against the fastest FLA prefill kernel for every family. The final run
used an 8 GiB workspace limit; the default 1 GiB rejected the first five rows.
Increasing that limit allowed execution but did not remove dense intermediates.

For scale, one FP32 token-state history here is
`B * H * T * K * V * 4 = 256 MiB`. Materializing token-level transforms, biases,
scan buffers and state histories accounts for multiple such buffers. This
explains the direction of the memory problem; it is not an allocation-by-allocation
profile. Oja's Graph time is 80.917 ms, so its gap persists without ordinary host
launch overhead. ABC's Graph times are 0.208 ms for Meta and 0.235 ms for FLA.

## Experiment: recover the existing low-rank path

GDN2, IPLR and DPLR author `Readout(StateTerm(memory, query, KEY))`.
`rank_one_factored_eligible()` in
[algorithm.py](../attention_engine/attn_engine/stateful/analyse/algorithm.py)
requires `StateContraction` instead. Meanwhile,
`requires_reference_lowering()` in
[reference.py](../attention_engine/attn_engine/stateful/lowering/reference.py)
classifies any `Readout` as needing reference execution unless a direct
structured executor handles it. At the measured long shape, the candidate list
for each of these three programs contains only `reference-stateful`.

A GPU probe changes only that readout to the equivalent post-transition
`StateContraction`. It uses identical inputs for each pair, B=1, H=8, T=512,
K=V=64, BF16 data and FP32 gates/state, zero initial state, and requests both
output and final state. Three warmups and ten repetitions exclude compilation.
All six executions pass the independent authored-order FP32 CPU token oracle.

| Family | Original ms | Canonical readout ms | Original / canonical | Original / canonical allocation MiB |
| --- | ---: | ---: | ---: | ---: |
| GDN2 | 10.420 | 0.817 | 12.76x | 832.0 / 17.2 |
| IPLR | 10.460 | 0.748 | 13.98x | 832.0 / 24.2 |
| DPLR | 10.823 | 0.844 | 12.83x | 832.0 / 17.2 |

All canonical variants select `tilelang-factored-recurrence`. IPLR satisfies
the scalar/rank-one compact-WY predicate. GDN2 and DPLR have row scaling, which
that predicate rejects; they use the existing ordered affine-summary/checkpoint
schedule. Canonical variants capture successfully, with Graph times of 0.259,
0.338 and 0.263 ms respectively. Original variants fail capture.

This is evidence for recovering an existing execution path, not a measurement
of an implemented compiler fix or a T=2048 speedup. The probe does not cover
nonzero initial states, continuation, packed inputs, gradients or unequal feature
sizes. Those remain required before changing shared compiler behavior.

Raw [probe records](measurements/stateful-structure-h20/readout-records.json),
[GPU probe](measurements/stateful-structure-h20/readout_probe.py) and
[structural eligibility records](measurements/stateful-structure-h20/eligibility.json)
are preserved alongside this report. To repeat the GPU experiment from the
repository root, on an idle GPU 0:

```bash
PYTHONPATH=. uv run --group fla-benchmark python \
  docs/measurements/stateful-structure-h20/readout_probe.py
```

It writes JSON to `/tmp/meta_slow_structure_probe.json`. The probe deliberately
uses the benchmark's internal input generation, oracle and timing helpers;
it is tied to the revision analyzed here.

## Compiler seam: a proved matrix-recurrence recipe

Extend the existing `RankOneFactoredIR` proof rather than adding a model-name
registry. The analyzer should return a complete recipe that lowerers consume;
it should not merely return eligibility and leave runtime dispatch to re-parse
the authored IR. Today `_execute_factored_recurrence()` in
[stateful_operator.py](../attention_engine/attn_engine/stateful_operator.py)
assumes exactly two injection factors and asserts every readout is a
`StateContraction`. Changing the predicate alone would leave these consumers
inconsistent.

The recipe should hide role orientation, expression loading and supported
factorization behind a small interface. It needs these semantic facts:

- The public state layout and the internal update axis, including any orientation
  mapping for initial state, final state and gradients.
- An ordered list of scalar/axis scales and rank-one propagations, with their
  original expression DAGs and dtype/rounding boundaries.
- Outer-product injection terms, their feature-free factors and authored
  multiplication order.
- Readout terms with state, operand, contracted axis, prior/post-transition timing
  and final output dtype, plus any supported downstream readout expression.

Its consumers are candidate enumeration and costing in
[plan.py](../attention_engine/attn_engine/stateful/compile/plan.py), factor
binding, forward execution and checkpointed backward. An unsupported program
retains its existing semantic fallback. Kernel choices belong below this seam:
short register recurrence, ordered chunk summaries/replay and compact WY.
Model names remain useful in examples and benchmarks, not in eligibility rules.

### 1. Recognize single-term readouts

Recognize a `Readout` whose entire expression is a post-transition `StateTerm`
contracting the same matrix state. Preserve its operand DAG, output name and
output dtype. Do not fold an arbitrary `Divide`, `Softmax`, prior-state term or
multiple-state expression into this case. Those require a separately proved
readout recipe.

The immediate beneficiaries are GDN2, IPLR and DPLR, plus the readout portion
of Delta Product. The implementation should keep the authored Algorithm IR and
its CPU/reference evaluation intact while giving the GPU executor the proved
readout view. A direct replacement of example nodes is insufficient: a small
CPU probe of all three canonical variants fails in `_lower_parallel_scan()`
when a zero-dimensional propagation coefficient is indexed as a token tensor
(`IndexError: too many indices for tensor of dimension 0`). This is a constraint
on dispatch and a regression case, not a failure of the mathematical equivalence.

### 2. Recognize weighted outer products

For two Delta Product writes, the authored injection factor counts are
`[2, 4, 2]`. The middle term has a KEY vector, a VALUE vector and two
feature-free coefficients, one of which is a `Contract` reduction.
`rank_one_factored_eligible()` currently accepts exactly two factors. The
canonical-readout probe therefore still rejects this program's factored path.

Represent the term as a weighted outer product whose coefficient factors are
kept separately from its two vector factors. Evaluate each input-only reduction
once per token, then consume the vectors and scalar factors without building
`[B,H,T,K,V]` injection tensors. Preserve multiplication order: moving a
coefficient into a BF16 vector or reassociating FP32 products can change rounding.
A coefficient-aware kernel is safer than an unconditional authored-DAG rewrite;
any reassociation needs the repository's numerical contract and oracle checks.

This gives a reusable representation for sums of weighted rank-one writes,
not just Delta Product. It also needs matching factor VJPs and resource estimates.
Existing evaluation/dispatch code must consume the recipe rather than unpack
`first, second = injection.factors`.

Recovering this proof does not automatically enable WY. Two propagations and
three expanded injections give five events; the current predicate allows at
most four. First reuse the general ordered low-rank executor. Then evaluate
adaptive event tiling or an input-only contraction that combines the transformed
write into one vector per original write. For two writes, that could reduce the
count to two propagation events and two write events, but it reassociates
arithmetic and is an untested optimization here. Increasing the WY event limit
alone increases padded local matrices and is not a performance argument.

The example's explicit transformed-write expansion grows as `2**n - 1` for
`n` writes. A later compiler composition recipe can retain a compact input-only
DAG of transformed vectors or derive ordered affine substeps. It must prove
that earlier writes receive later propagations; the current IR applies all
propagations before its injection sum. Public mid-transition writes are not
needed to optimize the current fixed small examples.

### 3. Support an independent readout axis

Oja updates the VALUE axis and reads along KEY. The existing factored proof
requires both rank-one updates and readouts on the first feature axis. Merely
transposing the state changes which condition fails; the structural probe
confirms both orientations remain ineligible.

For `S` of shape `[K,V]`, the example computes

```text
S' = S diag(exp(gate))
S  = S' - beta (S' v) v^T + beta k v^T
output = q^T S / sqrt(K)
```

With `U = S^T` of shape `[V,K]`, the equivalent update is

```text
U' = diag(exp(gate)) U
U  = U' - beta v (v^T U') + beta v k^T
output = U q / sqrt(K)
```

The update now fits a left-axis low-rank recurrence, while the readout contracts
the other axis. This motivates an orientation-aware recurrence and readout
recipe. It can also support future right-axis delta updates without introducing
an Oja-specific API.

The executor must change as well as analysis. Current checkpoint replay tiles
state columns and reduces rows for readout; `U q` reduces across column tiles.
For small K/V, a block holding the full state can perform both reductions.
Larger shapes need a different tile ownership or partial output reductions.
Expose those as resource-bounded kernel candidates. Carry/final-state layout
and backward must restore the public `[K,V]` orientation. Avoid materializing a
token-sized state history or a token-level `[V,V]` transform just to reuse GEMM.

No Oja GPU speedup is claimed by this diagnosis. Its orientation proof is
mathematical and structural; a correct implementation and shape sweep remain.

### 4. Reuse normalized readout regions for ABC

ABC already selects the chunked stateful executor ahead of reference execution.
It has independent `[K,M]`, `[M,V]` and `[M]` accumulator states, followed by a
normalization, slot softmax and second contraction. This belongs to the
multi-state readout-region machinery in
[readout_regions.py](../attention_engine/attn_engine/stateful/lowering/readout/readout_regions.py),
not the single-matrix rank-one proof.

Given its 0.208 ms Graph time versus 0.891 ms public time, first profile binding,
expression evaluation, launch submission and temporary allocation. The
public-minus-Graph difference is a diagnostic signal, not an exact attribution
to Python; Graph replay changes submission and allocation behavior too. Cache
static bindings and fuse input-only expressions where the profile supports it.
If device execution later needs optimization, extend the existing region recipe
to share the normalizer and chunk carries across both contractions.

Keep the three raw accumulator states and their continuation contract. FLA
exposes two normalized caches, so final-state/continuation comparisons remain
incomparable without a proved state conversion. Adding logcumsumexp stabilization
would change the state representation and belongs to a separate design.

## Schedule selection and implementation order

The common recurrence is an ordered composition of scale/rank-one maps plus
factored writes. Its dimensions for planning are update/readout axis, feature
sizes, event count, chunk length, expression cuts and requested output contract.
They determine resource use more directly than a family name.

1. Recover simple readouts with a recipe consumed consistently by analysis,
   dispatch and backward. Reuse current kernels; rerun GDN2/IPLR/DPLR at T=512
   and T=2048 under the default workspace budget.
2. Add coefficient-aware outer writes and reuse ordered chunk execution for
   Delta Product. Measure compact write composition and WY event tiling only
   after this correctness milestone.
3. Add independent readout orientation and resource-bounded state tiling for
   Oja. Measure unequal K/V to catch hidden transpose assumptions.
4. Reduce binding/packing overhead shared by the recovered paths and ABC.
   Consider diagonal-plus-low-rank chunk factorization for row-gated programs
   only after profiling the existing chunk executor.

Compact WY currently requires scalar scales, one to four events and feature
sizes no greater than 128. GDN2/DPLR row gates need the general ordered schedule.
A future gated-WY candidate cannot commute a diagonal gate through a rank-one
update; it must transform factors in authored order, bound intermediate values
and avoid unstable inverse gates. Existing inverse-free checkpointed backward
should remain the baseline.

Costing should charge the actual schedule's packing, chunk boundaries, readout
partials and output storage, with separate forward/backward context contracts.
The recovered probes still make 19–25 profiler activities per call, including
packing/expression work, despite only 3–5 classified kernel launches. That
suggests a follow-up opportunity but does not identify every activity's cost.
Workspace enforcement must cover tail and packed fallback schedules as well as
the ideal aligned case.

Before merging an engine change, require oracle output/final-state/continuation
checks with zero and nonzero initial state, grouped heads, unequal K/V, tails
and packed segments; compare gradients for every newly admitted training path.
Preserve CPU behavior and negative eligibility cases such as nonlinear readouts,
prior-state reads and genuinely unfactored injections. Record the runtime
schedule, public/Graph timing and incremental allocation across short/decode
and long/prefill shapes. Compare long-sequence candidates against FLA chunk
kernels where available. An eligibility unit test alone cannot establish that
the runtime consumed the new recipe or that dense allocations disappeared.
