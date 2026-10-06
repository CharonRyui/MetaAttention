# TileLang migration research for Stateful Operator

**Snapshot date:** 2026-10-05.  Sources below are first-party TileLang/TVM-FFI metadata or this repository.  “Current” means the upstream `main` branch visible at the snapshot date; the latest published release is v0.1.15.

The repository has now completed the Stateful lowerer rewrite against the released TileLang `0.1.15` API. Every Stateful kernel family that previously imported Triton—general diagonal, readout regions, low-rank/WY/checkpoint, optimized scalar, rank-one scan, and generated expression loading—now has a TileLang `PrimFunc` implementation. This is a kernel rewrite, not an import substitution; the independent numerical references and shape-specialized ABI remain.

The migration kept the safe sequence: pin the release and matching TVM-FFI, audit the existing TileLang JIT/PrimFunc/Kernel ABI, port one family at a time, and verify each family against authored-order references. Stateful source no longer imports Triton; external FLA/legacy paths are separate.

Current implementation status below supersedes the initial feasibility snapshot; the original matrix and historical notes remain useful for explaining why each family required a separate rewrite.

## Upstream version and release state

PyPI identifies `tilelang 0.1.15` as latest and requires **Python >=3.10**: [PyPI project metadata](https://pypi.org/project/tilelang/).

* The v0.1.15 release is tagged at commit [`a35f8ddf45eba16c21211ec8822d56ce5363036f`](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15), published 2026-09-30.  It reports four commits on `main` after that release at the time GitHub rendered the release page: [v0.1.15 release metadata](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).
* As of the requested 2026-10-05 snapshot, the official GitHub API reports `main` at [`d82101ff08acea8d3e3ca7d0ab40985a3ca2a781`](https://github.com/tile-ai/tilelang/commit/d82101ff08acea8d3e3ca7d0ab40985a3ca2a781), committed 2026-10-03: [GitHub commits API](https://api.github.com/repos/tile-ai/tilelang/commits/main).  Pinning `main` rather than v0.1.15 is therefore not equivalent to pinning the release; migration should record an exact commit.
* v0.1.12 (the release family corresponding to the repository's cache/version label) is tagged at [`2d63708c8ad57196051c4636a1167c5453c73a48`](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12).  Its release notes describe the LLVM backend, tile scheduler, backend registry, per-backend launch materialization, cross-host CUDA binary cache, and compile-option cache-key changes: [v0.1.12 migration/release notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12).

### This repository's baseline

* Before this migration, `pyproject.toml` allowed Python `>=3.10,<3.14`, selected the PyTorch CUDA 12.8 index, and overrode unversioned TileLang with git revision [`61f968b6c5c00c8986b9331d599f95dfee7a60cd`](https://github.com/tile-ai/tilelang/commit/61f968b6c5c00c8986b9331d599f95dfee7a60cd) (`pyproject.toml` history and the pre-migration lock). The pre-migration generated cache used the runtime identity `0.1.12_cuda_git61f968b6`. The current project declaration is `tilelang==0.1.15`, with the release recorded in `uv.lock`.
* Existing TileLang lowerers use the lazy-builder shape: import `tilelang`/`tilelang.language as T`, define a Python shape-specializing builder under `@tilelang.jit`, define `@T.prim_func`, annotate fixed `T.Tensor` shapes, launch through `T.Kernel(..., threads=...)`, and invoke the returned adapter.  Representative paths are [`stateful/lowering/scan/affine.py:23-47`](../attention_engine/attn_engine/stateful/lowering/scan/affine.py#L23-L47), [`scan/elementwise.py:79-116`](../attention_engine/attn_engine/stateful/lowering/scan/elementwise.py#L79-L116), [`scan/vector.py:28-62`](../attention_engine/attn_engine/stateful/lowering/scan/vector.py#L28-L62), and [`scalar/scalar.py:41-77`](../attention_engine/attn_engine/stateful/lowering/scalar/scalar.py#L41-L77).
* The former Triton families now use TileLang implementations: [`diagonal/diagonal_native.py`](../attention_engine/attn_engine/stateful/lowering/diagonal/diagonal_native.py), [`lowrank/checkpoint.py`](../attention_engine/attn_engine/stateful/lowering/lowrank/checkpoint.py), [`readout/readout_regions.py`](../attention_engine/attn_engine/stateful/lowering/readout/readout_regions.py), [`scalar/scalar_kernels.py`](../attention_engine/attn_engine/stateful/lowering/scalar/scalar_kernels.py), [`scan/rank1.py`](../attention_engine/attn_engine/stateful/lowering/scan/rank1.py), and [`expression/expression_kernels.py`](../attention_engine/attn_engine/stateful/lowering/expression/expression_kernels.py). No Triton reference remains in Stateful source.

## Installation and compatibility constraints

### Official requirements

The official installation guide says wheels/PyPI require **glibc 2.28 (Ubuntu 20.04 or later), Python >=3.10, and NVIDIA CUDA >=10.0 on the host, or the pip-provided CUDA toolchain >=13.0**.  ROCm requires a host ROCm installation providing `hipcc`: [official Installation Guide](https://www.tilelang.com/get_started/Installation.html) and its source text [Installation.md](https://www.tilelang.com/_sources/get_started/Installation.md.txt).

For source builds, the same guide requires Linux or Windows, Python >=3.10, and either a host CUDA toolchain or pip CUDA packages; source checkout should be recursive because customized TVM is a submodule.  It documents `USE_CUDA`, `USE_ROCM`, `USE_ASCEND`, and `USE_LLVM`, and says `TVM_ROOT` can select an existing TVM checkout (with path caveats): [Installation.md source](https://www.tilelang.com/_sources/get_started/Installation.md.txt).

The official README's backend matrix records CUDA as primary (SM70--SM120 code paths), HIP/ROCm as supported on Linux, Metal as supported on Apple silicon, LLVM CPU as experimental requiring LLVM 15+, and CuTe DSL as experimental.  It also says Ascend 950 requires a source build with CANN and `torch_npu`: [README.md, platform/backend matrix](https://github.com/tile-ai/tilelang/blob/main/README.md#platform-and-backend-support).

### PyTorch, TVM-FFI, and Triton

* TileLang 0.1.15's authoritative PyPI metadata lists `requires_python: >=3.10` and runtime requirements `apache-tvm-ffi>=0.1.11,<0.1.13`, `torch`, `torch-c-dlpack-ext` for Python <3.14, `cloudpickle`, `ml-dtypes`, `numpy>=1.23.5`, `psutil`, `tqdm>=4.62.3`, `typing-extensions>=4.10.0`, and `z3-solver>=4.13.0,<4.15.5`; its optional `nvcc` extra supplies `nvidia-cuda-nvcc>=13.0.48`, `nvidia-cuda-cccl>=13.0.50`, and Windows CUDA runtime packages.  This is the concrete package constraint set, rather than an inferred PyTorch/Triton matrix: [PyPI 0.1.15 JSON metadata](https://pypi.org/pypi/tilelang/0.1.15/json).  The repository's `<3.14` cap is therefore its own policy, not TileLang's declared Python upper bound (`pyproject.toml:6`).
* TileLang's official PyPI metadata declares Python >=3.10 and the PyPI description recommends installing `tilelang` separately from the chosen PyTorch build; the official quick start uses PyTorch CUDA tensors: [PyPI metadata](https://pypi.org/project/tilelang/), [README quick start](https://github.com/tile-ai/tilelang/blob/main/README.md#quick-start).  TileLang does not promise a single universal PyTorch version in its user-facing installation guide.  The v0.1.15 release changelog records a CI cap of `torch<2.14` for CUDA tests until flash-attn supports torch 2.14; this is a CI/test constraint, not a blanket TileLang runtime contract: [v0.1.15 compatibility/changelog](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).
* Current TileLang package metadata pins the runtime ABI dependency to `apache-tvm-ffi>=0.1.11,<0.1.13` (official upstream `pyproject.toml`, dependency table): [main `pyproject.toml`](https://github.com/tile-ai/tilelang/blob/main/pyproject.toml).  This matters for the repository's old git pin: v0.1.12 release notes explicitly mention pinning `apache-tvm-ffi` to compatible versions ([v0.1.12 notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12)), and upstream has documented import/core-dump hazards when mixing incompatible TileLang and `apache-tvm-ffi` versions ([TileLang issue #2367](https://github.com/tile-ai/tilelang/issues/2367), [issue #2552](https://github.com/tile-ai/tilelang/issues/2552)).  Do not independently upgrade `apache-tvm-ffi` around TileLang's bound.
* Triton is **not** the TileLang kernel API or a stated TileLang prerequisite.  The repository's Triton lowerers import `triton` directly; current TileLang's official dependency/API pages instead describe TVM/TIR and `apache-tvm-ffi` execution.  Therefore “upgrade TileLang” does not upgrade or validate the repository's Triton kernels, and “migrate Triton to TileLang” means replacing those kernels rather than changing an import.  Source evidence for the two sides: [TileLang `pyproject.toml`](https://github.com/tile-ai/tilelang/blob/main/pyproject.toml), [TileLang JIT API](https://tilelang.com/autoapi/tilelang/jit/index.html), and repository examples above.

## Current JIT, PrimFunc, launch, and layout APIs

### JIT and execution modes

The current JIT API documents `tilelang.jit` as an auto-tuning/JIT wrapper compiling TileLang PrimFuncs with TVM. `tilelang.compile` accepts `out_idx`, `execution_backend` (`auto`, `tvm_ffi`, `cython`, `nvrtc`, `torch`, `cutedsl`, or `pto`), `target`, `target_host`, `verbose`, `pass_configs`, and `compile_flags`; target and execution backend can also come from `TILELANG_DEFAULT_TARGET`, `TILELANG_EXECUTION_BACKEND`, and `TILELANG_VERBOSE`: [official `tilelang.jit` API](https://tilelang.com/autoapi/tilelang/jit/index.html).

The same API documents two modes: **lazy** mode returns a compiled kernel adapter when the decorated function returns a `PrimFunc`, while **eager** mode uses the builder pattern and compiles/executes on invocation.  Existing Stateful TileLang builders return a `PrimFunc`, so they align with lazy mode; a migration should not silently switch mode because output allocation/return behavior differs: [JIT execution modes](https://tilelang.com/autoapi/tilelang/jit/index.html#execution-modes).

### `T.prim_func`, `T.Tensor`, and `T.Kernel`

The language guide states that kernels are TVM TIR functions made with `@T.prim_func`; arguments are annotated by `T.Tensor` or `T.Buffer` with concrete or symbolic shapes/dtypes.  It shows PyTorch dtypes are accepted and normalized: [Language Basics, defining a kernel](https://www.tilelang.com/programming_guides/language_basics.html#defining-a-kernel-with-t-prim-func).

`with T.Kernel(grid_x, grid_y, ..., threads=...)` declares the target-independent program grid and returns program indices.  The current guide says CUDA treats a program as a thread block, CPU maps the grid to outer loops, and backend dialects own accepted launch annotations.  The CUDA dialect exposes `threads`, `prelude`, and `cluster_dims`; ROCm/Metal expose `threads` and `prelude`; CPU exposes `prelude`.  Kernels that query thread extents during tracing should provide `threads=` explicitly.  This is a migration hazard for code relying on old implicit or backend-specific launch parameters: [Language Basics, launching work](https://www.tilelang.com/programming_guides/language_basics.html#launching-work-with-t-kernel), [v0.1.15 compatibility notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).

The current language guide retains `T.Pipelined`, `T.Parallel`, `T.serial`, `T.unroll`, shared/fragment allocation, `T.copy`, and `T.async_copy`; current v0.1.15 compatibility notes remove the old `sync`/`group` arguments from `T.Pipelined` and `k_pack` from `T.gemm_sp`: [instructions/loops and memory](https://www.tilelang.com/programming_guides/language_basics.html#loops-and-control-flow), [v0.1.15 compatibility notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).

### Layout APIs

Current code and API docs expose layout helpers under `tilelang.layout`, including `make_swizzled_layout` and `make_volta_swizzled_layout`: [official swizzle API](https://tilelang.com/autoapi/tilelang/layout/swizzle/index.html).  The v0.1.12 notes specifically call out TMA lowering for arbitrary/swizzled shared-memory layouts and GMMA/UMMA lowering for sliced/arbitrary shared-memory layouts: [v0.1.12 CUDA/layout notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12).  Existing repository code already uses `tilelang.layout.make_swizzled_layout` in non-Stateful kernels ([`attention_engine/attn_engine/_flash_qla_hopper/fused_bwd.py:193-210`](../attention_engine/attn_engine/_flash_qla_hopper/fused_bwd.py#L193-L210)); a Stateful port should use documented layout objects rather than manually reproducing Triton's implicit lane mapping.

## Dynamic and symbolic dimensions

Current TileLang documentation supports two complementary forms:

1. `T.dyn["K"]` in a `T.Tensor` annotation for a named symbolic dimension; and
2. `T.dynamic("K", "int32")` for a term-level TIR variable usable in body expressions/loops.

The guide recommends reading a bound extent from `A.shape[0]` in the body, says symbols need not be separate kernel arguments because they are bound from argument shapes, and says under `@jit` concrete sizes come from actual tensor arguments at first call: [dynamic symbolic dimensions](https://www.tilelang.com/programming_guides/language_basics.html#dynamic-symbolic-dimensions-two-ways).

This is feasible for the Stateful contract's runtime Batch/Sequence/token extents **only if the kernel body and ABI are written against dynamic buffer shapes**.  Existing lowerers instead bake `token_count`, `sequence_count`, `rows`, and `columns` into Python closure values and annotations (for example `affine.py:9-17,35-40` and `elementwise.py:82-100`).  That is a specialization strategy, not dynamic-shape support: it creates a distinct compiled kernel/cache entry per shape.  A migration can either preserve that behavior (simpler, but not one dynamic binary) or rewrite signatures to `T.dyn`/`T.dynamic` and audit every `T.Kernel` extent, guard, allocation, and output ABI.

The v0.1.15 release also records a fix for dynamic-output allocation when the sizing input follows the output in the argument list ([PR #3207](https://github.com/tile-ai/tilelang/pull/3207)); this is evidence that dynamic output ABI ordering remains a concrete area to test during migration.  Current release notes also say `T.symbolic` is deprecated in favor of `T.dynamic` (and describe its compatibility history), so new code should use `T.dynamic`: [v0.1.15 compatibility notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).

## Migration hazards and history-derived checklist
1. **Pin exact versions together.** This migration selects the released v0.1.15 rather than moving to unreleased `main`; `uv.lock` records its matching `apache-tvm-ffi` dependency range. Avoid reusing the old cached 0.1.12 artifacts; TileLang's cache identity changes with the package/toolchain. Sources: [repository `pyproject.toml`](../pyproject.toml#L29-L40), [repository `uv.lock`](../uv.lock#L2335-L2358), [upstream pyproject](https://github.com/tile-ai/tilelang/blob/main/pyproject.toml), [v0.1.12 notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12).
2. **Audit launch signatures.** Add explicit `threads=` where code depends on thread extents; remove obsolete `T.Pipelined(sync=..., group=...)` and `T.gemm_sp(k_pack=...)`; backend-specific annotations now belong to the backend dialect. Source: [v0.1.15 compatibility notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15).
3. **Audit symbolic names and shape binding.** Use `T.dyn[...]`/`T.dynamic(...)`, not new uses of `T.symbolic`; decide per kernel whether runtime extents are truly dynamic or Python-specialized. Source: [dynamic dimensions guide](https://www.tilelang.com/programming_guides/language_basics.html#dynamic-symbolic-dimensions-two-ways).
4. **Do not translate Triton mechanically.** Triton's `tl.constexpr`, raw pointer arithmetic, `tl.load/store`, Triton `grid` tuples, and `num_warps` are not the equivalent of `T.Tensor`, `T.copy`, `T.Kernel`, and TileLang layouts. Each diagonal/lowrank/readout/expression family needs a TileLang PrimFunc and a separately reviewed schedule/ABI. Repository evidence: [diagonal_native.py](../attention_engine/attn_engine/stateful/lowering/diagonal/diagonal_native.py), [lowrank_kernels.py](../attention_engine/attn_engine/stateful/lowering/lowrank/lowrank_kernels.py), [readout_regions.py](../attention_engine/attn_engine/stateful/lowering/readout/readout_regions.py), [expression_kernels.py](../attention_engine/attn_engine/stateful/lowering/expression/expression_kernels.py).
5. **Treat layouts and synchronization as explicit.** TileLang `T.alloc_shared`, `T.alloc_fragment`, `T.copy`/`T.async_copy`, `T.Pipelined`, and `tilelang.layout.*` participate in compiler layout inference and synchronization. Recreate the intended memory layout with documented TileLang objects; do not assume Triton lane order carries over. Sources: [Language Basics memory/copy](https://www.tilelang.com/programming_guides/language_basics.html#memory-scopes-and-allocation), [layout API](https://tilelang.com/autoapi/tilelang/layout/swizzle/index.html), [v0.1.12 arbitrary-layout notes](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12).
6. **Separate feasibility from performance claims.** Official material documents supported backends, APIs, and release changes, but does not establish that a TileLang rewrite will outperform these Triton kernels. Any performance decision requires an apples-to-apples benchmark on the target GPU/toolchain; no such claim is made here.

## Feasibility matrix

| Stateful area | Current implementation | TileLang migration status | Required work |
| --- | --- | --- | --- |
| `scan/affine.py`, `scan/elementwise.py`, `scan/vector.py`, `scalar/scalar.py` | TileLang `jit` + `PrimFunc` + `Kernel`; Python-specialized extents | **Migrated and exercised** | Keep the released ABI and shape-specialized cache identity. |
| Identity/general diagonal and hierarchical carry | TileLang summary, carry, replay, softmax/shared readout and native loaders | **Migrated and exercised** | Preserve FP32 State arithmetic, authored BF16 rounding, prior flags, tails and final-State ownership. |
| Low-rank/WY/checkpoint and backward (`checkpoint.py`, `lowrank_kernels.py`, `wy_kernels.py`) | TileLang summary/carry/replay/VJP kernels | **Migrated; GPU evidence below** | Keep device-side metadata/temporary buffers and ordered operations. |
| Optimized scalar and rank-one scan (`scalar_kernels.py`, `scan/rank1.py`) | TileLang chunk/replay and vector scan kernels | **Migrated and exercised** | Preserve signed/zero gates, continuation and grouped/strided bindings. |
| Generated expressions and readout regions | TileLang expression IR/PrimFunc and native regions | **Migrated and exercised** | Preserve authored node rounding and shape-dependent readout ABI. |


## Repository change and verification record

The dependency declaration and lock now select released `tilelang==0.1.15` from PyPI instead of the old `61f968b6` git snapshot, with `apache-tvm-ffi 0.1.11`. Existing TileLang scan/scalar builders retain the lazy `@tilelang.jit` / `@T.prim_func` / `T.Kernel(..., threads=...)` interface.

### Initial identity/ProductFeature slice: historical measurements

- [`diagonal/identity.py`](../attention_engine/attn_engine/stateful/lowering/diagonal/identity.py): identity chunk summaries and contractions, including ProductFeature key factors. Paired numerator/normalizer summaries share a launch. FP32 GEMMs split operands into high/residual TF32 terms and use three products rather than a single precision-losing TF32 product.
- [`diagonal/identity_scan.py`](../attention_engine/attn_engine/stateful/lowering/diagonal/identity_scan.py): exclusive FP32 prefixes. Direct prefixes cover up to 64 chunks; longer sequences use 32-chunk recursive groups. The normalized pair shares each scan launch and reuses consumed summaries at direct levels; long applies keep source/output separate. Unrequested finals do not overwrite initial States.
- [`readout/product_kernels.py`](../attention_engine/attn_engine/stateful/lowering/readout/product_kernels.py): ProductFeature score and State contractions use TileLang FP32 GEMMs. The first normalized readout experiment was rolled back under the earlier performance gate; the subsequent correctness-first full migration replaces that Triton path too.
- The ProductFeature path preserves factored left-major/right-minor indexing, reusable chunk score ABI, explicit carry ownership, independent prior flags, and FP32 output accumulation.

These measurements predate the full migration. The user subsequently removed the performance gate: all Stateful families are migrated, without claiming latency parity with their previous schedules.

Verification for the retained ProductFeature slice:

- `tests/functional/test_product_contraction.py`: **6 passed** on TileLang 0.1.15.
- Same-shape H20 benchmark against the previous Triton implementation: `B=2,H=2,T=131,left=35,right=7,C=35` measured TileLang **0.0707 ms** vs Triton **0.0997 ms** (**1.41x**); `B=1,H=8,T=2048,left=32,right=32,C=64` measured TileLang **0.3741 ms** vs Triton **0.8614 ms** (**2.30x**).
- ProductFeature benchmark correctness differences were bounded by FP32/TF32 accumulation order: maximum absolute output difference `0.00031` on the short case and `0.00185` on the long case; denominator differences were `0.00014` and `0.00114` respectively. Independent authored-order correctness tests still passed.
- The initial normalized TileLang experiment was rolled back after the H20 `B=1,H=8,T=2048,rows=columns=64` comparison measured **0.1881 ms** vs Triton **0.0881 ms** (**0.468x**). This is historical evidence, not a benchmark of the final full migration.

### Earlier identity/affine verification

The carry and identity kernels preserve Python shape specialization; the migration does not introduce one dynamic Batch/Sequence binary. No general performance claim is made beyond the historical ProductFeature measurements above.


 - `tests/functional/test_stateful_readout_regions.py -k test_parallel_diagonal_block_fp32_oracle`: **40 passed** with scalar/row/column/dual gates, lengths 1/63/64/65/137 and both prior choices.
 - Long affine carry smoke at `T=4161` (**66 chunks**) with signed/zero row and column gates: **passed** against an independent FP32 recurrence; final State and all chunk carries matched.

Verification uses TileLang `0.1.15` from `/dev/shm/metaattention-tilelang-015`, PyTorch `2.11.0+cu128`, Triton `3.6.0` and `apache-tvm-ffi 0.1.11`. GPU work used an idle NVIDIA H20; compiler/test caches were directed to tmpfs.

- Earlier identity-slice `tests/unit`: **192 passed, 3 skipped**; current full-migration unit run: **192 passed, 6 skipped**.
- `tests/functional/test_stateful_normalized_pair.py` plus `test_stateful_hierarchical_carry.py`: **28 passed**. Coverage includes FP32/BF16, grouped BHT/BTH inputs, independent keys, all four prior combinations, tail chunks, continuation, initial-State immutability, FP32 finals, lossless-promotion invariance and the 65-chunk hierarchy boundary.
- Ruff checks on the seven changed lowering modules: **All checks passed**.
- Public `StatefulOperator.compile(..., mode="dense", optimized=True)` smoke: BF16 inputs at `B=2, H=2, T=130, D=17, DV=35`, nonzero initial States, output/final parity against an independent CPU recurrence, and CUDA Graph replay all passed. The CUDA profiler observed **3 launches** for the warmed call: paired summary, paired direct prefix, normalized readout. This is a launch-count observation, not a latency benchmark.

The repository `.venv` was not replaced: the initial upgrade attempt ran out of space on `/mnt/nvme`, so the release was exercised through a tmpfs overlay. The checked-in dependency/lock upgrade and the installed environment are separate facts; use the updated installation instructions to recreate the pinned environment. No old generated cache is evidence for the new release.

### Full migration integration evidence

- Stateful source audit: no `triton` imports, generated Triton source, or Triton calls remain under `attn_engine/stateful/` or `stateful_operator.py`.
- Rank-one/FLA integration run: **56 passed**, with 18 upstream deprecation warnings. The FLA baselines still use their own backend independently.
- `tests/functional/test_lowrank_wy.py`: **12 passed**, covering FP32/BF16, `T=1/17/67`, multiple rank/injection/readout events, noncommuting row scales, rank-only transitions without injections, strided/broadcast factors, zero/signed gates and continuation.
- Comprehensive Stateful regression: **215 passed** across readout regions, normalized pairs, hierarchical carries, ProductFeature contractions, scalar/vector scans, elementwise/two-axis recurrences, linear-attention examples, dense checkpoints and packed checkpoints.
- `tests/functional/test_gdn_h20.py`: **11 passed**, including GDN and compositional dense/packed forward/backward integration.
- Packed forward/backward CUDA Graph smoke passed after warming/creating autograd nodes on the capture stream. Replaying the same captured binary after changing device offsets from `[0,17,17,19]` to `[0,0,3,19]` matched eager outputs, every gradient and final State exactly; empty segments moved without rebuilding the graph.
- `uv lock --check --offline`: resolved **100 packages** with the existing release lock. Ruff checks passed for the migrated `stateful/lowering/` modules, selection contract and low-rank behavior tests; Python compile checks passed. Whole-Stateful Ruff also reports two pre-existing unused-symbol findings in `runtime/bindings.py`, unchanged by this cutover.
- Actual GPU smokes passed for public native multiple-injection prior/post readouts, BF16 authored scalar-node rounding, initial-State immutability, checkpoint forward/all-operand VJPs, scalar and vector public execution, expression packed/grouped/sliced bindings, diagonal signed/zero gates and ProductFeature/bonus readouts. Native, checkpoint, scalar, WY and generic mixed rank/row/column/whole-State recurrence CUDA Graph capture/replay passed.
- WY integration corrected missing gate storage, mutable scalar reductions (`T.alloc_var`), shared-memory synchronization, inverse-free interval products, and restored eligibility dispatch. Scalar gates commute with rank-one linear operators; zero/signed gates use direct products, never division by a prefix.
- A BF16-output boundary probe at `T=67` measured FP32-vs-FP64 maximum output error **1.9e-7**. At one midpoint, values `-0.1105956957` and `-0.1105957165` rounded to adjacent BF16 values. Regression coverage checks FP32 recurrence separately and bounds BF16 storage error against the unrounded oracle by half an ULP plus FP32 accumulation tolerance.

**Bottom line:** the Stateful backend cutover targets released TileLang 0.1.15, with shape-specialized kernels and independent correctness checks. Performance is not guaranteed; the current schedules must be benchmarked before making any throughput or latency claim.


