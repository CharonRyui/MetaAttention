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
| Low-rank/WY/checkpoint and backward (`checkpoint.py`, `lowrank_kernels.py`, `wy_kernels.py`) | TileLang summary/carry/replay/VJP kernels | **Migrated; targeted GPU evidence** | Preserve the established WY fast path; direct factor binding remains limited to non-WY eligible plans until its full ABI matrix is proven. |
| Optimized scalar and rank-one scan (`scalar_kernels.py`, `scan/rank1.py`) | TileLang chunk/replay and vector scan kernels | **Migrated; optimized scan candidate reverted** | The fragment-backed scan was rejected by TileLang layout inference; shared ping-pong remains the verified implementation. |
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
### Shape-specialized optimization pass

The optimization pass keeps public State/input/output ABIs and authored FP32 accumulation order unchanged:

- Scalar carry uses a 512-element fragment schedule for regular States with at least 4096 elements and an exact 512-element shape; smaller or irregular States retain the 256-element fallback.
- Checkpoint carry removes the redundant state-tile staging copy for exact 32/64-row States; padded and multi-tile States retain the generic path.
- ProductFeature readout uses a 16-column tile when `columns <= 16`; wider outputs retain the 32-column schedule.

Focused verification: `test_stateful_checkpoint.py`, `test_lowrank_wy.py`, `test_product_contraction.py`, and `test_scalar_parallel.py` completed **60 passed** on an idle NVIDIA H20; `test_gdn_h20.py` completed **11 passed**.

The post-change matrix was partially interrupted after valid records for linear, simple-gla, GLA, normalized-linear, retention, RWKV7, RWKV6, GSA, HGRN, Based, Rebased, GDN, and RetNet. The records showed large improvements for several chunked/product families, but RWKV7 was marked invalid by the independent oracle in that run and GDN reported approximately 1.26 s at `B=1,H=1/8,T=2048`; these are blockers, not performance successes. Full target parity with Triton/FLA is not established.
### Follow-up diagnosis

RWKV7's earlier invalid long-matrix result was not reproducible after isolating the benchmark invocation and restoring the original factored dispatcher. Focused GPU coverage passed **24 tests**; the representative `B=1,T=2048,D=64,DV=64` records were valid with max absolute output error `0.0306` at H=1 and `0.0305` at H=8.

The long GDN profile is a genuine blocker: one `kernel_kernel` consumes approximately **1,262,603 us** at `B=1,H=8,T=2048,D=64,DV=64`, while preparation/copy operations are below 0.1 ms. This isolates the issue to the factored recurrence kernel, not public-call preparation. The current optimization therefore preserves the proven WY eligibility boundary and does not claim GDN performance parity.

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


### RWKV7 inference schedule restoration (2026-10-06)

The previous claim that the RWKV7 benchmark exercised scalar WY was incorrect.
`examples/rwkv7_state.py` authors a row-scale followed by rank-one propagation;
scalar-only WY eligibility rejects it. Profiling the migrated inference path
observed one full-sequence kernel taking 9.174 ms at B1/H8/T2048/K64/V64.

`lowrank/lowrank_kernels.py` now routes non-WY inference through the existing
device-segmented checkpoint summary/carry/replay schedule rather than a
single-block token loop. `checkpoint.py` restores Tensor Core carry contractions
using the existing three-product FP32/TF32 macro, bounded K=32 tiles, and staged
fragment-to-shared copies. Authored operation order, inverse-free recurrence,
packed segmentation, local VJPs and initial/final-State ownership are unchanged.
This is the schedule used by the pre-migration Triton checkpoint implementation,
not a claim that row-scaled RWKV7 has scalar compact WY semantics.

H20 GPU 3, B1/K64/V64, same-input final-state A/B, three alternating rounds,
warmup 3 and 10 repetitions per round; table entries are medians of round medians:

| H | T | Before public ms | After public ms | Speedup | Before Graph ms | After Graph ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.682 | 0.656 | 1.04x | 0.289 | 0.083 |
| 1 | 2048 | 8.858 | 0.918 | 9.65x | 8.424 | 0.442 |
| 8 | 64 | 0.702 | 0.639 | 1.10x | 0.306 | 0.092 |
| 8 | 2048 | 9.671 | 1.445 | 6.70x | 9.492 | 1.008 |

Raw rounds: `benchmark_results/stateful_tilelang015_chunkcarry_ab.json`.
The separate 12-point output-only/final-state/continuation matrix is in
`benchmark_results/stateful_tilelang015_chunkcarry_after/`; all points passed
the runner's independent authored-order CPU recurrence check. FLA remains faster:
T2048 output-only Meta/FLA = 0.977/0.596 ms (H1), 1.474/0.680 ms (H8).
These are finite workload measurements, not a statistical confidence-interval gate.

Tradeoff: warmed incremental allocator peak rises from 8,708,096 to 13,725,696
bytes at H1/T2048 and 69,550,080 to 109,662,208 bytes at H8/T2048 (about 58%).
Chunk transform/bias/entry workspace replaces the near-workspace-free serial walk;
the existing factored candidate already charges affine chunk storage.

Isolated H8/128-chunk/K64/V64 carry Graph smoke: 0.629 to 0.427 ms (1.47x),
with independent FP64 chunk-entry parity; raw observation is in
`benchmark_results/stateful_tilelang015_carry_smoke.json`. Public nonzero-State
smoke B2/H2/T130/K64/V128 passed recurrence parity, split-at-63 continuation,
initial immutability, and exact warmed CUDA Graph replay. Profiler observed
summary/carry/replay durations 0.274/0.408/0.229 ms at H8/T2048 after cutover;
they remain the main device-side optimization targets.

Scalar-only WY separately uses chunk-parallel ordered event solves, compact U/W/V
carry contractions and chunk-parallel readout, with no dense affine detour.
Chunk length stays 16; single-rank-only event tiles pad to at least 32 to satisfy
the four-warp Tensor Core layout. Nonmultiple Feature/event tiles use explicit
zero padding, and GEMM accumulators are cleared before every independent product.
An attempted 32-token/128-row schedule exceeded H20 dynamic shared memory
(278,528 bytes); it was not retained.

Final GPU regression command covered `test_lowrank_wy.py`, `test_gdn_h20.py`,
`test_stateful_checkpoint.py` and `test_stateful_packed_checkpoint.py`: **54 passed**.
New regressions cover 65-row carry reduction tails and three-event WY padding
with a 31-row strided State. CPU unit suite: **192 passed, 6 skipped**. Changed
lowerers/tests passed Ruff; changed lowerers passed Python compilation.
Private scalar-only B1/H8/T2048/K64/V64 smoke passed independent FP64 output/final
parity with zero/signed gates and CUDA Graph replay; new Graph latency was
0.878 ms. Its HEAD serial baseline measured 1262.346 ms, not the previous live
checkpoint detour and not RWKV7; do not reuse that ratio as a public RWKV7 claim.
Raw observation/probe: `benchmark_results/stateful_tilelang015_wy_smoke.json`.

### Triton-guided scalar product scans (2026-10-07)

Compared the live TileLang scalar lowerer with `perf/structured-lowering` at
`1366a12`, whose `_chunk_summary` and `_chunk_readout` use
`tl.associative_scan(..., _multiply)`. TileLang 0.1.15 exposes cumulative sum/max,
not a general associative product scan. The migrated implementation therefore
uses an explicit six-level product tree, including a 64-by-64 interval matrix.

`scalar/scalar_kernels.py` now alternates source/destination shared buffers over
pairs of levels instead of copying each intermediate back. Both 1D gate scans
and 2D interval scans keep the same multiplication tree, padding and FP32 casts;
each scan has six explicit synchronization points instead of twelve. No gate division,
log-prefix subtraction, public ABI, dispatch or global workspace change is made.
The 2D scan declares a 16 KiB shared scratch tile instead of borrowing the score
fragment for intermediate values; measured identity kernels erase the scan.

Same-input raw-path Graph A/B on idle H20 GPU 3, PyTorch 2.11.0+cu128,
TileLang 0.1.15, nonzero initial State, FP32 output/final, three warmups and five
alternating rounds of 100 replays; entries are medians of round measurements:

| H | T | State rows/columns | Gate | Before ms | After ms | Speedup |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 2048 | 64/64 | log | 0.03557 | 0.03065 | 1.16x |
| 8 | 2048 | 64/64 | log | 0.10714 | 0.08661 | 1.24x |
| 8 | 8192 | 64/64 | log | 0.37603 | 0.30538 | 1.23x |
| 4 | 130 | 35/67 | signed/reset | 0.03741 | 0.03231 | 1.16x |
| 8 | 2048 | 64/64 | identity | 0.07621 | 0.07623 | unchanged |

All five A/B cases have bitwise-equal outputs and final States and preserve the
initial State. The baseline is the working-tree implementation immediately
before this optimization, including existing uncommitted changes, not HEAD or
the Triton branch. Raw rounds and measurement contracts are recorded in
`benchmark_results/stateful_tilelang015_triton_guided_scalar_scan.json`.

Public final-State A/B uses BF16 inputs/output and the independent authored-order
CPU FP32 token oracle. Five alternating rounds, three warmups and 20 event samples
per round; every output/final passed the oracle and remained bitwise equal:

| Family | H | T | Before Graph ms | After Graph ms | Before public event ms | After public event ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple-gla | 1 | 2048 | 0.03974 | 0.03486 | 0.12819 | 0.12773 |
| simple-gla | 8 | 2048 | 0.11000 | 0.09027 | 0.18136 | 0.16158 |
| retention | 8 | 2048 | 0.10989 | 0.09032 | 0.18074 | 0.16098 |
| simple-gla | 8 | 65 | 0.02122 | 0.02117 | 0.07618 | 0.07632 |

H8/T2048 public Graph latency improves about 1.22x; public event latency drops
about 11%. H1 public latency is host-launch dominated in this measurement. The
65-token workload retains the resident schedule and shows no meaningful change.
These are finite-workload observations, not a full matrix or confidence interval.

Direct same-input FP32 raw-path comparison against the fixed Triton branch:
T2048/K64/V64 log gates take 0.02353 ms (Triton) versus 0.03077 ms (TileLang)
at H1, and 0.07197 versus 0.08666 ms at H8. TileLang remains 31%/20% slower,
respectively; this optimization does not establish Triton performance parity.
Maximum output differences are 1.01e-6/1.25e-6; final-State differences are
4.77e-7/7.15e-7. Timing uses the same five-round/100-replay raw Graph protocol.


Verification: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ... python -m pytest
--run-gpu -q tests/functional/test_scalar_parallel.py`: **19 passed**. Covers
FP32/BF16, T63/64/65/130/2048, dynamic batch occupancy, grouped heads, signed/zero
gates, log-underflow resets, 35-by-67 State tails, nonzero initial immutability,
output-only/final-State, nonaligned continuation and public CUDA Graph replay.
The initial GPU-suite run exceeded its 600-second deadline after seven completed
cases; it is not counted as a completed suite. The completed run limits CPU
threading for the independent recurrence oracle. Changed kernel Ruff and Python
compilation checks passed. No regression tolerance or test code was changed.

### Further Triton-guided inference and staging pass (2026-10-07)

The live source at the start of this pass did **not** match the earlier inference
restoration narrative: `lowrank_kernels.execute` still selected its full-sequence
serial kernel for non-WY plans. Public RWKV7 T2048 measured approximately
8.4/9.3 ms Graph latency at H1/H8. Earlier records describe earlier artifacts,
not evidence that this starting working tree had the restored dispatcher.

Non-WY dense inference now invokes the existing device-segmented checkpoint
summary/carry/replay implementation with `record=False`. This follows the Triton
affine chunk decomposition, preserves authored row/rank/column/scale order, and
removes the obsolete full-sequence kernel. Factor packing retains FP32 promotion,
scalar/broadcast/strided bindings and token-major inputs; public outputs stay
B/H/T/V. Scalar-only WY eligibility, packed entry points and backward selection
are unchanged. Short dense inference uses 16-token chunks through T128; longer
inference uses 64-token chunks. Recorded, packed and supplied metadata retain
the 16-token checkpoint ABI.

Checkpoint carry exact 64-row tiles consume the shared staged State directly
instead of copying it into another shared tile. Other shapes keep the original
zero-padding/slicing path and triple-TF32 arithmetic. The 32-row candidate had
no benefit and was not retained; 64-row carry improves about 1.8–2.0% across
32/128 chunk measurements, with bitwise parity and independent FP64 entry checks.

Gated scalar readout splits the loaded FP32 query lane directly into TF32 high
and residual shared operands. It removes the intermediate query shared-memory
round trip, retaining the existing prepared-left triple-product order. Identity
readout keeps its previous measured schedule; no scalar tile-size cutover is
retained. Six FP32 raw cases have zero output differential and unchanged finals.
H8/T8192 raw Graph latency improves 0.30499 to 0.29190 ms (1.045x); public
simple-gla H8/T8192 improves 0.30086 to 0.29122 ms (1.033x). H8/T2048 raw and
public Graph improvements are approximately 1.4%; identity is unchanged.

Rejected experiments are recorded rather than hidden: global 32-row/column
scalar tiling regressed H8/T2048 by about 43% and H8/T8192 by about 54%, and
one identity output exceeded the strict FP32 differential threshold. Summary-only
32 tiling also regressed H8. Fused carry operand splitting improved 32-chunk
workloads but regressed 128-chunk workloads by 1.6–2.8%; neither fused variant is
retained. A temporary immutable query-lane binding produced invalid output;
the retained implementation uses explicit mutable `T.alloc_var` and passes
same-input differential checks without relaxing tolerances.

Raw rounds, rejected candidates and fixed-source hashes are recorded in
`benchmark_results/stateful_tilelang015_triton_guided_followup.json`.

Public final-State same-input A/B against the saved serial working-tree baseline,
H20 GPU 3, BF16 inputs/output, independent FP32 token oracle, five alternating
rounds of 20 warmed samples. Values are medians of round medians:

| Family | H | T | Before public event ms | After public event ms | Before Graph ms | After Graph ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RWKV7 | 1 | 2048 | 8.84597 | 0.75219 | 8.42198 | 0.28706 |
| RWKV7 | 8 | 2048 | 9.64792 | 1.16925 | 9.29938 | 0.74610 |
| compositional pressure | 8 | 2048 | 15.73578 | 1.68898 | 15.37150 | 1.25090 |
| RWKV7 | 1 | 64 | 0.65189 | 0.63853 | 0.28333 | 0.08056 |
| RWKV7 | 8 | 65 | 0.68042 | 0.61651 | 0.31278 | 0.13858 |
| RWKV7 | 8 | 130 | 0.96365 | 0.84155 | 0.58586 | 0.35814 |

The initial unconditional 64-token replay candidate regressed T65
(Graph 0.31003 to 0.35770 ms); the retained 16-token short schedule removes that
regression. All displayed cases pass independent output/final oracle checks;
low-rank outputs are tolerance-equivalent, not claimed bitwise equal because
chunked reductions change accumulation order. These ratios compare the starting
TileLang serial path, not Triton/FLA, and are not a full performance matrix.

The compiler now charges factored inference at the conservative 16-token chunk
granularity, covering short replay and eligible WY even when long non-WY calls
use 64 tokens. The workspace cap is unchanged; the resource-boundary regression
checks exact acceptance and one-byte-under-cap rejection for T65.
CPU unit verification: **193 passed, 6 skipped**; changed modules/tests passed
Ruff and Python compilation. The subsequent warmed GPU regression suite passed
**81 tests in 150.01 s** on H20 GPU 3. Additional memory, mixed-binding,
continuation and CUDA-Graph smokes passed; peak incremental allocation was
69,550,080 bytes before versus 71,714,816 bytes after. The earlier combined
run timed out during compilation and is superseded by the completed run.

### ProductFeature readout staging pass (2026-10-07)

`perf/structured-lowering` 的 Triton `_readout` 直接以 fragment 保存可复用的
64×64 local score tile；TileLang 原实现先把同一 Scores tile 搬到
`scores_shared`，再执行 FP32 GEMM。based/rebased ProductFeature readout 现改为
fragment score，去掉一次 shared-memory staging；因 `fp32_gemm_macro` 会在
调用处生成 TF32 high/residual shared operands，数值顺序与输出 ABI 不变。

H20 GPU 3 A/B（相同 carry、query、value、预计算 scores，5 轮、每轮 30 次
CUDA event，编译不计入）结果：H8/T2048/16×16×64 从 0.07650 ms 降至
0.07506 ms（1.019x）；H8/T2048/32×16×64 为 0.12483→0.12493 ms，视为
持平；H2/T131/5×5×3 为 0.02481→0.02519 ms，短尾端小幅回退。六项
ProductFeature functional GPU 测试全部通过，故保留为长序列中等 product
维度的保守优化，短序列不宣称收益。



### RWKV6 单 chunk output-only 路径（2026-10-07）

Triton `_replay` 对单 chunk output-only 调用直接从 initial State replay，
不需要 summary/carry。TileLang native diagonal 现对 `chunks == 1` 且不请求
final-State 的调用插入 zero-copy synthetic chunk axis；final-State、multi-chunk
和 continuation 路径保持原 summary/carry 逻辑。

H20 GPU 5、RWKV6 D=DV=64：H1/T64 public CUDA event 从 0.32304 ms 降至
0.23299 ms，Graph 从 0.05973 降至 0.05349 ms，total launches 从 3 降到 1；
H8/T64 public 从 0.32581 ms（既有矩阵记录）降至 0.23702 ms，Graph 为
0.05546 ms。H1/T2048 public 0.33115→0.32355 ms，Graph 0.07154→0.07138 ms，
说明优化只针对单 chunk，未改变长序列调度。

独立 authored-order oracle、RWKV6 FLA coverage（5 passed）、final-State 与
continuation 合同均通过。原始数据见
`benchmark_results/stateful_tilelang015_rwkv6_followup.json`。

### Rank-one direct replay and native codegen cache (2026-10-07)

`scan/rank1.py` now reuses a zero-copy initial-State carry for one-chunk
execution, including final-State requests. This restores the Triton direct
schedule without changing token recurrence or the multi-chunk path. Summary,
carry and their three vector workspaces are skipped. H8/T64 output-only peak
incremental allocation falls from 71,680 to 65,536 bytes.

Native summary, bonus replay and fused softmax look up compiled kernels before
loader expression emission and body/source generation. Keys contain ordered
loader semantics and source maps, current device/dtype/shape/strides/storage
offset/storage extent, head groups, B/H/T, prior flags and output/carry metadata.
Each invocation still validates public inputs and binds current pointers;
cache entries contain metadata and kernels, not tensors. Source caches also
include device. The existing unbounded source-cache convention remains; many
distinct layouts/storage extents can create additional compiled entries.

H20 GPU 3, fixed pre-edit modules, same compiled operator and inputs, five
alternating rounds of 30 warmed samples; compilation excluded. This is a
TileLang before/after comparison, not an FLA speedup claim:

| Family | H | T | Before public ms | After public ms | Before Graph ms | After Graph ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HGRN | 1 | 64 | 0.12891 | 0.07534 | 0.01411 | 0.01395 |
| HGRN | 8 | 64 | 0.13726 | 0.08042 | 0.01642 | 0.01363 |
| HGRN | 1 | 2048 | 0.13387 | 0.13358 | 0.01886 | 0.01893 |
| RWKV6 | 1 | 64 | 0.21402 | 0.18392 | 0.05376 | 0.05376 |
| RWKV6 | 8 | 2048 | 0.36555 | 0.31733 | 0.19595 | 0.19602 |
| GSA | 1 | 64 | 0.29616 | 0.25394 | 0.11485 | 0.11506 |
| GSA | 1 | 2048 | 0.48075 | 0.41234 | 0.13158 | 0.13176 |
| GSA | 8 | 2048 | 0.55077 | 0.47965 | 0.23272 | 0.23301 |
| normalized-linear | 8 | 2048 | 0.21598 | 0.21693 | 0.12853 | 0.12845 |

RWKV6/GSA public improvements are approximately 13–14%; Graph stays effectively
unchanged, consistent with CPU codegen removal. HGRN T64 public improves about
41%, with observed launches reduced 3 to 1. Long HGRN and normalized-linear are
unchanged within measurement noise. The first GPU 0 A/B overlapped regression
work and is superseded by the isolated GPU 3 numbers above.

All nine cases pass independent authored-order oracle checks and bitwise
before/after outputs/finals for output-only, final-State and continuation.
Additional changed-input CUDA Graph replay passes for HGRN/RWKV6/GSA; RWKV6/GSA
dynamic B/T rebinding passes B1/T64, B2/T65, B1/T130 and B1/T64 again, with retained
outputs unchanged. Consumer regressions add fresh contiguous storage with changed
offsets/extents for GSA and RWKV6 and retain noncontiguous-input rejection.
CPU suite: 193 passed, 6 skipped; Ruff and compilation passed.
The first combined rank-one/readout GPU run reported 132 passed and 74 failed:
test fixtures supplied noncontiguous tensors to the contiguous-only public API,
and two tests requested shared-scores for an unsupported dual-gate structure.
The fixtures now use legal contiguous continuation/offset views and enumerate
replay for dual gates, scores for the supported single axis; runtime validation,
candidate eligibility and numerical tolerances are unchanged. All 96 affected
native diagonal/softmax/bonus cases then passed (79 unrelated cases deselected)
in 470 seconds. The earlier run's other cases, including all rank-one tests,
passed; it is not reported as a passing full-suite run.

The GSA per-token shared-vector candidate is **not retained**. Four cases had
bitwise output parity, but raw replay Graph latency regressed 11.49–13.15%; the
compiler also emitted a ThreadSync hoist warning describing a potential race.
No tolerance was relaxed and no candidate arithmetic entered the repository.
Raw rounds and memory observations:
`benchmark_results/stateful_tilelang015_native_cache_followup.json`.

### Composed-IR generality matrix (2026-10-07)

Tested the committed `3e491e4` implementation on idle H20 GPU 0 with four
model-independent compositions from `testing/stateful_composition_profiles.py`:
dual-axis scaling (control), two product injections with two independent queries,
rank-one propagation with two queries, and ordered row-scale → rank-one →
column-scale with two injections and two queries. Dual-axis scaling and the base
rank-one algebra exist in FLA; “no direct adapter” for the composed interfaces
does **not** establish that FLA cannot express them by composition. These results
compare the same IR with `optimized=True/False`, not Meta versus FLA.

All inputs and outputs are FP32. Three B/H/source-head/K/V/T shapes were
`1/4/2/13/19/65`, `2/8/2/35/67/257`, and `1/8/2/64/128/2048`; four renamed-input
and renamed-role cases repeat the middle shape. Sixteen cases author
`Readout(StateTerm(...))`; four `1/4/2/64/128/129` cases author the legacy
`StateContraction` ABI. Its existing specialization rejects odd feature sizes;
the expression ABI accepts them. This ABI distinction is a measured boundary,
not a migrated or relaxed validation rule.

The main matrix records **15 VALID / 20 total**, two resource-envelope failures,
and three invalid-baseline comparisons. Every VALID case passes an independent
authored-order CPU FP32 token oracle, nonzero initial-State, split-at-17
continuation, and exact initial-State immutability. Non-reference optimized
paths also pass changed-query CUDA Graph replay. Tolerances remain
`rtol=2e-4, atol=2e-5`; FP32 input standard deviation is 0.01, gates are negative
absolute values, and coefficients remain signed. This is not an adversarial
numerical-stability sweep, nor BF16, packed, backward, or multi-State coverage.

Public-call timings exclude compilation and use ten alternating rounds, each
with three warmed CUDA-event samples. Peak incremental allocation is measured
before Graph capture and includes returned outputs/final-State.

| Composition / authored ABI | K×V / T | Actual baseline | Optimized public ms | Baseline public ms | Public speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Dual-axis control / expression | 13×19 / 65 | reference-stateful | 0.2850 | 0.9263 | 3.25× |
| Dual-axis control / expression | 35×67 / 257 | reference-stateful | 0.3964 | 1.1092 | 2.80× |
| Dual-axis control / expression | 64×128 / 2048 | reference-stateful | 0.5829 | 4.9395 | 8.47× |
| Two injections, two queries / expression | 13×19 / 65 | reference-stateful | 0.4225 | 1.1007 | 2.61× |
| Two injections, two queries / expression | 35×67 / 257 | reference-stateful | 0.4393 | 1.2630 | 2.88× |
| Two injections, two queries / expression | 64×128 / 2048 | reference-stateful | 0.6053 | 5.5306 | 9.14× |
| Two injections, two queries / contraction | 64×128 / 129 | tilelang-generic | 0.4139 | 0.7166 | 1.73× |

The injection contraction comparison also measures Graph replay
`0.10864 vs 0.17621 ms` (1.62×), and peak incremental allocation
`1.754 vs 89.452 MiB`. The long expression injection case allocates
`40.25 vs 2072.15 MiB`. These are different baseline implementations and must
not be combined into one kernel speedup claim. Renamed axis/injection cases
retain `tilelang-chunked-stateful` and pass correctness; their public ratios
are 2.76×/2.86× against reference-stateful.

Important limits exposed by the matrix:

- Expression rank-one/mixed cases select `reference-stateful` even when
  optimized. Smaller shapes and renamed variants pass, but both flags run the
  same fallback (approximately 1×), not a demonstrated structured optimization.
  Both long cases fail the 4 GiB compiled resource envelope before execution;
  no GPU OOM was observed or attempted.
- Initial reference-path Graph capture failed; subsequent reference Graph
  timings are explicitly unavailable, not silently approximated by events.
- The axis/rank-one/mixed contraction comparisons fail **only the
  `tilelang-generic` baseline**. Per-lowerer isolation passes optimized outputs
  and final-State (maximum error across these cases `1.26e-8`); baseline maximum
  output errors are `6.22e-5`, `8.07e-5`, and `5.86e-5`, respectively. No valid
  speedup is claimed for those comparisons, no tolerance is relaxed, and the
  generic-path numerical cause is not established by this investigation.

The three optimized contraction paths also pass a separate continuation,
initial-State immutability and changed-query Graph smoke; saved baseline failure
evidence is reused without rerunning the invalid baseline. Standalone warmed
public/Graph timings are axis `0.27470/0.07438 ms`, rank-one
`81.09063/80.26430 ms`, and mixed `1.08630/0.50688 ms`. The rank-one result
exposes a severe latency risk in this selected factored schedule; its cause
is not diagnosed here. Correctness does not establish acceptable performance.
Those timings are not paired comparisons and carry no speedup ratios.
See `optimized_contracts/records.json` and `conclusions.json`.

Raw rounds, memory, boundaries and per-lowerer attribution are retained in
`benchmark_results/stateful_generality_20261007/`, including the original
failed matrix rather than replacing it with a passing subset. No runtime code
or eligibility rule changed during this investigation.

### Short public-call preparation and direct final replay (2026-10-09)

Native diagonal replay now looks up its compiled kernel from ordered loader
semantics and current binding/output/carry metadata before emitting expressions
or generating source. Bonus replay also delays annotation formatting until a
cache miss. Identical `(input name, axis recipe)` uses share one raw source
within an invocation; distinct row/column uses remain separate. No cache entry
retains input tensors or pointers. Device, shape, strides, storage offset/extent,
head groups, prior flags and optional final-State metadata remain in the keys.
This removes repeated host code generation for GLA and redundant binding/view
preparation for RWKV6 without changing arithmetic or the long device schedule.

For one State and one ordinary or bonus readout, with no shared-score or
softmax schedule, sequences through 64 tokens can publish final-State directly
from the replay fragment. Summary/carry are skipped for these final-State calls
as they already were for output-only calls. Each State element has one owner;
the final write follows all valid token transitions, including the transition
after a prior-State readout. This removes two launches and three FP32
`B*H*rows*columns` workspaces; the distinct returned final-State allocation
remains. Longer sequences and other readout schedules keep summary/carry.
The existing authored-binding regression adds one-readout T64/T65 cases with
both prior flags, signed/zero dual gates, grouped heads, nonzero initial-State,
odd feature tails and continuation. Existing bonus regressions cover T1/63/64/65.

HGRN's vector-State recipe already uses raw bindings and one-chunk direct replay,
so no rank-two vector-propagation or model-name dispatch is added. Its public
host path now reuses compile-time unique State/output name tuples and the existing
trusted result-tuple constructor, builds the mutable role extent map once, and
shares one current-input gradient scan. Alias validation reads each current
storage pointer once rather than once per pair; byte intervals are only computed
for same-storage candidates. All public name, dtype, shape, layout, alias, State,
gradient, target and workspace checks still execute on each call.

Pre-edit source copies are retained under
`/tmp/metaattention_short_paths_baseline/attention_engine/attn_engine/`;
the integration baseline also exists at
`/tmp/metaattention_before_implementation/attn_engine/`. No timing, build,
lint or test was run by this implementation worker; measured public/Graph
latency and full integration verification must be supplied by the parent run.


### 集成实现证据（2026-10-09）

本轮 H20 smoke 使用 GPU 5，当前源码的 git commit 为
`c9dcd0a17c5fb4bf1a0455db31eef4664afa2a2e`；环境版本为
Torch `2.11.0+cu128`、CUDA `12.8`、TileLang `0.1.15`、FLA
`0.2.0`。runner 现在把实际测量源码 SHA256 和这些版本写入新 manifest。

低秩 GDN output-only 的 10 个 B1/H1,H8/T64,65,130,2048,8192
workload 均通过独立 oracle 与 Graph replay。H8/T2048 的 Meta public/Graph
为 `1.5864/1.2497 ms`，修改前 package 为 `44.3094/42.9491 ms`；
H8/T8192 为 `5.0594/4.8248 ms`，修改前为 `178.0668/176.9654 ms`。
同一 smoke 的最大 output absolute error 为 `0.00388`，使用既有 `0.02/0.05`
relative/absolute tolerance；这些是 Meta 对修改前 Meta package 的 A/B 数字，
不是 FLA speedup。

GLA、RWKV6、HGRN 的 B1/H1,H8/T64,2048,8192 18 项也均通过 oracle；短路径
变化不均匀，故不宣称统一加速。相对修改前 package，public 变化范围为
`0.65×` 到 `2.46×`（比值为旧/新），Graph 变化范围为 `0.76×` 到 `1.33×`。
ProductFeature GPU functional regressions now explicitly run with `--run-gpu`:
12 passed, including independent ProductFeature order, grouped Q/K/V,
FP32/BF16, nonzero initial/final States and continuation. Native GEMM results
stage into CTA shared tiles before reading a fixed denominator lane; this
avoids fragment layout conflicts without dropping causal chunk-local work.

The actual compiled public API also passes 30 contract smoke scenarios:
GDN/Based/ReBased/GLA/RWKV6/HGRN, B2/H2, T1/64/65/130/257, nonzero initial,
nonzero storage offsets, complete final State, continuation (T>1), CUDA Graph
replay and initial-State immutability. Evidence:
`benchmark_results/stateful_repair_contracts.json`. These scenarios are not
FLA comparisons or proof of training correctness. Historical FLA records
remain unchanged and are not counted as newly verified performance pairs.
