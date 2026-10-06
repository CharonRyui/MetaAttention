# MetaAttention

> The PPoPP'26 Artifact Evaluation is available on the [`PPoPP_AE` branch](https://github.com/SJTU-IPADS/MetaAttention/tree/PPoPP_AE).

## 0. Overview

MetaAttention provides users with pythonic interface to define customized attention flexibly and automatically generate device code with high performance. Examples of various attention mechanisms (e.g., softmax attention, RetNet, Mamba2, MLA) are provided in the examples/ folder.

# 1. Getting Started Guide

## Hardware Requirements

To run generated kernels, you still need a supported GPU runtime:

- 1x NVIDIA Hopper GPU, or
- 1x AMD MI200 Series GPU

The dedicated Gated Delta Rule engine additionally supports NVIDIA H20. Its
first-release contract is fixed-length, head-first BF16 training with FP32
gate/beta/state, 128-dimensional key/value heads, and 64-token chunk alignment.

## Local Quickstart with `uv`

Use this path if you already have a working Python environment and a PyTorch build that matches your target runtime (CUDA, ROCm, or CPU) on the host machine.

### Install
```bash
uv venv --python 3.12
. .venv/bin/activate
# install a platform-appropriate PyTorch build first
uv pip install torch
# install the pinned TileLang release with its runtime dependencies
uv pip install "tilelang==0.1.15"
# then install this project without re-resolving torch or TileLang
uv pip install --no-deps -e .
```

This installs MetaAttention itself from the local checkout. PyTorch is intentionally host-selected so CUDA, ROCm, and CPU environments can provide their own compatible build, and `--no-deps` avoids replacing that choice during editable install.
The project pins the released TileLang `0.1.15` API in `pyproject.toml` and `uv.lock`. Stateful identity summary/carry helpers and the native normalized identity inference path now use TileLang's lazy `@tilelang.jit`/`@T.prim_func` interface. Other lowerers still use Triton; a package upgrade does not translate their kernels. See the [migration research and verification record](docs/tilelang-stateful-operator-research.md) for APIs, compatibility constraints and the exact migrated scope.

### Verify the installation
```bash
# run from the repo root
python examples/retention_parallel.py
```

Expected output includes `AttentionEngine Succuessfully created.`

# 2. Step-by-Step Instructions

## Functional tests

Run the unified pytest suite from the repository root. The unit suite is CPU-safe and checks imports, code generation, benchmark dispatch, CSV schemas, and PDF plotting without compiling GPU kernels:
```bash
uv run --extra test pytest -q tests/unit

# imports for all 14 official example factories
uv run --extra test pytest -q -m examples tests/unit/test_example_imports.py
```

Run the GPU correctness suite explicitly:
```bash
uv run --extra test pytest -q --run-gpu -m 'functional and gpu' tests/functional
```
This executes the legacy 10-case parity matrix plus direct-engine and official v2 factory coverage. Forward results, backward gradients where supported, reference implementations, dtypes, and operator-specific tolerances are checked. Without `--run-gpu`, GPU tests are collected but skipped with an explicit reason; without a CUDA/HIP-visible device, the hardware fixture skips them.

Run the H20 GDN functional suite on an otherwise idle H20:
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test pytest -q --run-gpu \
  -m 'functional and h20' tests/functional/test_gdn_h20.py
```
This executes the FlashQLA-derived fused kernel for single- and multi-chunk GVA,
initial/final state, custom scale, and all supported gradients. See
`examples/gated_delta_rule.py` and `docs/API.md` for the public interface.

**Expected output**: all unit tests pass. On a supported GPU, all functional tests run without unconditional skips. Optional baseline warnings do not replace MetaAttention correctness checks.


### Stateful Operator 与 FLA 完整矩阵

对 `examples/` 中已有 FLA Stateful DSL family 运行完整对照矩阵：

```bash
PYTHONPATH=/dev/shm/metaattention-tilelang-015:. \
TILELANG_CACHE_DIR=/dev/shm/stateful-fla-full-tilelang \
TRITON_CACHE_DIR=/dev/shm/stateful-fla-full-triton \
XDG_CACHE_HOME=/dev/shm/stateful-fla-full-xdg \
../bench.sh \
  --families linear,simple-gla,gla,normalized-linear,retention,rwkv7,rwkv6,gsa,hgrn,based,rebased,gdn,retnet \
  --seqlens 64 2048 \
  --batches 1 \
  --heads 1 8 \
  --contract all \
  --timing e2e \
  --warmup 3 \
  --repetitions 10 \
  --output-dir benchmark_results/stateful_fla_meta_all
```

该 runner 同时记录 MetaAttention 的独立 authored-order FP32 token recurrence correctness、FLA baseline 状态合同可比性、public CUDA latency、CPU wall time、peak allocator memory、CUPTI activity/kernel launch 和 CUDA Graph（启用 `--timing both` 时）。FLA 只作为执行性能 baseline，不作为 correctness oracle；`pressure`/`mamba2` 没有等价的 FLA adapter，它们只能作为 MetaAttention 结构/正确性诊断，不应伪造 baseline。

运行期间必须使用真正空闲 GPU；runner 默认拒绝已有 compute PID、显存占用超过 1 GiB 或 GPU utilization 非零的设备。结果目录含 `manifest.json`、`records.json`、`summary.csv` 和可用的 PNG 图表。本次实际矩阵结果写入 `benchmark_results/stateful_fla_meta_all_correctness/`；Meta 独立 FP32 recurrence correctness 为 156/156 `VALID`，FLA baseline 为 150/156 成功，HGRN 的 `H=8` 六条 baseline 记录保留失败原因。

## Performance tests

We consider Figure 11 and Figure 14 to be the key results of our paper, demonstrating the performance of MetaAttention-generated operators on hardware.
The following are the steps to replicate these experiments.

**Note for reproducing the result**:
Baseline libraries (e.g., FlashAttention, FlashLinearAttention) are frequently updated, thus slight variations in performance numbers compared to the static plots in the paper are expected. However, the overall conclusion should remain consistent: MetaAttention achieves performance comparable to hand-written libraries and better than native PyTorch.


## Figure 11

- Target Device: NVIDIA H100-80GB GPU
- Description: Evaluates the performance of MetaAttention against baselines on the H100.

Run the guarded long-run pytest entrypoint (about 90 minutes on the target hardware):
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench pytest -q \
  --run-benchmarks -m 'benchmark and h100' tests/benchmarks/test_h100.py
```
The runner executes all 62 Figure 11 configurations in an isolated pytest temporary directory, validates 24 CSV files containing `MetaAttention` rows, and renders a non-empty `figure11_h100.pdf`. Missing optional baselines are reported and omitted rather than replaced with fabricated values.

## Figure 14

- Target Device: AMD MI250X GPU
- Description: Evaluates the performance of MetaAttention on the AMD backend.

Run the guarded long-run pytest entrypoint (about 20 minutes on the target hardware):
```bash
HIP_VISIBLE_DEVICES=0 uv run --extra test --extra bench pytest -q \
  --run-benchmarks -m 'benchmark and mi250' tests/benchmarks/test_mi250.py
```
The runner executes all 25 Figure 14 configurations in an isolated pytest temporary directory, validates 10 CSV files containing `MetaAttention` rows, and renders a non-empty `figure14_mi250.pdf`. On a non-HIP runtime or non-MI250 device, it skips with the unmet hardware prerequisite.
