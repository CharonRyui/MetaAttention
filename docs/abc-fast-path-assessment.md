# ABC 快速路径实现评估

评估基于 SSH 工作区 `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl` 的当前实现。记录 ABC lowering 的方案、实现和验证；数学定义与三个 raw accumulator State 的公共合同保持不变。

## 结论

可以扩展现有路径兼容 ABC。除法已经是表达式 IR 和 TileLang emitter 支持的逐元素运算，不需要 ABC 专用除法 kernel、公共 IR 节点或模型名称分流。缺少的能力是：在 readout 内绑定 rank-1 State 值，并允许逐元素 DAG 出现在 contraction、softmax、下一次 contraction 之间。

应优先扩展现有 `native-diagonal`/readout-region 能力，保留 identity chunk GEMM 作为长序列候选。完整融合 replay 是另一个 schedule，不能仅因 launch 更少而默认取代 GEMM。

## 实现前的限制与复用位置

- `stateful/lowering/expression/dense_expression.py::compile_dense_expression` 已支持 Add/Multiply/Divide，记录 dtype 与作者给定的表达式顺序。它目前只接受 Input/Constant 叶子，不接受 StateTerm。
- `diagonal/native_tile_kernels.py::emit_dense_loader` 已把 Divide 生成为 TileLang `/`，并保留 BF16 的逐节点 rounding。
- `expression/expression.py::CompiledExpression` 可将 StateTerm 作为叶子，但要求调用者提供已 materialize 的张量。`expression_kernels.py::launch_expression` 将 Softmax/Contract 等 reduction 留给外层 lowering。
- `diagonal/diagonal_native.py::compile_native_diagonal` 在 State.rank != 2 时拒绝 recipe；当前 softmax recipe 只接受 `StateTerm(second, Softmax(StateTerm(first,...)))`。
- `diagonal/native_tile_softmax.py::_fused_softmax` 已实现两个 State 的 chunk-local replay、feature reduction 和 softmax。它固定持有两个矩阵 State，未绑定辅助 rank-1 State，也没有 score/weight 阶段的 typed pointwise 表达式。
- `diagonal/identity.py::identity_readout` 是 chunk GEMM contraction。ABC 的三个 State 目前各自 summary/carry/readout，再由 `chunked.py::evaluate_uncached` 发射 divide/softmax。

因此，放宽 `isinstance` 检查不足以使现有 kernel 正确执行 ABC；仍需扩展 recipe、State 绑定、codegen 与 resource accounting。

## 建议的内部表示

保留现有 Algorithm IR。分析 readout 的 typed DAG，形成有序局部操作：

1. contraction 的输入与输出 Feature roles、来源 State 和 prior/post 标志；
2. rank-1 StateTerm 的叶子绑定；
3. contraction 输出间的逐元素表达式；
4. Softmax 的归约 Feature role；
5. 后续 contraction 的 operand 表达式；
6. 输出 dtype 与每个节点的 rounding。

ABC 对应：`scores = contract(q * scale, key_slot)`；`logits = scores / normalizer`；`probability = softmax(logits, SLOT)`；`weights = probability / normalizer`；`output = contract(weights, slot_value)`。

normalizer 是按 token 演化的 slot 向量，不能作为 Input loader 的常量或仅使用 chunk-entry carry。多个引用共享同一时序的 StateTerm，只推进一次。每个 prior/post 选择独立保留。rank-1 State 可在 lowering 内视为 `[SLOT, 1]`，public initial/final State 仍为原 rank-1 shape。

实现上复用 typed pointwise 节点与 emitter，增加由调用者绑定的局部 State/中间值叶子。编译器证明角色兼容、时序和资源合法性，不识别 ABC 名称，也不匹配整条模型公式。第一版可限定现有“两次 contraction + 一次 softmax”的 region，允许其间的 Add/Multiply/Divide 与辅助向量 State；超出范围继续使用当前合法执行路径。

## 两种 schedule

### 保留 GEMM 的 prefill schedule

复用 identity summary/carry 和两阶段 chunk GEMM。第一 contraction 的 epilogue 计算 `scores / z`，softmax 阶段计算 `softmax(logits) / z`，第二 contraction 读取 weights。rank-1 normalizer 用向量 prefix，避免把逐 slot 累加变成 64×64 GEMM。

若 softmax 能在完整 slot tile 的第一 contraction epilogue 中完成，可合并阶段；slot 跨多个 tile 时保留 reduction cut。中间 logits/weights 的物化是有意选择的 schedule，不把每次 divide 变成独立 kernel。

原始 layout 的 query/key/value/logit 由现有 raw binding/loader 读取；exp、单位 injection factor 与 scale 在 kernel 内生成，减少 FP32 copy、contiguous 和常量 fill。跨 summary/readout 共享 exp(logits) 是否物化为一次 weight tensor，由复用次数与带宽成本决定，避免每个 State element 重复计算 exp。

### 完整 chunk-local replay schedule

扩展现有 `_fused_softmax`，同时持有 key_slot、slot_value 和 normalizer；每 token 更新三个 State，然后依序 contraction/divide/softmax/divide/contraction。两处 Divide 只是局部 FP32 `/`。没有全局 score/weight tensor。

单 chunk、output-only 时可直接把 initial States 用作 entry，省去无消费者的 summary/carry。requested finals 需保持当前 State 数值合同和 initial 不变性。

此 schedule 的风险是 serial token replay 与寄存器压力。默认 K=V=64、SLOT=16 时，两矩阵共 2048 FP32 elements，normalizer 仅增加 16，但实际 live storage 还包括 products、softmax 和 query，不能用这两个数字代替资源估计。更大 slot/feature 尺寸必须保留切分执行候选。

## 数值与验证

- 保持 `query * scale` 的 BF16 舍入后才进入 FP32 contraction。
- 保持 `scores / z` 在 softmax 之前、`probability / z` 在第二 contraction 之前；不交换除法和归约、不替换成改变舍入的全局 State 归一化。
- 不自动插入 epsilon/clamp，不替换 raw accumulators 为 FLA normalized caches；当前例子仍要求 exp(logits) 与累加值为有限正数。
- 使用独立 authored-order oracle，覆盖非零 initial、三个 finals、continuation、tails、grouped heads、不同 State 名称与不同点运算组合；FLA 仅为 output-only 性能基线。
- 同一输入交错比较现有路径、GEMM epilogue 与完整 replay；分别记录 public/Graph、launch、峰值、register/shared-memory/spill。验证 T=63/64/65/130/2048/8192 和 B/H 变化。

## 实施顺序

先增加辅助向量 State 和 typed pointwise bindings，使现有 readout-cut 路径合法执行 ABC，同时保留 GEMM。再增加完整 replay schedule，按测量选择短序列与资源允许的配置。最后合并多个 identity summary/carry 的调度与共享输入加载。

核心 seam 放在编译后的 readout region：分析负责语义，执行器负责 State 准备，codegen 负责局部 arithmetic/reduction。候选名称可以保持 `native-diagonal` 和 readout-cut 系列；若需新增 schedule 标识，应描述执行方式，而非增加 ABC 专用 lowerer。

## 本轮 benchmark 验证与边界

本轮实现了 `--fla-baseline recurrent|prefill|both`，默认 both，schema 6 在每条比较记录、CSV 和图中明确基线身份。缺失接口不回退到另一 schedule。ABC 的 prefill 使用 `fla.ops.abc.chunk_abc`，recurrent 记录 UNAVAILABLE。以下为快速路径实现前的 benchmark 接入结果；新的实现和测量见下节。

远程 H20 GPU 0，B=H=1、T=65、K=V=64、SLOT=16、output-only，warmup=1、repetitions=3 的独立 smoke：ABC prefill 三项校验（Meta/oracle、FLA/oracle、两者输出）通过，public Meta/FLA 为 0.804/0.339 ms，Graph 为 0.0698/0.0421 ms；这不是稳定性能结论。原始产物在远程 `/tmp/meta_abc_graph_final/{records.json,manifest.json,summary.csv}`。

相关 CPU 单元测试 106 passed、4 GPU tests deselected，Ruff check/format 与 diff whitespace 检查通过。新增测试覆盖接口选择、禁止缺失接口回退、baseline-specific Comba 转换、FP32 authored transforms、GLA neutral padding、CSV 基线身份。GPU smoke 全 family output-only 初轮产物为 `/tmp/meta_dual_baseline_smoke`；其中 Linear/Simple GLA/GLA/RWKV7 的 prefill mixed-dtype 错误在适配器中保留 FP32 变换并统一 q/k/v dtype 后解决，独立复测产物为 `/tmp/meta_dual_baseline_fixed`。

GLA FP32 短 chunk 的 FLA output kernel autotuning 曾触发 misaligned address，独立复现记录在 `/tmp/meta_gla_prefill_continuation_probe.log`。适配器对长度 <64 的调用补齐零 q/k/v/log-gate，尾部为单位转移、零写入，裁掉多余输出；padding 成本计入 public 调用并写入 preprocessing。Linear/Simple GLA 的三合同已通过（`/tmp/meta_dual_baseline_contracts.log`）；RWKV7/Comba 的三合同通过（`/tmp/meta_dual_baseline_contracts_fixed`）；GLA 修正后 continuation 的输出和 final State 校验通过（`/tmp/meta_gla_continuation_final`）。这仅验证所列 B/H/T，不宣称全矩阵或所有 contract 已验证。

初轮 Delta Product 选择 reference-stateful 后出现 Meta/oracle 校验失败和 Graph capture 错误，继而污染同进程后续记录。失败原样保留，不计入有效比较；ABC 已在独立进程成功复测。当前 IR/ordered-transition 实现由另一工作范围修改，本轮不改生产 lowering 或放宽容差。benchmark 对 CUDA context 致命错误的进程隔离、增量产物落盘可另行改进；本轮未实现。


## 已实现的通用 softmax region

`readout/softmax_recipe.py` 在 typed IR 上证明“两次 contraction + 一次 softmax”的局部 region。score 和 weight 之间允许 DenseExpression 的 pointwise 节点，rank-1 StateTerm 可作为局部叶子；结构证明不依赖 ABC 或 State 名称。第一版向量叶子要求 scalar operand 为 1 且不收缩 Feature，其他形式保留 generic chunked 路径。Input 的 Feature roles 必须匹配 slot 轴。

`dense_expression.py` 与 raw emitter 接受调用者绑定的 local values，并复用原有逐节点 dtype/rounding。`diagonal_native.py` 将辅助向量 State 内部表示为 `[SLOT,1]`，对外保持 rank-1 initial/final shape。多次引用相同 State/时序只推进一次；不同 prior/post 视图在同一推进中快照。

两种执行候选已实现：

- `tilelang-native-diagonal-readout-cut`：两个 identity contraction 使用现有 three-TF32 GEMM macro，直接读取 raw stride/grouped-head bindings 与 typed loaders；向量使用 chunk prefix。softmax kernel 内依次执行 score pointwise → stable softmax → weight pointwise，两处除法不再各发一个 kernel。
- `tilelang-native-diagonal`：扩展原 fused softmax replay，局部持有两个矩阵及向量 State。单 chunk、所有 State 都由 region 消费时可直接发布三个 finals，避免 summary/carry；initial 不被修改。大尺寸超过 fused 资源证明时使用分阶段执行。

对带辅助向量的 identity region，static estimate 计入 replay 的串行关键路径代理成本，而不仅按 launch 数选择；默认 ABC 使用 GEMM cut，replay 保留供 autotune 比较。代理成本不计入实际 workspace。cut 的两阶段 FP32 buffers、不同 vector temporal views、vector prefix launches 均计入资源估计。现有 GSA 两 State gated region 不使用该额外成本。

首次融合 kernel 的测试捕获了 TileLang 在 shared weights 两次赋值之间插入、位于单 warp 条件分支内的 `__syncthreads()`。实现改为在 fragment 中完整计算 weights，再一次性 `T.copy` 到 shared；原复现及完整测试已通过。该修改也应用于现有 softmax region，GSA 回归通过。

## 快速路径验证

远程 H20，新增 ABC GPU tests 共 24 passed：FP32/BF16，B=2、State H=4，query/key/value H=2/1/2，K/SLOT/V=13/19/11，T=1/63/64/65/130；覆盖非零 initial、output-only、三个 finals、非对齐 continuation、两种 schedule、不同 pointwise 组合、normalizer 的 prior/post 混合引用。oracle 保持 BF16 `query * scale` 的舍入后再 FP32 contraction。输出容差 FP32 rtol/atol=2e-4/2e-5、BF16=2e-2/2e-3；finals 为 2e-4/2e-5。最初日志 `/tmp/meta_abc_last_tests.log` 同时含 3 项 Abs/Cast tests，共 27 passed。最终 `/tmp/meta_abc_handoff_gpu.log` 为 29 passed，包含非零 storage offset 的 24 个核心 ABC 用例、1 个资源超限 cut 用例、4 个 Abs/Cast 用例（包括 public backward）。

CPU recipe tests 4 passed，覆盖 State 重命名、辅助 vector cut 计费、默认 GEMM 候选、资源超限分阶段执行、未支持的向量 operand 回退。现有 GSA softmax tests 22 passed（`/tmp/meta_gsa_region_regression.log`）。额外 GPU 验证 K/SLOT/V=96/96/33、T=7 的资源超限 cut 输出与三个 finals，以及 public Abs/Cast backward，记录 `/tmp/meta_abc_extra_checks.log`。最终全量 CPU 回归为 503 passed、4 skipped、92 GPU deselected（`/tmp/meta_abc_cpu_handoff.log`）；它包含其他会话同时实现的 staged 功能。新增 4 recipe tests 与 2 个 CPU Abs/Cast tests 也单独通过，Ruff check/format 和 diff whitespace 检查通过。

## 性能测量与限制

独立进程比较 generic chunked、完整 replay 和 GEMM cut，B=H=1、K=V=64、SLOT=16、BF16 q/k/v、FP32 logits，warmup=3、repetitions=10。编译不计时；每次使用相同随机种子，并与独立 reference operator 比较输出。Graph 测量表明 replay 的 serial token critical path 显著高于 GEMM，不能因 launch 更少默认取代它。T=65/130/2048/8192 的 replay Graph 分别约 0.162/0.161/0.169/0.198 ms；generic 为 0.0666/0.0680/0.0760/0.123 ms，GEMM cut 为 0.0485/0.0489/0.0596/0.106 ms。产物为远程 `/tmp/meta_abc_perf_{T}_{schedule}.log`；cache 优化后的 cut 复测在 `/tmp/meta_abc_perf_final_{65,2048}_cut.log`。

公开 benchmark 独立复测 H20 GPU 3（启动时无其他 compute PID），B=H=1、K=V=64、SLOT=16、zero initial、output-only、prefill、warmup=3、repetitions=10。T=63/64/65/130/2048/8192 均通过 Meta/oracle、FLA/oracle 和 pair 输出校验，全部选择 `tilelang-native-diagonal-readout-cut`。原始 JSON/CSV/manifest 在 `/tmp/meta_abc_native_public`。

| T | Meta public ms | FLA public ms | Meta Graph ms | FLA Graph ms | Meta peak incremental bytes | FLA peak incremental bytes |
|---|---:|---:|---:|---:|---:|---:|
| 63 | 0.302 | 0.342 | 0.0383 | 0.0363 | 20,480 | 45,056 |
| 64 | 0.311 | 0.352 | 0.0344 | 0.0376 | 20,480 | 45,056 |
| 65 | 0.532 | 0.360 | 0.0512 | 0.0433 | 40,448 | 53,760 |
| 130 | 0.591 | 0.356 | 0.0509 | 0.0465 | 69,120 | 98,816 |
| 2048 | 0.477 | 0.343 | 0.0618 | 0.160 | 923,648 | 1,441,792 |
| 8192 | 0.675 | 0.621 | 0.108 | 0.530 | 3,694,592 | 5,767,168 |

这是固定小 batch/head workload 的测量，不代表全部形状。多 chunk 的 public 调用仍主要受 dispatch/summary/carry 多 kernel 开销影响，部分点慢于 FLA；Graph 与 public 结果分别报告。CUPTI 对 foreign launches 的计数不稳定，`total_launches` 只能作为单次 profile 证据，不能凭上述表宣称所有 kernel 均融合。此次未完成 register/shared/spill 全矩阵 profile，也未合并多个 State 的 summary/carry；这两项是后续优化。训练与 packed 继续走既有合法路径，本次 native softmax region 仅用于 dense inference。

同步完成通用 `Abs`/`Cast` 表达式编译支持：分析/type、State 依赖遍历、VJP input 依赖、CPU/autograd、TileLang pointwise 与 raw loader 均接受节点；Cast 保留每个 BF16/FP32 转换边界，Abs 在 FP32 计算且零点梯度为 0。公共 IR 定义/export/reference evaluation 由主会话范围提供。
