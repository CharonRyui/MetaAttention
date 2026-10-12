# ABC 快速路径实现评估

评估基于 SSH 工作区 `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl` 的当前实现。这里只评估 ABC lowering；benchmark 的 recurrent/prefill 选择另行实现，不改变 ABC 数学或生产 kernel。

## 结论

可以扩展现有路径兼容 ABC。除法已经是表达式 IR 和 TileLang emitter 支持的逐元素运算，不需要 ABC 专用除法 kernel、公共 IR 节点或模型名称分流。缺少的能力是：在 readout 内绑定 rank-1 State 值，并允许逐元素 DAG 出现在 contraction、softmax、下一次 contraction 之间。

应优先扩展现有 `native-diagonal`/readout-region 能力，保留 identity chunk GEMM 作为长序列候选。完整融合 replay 是另一个 schedule，不能仅因 launch 更少而默认取代 GEMM。

## 当前限制与复用位置

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

本轮实现了 `--fla-baseline recurrent|prefill|both`，默认 both，schema 6 在每条比较记录、CSV 和图中明确基线身份。缺失接口不回退到另一 schedule。ABC 的 prefill 使用 `fla.ops.abc.chunk_abc`，recurrent 记录 UNAVAILABLE。尚未实现上述 ABC 快速路径，以下数字仅验证 benchmark 接入。

远程 H20 GPU 0，B=H=1、T=65、K=V=64、SLOT=16、output-only，warmup=1、repetitions=3 的独立 smoke：ABC prefill 三项校验（Meta/oracle、FLA/oracle、两者输出）通过，public Meta/FLA 为 0.804/0.339 ms，Graph 为 0.0698/0.0421 ms；这不是稳定性能结论。原始产物在远程 `/tmp/meta_abc_graph_final/{records.json,manifest.json,summary.csv}`。

相关 CPU 单元测试 106 passed、4 GPU tests deselected，Ruff check/format 与 diff whitespace 检查通过。新增测试覆盖接口选择、禁止缺失接口回退、baseline-specific Comba 转换、FP32 authored transforms、GLA neutral padding、CSV 基线身份。GPU smoke 全 family output-only 初轮产物为 `/tmp/meta_dual_baseline_smoke`；其中 Linear/Simple GLA/GLA/RWKV7 的 prefill mixed-dtype 错误在适配器中保留 FP32 变换并统一 q/k/v dtype 后解决，独立复测产物为 `/tmp/meta_dual_baseline_fixed`。

GLA FP32 短 chunk 的 FLA output kernel autotuning 曾触发 misaligned address，独立复现记录在 `/tmp/meta_gla_prefill_continuation_probe.log`。适配器对长度 <64 的调用补齐零 q/k/v/log-gate，尾部为单位转移、零写入，裁掉多余输出；padding 成本计入 public 调用并写入 preprocessing。Linear/Simple GLA 的三合同已通过（`/tmp/meta_dual_baseline_contracts.log`）；RWKV7/Comba 的三合同通过（`/tmp/meta_dual_baseline_contracts_fixed`）；GLA 修正后 continuation 的输出和 final State 校验通过（`/tmp/meta_gla_continuation_final`）。这仅验证所列 B/H/T，不宣称全矩阵或所有 contract 已验证。

初轮 Delta Product 选择 reference-stateful 后出现 Meta/oracle 校验失败和 Graph capture 错误，继而污染同进程后续记录。失败原样保留，不计入有效比较；ABC 已在独立进程成功复测。当前 IR/ordered-transition 实现由另一工作范围修改，本轮不改生产 lowering 或放宽容差。benchmark 对 CUDA context 致命错误的进程隔离、增量产物落盘可另行改进；本轮未实现。
