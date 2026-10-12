# Delta-product benchmark 正确性修复计划

状态：已完成 oracle 顺序修复及回归验证。显存改动已提交为 `3d032cb`；本任务修复 benchmark 正确性判断，不改变已验证的显存优化。

实施结果：共享 oracle 按 `ordered_steps` 执行；CPU 全量 551 通过、96 跳过；GPU inference 23 通过，新增长度边界独立对照 16 通过。原 50 条矩阵 Meta 全部 VALID，META_INVALID 与 META_RUNTIME_REJECTED 均为 0；delta-product 三种合同及 FLA prefill 全部通过。详细证据见 [H20 修复报告](../../measurements/delta-product-oracle-fix-h20/README.md)。以下保留原诊断依据与实施计划。

## 已确认的问题

`stateful_fla_meta_staged_fix_final_20261012` 中 delta-product recurrent/prefill 的 `META_INVALID` 都来自同一次 Meta correctness 检查，现有 FP32 oracle 报告 output max_abs=0.282324，容差为 rtol=0.02、atol=0.05。

`examples/gated_delta_product.py` 声明的每 token 操作为：

1. 对 State 应用 scalar decay。
2. 对每个 inner step j，先应用 RankOnePropagation，再应用对应的 ProductInjection。
3. 所有 inner steps 完成后再 readout。

`testing/benchmark_stateful_plans.py::_token_oracle` 却分别遍历 `transition.propagations` 和 `transition.injections`：所有 rank-one actions 先执行，所有 writes 后执行。它忽略 `transition.ordered_steps` 的交错顺序，因此后一 rank-one action 没有作用于前一 write。该 oracle 同时被 `testing/benchmark_fla_stateful.py` 调用。

最小 CPU 反例（B=H=T=1，K=V=2，zero initial State，gate=0，beta_0=beta_1=1）：

- key_0=[1,0]，value_0=[1,2]。
- key_1=[0.5,0.5]，value_1=[0,0]。
- 独立有序残差递推的 State=[[0.75,1.5],[-0.25,-0.5]]。
- 当前 runner oracle 的 State=[[1,2],[0,0]]，max_abs=0.5。

这是已经执行的确定性复现，不是从长序列误差推断出来的假设。引擎的 factored recipe 则记录 injection 后续 propagation 并转换其贡献，不能把 oracle 中丢失的顺序归咎于该转换。

## 原失败尺寸的独立核对

在 `3d032cb` 上，H20 GPU 3、seed 42、B2/H16/T8192、K=V=64、BF16 输入、1 GiB 默认预算、zero initial State，直接运行 public compiled inference，比较独立 CPU FP32 recurrence：

`S <- exp(g) S; S <- S + k_j [beta_j (v_j - k_j^T S)]^T`，按 j 顺序更新，然后计算 q^T S / sqrt(K)。

结果：output max_abs=0.0039052963、超既有容差元素=0；FP32 final State max_abs=2.3841858e-7，符合既有 State 容差 rtol=0.01、atol=0.02。原始证据及可重跑脚本见 `../../measurements/delta-product-oracle-h20/`。

因此，先前“delta-product BF16 长序列数值问题”的归因需要更正：当前证据确认 benchmark oracle 的顺序错误；原失败输入上的 Meta inference 通过独立有序 recurrence。尚未检验所有 shape/dtype/State/step 数的组合，不宣称所有 numerical coverage 已完成。已有 6 项 BF16 backward 与 2 项 two-axis GLA 测试失败仍是独立问题。

## 实施顺序

1. **先锁定实际 oracle seam。** 在调用 `_token_oracle` 的单元测试中加入上述非交换两步写入反例，同时检查 output 和 final State。预期为手算矩阵；不复用引擎 evaluator、factored recipe 或 kernel 构造 expected。观察现有 oracle 失败。
2. **修复 ordered steps。** 检查 `_token_oracle` 的所有调用方和它承诺支持的 IR 节点，按 authored `ordered_steps` 执行并在每一步更新局部 State。legacy propagations-then-injections 的顺序应自然保持。保留 token 的 prior-State 快照、所有 readout 时序以及 HeadMapping。确认 Identity/StateInjection 等节点的支持范围；不把 unsupported 节点静默跳过。
3. **核对精度合同。** 从现有 IR spec、显式 Cast、节点类型和已接受的 functional oracle 确认 FP32 State 与 BF16 输入表达式的舍入位置。当前全 FP32 runner oracle 已能在原输入下通过独立对照；不把本次顺序修复扩大为全局精度策略变更。只在单独对照证据要求时增加精度修复，并保留每个原因的独立回归。
4. **回归独立有序递推。** 覆盖 num_householder=1/2/3、T=1/16/17/128/129/1023/1024/1025/8192、FP32/BF16、zero/nonzero initial State、非方形 State、grouped heads、output-only/final-State/非对齐 continuation、多个 readout 与 CUDA Graph。长序列以原 B2/H16/T8192 seed 42 为必要验收点；先用小 B/H 跑边界，再做原始大尺寸。测试 expected 用直接 residual token recurrence；不依赖 FLA。
5. **重新生成 benchmark 证据。** 先跑 delta-product 两种 FLA baseline 与三个 contracts，再复跑原 50 条矩阵，保留旧记录。修正 Meta oracle 可能暴露 FLA API 或精度问题，按实测状态报告；不强制所有记录变成 OK。对受 ordered steps 影响的其他算子（尤其 staged 子算子使用的独立 oracle）检查是否共享同一错误。
6. **只在修正 oracle 后仍存在 Meta 失败时调查 kernel。** 按下面的单变量对照定位；没有残余失败时不更改 GPU kernel。

## 残余 kernel 问题的候选根因与区分实验

以下均未被当前 seed 42 独立验证支持；作为后备诊断顺序：

1. 若误差来自 BF16 表达式舍入，保持同一量化输入，将叶值无损提升至 FP32并保持输出 dtype，误差分布应改变；逐节点对照 query scaling、beta*value 及 suffix factors，区分合法精度合同与实现不一致。
2. 若误差来自 ordered-write suffix 转换，1-step 应通过而非交换 2/3-step 在 direct/WY 都失败；用短序列、非零 State 和手算 State 逐步对照 prepared factors 与独立 ordered residual recurrence。
3. 若误差来自 WY solve/carry/readout，使用完全相同 prepared FP32 factors 的 direct 路径应通过、WY 失败；检查 T=128/129 与 16-token chunk boundary，定位首个 State 或输出分歧，并核对 triple-TF32/GEMM association。
4. 若误差来自窗口 carry，改变窗口边界和续算 split 应改变首个失败位置；比较同一输入整段与分段运行的最终 FP32 States，再检查 output cast。不得仅用缩短窗口掩盖错误。

## 验收与提交范围

- 原 B2/H16/T8192 seed 42 的 Meta output-only/final-State/continuation 通过独立有序 oracle；output 容差保持 rtol=0.02、atol=0.05，runner State 容差保持 rtol=0.01、atol=0.02。更严格短序列 functional 容差也不放宽。
- oracle 单 token 反例通过，并且改变两次 rank-one/write 顺序会改变结果；已有 legacy oracle 用例通过。
- 默认 1 GiB 仍可运行，显存回归 `test_factored_memory.py` 通过；记录 public/Graph latency 与 peak memory，任何 kernel 变更不得恢复全长 scratch。
- 保留 old/new correctness 和版本信息，明确区分 Meta numerical acceptance 与 FLA availability/comparability。
- 按实际修改提交 oracle、必要回归测试和观测证据。GPU kernel 仅在有残余可复现问题且修复通过上述对照时纳入提交。不按 family 名称添加特判，不修改模型数学或放宽容差。

正式 runner 重跑模板（`<idle_gpu>` 由运行前 idle query 选择）：

```bash
PYTHONPATH=attention_engine .venv/bin/python testing/benchmark_fla_stateful.py \
  --families delta-product --batches 2 --heads 16 --seqlens 8192 \
  --contract all --fla-baseline both --timing both \
  --gpu <idle_gpu> --output-dir benchmark_results/delta_product_oracle_fix
```

计划已实施：修改 oracle、必要回归测试和观测证据；GPU kernel 无需修改。benchmark oracle 的耦合 StateInjection 等其他 IR 支持扩展不属于本次 delta-product 修复范围，遇到 unsupported step 会显式报错。
