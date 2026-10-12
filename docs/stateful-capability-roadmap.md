# Stateful IR 下一阶段能力增强计划

状态：P2 能力交付已实现，schema v5。Scan Stage DAG、typed binding、Abs/Cast、跨阶段梯度、GDN head sharing 与稳定 LogAddExp 已通过公开调用验证；checkpoint、资源估计和独立 benchmark 已接入。长序列性能仍需后续优化，实测记录见 [阶段基准](stateful-stage-benchmarks.md)。

## 当前起点

[ADR 0004](adr/0004-ordered-affine-state-transitions.md) 记录有序 affine steps、
Prior/Current StateReference 与跨 StateInjection。Delta Product 的 public IR
为 O(n)，factored 路径的后续动作因子准备可为 O(n²)；Momentum 用联合
block-axis summary 做可微 segmented scan。

当前有两个明确边界：

- current-token Transition DAG 描述一个联合仿射更新；它不能表达非线性辅助状态变换后生成下游 scan 系数。
- 耦合要求相同 State roles、单一共享动作轴；通用 PyTorch scan 是数学与梯度基线，不能据此宣布高效 GPU 覆盖。训练时 autograd 会保留 scan 各级中间值，workspace 估计还需覆盖这些实际存活张量。

基线覆盖分类见 [FLA 评估](fla-ir-coverage-assessment.md)。P2 的接口与数值合同见 [ADR 0005](adr/0005-scan-stage-dag-and-logaddexp.md)。下一步能力主线为 P3；现有 stage 与 readout lowering 性能增强按独立范围推进。

## 实施顺序与交付边界

| 顺序 | 能力与交付 | 验收 example | 完成条件 |
| --- | --- | --- | --- |
| 1 | Scan Stage DAG、typed stage result binding、Abs、跨 stage VJP 和资源估计 | Preconditioned KDA | 输出、两组 final State、continuation、packed、输入/辅助 initial State 梯度通过 |
| 2 | Stage 间 Head Mapping 与 auxiliary head sharing | Preconditioned GDN | ATK 按 key heads、主 memory 按 value heads 保存；原生 State 形状与共享梯度归约通过 |
| 3 | 稳定 LogAddExp prefix、每种 scan algebra 的 State identity、首 token 与 prior 轨迹处理 | 稳定 LightNet | 极端 logits、非零缓存、`-inf` identity、packed 空段、梯度与 FLA 缓存合同通过 |
| 4 | H20 stage execution、轨迹 lifetime/checkpoint、公开调用 benchmark 注册 | 上述三种 examples | correctness 通过后报告 prefill/decode/backward、workspace 与 FLA recurrent/prefill 对比 |
| 后续 | Phase/Sin/Cos/rotary、轴变换、history continuation，之后增加 MIMO | CyFA、Mamba3 | 全部数学状态和历史缓存分段一致；Mamba3 外部 kernel 先单独审计 |

单独的现有能力性能任务：联合 affine scan 的 chunk/checkpoint 降低、Delta Product 因子准备，以及 ABC readout region。ABC 的具体 lowering 计划见 [ABC fast-path assessment](abc-fast-path-assessment.md)。它改善已有 raw-state 算子的执行，不等同于稳定 ABC 或新增 Scan Stage DAG。

## 第一步：阶段组合的最小语义

推荐采用显式 typed stage 描述作为首版 authoring 形式。每个 scan stage 内复用现有 AlgorithmIR；组合层声明 upstream trajectory/result 到 downstream 内部 binding 的关系。它们不成为调用者必须提供的外部 TensorInput，也不允许 runtime 任意张量替换改变 Program 语义。

需先定义并实现：

1. **结果类型**：State trajectory 的 Feature/Head roles、dtype、prior/current 时序与 packed sequence 身份。Final State 用于 continuation，不能代替逐 token trajectory。
2. **依赖规则**：单向、无环；downstream 系数只依赖外部 Inputs 和已完成 upstream stage 的轨迹。禁止通过 stage binding 读取 downstream 自身待更新 State。
3. **局部证明**：每个 stage 单独证明其可扫描 algebra。当前 Transition DAG 的联合 affine summary 与 Scan Stage DAG 的阶段执行分别建模，不将 state-derived 非线性系数误判为 input-driven 联合仿射更新。
4. **梯度**：先完成 downstream VJP，再经系数变换传回 upstream trajectory 和 initial State；同时汇合各 stage Readout 与 final-state loss 的 cotangent。Abs 在零点采用 PyTorch 的零 subgradient。
5. **公开 State**：按声明的顺序汇总各 stage 的 initial/final State；保留现有单 stage operator 的调用合同与缓存身份。
6. **资源**：先采用明确物化的轨迹基线。编译估计和 runtime envelope 包括每个 stage summary、轨迹、表达式中间值和反向保存值，记录 overlapping lifetime；随后才加入 checkpoint/recompute 选择。

已采用 `ScanStage`、`StageBinding`、`StageOutput` 和 `StagedStatefulOperator`，输入 binding 保留 logical roles/dtype，并转换物理 axis 顺序；`Cast` 明确转换位置。每个 stage 导出 Readout，从而复用 prior/current 选择和类型检查。接口、限制与用法见 [API](API.md#scan-stage-dag-schema-v5)。不增加任意非线性 recurrence interpreter。

Preconditioned KDA 的首个完整链路：

```text
A scan: A = exp(g_atk) * A_prior + beta_atk * k^2
pointwise: r = log(A + eps) - center
           m = exp(-log(x) * r / (1 + abs(r)))
           k_write = k * m
S scan: decay S, predict with original k, write with k_write
readout: q^T S
```

沿用 FLA 的 eps、center、query scaling 和 FP32 变换约定。非对称 RankOnePropagation 已能承接主更新；主要新增的是 trajectory binding 和阶段反向传播。先同 Head role 验证 KDA，再处理 GDN 的 key/value head sharing，避免用重复 ATK 缓存冒充原生 State 兼容。

## 稳定 LightNet 的专门要求

LogAddExp 是有限的 associative scan algebra。不能仅新增 `LogAddExp` 表达式然后放开一般 State-dependent Transition；需定义 reduction State、初始化与 scan summary 的合法形式。

- Log-space accumulator 的默认 identity 为 `-inf`。不通过放开 Constant 的所有无穷数来实现；identity 属于 State/scan algebra 的合同，显式 initial State 仍逐项验证。
- 首 token 的 gate/key 与 FLA gate scan 对齐。明确 Prior `z=-inf`、显式有限 initial z、非零主 memory 的行为，避免直接计算 `-inf - -inf`。
- 保留 FLA 的 `(S,z)` continuation 表示；当前数学改写 `(U,C)` 只作为有限数值范围的对照，不能成为稳定性的验收 reference。
- `logaddexp(-inf,-inf)` 的输出 identity 和梯度需要显式约定；padding、空序列和无消费者分支不能向梯度注入 NaN。
- 稳定辅助 scan 完成后再生成 normalized key/gate 并扫描主 memory。大正负 logits 下检查输出和梯度，禁止用 raw Exp sums 冒充稳定实现。

## 验收矩阵

每个交付跨公开 compile/invoke seam，以独立数学 reference 为主要正确性 oracle，FLA 指定入口作互验。

- Dense 多 batch、grouped heads、K≠V；packed 含空序列、全空输入、尾部 token。
- 从零及非零 initial State 开始；两段和多段 continuation；State 表示转换单独测试。
- Output-only、final-state-only、联合 loss；所有输入和 auxiliary initial State 梯度。
- FP32/BF16、极端 gate/logits、Abs 零点、identity 和 dtype rounding。
- 无效依赖、roles、dtype 和资源 envelope 在编译或执行前拒绝。

CPU 数学检查通过后再运行 H20。性能测试分别记录 prefill、decode 与 backward；FLA recurrent 和 chunk/prefill 分开，声明 timed preprocessing、selected lowerer、峰值内存和 source revision。先覆盖短序列、chunk 边界和 continuation，随后测 T=2048/8192；不把有限形状正确性或单次 reference latency 当作性能完成条件。

TTT/Titans 的一般 Nonlinear State Feedback、Mesa solve 与稀疏 routing 保持后置，不纳入这轮阶段组合的最小范围。
