# FLA 覆盖评估与 Algorithm IR 扩展建议

状态：保留基线评估结论与扩展建议；第一、二阶段实现进展另列于下节。基线能力矩阵对应下述 commit，不应当作新增 schema v4/v5 的能力清单。

## 范围与证据

评估对象为 `allinai2plus:/mnt/nvme/linyanxuan/MetaAttentionImpl`，源码基线为
`7e8ea16865623e7da1afa061d75cc6dc42c06d59`。相邻 FLA checkout 为
`../flash-linear-attention`，基线为 `f3bec720f7847d4f176872b34f002cc580e2cc6e`。
FLA 的 layers、models 和 ops 分别包含完整层、模型组合及核心算子，目录数量不能直接视为不同 attention 数学类型的数量。

本文将前期源码评估归约为能力分类、覆盖矩阵和扩展顺序。“结果归约”指评估结论的整理；下文 `Contract` 等归约操作另有明确数学含义。

术语使用根目录 [GLOSSARY.md](../GLOSSARY.md)。每项覆盖分别判断：数学表达、数值范围、状态兼容、完整 layer、实际执行性能。模型名相同、输出接近或调用成功，都不能单独证明其余契约。

已执行相关 CPU 检查：

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_stateful_v2.py tests/unit/test_stateful_v3.py \
  tests/unit/test_delta_family_examples.py tests/unit/test_example_imports.py \
  -m 'not gpu'
```

基线评估的结果为 **91 passed，18 deselected**。另通过公开 operator 接口临时验证了 LightNet 原始累加状态改写，见下文。基线评估未运行 GPU 正确性或性能测试；后续第一阶段实现的 GPU 正确性检查记录在下节，既有测量仅作为历史证据。

## 第一阶段实现进展

在完成本文基线评估后，schema v4 新增 `StateTransition.steps`、
`StateReference` 和 `StateInjection`，支持 propagation/injection 交错、
prior/current 引用与 current-state Transition DAG。接口与决定见
[API](API.md) 和 [ADR 0004](adr/0004-ordered-affine-state-transitions.md)。

- Delta Product 已改为有序写入，IR 规模为 O(n)。factored lowering 将早期写入的因子经过后续动作变换，保留原有编译路径；因子准备可为 O(n²)，不再进行指数展开。
- Momentum Delta Rule 已有 example，按照 FLA 的符号约定更新两个 matrix State。相同 roles、沿一个共享 Feature 轴的耦合更新组合为联合 block-affine summary，再做 segmented parallel scan。
- 本阶段允许引用 Prior State 或另一 State 的完整 Current State；中间 target 值由 steps 顺序表达，未增加任意命名中间版本。
- 单 token 联合 affine 更新仍不支持混合轴耦合或不同 State roles。Momentum 使用 reference-grade differentiable scan，尚无性能对比；跨 scan 的非线性系数变换已由下述第二阶段补齐。

新增验证覆盖输出、两组 initial/final State、continuation、packed 空序列、依赖环拒绝与辅助 State 梯度。H20 检查覆盖 Momentum 输出/梯度和 Delta Product 的编译路径跨 chunk 行为。直接调用相邻 FLA Momentum reference，在 FP32、B=H=2、T=17、K=3、V=5 下，两种 `use_p_times_alpha` 设置最大输出误差为 `1.49e-8` 和 `1.30e-8`，final states 也通过误差检查。最终 CPU 单元回归为 **405 passed，76 deselected**；两组相关 H20 检查共 **18 passed**，覆盖新增 examples 和既有 factored 路径。这些检查不是新增性能测量。

## 基线领域模型与实现边界

Algorithm IR 是数学描述，Stateful Operator 是其编译与调用入口。现有 AttentionEngine 的 pairwise score/online reduction 路径与 Stateful Operator 的 state scan 路径都需要纳入 FLA 覆盖评估。

| 概念 | 当前实现 | 需要保留的区别 |
| --- | --- | --- |
| State | 多个独立 named State；每个一个或两个 Feature 轴 | 多 state 数量不等于跨 state transition 依赖 |
| Product Feature | 两个 Feature 的扁平乘积，可占 State 的一个轴 | `[K*K,V]` 仍是两个 State Feature 轴 |
| Transition | 有序 propagation 后统一加 injections；表达式由 Inputs 驱动 | 尚不支持 propagation/injection 交错或 state-derived 系数 |
| Propagation | AxisScale、RankOnePropagation；有序执行 | 作用于已有 State，不能代替完整 Transition |
| Injection | 一项或多项 ProductInjection；每项可有多个因子 | 缺少自然的“写入后再传播”语义，不是缺少第三个因子 |
| Readout | 可组合多个 state、嵌套 contraction、算术和特征 Softmax | 非线性 Readout 不会使 State Transition 变为非线性 |
| Prior/Current State | Readout 支持 `prior=True`；默认读取完整更新后的 State | Transition 引用和中间 state 版本仍缺失 |
| Shift | 同一次 invocation 的序列内延迟，起点补零 | 不自动保存前一段 input history |
| Head Mapping | 输入 head sharing；reference executor 要求所有 State 共用一个 HeadRole | 辅助 state 按 key heads 共享的原生缓存契约仍有限制 |
| Scanable | 当前接受的 input-driven affine transition 有 associative summary | 数学可扫描、schema 接受、lowering 支持与硬件可行性分开判断 |

依据：[IR 定义](../attention_engine/attn_engine/stateful_operator.py)、
[validation](../attention_engine/attn_engine/stateful/analyse/validation.py)、
[analysis](../attention_engine/attn_engine/stateful/analyse/algorithm.py)、
[reference executor](../attention_engine/attn_engine/stateful/lowering/reference.py)。

领域模型需要明确三种不同依赖：

1. **Coupled Affine Transition**：多个 State 相互作用，但联合更新仍对联合 Prior State 仿射，可构造一个联合 summary。
2. **Scan Stage DAG**：先完成辅助 scan，再将它的轨迹非线性变换为下游 scan 的系数；不必具有单一联合仿射 summary。
3. **Nonlinear State Feedback**：被更新的 State 的非线性函数参与自身后续更新，现有 summary 证明不适用。

以上是数学分类；基线时后两类名称不对应新增 IR 节点。Schema v5 已实现有限的 Scan Stage DAG；Nonlinear State Feedback 仍未支持。Token 内的有序赋值依赖与整段序列的 Scan Stage DAG 不能混为同一个调度规则。

## FLA 类型覆盖归约

下表覆盖当前 checkout 中的主要语义类别。组合模型与同核 layer 按其核心运算归类，完整 layer 的投影、norm、卷积、路由和 cache 仍需逐项验收。

| 类型 | 数学表达与当前 example | 差距或限制 |
| --- | --- | --- |
| Linear/normalized linear、Simple GLA、GLA、Retention、Lightning | 核心 scan 可表达，已有 examples | 部分 linear examples 显式注入 `key + 0.1`，与 FLA 比较需要相同输入准备；不能直接视为原始 key 的 parity |
| HGRN、HGRN2、Rodimus GLA 核心、YOCO gated retention | 核心递归属于现有向量或矩阵仿射能力 | 主要是 example/layer 组合；短卷积和完整缓存另验 |
| Delta Rule、GDN、KDA、GDN2、ComBa、IPLR/DPLR、Oja | 已有核心 examples | DPLR 因式分解含 `exp(-gate)`；数值范围与具体 lowerer 单独判断 |
| RWKV6、RWKV7 | 已有 state algebra examples | 不能据此认定完整 token mixing、输出修正、layer cache 已覆盖 |
| GSA、Based、Rebased | 已有多 state readout/product-feature examples | Product Feature 增大状态与 workspace；表达能力不保证大维度高效执行 |
| ABC | 原始 weighted sums 与 readout 可表达，已有 example | 缺稳定 logcumsumexp 表达；三组 raw state 与 FLA normalized cache 不兼容 |
| Delta Product | 小固定写入次数可表达，已有 example | 传播与写入分组导致早期写入显式展开，规模随写入次数指数增长 |
| Momentum Delta Rule | 联合数学更新是仿射的 | 缺跨 state action、state 版本与联合 summary |
| LightNet | 原始累加状态的等价输出可用现有 IR 表达 | 稳定 log-space 版本、FLA state 表示与首 token identity 尚未覆盖 |
| Preconditioned KDA/GDN | 辅助 ATK affine scan 与主 affine scan 分别可表达 | 缺 scan 结果供下游 transition 使用的阶段组合；缺 Abs；共享 ATK head 契约另验 |
| Mamba/Mamba2 | 当前 Mamba2 example 覆盖 selective scan profile | 不等于完整 layer；discretization、卷积、输出与缓存须分别核对 |
| Mamba3、CyFA | 部分累积/缩放/slot readout 可复用 | 缺 phase/rotary、轴变换、MIMO 和 continuation history |
| MesaNet | 两组统计量积累可表达 | state-derived linear solve、matrix read/diagonal 和求解数值契约缺失 |
| NSA、MoBA、DSA | 注意力消费部分可考虑已有 score/mask 能力 | 完整流程涉及 compression/top-k/索引和动态稀疏访问；当前 block mask 不等于这些能力 |
| MoM、Raven | 路由后的 GDN/GSA 核心可表达 | 完整路由的 top-k、重排、gather/scatter 与多 memory 状态组织另缺 |
| Forgetting Attention | prefix scalar gate + pairwise softmax 属于 AttentionEngine 候选路径 | 当前同名 Stateful example 只是 Simple GLA，未覆盖 FLA 语义 |
| Wall、Parallax | prefix/channel gate 或双 score 统计的 pairwise attention | 需要核对 score construction、多组加权输出与 online reduction；尚无完整 parity 证据 |
| Path、Stick-breaking、DeltaFormer | 具有序列路径变换、query-specific 累积或三角依赖 | 需要 sequence-axis scan/solve 与多阶段 pairwise pipeline；普通 Feature Contract 不足 |
| Log-linear Attention/Mamba2 | 层级加权与历史缓存 | 状态层数/位置/未完成 chunk 历史不能直接视为固定大小 matrix scan |
| TTT、Titans | 非线性 State Feedback；TTT 有数学意义的 mini-batch 更新 | 需要独立 recurrent/inner-loop 决策，尚无 compact summary 证明 |

核对来源包含 FLA 的 [layers](../../flash-linear-attention/fla/layers)、
[ops](../../flash-linear-attention/fla/ops) 和 [models](../../flash-linear-attention/fla/models)。
这是一份语义分类，不是每个 FLA 入口、dtype、gate mode 或 backend 的逐项兼容清单。

### 必须修正的 example 映射

[现有 forgetting_attention](../examples/linear_attention.py) 委托 Simple GLA，
而 [FLA Forgetting Attention](../../flash-linear-attention/fla/ops/forgetting_attn/naive.py) 计算
`softmax(scale * q_i^T k_j + c_i - c_j) v_j`。本轮记录差异，不修改公开 factory。

ComBa 的不同 FLA 路径有 prior/decayed prediction 差异；DPLR 示例通过逆 decay 因子还原 prior-state 低秩项；RWKV7 的 state algebra 也不能替代完整 layer。任何 parity 文档都应说明被比较的公式、入口及输入准备，详见
[example guide](../examples/README.md) 与 [既有测量](stateful-example-benchmarks.md)。

## 四类关键扩展的数学依据

### Momentum：跨 State 不等于非线性

FLA reference 的约定是 `w = -(v - p^T S_prior)`，随后
`M_current = mu M_prior + eta k w`、`S_current = alpha S_prior - beta M_current`。
令 `L = eta k p^T`、`J = eta k v^T`，则

```text
M_current = L S_prior + mu M_prior - J
S_current = (alpha I - beta L) S_prior - beta mu M_prior + beta J
```

两个 matrix State 沿 KEY 轴堆叠后，更新有 input-driven block-affine summary。
Token 内 `M_current → S_current` 需要有序语义，但没有必要因此引入通用串行 token fallback。

需补跨 state affine action、明确 state 版本、联合 summary、反向 cotangent 传递及结构化 lowering。
只放开 validation 中的 StateTerm 禁令不能使当前逐 State executor 正确执行。

公式符号与用户提供的参考存在约定差异；公开 FLA 入口还有 `use_p_times_alpha`。
验收必须固定所用约定。依据：
[reference](../../flash-linear-attention/fla/ops/momentum_delta_rule/naive.py)、
[公开 recurrent 入口](../../flash-linear-attention/fla/ops/momentum_delta_rule/fused_recurrent.py)。

### LightNet：有限范围数学覆盖与稳定实现分开

稳定形式维护 `z_current = logaddexp(z_prior, x)`，再生成 normalized key 和 gate，见
[FLA gate scan](../../flash-linear-attention/fla/ops/lightnet/gate.py) 与
[GLA 组合](../../flash-linear-attention/fla/ops/lightnet/chunk.py)。

从零 memory 开始，可用两个 input-driven raw accumulator 改写：

```text
C_current = C_prior + exp(x)
U_current = U_prior + exp(x) v^T
output = (q / C_current)^T U_current
```

其中 `C` 为 KEY 向量，`U` 为 `[KEY, VALUE]` 矩阵；归一化必须逐 KEY 通道完成。
当前嵌套 StateTerm readout 可表达该式。临时 CPU 检查采用 FP32、B=H=1、T=7、K=3、V=2、随机 seed 2；不含默认 query scale，双方同用未缩放 query。最大输出误差为 `2.384185791015625e-07`，在 token 3 分段的 suffix 输出误差为 `0`。此检查没有新增正式 example 或测试文件。

`(U,C)` 与 FLA `(S,z)` 缓存不同。已有有限 `z` 的状态可按 `C=exp(z)`、`U=C*S` 转换，但需要显式转换与范围约定。raw sums 可能溢出或下溢；稳定覆盖需要 LogAddExp scan、`-inf` 初始 identity、首 token 分支与梯度语义，不能用 `Log(Exp(a)+Exp(b))` 代替稳定原语。

### Preconditioned KDA/GDN：辅助 Scan Stage → 主 Scan Stage

[FLA reference](../../flash-linear-attention/fla/ops/precond_kda/naive.py) 的辅助状态和变换为：

```text
A_current = exp(g_atk) A_prior + beta_atk k^2
r = log(A_current + eps) - center
m = exp(-log(x) * r / (1 + abs(r)))
k_write = k * m
```

主 memory 用原始 `k` 预测，用 `k_write` 写入。主更新在给定辅助轨迹后仍是仿射的，现有非对称 RankOnePropagation 可以承接其代数。
[FLA chunk 实现](../../flash-linear-attention/fla/ops/precond_kda/chunk.py) 也先计算 ATK，再运行主更新。

联合 `(A,S)` 更新不是 input-driven affine map，但可以通过 Scan Stage DAG 执行：先扫描 A，逐 token 生成 key，再扫描 S。需补内部 stage result binding、Abs、依赖检查、反向跨 stage 链路和 memory/checkpoint 策略。

[Preconditioned GDN reference](../../flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py) 还规定 ATK 按 key heads 共享、主 memory 按 value heads 保存。完整状态契约需要跨 State Head Mapping；复制辅助 state 可作显式改写，不等于原生缓存兼容。

### Mamba3/CyFA：Phase、变换与历史缓存

[Mamba3 layer](../../flash-linear-attention/fla/layers/mamba3.py) 维护 angle、SSM、K、V 状态，支持 SISO/MIMO，并将核心计算交给外部 `mamba_ssm` kernel。当前 Mamba2 example 不能证明这些能力已覆盖；此次未审计外部 kernel 的完整数学实现。

需要 phase scan、Sin/Cos、配对/分块 rotary、MIMO rank 的变换与归约、历史 K/V 的读取和更新，以及跨调用 continuation。
[CyFA reference](../../flash-linear-attention/fla/ops/cyfa/naive.py) 同样使用累积 phase、slot rotation 与多 state readout，可作为可复用性验收 example。

## 第二阶段实现进展（schema v5）

- `ScanStageIR` 组合单独证明的 affine `AlgorithmIR` 和有限 `LogAddExpScanIR`。Typed `StageBinding` 导出 Readout 轨迹、保留 prior/current、roles、dtype 与 sequence 身份；依赖无环且自动排序，内部输入不由调用者提供。
- `Abs` 与显式 `Cast` 支持 FP32 preconditioning 与 authored rounding。阶段 VJP 经 Torch autograd 汇合 output/final loss、传回 upstream initial State；可选 stage checkpoint 重算 child intermediates。
- [Preconditioned KDA/GDN](../examples/preconditioned_delta.py) 用 ATK scan 生成非对称写入 key，再扫描主 memory。ATK 按 key heads 保存，memory 按 value heads 保存；HeadMapping 共享轨迹并在反向汇总梯度。KDA 遵循 naive raw q/k，GDN 使用 FP32 q/k L2 normalization。
- [稳定 LightNet](../examples/lightnet_stable.py) 使用 log-space accumulator，identity 为 `-inf`，安全 identity VJP，cold 首 token gate=0；支持 finite initial z、非零 memory、continuation 和 FLA `(S,z)` 缓存显式转换。
- CPU 独立 reference 和 H20 输出、initial/final、packed empty 与梯度检查通过；直接 FLA naive KDA/GDN 和 LightNet gate/full operator 互验已完成。性能与入口合同差异见 [stage benchmarks](stateful-stage-benchmarks.md)。这仍是 stage 能力基线，reference scan 的训练存储和长序列调度需要后续优化。

接口与取舍见 [ADR 0005](adr/0005-scan-stage-dag-and-logaddexp.md)。General Nonlinear State Feedback、history、rotary 与 MIMO 仍未新增；阶段组合不构成对这些语言能力的隐式支持。

## 建议优先级

以下为基线评估提出的优先级。P1、P2 的能力范围已落地，见上节；P3–P5 仍是后续建议。已实现合同、验收矩阵与后续性能任务见 [能力增强实施计划](stateful-capability-roadmap.md)。

| 优先级 | 能力 | 首要验收收益 |
| --- | --- | --- |
| P0 | 修正术语和覆盖映射；补现有 IR 可表达的 examples；独立跟进已知 lowering 瓶颈 | 降低虚假覆盖，扩大真实数学覆盖 |
| P1 | 有序 affine steps：传播/写入交错；state 版本；跨 state action；联合 affine summary | 自然表达 Delta Product，避免指数展开；覆盖 Momentum |
| P2 | Scan Stage DAG；辅助轨迹生成 downstream coefficients；Abs；稳定 LogAddExp scan 与 identity | Preconditioned KDA/GDN、稳定 LightNet；为 ABC/RWKV4 稳定化提供基础 |
| P3 | Phase/rotary、轴变换、continuation history、MIMO 轴与归约、跨 state head sharing | Mamba3、CyFA 与更完整的 SSM/layer cache |
| P4 | 根据覆盖目标分别设计 state-derived solve、稀疏索引/路由或多阶段 pairwise pipeline | MesaNet；NSA/MoBA/DSA/MoM/Raven；Path/Stick-breaking/DeltaFormer 等 |
| P5 | Nonlinear State Feedback、mini-batch/inner-loop 更新与其 VJP | TTT/Titans；需单独确认执行与性能契约 |

P1 与 P2 可共享类型、表达式和依赖检查，但必须分别证明联合 affine summary 或各 Scan Stage 的合法性。LogAddExp 本身非线性但具结合性；增加非线性表达式不等于支持一般 Nonlinear State Feedback。

有序 steps 属于 Algorithm IR 的数学语义。一个 token 内多次写入、TTT 的 mini-batch 更新都不能当作可任意改变的 kernel chunk 参数。

## 验收与后续决策

每个 example 应有独立 recurrence/reference，并在公开 StatefulOperator 或 AttentionEngine 调用入口验证：

- 输出、非零 initial state、final state 与两段 continuation；状态转换另行说明。
- output-only、final-state-only 与联合 loss 的梯度，包括所有辅助 initial state。
- grouped heads、不同 Feature 大小、dense/packed、空序列与尾部 token 的适用范围。
- 稳定性范围：极端 logits、零/极小 gate、初始 identity，以及输入与 State 精度。
- 所选 lowerer、workspace、public-call latency 与明确的 FLA 入口；prefill、decode、backward 分别测量。

CPU 检查和已有 forward benchmark 不足以确认新能力的 GPU 性能。reference lowering 能执行也不代表性能覆盖；历史数据见
[stateful example benchmarks](stateful-example-benchmarks.md)，结构优化方向见
[stateful specialized paths](stateful-specialized-paths.md)。

P1 的首个接口范围由 ADR 0004 记录，P2 的显式 stage authoring、有限 LogAddExp algebra、State 转换和验收范围由 ADR 0005 与阶段基准记录。P3 的 history、rotary/MIMO 与其他新 State 合同仍需设计和独立验收。
