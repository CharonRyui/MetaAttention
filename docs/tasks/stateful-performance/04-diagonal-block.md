# P2-A diagonal block algebra 与分层 carry

## 状态
已实现 inverse-free parallel block lowering；本轮 bounded 2-target readout 已通过独立 oracle、final-State/continuation 和完整同合同 latency/allocator 矩阵。完整 activity 证据另用每 workload 新进程采集，不将缺失 device events 当作零时间。

## 依赖
00

## 文件归属
_stateful_diagonal_kernels.py; _stateful_chunked.py integration owner

## 实现合同
AxisScale + separated injection + contraction 使用稳定 chunk-local decay 与 masked block contraction；保持两个轴因子；短序列 streaming carry，长序列 segmented hierarchical prefix；禁止通过 inverse gate 恢复。

## 验收
scalar/row/column/dual gate，零 gate、strong decay、tails、prior、多个 batch 正确；实测降低 serial replay 成本。

## 当前实现与成本
`_stateful_diagonal_block.py` 按 32×32 State tile 执行 summary：对 shifted row/column gates 分别做 reverse associative forward-product scan，再以 TF32x3 weighted-key/value GEMM 得到 bias。readout 按 `(batch×head, 2-target-token tile, column-tile)` 并行；同 tile 共享源 gate/key/value/State loads，直接 FP32 multiply/reduce 得到 scores、incoming-State contraction 与 weighted-value contraction，不复制 query 成 16 行 dot。prior 的终点是 `t-1`，不对零 gate 求逆、不使用 log-prefix subtraction；row/column 因子仍在 contraction 前分离，column suffix 在 row loop 完成后生成以缩短 register live range。summary、FP32 carry、named finals 与 workspace 收费不变；最后的 target tile masked，不启动整块无消费者 tail CTAs。

令 `N=B×H`，`C=ceil(T/64)`，`R`/`D` 是两个 State 轴。summary/carry 的 global FP32 workspace 为 `3×4×N×C×R×D` bytes（scale、bias、carry），另加可选 final `4×N×R×D` bytes。分层 scan 每个递归层额外分配 `3×4×N×ceil(C/32)×R×D` bytes，直到 chunks≤64。readout 只分配 `4×B×T×H×D` output bytes；没有 token-sized matrix States 或 decay 矩阵。上述不含输入 `.float().contiguous()` 的 dtype/layout 转换。

summary 是一次 kernel launch；短序列 carry 一次，readout 一次，长序列 carry 每个递归层增加两次 launches。令 `Nr=ceil(R/32)`、`Nd=ceil(D/32)`，summary dot FLOPs 仍为 `131072×N×C×Nr×Nd`。旧 dual-gate target kernel 的 padded readout dot FLOPs 为 `N×T×Nd×(98304×Nr+65536)`，只保存 16-row dot 的第零行；新 readout 以固定 2 targets/32 rows/32 columns 的 FP32 reductions 消除这份 replicated arithmetic，不增加 global decay/score/State storage。readout reduction association 不要求对旧路径逐位相同，必须通过既有独立 FP32/BF16 oracle 容差；所有返回的 final States 因 summary/carry 未修改而要求逐位 A/B 相同。永久回归覆盖 T1/63/64/65/137、scalar/row/column/dual、zero/signed/strong gates、prior、B>1、非方形 axes、separated product Features，以及 split=17 改变 chunk/target tile 相位的 continuation。

## 候选筛选记录（2026-10-02）

固定 baseline `d386d2ee2946c081e2cbb840d82a186e4f18f562`，旧 diagonal 源码 SHA256 `693d27a965d69503806e49c463be53ff4524fdcbd2726e2d39c2b975eaaa6da1`，不重复归因 normalized 前几轮收益。早期实际 public smoke 对 RWKV6/GSA、B1/H8、T64/2048 的 output/finals 独立 FP32 oracle 均通过，final States 对旧路径逐位相同。

4-target/4-warps 的 row-only readout 编译产生 6 spills；增至 8 warps 后虽零 spills，但 T2048 Graph 回退，拒绝这两个版本。1-target/4-warps 零 spills，但长序列不及 2-target。最终 2-target/4-warps 将 column interval tile 延迟到 row loop 后，早期默认形状资源为 145–216 registers/thread、0 spills、16384 bytes shared memory；这些 smoke 不代替完整 public/Graph/allocator/CUPTI 验收。所有候选原始数据保留在 `benchmark_results/diagonal_target*_smoke.json`。

## 完整同合同测量（2026-10-02）

矩阵为 RWKV6/GSA × B(1,2) × H(1,8,16) × T(63,64,65,130,2048,8192) × output-only/final-state/continuation，共 216 点；同一 compiled operator、输入、独立非零 initial States，只切换 private diagonal readout。两侧统一 4 GiB planner cap，生产默认仍为 1 GiB；6 个 T8192 大形状默认 bound 拒绝，不能把补充 A/B 宣称为默认 cap 全矩阵成功。原始记录 `benchmark_results/diagonal_readout_ab.json`。

| contract | 点数 | public CUDA-event 几何平均下降 | CPU wall 下降 | Graph 下降 |
| --- | ---: | ---: | ---: | ---: |
| output-only | 72 | 7.88% | 7.87% | 16.46% |
| final-state | 72 | 7.46% | 7.52% | 16.38% |
| continuation | 72 | 9.39% | 9.33% | 13.97% |
| 全部 | 216 | 8.25% | 8.24% | 15.61% |
| T2048/8192 | 72 | 17.83% | 17.75% | 23.02% |

216 点的 peak allocated/incremental 与旧路径逐点相同；这是算术减少，不是内存或 launch 减少。144 个返回 final-State 场景逐位 A/B 相同，输入/initials 均不变。完整默认 readout 资源为 128–217 registers/thread、0 spills、16384 bytes shared memory、4 warps。

第一轮 22 个 public 点和 1 个 Graph 点比值 >1.03，全部保留；对这 23 点各做两轮 ×40 次交错 public 采样及独立 Graph 复测。合并 public 样本的最差 new/old 为 1.0173，Graph 最差 1.021；单轮仍有噪声，不宣称所有逐点 public 都加速。原始复测 `benchmark_results/diagonal_readout_retests.json`，不替换原矩阵均值。

行为矩阵 `benchmark_results/diagonal_readout_behavior.json` 240/240 场景通过，覆盖 FP32/BF16、row/column/dual、prior/post、非方形 Features、grouped heads、signed/zero/strong gates 与非对齐 continuation。实际 GPU expression/readout/hierarchical suite 76 passed；CPU expression/v2/v3/binding/workspace/selection 47 passed、4 deselected；相关 Ruff 通过。

完整 T8192/B2/H16 的两家族 ×三合同独立 CPU oracle 6/6 通过，continuation split=4097；证据 `benchmark_results/diagonal_readout_long_oracle.json`。RWKV6 bonus 的 authored BF16 multiply 必须逐节点舍入，再进入 FP32 Contract；原 all-FP32-expression oracle 在旧、新路径均失败，修正一次性 oracle 的 dtype 合同后通过，既有 output/State 容差未变。该文件同时保留错误 oracle 的对照，不将其失败归因到 readout。

原矩阵 `FAILED_GATE` 原因是同进程反复 profiler 后开头 device events 渐进缺失；保留完整失败证据。新进程 GSA/B2/H16/T65 探针 6/6 捕获全部 43 次 runtime launches 和 43 个 kernel events，并单独排除一个 priming spin，完整 trace 为 `benchmark_results/diagonal_activity_trace_probe.json`。最终 `benchmark_results/diagonal_readout_activity_isolated.json` 对全部 72 workloads 分别新建进程：216 合同点、1296 个原始样本全部完整捕获，逐样本 runtime launches/device kernel counts 相等，spin 始终独立排除。kernel activity 时间几何平均下降 16.66%，launch 数不变；不修改原失败记录。采集中途 `/tmp` 所在 filesystem 无可用空间，改到有空间的 NVMe 临时目录完成剩余 5 个 workloads，失败日志保留。

默认 Target 实际 CLI 为 `../bench.sh --families rwkv6,gsa --seqlens 64 2048 8192 --batches 2 --heads 8 --contract all --timing both --warmup 3 --repetitions 10 --gpu 6 --output-dir benchmark_results/diagonal_readout_public_smoke`。18 条记录中 14 条 Meta OK，4 条 T8192 的 output-only/final-state 因现有 1 GiB workspace bound 拒绝；两家族 T8192 continuation 均成功。该限制不通过提高生产 cap 绕过。FLA 仅性能基线，该 smoke 不代替本轮同合同 A/B，也未证明 Meta 快于 FLA。

综合摘要 `benchmark_results/diagonal_readout_acceptance.json` 列出全部数值、复测、默认 cap 拒绝和证据源。原始 JSON 保留本地，交付同时归档 gzip 压缩记录；一次性 GPU/oracle/复测脚本已删除，用户既存 `testing/benchmark_fla_meta_stateful.py` 未改动。


## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
