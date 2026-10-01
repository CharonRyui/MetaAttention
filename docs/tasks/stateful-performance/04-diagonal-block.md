# P2-A diagonal block algebra 与分层 carry

## 状态
已实现 inverse-free parallel block lowering；GPU correctness/performance 验收由集成负责人执行，未测量前不宣称完成。

## 依赖
00

## 文件归属
_stateful_diagonal_kernels.py; _stateful_chunked.py integration owner

## 实现合同
AxisScale + separated injection + contraction 使用稳定 chunk-local decay 与 masked block contraction；保持两个轴因子；短序列 streaming carry，长序列 segmented hierarchical prefix；禁止通过 inverse gate 恢复。

## 验收
scalar/row/column/dual gate，零 gate、strong decay、tails、prior、多个 batch 正确；实测降低 serial replay 成本。

## 当前实现与成本
`_stateful_diagonal_block.py` 按 32×32 State tile 执行 summary：对 shifted row/column gates 分别做 reverse associative forward-product scan，再以 TF32x3 weighted-key/value GEMM 得到 bias。readout 按 `(batch×head, target-token, column-tile)` 并行，直接扫描 source→target 区间；prior 的终点是 `t-1`，不对零 gate 求逆、不使用 log-prefix subtraction。row/column 因子在 contraction 前保持分离。FP32 carry 和 TF32x3 dot 的合同不变；非 identity 不再串行 replay 64 个 token。

令 `N=B×H`，`C=ceil(T/64)`，`R`/`D` 是两个 State 轴。summary/carry 的 global FP32 workspace 为 `3×4×N×C×R×D` bytes（scale、bias、carry），另加可选 final `4×N×R×D` bytes。分层 scan 每个递归层额外分配 `3×4×N×ceil(C/32)×R×D` bytes，直到 chunks≤64。readout 只分配 `4×B×T×H×D` output bytes；没有 token-sized matrix States 或 decay 矩阵。上述不含输入 `.float().contiguous()` 的 dtype/layout 转换。

summary 是一次 kernel launch；短序列 carry 一次，readout 一次，长序列 carry 每个递归层增加两次 launches。令 `Nr=ceil(R/32)`、`Nd=ceil(D/32)`，summary dot FLOPs 为 `131072×N×C×Nr×Nd`；当前 dual-gate target kernel 利用硬件最小 16-row dot（只保存第零行），readout dot FLOPs 为 `N×T×Nd×(98304×Nr+65536)`，还包含 forward-product scans 和逐元素乘法。该 FLOP 数明确计入 padded/duplicated arithmetic；没有未经实测的 latency 改善声明。独立 CPU FP32 token oracle cases 位于 `tests/functional/test_stateful_readout_regions.py`，涵盖 scalar/row/column/dual、zero/signed gates、strong decay、tails、prior、B>1；子代理未运行检查。


## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
