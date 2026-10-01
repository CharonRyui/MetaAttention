# P2-A diagonal block algebra 与分层 carry

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00

## 文件归属
_stateful_diagonal_kernels.py; _stateful_chunked.py integration owner

## 实现合同
AxisScale + separated injection + contraction 使用稳定 chunk-local decay 与 masked block contraction；保持两个轴因子；短序列 streaming carry，长序列 segmented hierarchical prefix；禁止通过 inverse gate 恢复。

## 验收
scalar/row/column/dual gate，零 gate、strong decay、tails、prior、多个 batch 正确；实测降低 serial replay 成本。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
