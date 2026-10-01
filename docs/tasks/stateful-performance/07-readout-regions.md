# P2-D readout DAG region fusion

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
01,04,06

## 文件归属
_stateful_chunked.py; private fused readout kernels

## 实现合同
按 typed DAG fusion region；独立 contractions 共享 State replay；arithmetic epilogue fusion；StateTerm→Softmax→StateTerm 依赖保持方向和 reduction；资源成本约束，避免 all-in-one 寄存器爆炸。

## 验收
GSA、normalized、prior readouts、多个输出独立 oracle；launch/peak memory 与 latency 改善，非法 fusion 不改变 semantics。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
