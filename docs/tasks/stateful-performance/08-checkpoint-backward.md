# P3 checkpointed backward

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
01,03,04,05

## 文件归属
_stateful_backward.py; _stateful_factored.py; private reverse chunk kernels

## 实现合同
forward 保存 FP32 chunk checkpoints；chunk-local replay 恢复状态后 VJP；反向 summary/prefix 传播 adjoint；所有 input/State/readout gradients；deterministic head reductions；不保存完整 BF16 token States、不逆 recurrence。

## 验收
FP64 autograd oracle、所有 gradients、grad_final、zero/strong decay、非对齐 continuation；training latency/memory 改善。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
