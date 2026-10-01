# P2-C ProductFeature 收缩顺序

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00

## 文件归属
_stateful_diagonal_kernels.py product regions; new private product kernels

## 实现合同
从 typed Outer/Contract 证明 (a⊗b)·(c⊗d)=(a·c)(b·d)；chunk-local scores 使用基础 feature GEMM；inter-chunk 保持显式 State；共享 numerator/denominator；默认保持线性 T 复杂度。

## 验收
非同一 left/right、ProductFeature order、Based/ReBased、initial/final-State、continuation 正确；减少 expanded-axis 计算，不引入 quadratic T 默认。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
