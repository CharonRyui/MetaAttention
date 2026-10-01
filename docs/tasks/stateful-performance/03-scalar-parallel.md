# P1-C scalar/identity 序列并行

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00

## 文件归属
_stateful_scalar.py; private scalar chunk kernels; planner integration by owner

## 实现合同
实现原输入加载的 sequence-parallel summary/prefix/readout；保留短序列高 B*H fused streaming；长序列低 B*H 使用 chunk grid；支持 tails 与 nonzero State；候选选择不用模型名。

## 验收
T=63/64/65/130/2048，B=1/2，FP32 final-State 与 continuation oracle；public/Graph 对比，短序列不退化。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
