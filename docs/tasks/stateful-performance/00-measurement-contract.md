# P0 公平性能评测合同

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
无

## 文件归属
testing/benchmark_stateful_plans.py; testing/benchmark_linear_attention.py

## 实现合同
统一 output-only、final-State、continuation 合同；FLA 错误输出不阻止计时；区分 CPU wall、CUDA-event public call、Graph、native-layout preparation；记录实际 peak memory 与总 launch 数。所有 FLA adapters 明确 preprocessing 是否计时，不把无等价 final-State 的调用标成可比。

## 验收
小规模 H20 CLI 完整输出；baseline 错误不影响 Meta oracle 检查；计时外编译；无 GPU 时明确失败。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
