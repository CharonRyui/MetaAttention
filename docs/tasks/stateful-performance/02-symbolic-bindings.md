# P1-B 符号化布局与 HeadMapping

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
01

## 文件归属
stateful_operator.py; stateful/lowering/reference.py; lowerer bindings

## 实现合同
保留逻辑 roles、物理 strides、broadcast 与 head mapping；kernel 原布局加载 BF16 后 FP32 运算，不提前 head repeat/完整 FP32 copy；operation codes 编译期绑定；保持公共 contiguous contract。

## 验收
B>1、grouped heads、非方形 State、continuation、grad reduction 正确；实际 copy/launch/peak memory 下降。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
