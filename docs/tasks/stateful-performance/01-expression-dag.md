# P1-A 表达式 DAG 与 kernel 融合

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00

## 文件归属
_stateful_chunked.py; new private expression compiler

## 实现合同
将现有表达式树编译成共享 typed DAG；跨 State/input/readout CSE，不改变 FP association；kernel prologue/epilogue 执行 elementwise chains；reduction 和多消费者决定 materialization cut；消除 token-sized 常量。

## 验收
公共 callable 覆盖多个 State 和重复 subexpression；独立 FP32 oracle；有效减少物化/launch，public/Graph 不回退。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
