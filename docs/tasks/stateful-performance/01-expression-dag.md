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

## CUDA kernel-cache identity 修复（2026-10-02）

`_stateful_expression_kernels.py` 原 key 仅含 node count、outputs count、dtype 与 logical shape；同进程先执行 `x+y` 再执行 `x*y` 时，后者错误复用加法 kernel。修复以 `CompiledExpression` 构造时预计算的 exact immutable DAG identity（node kind、expression、children、roots）区分 authored operators、constants 与输出顺序；descriptor 的 role order、feature extents、target/grouped heads、mode、sequence length、storage offset 一并进入 key。tensor identity/data pointer 不进入 key，strides 仍作为 runtime launch 参数，不改变 arithmetic、dtype rounding 或 public layout 合同。

CUDA 消费者 smoke 实际返回 addition `[4,6]`、multiplication `[3,8]`、division `[0.3333333432674408,0.5]`。永久回归覆盖不同 operators/constants/root order、相同 logical shape 的不同 descriptor 地址公式及同公式的 dynamic strides；expected 由独立 tensor indexing/arithmetic 构造，global cache 隔离。最终实际执行 `--run-gpu` 的 expression/readout/hierarchical GPU suite 76 passed；CPU expression/v2/v3/binding/workspace/selection 47 passed、4 deselected。相关 Ruff 通过；这些验证不代表整个 DAG performance 任务已完成。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
