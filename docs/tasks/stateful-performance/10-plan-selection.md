# 计划组合、真实成本与调度选择

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00,01,03,04,05,06,07,08,09

## 文件归属
stateful/compile/plan.py; stateful/compile/tuning.py; stateful/compile/benchmark.py

## 实现合同
按 region 组合数学计划，估计实际 DAG bytes/flops/launch/live workspace；探索有效 chunk/tile/schedule configs；修复 candidate benchmark bindings；动态 B/T 不重新 Mathematical Plan；权衡 public/Graph/memory，不按例子名。

## 验收
多个结构候选均实际执行；选择可解释且 deterministic；无 resource 误价或动态 replanning；真实 H20 smoke，用户合同不变。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
