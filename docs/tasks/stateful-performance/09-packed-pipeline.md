# P3 packed chunk pipeline

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
02,04,05,08

## 文件归属
packed metadata; lowerer segmented kernels; stateful_operator.py integration owner

## 实现合同
建立 device sequence/chunk metadata，segmented prefix、masked tails；dense 与 packed 共享数学 region；无 CPU offset 同步；empty/all-empty；buffer/resource 实际计价。

## 验收
ragged/long-tail/empty/all-empty、grouped heads、final/initial-State、所有 gradients 独立 oracle；相同 compiled artifact 动态 T/offsets。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
