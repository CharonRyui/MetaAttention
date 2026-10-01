# P2-B 有序低秩 compact WY

## 状态
待实施；只有通过下述验收才能改为完成。

## 依赖
00

## 文件归属
_stateful_lowrank_kernels.py; private WY kernels

## 实现合同
对 ordered RankOnePropagation 链推导 compact WY/triangular form；summary、injection、intra-chunk output 使用 block GEMM；支持 left!=right、多 injection/readout；仅对有证明的 scale region 应用；其他结构现有 lowering。

## 验收
独立 FP64/FP32 oracle，顺序敏感两 rank-one、tails、nonzero State、continuation；实际 computation/latency 改善；不逆 gate。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
