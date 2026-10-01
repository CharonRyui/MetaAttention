# P2-D readout DAG region fusion

## 状态
已实现 resource-bounded private fusion kernels 与独立 FP32 oracle；GPU correctness/resource/performance 验收由集成负责人执行，未测量前不宣称完成。

## 依赖
01,04,06

## 文件归属
_stateful_chunked.py; private fused readout kernels

## 实现合同
按 typed DAG fusion region；独立 contractions 共享 State replay；arithmetic epilogue fusion；StateTerm→Softmax→StateTerm 依赖保持方向和 reduction；资源成本约束，避免 all-in-one 寄存器爆炸。

## 验收
GSA、normalized、prior readouts、多个输出独立 oracle；launch/peak memory 与 latency 改善，非法 fusion 不改变 semantics。

## Kernel interface 与资源边界
`_stateful_readout_regions.py` 提供 `fused_readout_region(first, second, query, *, sequence_length, prior_first=False, prior_second=False, output_dtype=torch.float32)`，其中 prepared State 为 `(carry, factors, identity)`。要求 `first.columns == second.rows`；一 CTA 持有两个 FP32 State，执行 first row contraction→first columns softmax→second row contraction，两个 prior flag 分别选择更新前 State。softmax reduction 不沿 token 或 second columns；没有 global logits/softmax tensors。无法合法融合时返回 `None`，调用者执行现有 DAG materialization cut。

`fused_normalized_region` 同签名，要求 rows 相同与 denominator.columns=1，融合两个 contraction 与 scalar division；不添加 epsilon/clamp。`fused_shared_readouts(prepared, queries, *, sequence_length, priors=None, output_dtypes=None, epilogue=None)` 对同一 State 的 2–4 个 contractions 共享一次 replay；返回 outputs tuple，或资源不满足时 `None`。`epilogue` 可为 `add`/`multiply`/`divide`，要求两个 queries，只输出一个 tensor；`None` 保留独立的多个输出与独立 dtype。所有 cast 在 FP32 contraction/arithmetic 之后。

令 `a,b,c,d` 为两个 State rows/columns 的 next-power-of-two。当前结构准入为 `max(a,b,c,d)≤256` 且 `a×b+c×d+max(a×b,c×d)+4×(a+b+c+d)+3×b≤16384` live FP32 elements（normalized 最后一项改为 `b`）。额外的 State tile 项计入 live update temporary；shared readouts 的预算为 `2×a×b+4×(a+b)+M×(a+2×b)≤16384`、`M∈[2,4]`。这是保守结构预算，非已测编译器 register/spill 数；GPU 验收还必须记录实际资源使用。

每个成功 fusion call 一次 kernel launch；global workspace 只有最终 output：softmax 为 `B×T×H×second.columns×dtype.itemsize` bytes，normalized 为 first.columns，同 State 的独立 contractions 为各 output 之和、binary epilogue 仅一个 output。另可能有 query 的 FP32/layout conversion；没有 token-sized State 或 softmax allocation。已有 carry/summary 的成本归 04。每 token two-State update 至多约 `4×(R1×D1+R2×D2)` scalar FLOPs；softmax 的两次 contraction 约 `2×(R1×D1+R2×D2)`，另有 `D1` exp/reduction/division。shared replay 每 token 约 `4×R×D+2×M×R×D`，另加选择的逐元素 epilogue。tail CTA 用单位 gates/零 injection 不改变 State。尚无 latency/peak-memory 改善的实测声明。

`tests/functional/test_stateful_readout_regions.py` 使用独立 CPU FP32 token recurrence，覆盖不相等 axes、真实 reduction 方向、四种 prior 组合、identity/row/column/dual、zero/signed/strong gates、tails、B>1、normalized 与多输出 arithmetic，以及超出资源边界后的既有 executor。子代理未运行检查。


## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
