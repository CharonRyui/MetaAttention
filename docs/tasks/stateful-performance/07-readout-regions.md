# P2-D readout DAG region fusion

## 状态
normalized shared-query readout 已接入编译期 recipe 与主执行路径，并完成 H20 Meta A/B；GSA/shared-readouts 仍只有 private kernels，不在本次切换范围。

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

`fused_normalized_region` 增加 `epsilon=0.0`，要求 rows 相同与 denominator.columns=1，融合两个 contraction、authored epsilon 加法、scalar division 与最终 cast；默认 0 保持裸除法，normalized-linear recipe 传递表达式 `Constant(EPS)` 的值（`EPS=1e-10`），不插入 clamp。identity prepared States 使用 64-token chunk GEMM，将 shared query/key scores、memory carry contraction、key_sum carry contraction 与 epilogue 保留在同一个 output kernel；独立 keys/两个 prior flag 仍合法。diagonal States 保留 resource-bounded replay。`fused_shared_readouts(prepared, queries, *, sequence_length, priors=None, output_dtypes=None, epilogue=None)` 对同一 State 的 2–4 个 contractions 共享一次 replay；返回 outputs tuple，或资源不满足时 `None`。`epilogue` 可为 `add`/`multiply`/`divide`，要求两个 queries，只输出一个 tensor；`None` 保留独立的多个输出与独立 dtype。

令 `a,b,c,d` 为两个 State rows/columns 的 next-power-of-two。当前结构准入为 `max(a,b,c,d)≤256` 且 `a×b+c×d+max(a×b,c×d)+4×(a+b+c+d)+3×b≤16384` live FP32 elements（normalized 最后一项改为 `b`）。额外的 State tile 项计入 live update temporary；shared readouts 的预算为 `2×a×b+4×(a+b)+M×(a+2×b)≤16384`、`M∈[2,4]`。这是保守结构预算，非已测编译器 register/spill 数；GPU 验收还必须记录实际资源使用。

每个成功 fusion call 一次 kernel launch；global workspace 只有最终 output：softmax 为 `B×T×H×second.columns×dtype.itemsize` bytes，normalized 为 first.columns，同 State 的独立 contractions 为各 output 之和、binary epilogue 仅一个 output。另可能有 query 的 FP32/layout conversion；没有 token-sized State、numerator、denominator 或 softmax allocation。已有 carry/summary 的成本归 04，两个 State 的 summary/carry 未合并。identity normalized 使用固定 64-token/32-column GEMM tile，未做 autotuning；diagonal 每 token two-State update 至多约 `4×(R1×D1+R2×D2)` scalar FLOPs。tail masked loads/stores 保持 State 与输出语义。

`tests/functional/test_stateful_readout_regions.py` 使用独立 CPU FP32 token recurrence，覆盖不相等 axes、真实 reduction 方向、四种 prior 组合、identity/row/column/dual、zero/signed/strong gates、tails、B>1、normalized 与多输出 arithmetic，以及超出资源边界后的既有 executor。子代理未运行检查。

### Normalized shared-query 实测（2026-10-02）

基线 `097b7e8` 的 `_stateful_chunked.py` 在独立模块加载，前后仅替换 recipe/executor，使用同一 operator、同一输入、同一 public validation/target；编译与 warmup 不计时。H20 GPU 6，B=1/2，H=1/8/16，T=63/64/65/130/2048/8192，output-only/final-state/continuation，共 108 点；3 次 warmup，20 次 repetitions，public 交错测量，Graph、CUPTI activity、单调用 allocator peak。peak 在 Graph capture 前采样、删除返回值，避免 graph pool 混入。矩阵前后均使用 `Target(workspace_bytes=4<<30)` 覆盖默认 1 GiB 保守 estimate 拒绝的高 B/H 长序列；生产默认和 estimate 未变。最大实测 peak 613 MiB，远低于空闲卡安全余量。原始记录 `benchmark_results/normalized_readout_ab.json`。

| contract | 点数 | public 几何平均下降 | Graph 几何平均下降 | peak allocated 几何平均下降 | peak incremental 几何平均下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| output-only | 36 | 23.80% | 24.61% | 19.64% | 25.27% |
| final-state | 36 | 22.70% | 24.54% | 18.64% | 23.51% |
| continuation | 36 | 22.70% | 25.39% | 16.55% | 22.42% |

全矩阵 public 几何平均下降 23.07%，Graph 下降 24.85%，peak allocated 下降 18.28%；108 点 Graph/peak 均未增加。单调用 activity 减少 5（16→11、19→14、20→15、23→18），continuation 减少 10（33→23、39→29，包含 concat）。两点 public 单轮回退：B2/H8/T8192 continuation 1.561→1.828 ms，B2/H16/T130 continuation 1.415→1.461 ms；保留记录，不宣称逐点 public 无回退。前后 output parity 通过 `rtol=1e-2, atol=1e-4`，所有返回的 memory/key_sum final States 逐位一致。FLA 不作正确性 oracle 或本次收益分母。

独立 CPU FP32 recurrence 回归覆盖非零 initial、exact-zero denominator + EPS、B2/H2、非方形/非二次幂 17×35 State、BF16/FP32、T=63/64/65/130、output-only、两个 final States、64+1 tail 和 continuation；oracle 保留 authored 输入乘法的 CUDA scalar/BF16 舍入。identity helper 另覆盖独立 keys 与四种 prior 组合；既有 diagonal/softmax 资源回退测试保持通过。GPU normalized focused 15 passed，readout 全套 33 passed，CPU v2/selection 25 passed；使用 `--run-gpu` 实际执行，不将 skip 记作验证。

最终默认 target CLI smoke 使用 `../bench.sh --families normalized-linear --seqlens 64 2048 --batches 1 --heads 1 8 --contract all --timing both --warmup 3 --repetitions 10 --gpu 6 --output-dir benchmark_results/normalized_readout_public_smoke`，12/12 Meta records OK；FLA final-state/continuation 被明确标为 output-only、不具可比 State 合同。64×64 shared-key identity readout 编译资源：206 registers/thread、0 spills、57344 bytes shared memory（4 warps）。

串行 replay 融合探针曾使 H1/T64 Graph 0.049→0.112 ms、H8/T2048 0.184→0.300 ms；已改为并行 chunk GEMM，回退实现不提交。

### 下一个重点：原输入 symbolic binding 与 staging 消除

H1/T64 output-only activity 实测 11：两个 State 各 summary/carry 共 4，融合 readout 1，FP32 copy 3、query multiply 1、fill 2 共 6。public 0.456 ms 对 Graph 0.036 ms，短序列需同时减少 Python tensor bookkeeping 与 staging launches。优先扩展编译 recipe，为 key/value/query 绑定原 B/H/T strides、head mapping 与 authored operation/dtype；kernel 直接 load BF16→FP32，query scale 保留原 scalar 转换与 BF16 rounding，rank-1 单位 value/常量在 kernel 内合成。保持两个 State 独立 summary/carry、public contiguous/grad/packed 合同；不以改默认 workspace 上限或 tile autotuning 代替优化。

验收沿用同一 108 点 A/B、三 contracts、public/Graph/activity/peak，新增 grouped heads 和 authored BF16 舍入回归。先核对 copy/fill activity 消除及 workspace 真实 charge，再以实测选下一调度；summary/carry 跨 State 合并是后续独立优化，不在本步同时实施。



## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
