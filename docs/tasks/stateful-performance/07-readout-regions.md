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

### 原输入 symbolic binding 与 staging 消除实测（2026-10-02）

`_stateful_native.py` 从 typed IR 证明两个 identity States、原输入 key/value 和共享 Input 或 scalar-scaled Input query；编译 recipe 在资源允许时启用，其他结构保留通用路径。summary/readout 直接按 BindingDescriptor 的 B/H/T/Feature strides 与 grouped HeadMapping load 原 BF16/FP32，不生成 FP32 copy、query multiply 或常量 fill tensor。query scale 保留 CUDA scalar 转换及 authored BF16 rounding；单位 denominator value、epsilon 在 kernel 内合成。memory/key_sum 仍各自 summary/carry，final States 与 continuation 不变；未做跨 State 合并或 tile autotuning。

基线 `895d0ea` recipe、binding、executor、summary/readout helpers 独立加载，使用同一 operator、输入、public validation 和 target。H20 GPU 6，B=1/2，H=1/8/16，T=63/64/65/130/2048/8192，三个 contracts 共 108 点，warmup=3，repetitions=20；public 交错采样，peak 在 Graph capture 前测量。两侧均用 4 GiB cap 以排除基线 estimate 拒绝，生产默认仍为 1 GiB。原始记录 `benchmark_results/normalized_native_ab.json`。

| contract | 点数 | public 几何平均下降 | Graph 几何平均下降 | peak allocated 几何平均下降 | peak incremental 几何平均下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| output-only | 36 | 35.65% | 39.24% | 40.26% | 55.74% |
| final-state | 36 | 33.01% | 39.03% | 37.99% | 51.02% |
| continuation | 36 | 35.48% | 41.85% | 41.84% | 62.32% |

全矩阵 public 下降 34.72%，Graph 下降 40.05%，peak allocated 下降 40.05%，peak incremental 下降 56.61%；108 点四项均无回退，最大实测 peak 273.52 MiB。单调用 activity 11/14→5，T8192 递归 carry 15/18→9，continuation 23/29→11。前后全部 output 与所有返回的 memory/key_sum 均逐位一致，包括 continuation；FLA 不作本次正确性 oracle 或收益分母。

未约束 source-bitwidth 的 BF16 raw-load 探针使 Triton dot operand `kWidth=1→2`，改变 summary 累加顺序；B1/H8/T8192 continuation 一个 State 元素越过既有容差。纯 `mov.b32` 不阻止 compiler 回溯，已删除；最终 FP32 seam 使用非 pure bitwise move，恢复 `kWidth=1` 和逐位 parity，不新增 global allocation 或 kernel launch。此处编译器约束必须保留，不能因 load 已转 FP32 而删除。

workspace 按实际 output 与独立 summary/carry、递归 group buffers 收费：`4*H*V` bytes/token，`12*H*K*(V+1)` bytes/chunk/sequence，额外 boundary 覆盖 final States。默认 target 的 B2/H16/T8192 估计 273178624 bytes，实测单调用 incremental output-only 140509184、final-state 141033472、continuation 84942848 bytes；不提高默认上限。

独立 FP32 recurrence 回归保留 T=63/64/65/130、17×35 State、非零 initial、exact-zero denominator + EPS、output-only/final-state/continuation，扩展 B/T/H 原布局与 query/key/value 2/1/2→4 grouped heads；BF16 输入无损提升为 FP32 后的 final States 逐位不变式锁定累加 seam。GPU normalized 17 passed，readout/rank-1 全套 57 passed，CPU selection/v2/binding/workspace 34 passed，Ruff 通过。额外 public smoke 验证 independent keys、四种 prior/post 组合、FP32/BF16、65+6 continuation，8/8 通过。B2/H16/T8192 continuation 独立 FP32 recurrence 的 memory 最大误差 0.001007080078125、key_sum 0，满足既有长序列 benchmark State 容差（rtol=0.01, atol=0.02）。两个不同 grouping 的长 scan 不要求短序列 oracle 的绝对容差；本次前后同合同 A/B 全部逐位一致。

最终默认 target CLI：`../bench.sh --families normalized-linear --seqlens 64 2048 8192 --batches 2 --heads 16 --contract all --timing both --warmup 3 --repetitions 10 --gpu 6 --output-dir benchmark_results/normalized_native_public_smoke`，9/9 Meta records OK，覆盖最长默认 envelope。FLA final-state/continuation 仍被明确标为 output-only、不具可比 State 合同。64×64 native BF16 summary/readout 资源分别为 125/170 registers/thread、0 spills、32768/57344 bytes shared memory，均 4 warps。

### 下一个重点：双 State summary/carry 调度

短序列 staging activity 已移除，输出 kernel 已融合 contraction、EPS、divide、cast。下一步独立评估两个 State 的 summary/carry 合并：先减少 4 个 summary/carry launches 和 Python tensor bookkeeping，再比较长序列带宽、register/shared-memory 与递归 carry 成本。保留两个 named final States、独立初值和 continuation；不把跨 State 合并与 tile autotuning 同时引入。仍沿用完整 108 点 A/B 和独立 FP32 oracle，不以单点 Graph 收益替代 public 验收。



## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
