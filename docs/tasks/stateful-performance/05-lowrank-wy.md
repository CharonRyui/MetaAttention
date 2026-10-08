# P2-B 有序低秩 compact WY

## 状态
scalar scale + ordered rank-one 的 TileLang compact WY 已接入 chunk solve/carry/readout；非交换 row/column-scale 使用 affine checkpoint schedule。2026-10-06 GPU lowrank/checkpoint/packed/GDN 回归 54 passed；历史 Triton 数据与当前 TileLang 证据分列如下。仍不宣称达到 FLA 延迟或完成 spills 优化。

## 依赖
00

## 文件归属
stateful/lowering/lowrank/lowrank_kernels.py; stateful/lowering/lowrank/wy_kernels.py

## 实现合同
对 ordered RankOnePropagation 链推导 compact WY/triangular form；summary、injection、intra-chunk output 使用 block GEMM；支持 left!=right、多 injection/readout；仅对有证明的 scale region 应用；其他结构现有 lowering。

## 验收
独立 FP64/FP32 oracle，顺序敏感两 rank-one、tails、nonzero State、continuation；实际 computation/latency 改善；不逆 gate。

## FP32 factor 直接绑定优化（2026-10-03）

固定基线 `4152413`。eligible WY 在 `stateful.lowering.lowrank.lowrank_kernels.execute` 的 packing 前分流：rank factors、coefficients、injections、queries 以 tensor tuples 与各自 strides 送入同一 triangular solve/readout。广播 head、标量、非连续 Feature/storage offset 都按绑定读取；原 FP32 promotion 保留，避免 BF16 指针改变 Triton dot lowering。操作顺序、chunk 长度、zero/signed gate、FP32 carry/final、output cast、eligibility 均不变。非交换 row/column scales、packed、backward 与保守 workspace 收费不改。

H20 GPU 6，`B={1,2}`、`H={1,8,16}`、`T={63,64,65,130,2048,8192}`、K=V=64、output-only/final-state/continuation，108 点交错 A/B。两侧用 4 GiB cap 比较原 1 GiB 估计拒绝的七点；生产默认 cap 不变。warmup=3、repetitions=20；逐点交替 before/after，同一输入与非零 initial；峰值在 capture 前测量；每次 Graph 使用 fresh pool。原始证据、源码 SHA256、probe 源码、失败、资源与所有复测保存在 `benchmark_results/lowrank_wy_binding_ab.json`。

| contract | 点数 | public CUDA event 下降 | CPU wall 下降 | Graph 下降 | peak incremental 下降 | peak allocated 下降 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| output-only | 36 | 8.97% | 8.59% | 5.35% | 22.83% | 20.42% |
| final-state | 36 | 9.16% | 9.34% | 5.36% | 22.83% | 20.42% |
| continuation | 36 | 9.03% | 9.43% | 5.69% | 20.28% | 15.48% |
| 全部 | 108 | 9.05% | 9.12% | 5.46% | 21.99% | 18.81% |

表为首轮 raw ratio 几何平均。B1/H1/T8192 continuation 首轮 public/CPU 增加 16.62%/18.65%；保留该回退，两轮交错复测后中值 ratio 为 0.95424/0.96777。108/108 逐点三轮中值 gate≤1.05；全矩阵最差 public/CPU/Graph ratio=0.99595/1.00375/0.99784。只主张约 9% public 改善，不主张达到 20% 目标。

所有 108 点 outputs/requested finals 与固定基线逐位一致且通过独立 authored-order FP32 CPU recurrence；initial 在计时/profiling/Graph 后不变。默认 `Target()` 的 108 点两侧均为 101 VALID、7 个相同 `RUNTIME_RESOURCE`；不能把扩大 cap 的 A/B 当作默认范围扩展。private 双 rank、双 injection/readout、BF16/FP32、T1/17/67/2048 的全调用及 continuation 8 场景也与基线逐位一致。新增行为回归覆盖 strided Feature、storage offset、broadcast head、scalar gate 与 T33/34 continuation，容差保持 2e-4。

独立 warmed private T67 活动：FP32 14→3（删除 8 次 DtoD 和 3 次 stack/cat）；BF16 19→15，保留 promotion 和 output conversion。public profiler 有 cached expression 与漏捕 `_solve`，只保留 raw activity，不把不完整计数作为精确 launch gate。public solve/carry/readout 资源分别仍为 255/190–241/255 registers；solve 70 spills、carry 0、readout 236–260 spills；共享内存 solve 65536、carry 16384–98816、readout 16384 bytes。本轮不宣称解决 spills。

CPU selection/measurement/workspace/operator：77 passed，1 expected tuning warning；修改文件 Ruff 通过。GPU `test_lowrank_wy.py`：10 passed、1 pre-existing BF16 T67 oracle-boundary failure。GPU `test_gdn_h20.py`：10 passed、1 unrelated pre-existing `stateful.lowering.lowrank.checkpoint._local_vjp` Triton branch merge failure：`dl` fp32[16] 与 fp32[128] 类型不一致；checkpoint/factored/backward 文件与基线字节相同。不修改这些测试或容差，不宣称全 suite 通过。

拒绝的实验：直接 BF16 指针、load 时转 FP32，T2048 before/after 出现 2/393216 元素超原容差，最大 0.00048828125。回退到原 FP32 promotion 后，8 场景 differential output/final max_abs 全为 0；不保留改变数值的版本。

## TileLang 调度修复（2026-10-06）

上面的 Triton 证据仅为历史记录。当前 RWKV7 为 row-scale + rank-one，
不满足 scalar-only WY eligibility；公共 inference 已恢复原 Triton 的
chunk summary / Tensor Core carry / parallel replay，而非全序列单 block 循环。
同输入、三轮交错 A/B：B1/K64/V64/T2048，H1 公共调用 8.858→0.918 ms
（9.65x），H8 9.671→1.445 ms（6.70x）。增量 allocator peak 约增加 58%，
仍慢于 FLA。原始数据：`benchmark_results/stateful_tilelang015_chunkcarry_ab.json`；
完整合同矩阵与限制见 `docs/tilelang-stateful-operator-research.md`。

scalar-only WY 独立恢复 solve / compact carry / parallel readout，删除通用
dense checkpoint detour。rank/injection event order 保留，zero/signed gate
使用 forward products，不逆 gate；不把 row-scale 混入 scalar 的交换性证明。
TileLang triple-TF32 staging 与 Triton 的寄存器布局不同，chunk 固定 16 tokens，
避免 128-row State 的共享内存超限。

最终 GPU lowrank/WY/GDN/dense checkpoint/packed checkpoint：54 passed。
新增 65-row carry 尾块与 31-row/three-event WY padding 回归，不放宽原容差。
长 scalar-only private recurrence 的独立 FP64 output/final、zero/signed gates
和 Graph replay 通过；Graph 0.878 ms，不作为 RWKV7 性能证据。
CPU unit：192 passed、6 skipped；修改代码 Ruff/Python compile 通过。
当前 FP32 factor staging 仍有 stack/copy；历史“直接绑定”不代表此 TileLang 实现。

### 当前源码再接入与资源合同（2026-10-07）

本轮起始源码的 `lowrank_kernels.execute` 仍走 non-WY 全序列串行 kernel，
与上面历史 cutover 描述不一致。现在已实际接入已有 checkpoint
summary/carry/replay，删除旧 serial kernel。短 dense T≤128 使用 16-token
chunks，长 dense 使用 64；recorded/packed/supplied metadata 保持 16-token ABI。
WY eligibility、autograd 与 packed 入口不改。

同输入、五轮交错、BF16 public final-State：RWKV7 B1/K64/V64/T2048，
H1 public 8.846→0.752 ms，H8 9.648→1.169 ms；pressure H8
15.736→1.689 ms。短 RWKV7 H8/T65 Graph 0.313→0.139 ms，
不保留曾退化的 unconditional 64-token schedule。outputs/finals 通过独立
CPU authored-order recurrence；chunk reduction 不宣称逐位等价。

`WorkspacePolicy` 对 factored inference 按保守 16-token chunk 计费，
不扩大 workspace cap；T65 exact-cap/one-byte-under-cap 回归覆盖短 replay。
64-row carry 去除冗余 shared staging，约 1.8–2.0% kernel 收益；
32-row/fused operand split 实验未获稳定收益，不保留。

GPU 回归：H20 GPU 3 上指定 lowrank/WY、checkpoint、packed、GDN、scalar
测试 **81 passed / 150.01 s**。长度 65/130 的 output-only 与 continuation
memory smoke 均通过 oracle 与 CUDA Graph；strided/scalar coefficient/
multiple readout/continuation/Graph/initial immutable 混合 binding smoke
均通过（rows/columns/length = 35/67/130、65/35/65、64/128/130）。
增量 peak allocation 69,550,080→71,714,816 bytes；该变化已记录，不宣称
内存下降。
完整数据与限制见 `docs/tilelang-stateful-operator-research.md` 和
`benchmark_results/stateful_tilelang015_triton_guided_followup.json`。




## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
