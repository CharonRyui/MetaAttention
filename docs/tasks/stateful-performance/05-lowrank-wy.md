# P2-B 有序低秩 compact WY

## 状态
scalar scale + ordered rank-one 的 compact WY 已接入；本轮完成 FP32 factor 直接绑定、去除 packing/stack。性能与行为证据见下节。完整任务仍不标为完成：原有 BF16 oracle 边界失败和 WY solve/readout spills 尚在，不放宽容差。

## 依赖
00

## 文件归属
_stateful_lowrank_kernels.py; _stateful_wy_kernels.py

## 实现合同
对 ordered RankOnePropagation 链推导 compact WY/triangular form；summary、injection、intra-chunk output 使用 block GEMM；支持 left!=right、多 injection/readout；仅对有证明的 scale region 应用；其他结构现有 lowering。

## 验收
独立 FP64/FP32 oracle，顺序敏感两 rank-one、tails、nonzero State、continuation；实际 computation/latency 改善；不逆 gate。

## FP32 factor 直接绑定优化（2026-10-03）

固定基线 `4152413`。eligible WY 在 `_stateful_lowrank_kernels.execute` 的 packing 前分流：rank factors、coefficients、injections、queries 以 tensor tuples 与各自 strides 送入同一 triangular solve/readout。广播 head、标量、非连续 Feature/storage offset 都按绑定读取；原 FP32 promotion 保留，避免 BF16 指针改变 Triton dot lowering。操作顺序、chunk 长度、zero/signed gate、FP32 carry/final、output cast、eligibility 均不变。非交换 row/column scales、packed、backward 与保守 workspace 收费不改。

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

CPU selection/measurement/workspace/operator：77 passed，1 expected tuning warning；修改文件 Ruff 通过。GPU `test_lowrank_wy.py`：10 passed、1 pre-existing BF16 T67 oracle-boundary failure。GPU `test_gdn_h20.py`：10 passed、1 unrelated pre-existing `_stateful_checkpoint._local_vjp` Triton branch merge failure：`dl` fp32[16] 与 fp32[128] 类型不一致；checkpoint/factored/backward 文件与基线字节相同。不修改这些测试或容差，不宣称全 suite 通过。

拒绝的实验：直接 BF16 指针、load 时转 FP32，T2048 before/after 出现 2/393216 元素超原容差，最大 0.00048828125。回退到原 FP32 promotion 后，8 场景 differential output/final max_abs 全为 0；不保留改变数值的版本。


## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
FLA 仅性能基线。Meta 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
