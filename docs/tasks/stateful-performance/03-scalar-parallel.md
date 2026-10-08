# P1-C scalar/identity 序列并行

## 状态
原生 rank-1 diagonal/identity-readout 路径与 direct chunked runtime 已实现并通过 H20 验证。Rank-2 scalar-factorized raw 路径已移植原 Triton 的 summary/exclusive-carry/readout 分解至 TileLang；本轮未运行 GPU 验证，不声称性能提升。

## 依赖
00

## 文件归属
stateful/lowering/scalar/scalar.py; private scalar chunk kernels; planner integration by owner

## 实现合同
实现原输入加载的 sequence-parallel summary/prefix/readout；保留短序列高 B*H fused streaming；长序列低 B*H 使用 chunk grid；支持 tails 与 nonzero State；候选选择不用模型名。

## 验收
T=63/64/65/130/2048，B=1/2，FP32 final-State 与 continuation oracle；public/Graph 对比，短序列不退化。

### Rank-1 实测验收

`stateful/lowering/scan/rank1.py` 按 typed IR 匹配单 rank-1 State、一个原输入 vector injection、可选 vector gate/Exp gate、一个 post-State identity readout。原输入 symbolic binding 保留 HeadMapping 和 strides；T≤64 单 kernel，T>64 三 kernel：64-token vector affine summary、exclusive carry、replay/output。组合不使用逆递推；FMA fusion 关闭。输出 dtype cast 与可选 FP32 final-State 写出在 replay kernel 内完成。

`stateful_operator.py` 的 `_execute_chunked_stateful` 使用 `stateful/compile/plan.py` 编译期保存的 recipe，跳过 reference segment/normalization plumbing；其他 chunked State 使用预绑定布局。reference executor 不再尝试 chunked inference。public validation、contiguous 输入合同、packed/grad fallback、dense 空序列拒绝保持不变。

H20 GPU 3：B=1/2，H=1/8/16，T=63/64/65/130/2048/8192，output-only/final-state/continuation，共 108 records。三次 warmup、五次 repetitions、e2e 与 Graph。106 个可配对 workload 无 public latency 回退；各 contract public latency 几何平均分别下降 82.5%、82.2%、80.2%。

独立 CPU FP32 recurrence 回归覆盖边界长度、非 2 次幂 feature、head sharing、BF16/FP32、identity/log/signed/zero gates、非零 initial State、不修改 initial、output-only、final-State、不同长度 continuation；focused suite 通过。外部库不作为独立 recurrence oracle。

### Rank-2 scalar-factorized TileLang schedule

`tilelang_scalar_factorized_dense_raw` 对 T≥128 且 B×H<32 使用三个 kernel；不满足 32-row GEMM 对齐的 State 也使用 padded chunk 路径（包括短 continuation）。其余形状保留 resident GEMM schedule，dispatch 只取当前调用的形状。Raw tensor strides、grouped heads、FP32 State、optional final-State 与 output dtype ABI 不变。

1. Summary grid `(B×H, ceil(T/64), ceil(rows/32)×ceil(columns/32))`：每个 CTA 计算 32×32 FP32 chunk bias；gate suffix 通过六级 parallel product scan 得到，支持 signed/zero/log-underflow gates，不使用 inverse 或 prefix ratio。
2. Exclusive-carry grid `(B×H, ceil(rows×columns/256))`：每个 CTA 持有 256 个 FP32 State 元素，只在 chunk 维度串行，先读取 bias 再原地覆盖为 prior-chunk State，并按请求写 final-State。
3. Readout grid `(B×H, ceil(T/64), ceil(columns/32))`：CTA 按 32-row tiles 累加 query×carry 和 query×key；64×64 causal scores 的 gate 权重通过六级 parallel reverse product scan 形成，再计算 scores×value。无 Python per-token launch。

GEMM 复用 `diagonal.identity.fp32_gemm_macro` 的三 TensorFloat 产品和 residual，禁止 single-TF32 rounding；编译使用 `-fmad=false`。Tail token 的 gate 为 1，key/query/value 与 row/column padding 为 0。Scratch 为一个 FP32 `(B,H,ceil(T/64),rows,columns)` buffer 与 gated 路径每 chunk 一个 scalar；bias 与 exclusive carry 共用 storage，identity 不分配 scale tensor。代价是三次 launch、chunk-State global bandwidth 与每个 column tile 重算 query/key scores；收益是 summary/readout 的 sequence parallelism 与固定 CTA 资源，不再每线程串行遍历 State rows。短序列/高 B×H 保留原单 kernel 路径。

Raw 35×67 State、head sharing、signed/zero/log gates、FP32/BF16、final/output-only/65-token continuation 的行为回归位于 `tests/functional/test_scalar_parallel.py`；mock-only launcher/dispatch tests 不保留。集成后的 GPU 和性能证据见 `docs/tilelang-stateful-operator-research.md`。


## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
