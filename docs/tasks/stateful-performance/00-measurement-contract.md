# P0 公平性能评测合同

runner 已补充 manifest provenance：git commit、测量源码 SHA256、Torch/CUDA、TileLang 与 FLA distribution 版本。历史 `benchmark_results/stateful_fla_full/` 的矩阵完整（468/468），但不因记录完整而把无效 Meta/FLA/pair 记录计入性能结论。

2026-10-09 runner smoke：`benchmark_results/stateful_repair_runner_smoke/`，24/24 records，10 valid pairs。CSV 对 FLA failed、not-comparable 记录的 latency/speedup 全部留空；manifest 的 `valid_pairs` 与 eligibility rule 一致。HGRN grouped-head output/final/continuation records 均可比较；retention adapter 在本环境仍全部 `FLA_FAILED`，不能据此宣称 retention speedup。

## 依赖
无

## 文件归属
testing/benchmark_stateful_plans.py

统一 output-only、final-State、continuation 合同；区分 CPU wall、CUDA-event public call、Graph 与 workspace；记录实际 peak memory 与总 launch 数。

绘图仅使用同时具备 Meta latency、FLA latency 和 speedup 的记录；speedup 矩阵采用以 1× 为中心的对数色阶。
工作区异常分类为 `META_RUNTIME_REJECTED`，不再误标为编译拒绝；Meta、FLA、pair correctness 必须分别通过才算有效对比。

验收：小规模 H20 CLI 完整输出；baseline 错误不影响 Meta oracle 检查；计时外编译；无 GPU 时明确失败。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
