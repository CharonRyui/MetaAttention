# Stateful Operator 结构性能任务

起点：`c0e1fd7`。任务按 typed IR 结构匹配，不按 FLA 家族 dispatch。

当前文件归属以 `attention_engine/attn_engine/` 为根：`stateful/analyse/`、
`stateful/compile/`、`stateful/runtime/` 按阶段分类；`stateful/lowering/` 内按
expression、diagonal、scalar、lowrank、scan、readout 保持功能 locality。
公开 interface 仍在 `stateful_operator.py`；lowering recipe 与执行不因目录迁移拆开。
下文源码引用使用当前路径；历史提交中的同一实现可能仍位于旧 `_stateful_*` 路径。


- [P0 公平性能评测合同](00-measurement-contract.md)；依赖 无。
- [P1-A 表达式 DAG 与 kernel 融合](01-expression-dag.md)；依赖 00。
- [P1-B 符号化布局与 HeadMapping](02-symbolic-bindings.md)；依赖 01。
- [P1-C scalar/identity 序列并行](03-scalar-parallel.md)；依赖 00。
- [P2-A diagonal block algebra 与分层 carry](04-diagonal-block.md)；依赖 00。
- [P2-B 有序低秩 compact WY](05-lowrank-wy.md)；依赖 00。
- [P2-C ProductFeature 收缩顺序](06-product-contraction.md)；依赖 00。
- [P2-D readout DAG region fusion](07-readout-regions.md)；依赖 01,04,06。
- [P3 checkpointed backward](08-checkpoint-backward.md)；依赖 01,03,04,05。
- [P3 packed chunk pipeline](09-packed-pipeline.md)；依赖 02,04,05,08。
- [计划组合、真实成本与调度选择](10-plan-selection.md)；依赖 00,01,03,04,05,06,07,08,09。
- [Delta-product benchmark 正确性修复计划](11-delta-product-correctness.md)；已修复 ordered-step oracle，原 50 条矩阵 Meta 全部通过。

## 执行与提交
子代理实现互不重叠的文件集合，负责人拥有共享接口与集成、H20 smoke 和原子提交。每个提交必须包含可运行的行为及证据，不能只交 scaffold。性能数据需要区分真实观测和推断。短序列 public latency 和长序列 Graph 均需验收；不强制所有结构使用同一 launch 数。

### ProductFeature readout staging（2026-10-07）

based/rebased 的 ProductFeature readout 对照 Triton `_readout`，将可复用
64×64 Scores 从 shared staging 改为 fragment，减少一次 shared-memory copy。
六项 GPU functional tests 通过；H8/T2048/16×16×64 测得 1.019x，其他
长/短形状持平或轻微回退，未扩大优化适用性。原始 A/B 见
`benchmark_results/stateful_tilelang015_product_readout_followup.json`。
