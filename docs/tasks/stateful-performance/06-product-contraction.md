# P2-C ProductFeature 收缩顺序

已实施 typed native normalized ProductFeature 路径。H20 GPU 5 显式执行
`pytest --run-gpu tests/functional/test_product_contraction.py`：**12 passed / 1085.66 s**。
覆盖下面的 public/internal consumer 合同；历史 468 条记录不替代新的性能矩阵。

## 依赖
00

## 文件归属
stateful/lowering/diagonal/diagonal_kernels.py product regions; new private product kernels

## 实现合同
从 typed Outer/Contract 证明 (a⊗b)·(c⊗d)=(a·c)(b·d)；chunk-local scores 使用基础 feature GEMM；inter-chunk 保持显式 State；共享 numerator/denominator；默认保持线性 T 复杂度。

实现：`readout/product_native.py` 证明 identity 更新、匹配的 numerator/denominator query、typed Outer 的 left/right 和行优先 ProductFeature order；支持零阶/一阶/二阶 StateTerm 加和、常数系数与 epsilon，并保留 authored readout 树。`product_native_kernels.py` 复用 DenseExpression / RawLoaderContext，在 summary 与 readout 的 native load 中执行 authored 运算与 BF16 节点舍入，避免输入转换和表达式物化。chunk-local 分数继续使用两个基础 Feature GEMM 的乘积；value tile 为 16，分母使用额外 lane。GEMM fragment 完成 carry 和 local 两部分后复制到 CTA shared tile；最终表达式从 shared 读取 numerator 与分母，避免固定分母列与输出列在 fragment layout inference 中冲突。FP32 除法与 output dtype conversion 保留在同一 readout launch 中。public 两个/六个 State 均保留；多 chunk 使用 existing paired hierarchical identity scan，单 chunk output-only 直接读取非零 initial，不生成 summary/carry。

workspace API：`native_product_resources(recipe, sizes, heads)` 返回 `(bytes/token, bytes/chunk/sequence)`，按 `B*T` 和 `B*(ceil(T/64)+1)` 收费；token 只含最终输出，chunk 保守保留三份完整 public State，包括保留的 summaries、long carries、recursive scan 与 final。scan 后释放 summary 引用。没有完整 ProductFeature token buffer，也没有 T×T buffer；默认复杂度保持线性 T。kernel cache hit 只重绑指针与 metadata，不重新生成 TileLang 源码。

planner cost API：`native_product_estimate(recipe, sizes, heads)` 复用 `NativeDiagonalResources`，不计不存在的 expanded token traffic / score buffer；launch 数是 direct scan 的代表值（summary + 每组 paired scan + fused readout），单 chunk output-only 为一个 launch，递归 scan 按层增加。保留 `recipe=None` direct token-major executor ABI，只有 public 已绑定 recipe 使用 native authored-layout 路径。

consumer 回归：public Based/ReBased、FP32/BF16、31/131 tail、256×64 product State、grouped Q/K/V heads、nonzero initial、output-only、完整 final 和 continuation；direct native executor 另测 strided feature storage，public 非连续输入仍按既有合同拒绝为 RUNTIME_LAYOUT。独立 left/right、反向 ProductFeature order、prior/post 和 FP64 oracle 均通过；FP32 既有 `atol=2e-5, rtol=2e-4` 未放宽。reference Outer reshape 也保留全部非 Feature 轴，不再丢掉 dense head 轴。
性能仍须从新 runner 结果报告，不能用 functional 通过推导 FLA speedup。

## 验收
非同一 left/right、ProductFeature order、Based/ReBased、initial/final-State、continuation 正确；减少 expanded-axis 计算，不引入 quadratic T 默认。

## 原子提交
本任务的有效代码、必要行为回归测试和观测证据一起提交。性能回退或错误实现不提交；明确记录失败，不缩小合同来宣称完成。共享文件由集成负责人修改；子代理不提交，不运行中途 build/lint/tests/formatters。GPU 验证由负责人检查空闲 GPU 与 workspace 风险后串行执行。

## 正确性规则
Stateful Operator 使用独立 token recurrence / FP64 autograd oracle；保持 authored propagation order、floating-point contract、prior/post-State、initial/final-State、continuation 与 HeadMapping。禁止 model-name dispatch、inverse recurrence、放宽已有容差掩盖错误。
