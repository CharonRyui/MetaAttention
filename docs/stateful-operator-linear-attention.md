# Stateful Operator DSL：Linear Attention 用法、能力边界与性能

本文说明当前分支相对 `main` 新增的 Stateful Operator DSL，以及它如何表达 Linear Attention 类线性递推。文中的“当前实现”指 `main...HEAD` 的源码与已提交的 H20 测量记录；性能数字不是跨硬件或跨软件栈的承诺。

## 1. 结论先行

- `main` 的线性注意力入口是 modifier-based `LinearAttentionEngine`：用户提供 Q/K/V meta、若干 modifier 和 `custom_io`，构造时直接生成并加载 TileLang Python callable。
- 当前分支新增了一条 IR-first 路径：用户用 typed Algorithm IR 描述输入、State Transition 和 Readout，再通过 `StatefulOperator.compile(...)` 生成一个可复用的 metadata-specialized callable。
- Stateful Operator **不替换** `LinearAttentionEngine` 的现有调用路径。后者仍导出，但已标记 deprecated，且没有路由到 Stateful compiler；两者的兼容性和性能保证相互独立。
- DSL 的核心能力是把“每个 token 的固定大小 State 递推”表达为可并行 scan 的代数程序。常见的 Linear Attention、GLA、GDN、RetNet、Mamba2-style selective State、RWKV、normalized linear attention、Based/ReBased、GSA/HGRN 等已经有独立示例工厂。
- 当前生产 Target 是 NVIDIA H20 CUDA。输入和输出支持 BF16/FP32，State 在执行中保持 FP32；生产用例要求连续布局和显式 role/head/feature 形状。
- 性能取决于 typed structure、Feature size、Batch/Sequence、是否 packed、是否训练、是否请求 final State，以及编译器选中的 lowering。不能用单个数字代表所有 DSL 程序。
- 已有 H20 记录显示：结构化 chunked/factored lowering 相对同一 IR 的 generic lowering 可有约 `2.6x–22x` CUDA-event 优势，并显著降低临时分配；与 stock FLA 的比较则是混合结果，较长序列上 FLA 仍常常更快。

## 2. 与 `main` 的差异

### 2.0 对比矩阵

| 维度 | `main` | 当前分支 |
|---|---|---|
| 编程抽象 | modifier/callback 驱动的线性 attention lowering | typed Algorithm IR：Input、State、Transition、Injection、Readout |
| 编译时输入 | `qkv_meta` 和 modifier；构造时生成 callable | Head/Feature `role_sizes`、Target、`mode`、优化/反向选项；不接 runtime Tensor |
| 运行时返回 | 旧接口直接返回 attention output（具体形状由旧 engine 约定） | 固定 `ExecutionResult`，包含 immutable named outputs 和可选 named final State |
| State 语义 | 由旧 linear lowering/template 隐含 | caller 显式传入 `initial_state`，按需请求 `return_final_state`，支持 continuation |
| 变长执行 | 旧 `LinearAttentionEngine` 接口没有 Stateful packed contract | `mode="packed"` + device-local contiguous int32 offsets，允许 empty/all-empty sequence |
| Backward | 由旧 generated linear callable 的机制决定 | 由 compiler 从 Algorithm IR 推导 Backward IR，并按 compile 选项生成匹配 context/kernel |
| 选择与 fallback | 以旧 lowering/template 为中心 | 按 typed structure 和 shape-aware estimate 选择 candidate；generic dense affine 是语义 fallback |
| 生产 Target | main 的旧 LinearAttentionEngine 不属于 H20 Stateful contract | 当前 Stateful production gate 是 NVIDIA H20；旧 LinearAttentionEngine 仍独立存在 |

因此，当前分支的核心变化不是“给旧 LinearAttentionEngine 加几个参数”，而是新增一个更深的 IR/compiler module：caller 只描述数学递推，validation、scan analysis、VJP、plan、kernel configuration、cache 和 execution 由 Stateful Operator 持有。

### 2.1 `main` 的旧路径

`main` 只有以下两类公开入口：

1. `AttentionEngine`：以 score/mask/online function 为中心，适合 softmax、retention 等按 attention score 定义的算子。
2. `LinearAttentionEngine`：
   - 构造参数包含 `qkv_meta`、`q_mod`、`k_mod`、`v_mod`、`decay_mod`、`custom_io`、tuning 选项；
   - 内部调用 `core.lower.lower_linear.lower_tl(...)` 生成源码；
   - 将生成的 `linear_attention` callable 动态加载后转发调用。

对应基线源码是 `main:attention_engine/attn_engine/linear_attn_engine.py` 和 `main:attention_engine/attn_engine/__init__.py`。`main` 的 examples 主要是 `gated_retention.py`、`mamba2.py`、`retention_parallel.py`，没有当前分支的 Stateful IR 和 FLA coverage map。

### 2.2 当前分支新增的路径

当前分支新增：

- `attention_engine/attn_engine/stateful_operator.py`：公开 IR 类型、编译入口、运行时校验、`ExecutionResult` 和结构化错误；
- `stateful/analyse/algorithm.py`：类型推导、scanability/结构分析、canonical identity、compiler-derived Backward IR；
- `stateful/compile/plan.py`：Mathematical Plan、Lowered Candidate、KernelConfig、workspace policy 和候选选择；
- `stateful/runtime/`：invocation 校验和逻辑角色到物理张量的 bindings；
- `stateful/lowering/`：按 expression、diagonal、scalar、lowrank、scan、readout 功能收拢 implementation；chunked、packed、reference、structured 和 backward 编排留在该目录；
- `examples/*.py`：每个 FLA 线性递推 profile 的独立 Algorithm IR；
- `examples/fla_coverage.py`：FLA operator 到 DSL factory 的 coverage map；
- `testing/benchmark_stateful_plans.py` 和 `testing/benchmark_linear_attention.py`：统一的正确性与性能测量入口；
- `docs/stateful-operator-ir-spec.md`：完整 DSL/compiler specification。

当前分支还把 `LinearAttentionEngine` 改为 deprecated adapter，源码位置为 `attention_engine/attn_engine/linear_attn_engine.py`。因此迁移方向是“新程序优先使用 Stateful Operator”，不是把旧接口悄悄改成同一实现。

### 2.3 Stateful TileLang 迁移范围

依赖已从旧 git snapshot 更新到发布版 `tilelang==0.1.15`。Identity 的 chunk summary、exclusive hierarchical carry helper，以及 native normalized identity dense inference 的 summary/carry/readout 已改为 TileLang；general affine chunk carry 也已改为 TileLang。保留 FP32 State、BF16 authored query rounding、独立 denominator key/prior、grouped heads 和 BHT/BTH 输入布局。

Affine carry 使用 direct（不超过 64 chunks）和 32-chunk recursive hierarchy，支持 zero/negative scale、tail chunk、continuation 和 final-State ownership；weighted general-diagonal summary/readout 仍由 Triton 提供。

这不是全后端切换：general diagonal/native single-State 的 summary/readout、ProductFeature/softmax/shared readout、optimized scalar、rank-one scan、lowrank 和 generated expression 仍有 Triton kernels；packed/training 没有迁移。详细来源、边界与实测验证见 [TileLang 迁移调研](tilelang-stateful-operator-research.md)。

## 3. 心智模型：用一个固定大小 State 表示线性递推

一个最基本的矩阵 State 程序是：

\[
S_t = P_t(S_{t-1}) + \sum_j I_{t,j},
\qquad
Y_t = R_t(S_t).
\]

其中：

- `S` 是每个 batch/逻辑序列/head 独立维护的 State；
- `P_t` 是有序 Propagation，如 scalar/axis scale 或 rank-one update；
- `I_t` 是一个或多个 Product Injection；
- `R_t` 是 post-transition Readout，除非显式声明 `StateTerm(..., prior=True)`；
- State 的逻辑大小依赖 Head/Feature role，不依赖 Sequence length。

Linear Attention 的常见形式：

\[
S_t = G_t(S_{t-1}) + k_t v_t^T,
\qquad
o_t = q_t S_t / \sqrt{d_k}.
\]

在 DSL 中，`k_t` 和 `v_t` 不是特殊的 “key/value operator”；它们只是分别携带 State 的两个 Feature role 的 Product Injection factors。Gate 是 `AxisScale`，输出是 `StateContraction`。

程序首先在 target-independent IR 上做类型分析和代数分析。对于可接受的 input-driven Transition，编译器可以构造固定大小的 affine summary：

\[
\operatorname{summary}_t(S) = M_t S + b_t.
\]

随后通过 segmented parallel scan、chunk boundary carry 和适用的专用 lowering 执行。模型名称、example 名称和完整 profile 名称不参与 lowering dispatch；候选由 IR 结构和固定 role sizes 决定。

## 4. 最小可用例：plain Linear Attention

### 4.1 定义 Algorithm IR

`examples/linear_attention.py` 中的 `linear_attention(...)` 是完整版本。下面是同一结构的关键部分：

```python
import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")

algorithm = AlgorithmIR(
    inputs=(
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), torch.bfloat16),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.bfloat16),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), torch.bfloat16),
    ),
    states=(
        StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
    ),
    transition=StateTransition(
        "memory",
        injections=(
            ProductInjection((
                ProductFactor(Input("key"), (KEY,)),
                ProductFactor(Input("value"), (VALUE,)),
            )),
        ),
    ),
    readouts=(
        StateContraction(
            "output",
            "memory",
            Input("query") * (64 ** -0.5),
            KEY,
            torch.bfloat16,
        ),
    ),
    head_mapping=HeadMapping({
        QUERY_HEADS: STATE_HEADS,
        KEY_HEADS: STATE_HEADS,
        VALUE_HEADS: STATE_HEADS,
        STATE_HEADS: STATE_HEADS,
    }),
)

operator = StatefulOperator(algorithm)
```

这里的语义是：

```text
memory_t = memory_(t-1) + key_t outer value_t
output_t = query_t / sqrt(64) contract memory_t over KEY
```

`ProductFactor(..., (KEY,))` 表示该 factor 落在 State 的第一个 Feature 轴；`VALUE` factor 落在第二个 Feature 轴。DSL 不从广播形状猜测 outer product，Feature role coverage 是显式契约。

### 4.2 Metadata-only compile

当前接口编译时不接收 example Tensor：

```python
from attn_engine import Target

compiled = operator.compile(
    role_sizes={
        QUERY_HEADS: 2,
        KEY_HEADS: 2,
        VALUE_HEADS: 2,
        STATE_HEADS: 2,
        KEY: 64,
        VALUE: 128,
    },
    target=Target(
        # 超过此 envelope 的运行 workload 在执行前显式拒绝。
        workspace_bytes=1 << 30,
    ),
    mode="dense",
    optimized=True,
    autotune=False,
    backward=False,
)
```

`role_sizes` 只绑定 Head/Feature role。Batch、Sequence length、packed sequence count、packed total token count 和 packed offsets values 属于运行时数据。一个 dense handle 可以复用到不同的正 Batch 和 Sequence length；一个 packed handle 可以复用到不同的 offsets values，只要编译身份所需的 metadata 不变。

注意：当前生产 Target 的矩阵 specialized path 对 matrix State 的 Feature pair 支持 `{64, 128} × {64, 128}`。Algorithm IR 本身可描述更一般的正 Feature size，但 concrete H20 specialization 可能以 `SPECIALIZATION` 结构化错误拒绝。ProductFeature/multi-State 等进入 reference 或 chunked path 时，具体约束由对应 lowering 决定。

### 4.3 Dense invocation

示例工厂使用 head-first contiguous layout：

```python
B, H, T, K, V = 2, 2, 257, 64, 128
query = torch.randn(B, H, T, K, device="cuda", dtype=torch.bfloat16)
key = torch.randn(B, H, T, K, device="cuda", dtype=torch.bfloat16)
value = torch.randn(B, H, T, V, device="cuda", dtype=torch.bfloat16)

result = compiled(query=query, key=key, value=value)
output = result.outputs["output"]
# output.shape == (B, H, T, V)
# result.final_state is None: final State 未请求，也不会 materialize。
```

请求 continuation State：

```python
result = compiled(
    query=query,
    key=key,
    value=value,
    return_final_state=True,
)

output = result.outputs["output"]
final_memory = result.final_state["memory"]
# final_memory.shape == (B, H, K, V)
# final_memory.dtype == torch.float32
```

省略 `initial_state` 等价于每个逻辑序列一个 FP32 zero State。显式 initial State 必须通过 immutable `StateTuple` 传入：

```python
from attn_engine import StateTuple

initial = torch.randn(
    B, H, K, V, device="cuda", dtype=torch.float32
).contiguous()

result = compiled(
    query=query,
    key=key,
    value=value,
    initial_state=StateTuple(("memory",), (initial,)),
    return_final_state=True,
)
```

prefix/suffix continuation 不需要重新 compile：

```python
split = 128
prefix = compiled(
    query=query[:, :, :split].contiguous(),
    key=key[:, :, :split].contiguous(),
    value=value[:, :, :split].contiguous(),
    initial_state=StateTuple(("memory",), (initial,)),
    return_final_state=True,
)

suffix = compiled(
    query=query[:, :, split:].contiguous(),
    key=key[:, :, split:].contiguous(),
    value=value[:, :, split:].contiguous(),
    initial_state=prefix.final_state,
    return_final_state=True,
)
```

### 4.4 Packed invocation

packed mode 编译一次：

```python
packed = operator.compile(
    role_sizes={
        QUERY_HEADS: 2,
        KEY_HEADS: 2,
        VALUE_HEADS: 2,
        STATE_HEADS: 2,
        KEY: 64,
        VALUE: 128,
    },
    target=Target(),
    mode="packed",
)
```

所有带 `Sequence` role 的 input 使用 token-major layout；以本例为例：

```python
# 三条逻辑序列，长度为 3、0、2；total_tokens == 5。
sequence_offsets = torch.tensor(
    [0, 3, 3, 5], device="cuda", dtype=torch.int32
).contiguous()
query = torch.randn(5, 2, 64, device="cuda", dtype=torch.bfloat16).contiguous()
key = torch.randn(5, 2, 64, device="cuda", dtype=torch.bfloat16).contiguous()
value = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16).contiguous()

result = packed(
    query=query,
    key=key,
    value=value,
    sequence_offsets=sequence_offsets,
    return_final_state=True,
)

# result.outputs["output"].shape == (5, 2, 128)
# result.final_state["memory"].shape == (3, 2, 64, 128)
# 第 2 条空序列的 final State 保持其 initial/zero State。
```

packed offsets 的要求：

- 一维、device-local、contiguous、`torch.int32`；
- `offsets[0] == 0`；
- `offsets[-1] == total_tokens`；
- 单调不降，因此允许 interior empty sequence 和 all-empty workload；
- offsets 的 shape 进入 specialization metadata，offsets 的 values 保持 runtime data，不作为 artifact identity。

### 4.5 Backward

先在 compile 时打开 backward。以下沿用 4.2 中已定义的 `role_sizes` 字典（实际代码中应将上节的 inline mapping 先赋给该变量）：

```python
compiled = operator.compile(
    role_sizes=role_sizes,
    target=Target(),
    mode="dense",
    optimized=True,
    backward=True,
)

query = query.detach().requires_grad_(True)
key = key.detach().requires_grad_(True)
value = value.detach().requires_grad_(True)
result = compiled(query=query, key=key, value=value)
loss = result.outputs["output"].float().square().mean()
loss.backward()
```

`backward=False` 时，grad-enabled input 或 requires-grad initial State 会被 `RUNTIME_GRAD` 拒绝。`backward=True` 时，编译器从 IR 节点组合 VJP，保存与 selected forward family 匹配的 context，并通过 PyTorch autograd 传递 cotangent。

当前边界：

- v1 single-matrix-State 结构有明确的 TileLang backward candidate；
- packed/参考 lowering/尚未有专用 backward schedule 的新组合可能保留 generic/reference execution，不能按 specialized H20 kernel 的性能估计；
- 输出、final State、initial State 和各 differentiable input 的梯度是否被某一具体组合支持，应以对应 functional tests 为准；
- 不要把 `backward=True` 理解为“所有未来 IR 节点自动获得高性能 backward”。`stateful/lowering/backward.py` 当前支持的 canonical VJP operation 包括 `axis_scale`、`rank_one`、`product_injection`、`state_contraction`、`readout`、`head_reduce` 和 `state_recurrence`。

## 5. DSL 构件与表达能力

### 5.1 Logical roles

```python
Batch                  # batch/逻辑序列数
Sequence               # dense sequence 轴；packed 时被 offsets 替代
One                    # 固定 singleton 轴
HeadRole("...")       # 语义化 head 角色
FeatureRole("...")    # 语义化 feature 角色
ProductFeatureRole(a,b)# 展平的 a × b feature 角色
```

`TensorInput` 用 role tuple 声明逻辑 shape 和 dtype。输入名称必须唯一且是 Python identifier；输入 dtype 仅允许 BF16/FP32。`differentiable=False` 可显式禁止该 input 参与 requires-grad invocation。

`StateSpec` 当前接受：

- rank-2 matrix State：`(Batch, HeadRole, FeatureRole, FeatureRole)`，两个 Feature 必须不同且有序；
- rank-1 vector State：`(Batch, HeadRole, FeatureRole)`，用于 denominator、HGRN hidden 等 scalar/vector accumulator。

所有 State 必须共享一个 State Head role，且每个 State 由恰好一个 Transition 驱动。

### 5.2 Input Expressions

当前导出的表达式节点：

- `Input(name)`；
- `Constant(value, dtype=torch.float32|torch.bfloat16)`；
- `Add`、`Multiply`、`Negate`、`Exp`；
- `Divide`、`Log`、`Sigmoid`、`Sqrt`、`Reciprocal`、`Clamp`；
- `Shift(expr, lag)`：同一逻辑序列内向前取 `lag` token，越过序列开头时取 typed zero；
- `Outer(left, right, ProductFeatureRole)`；
- `Contract(left, right, FeatureRole)`；
- `Softmax(operand, FeatureRole)`；
- `StateTerm(state, operand, role, prior=False)`。

表达式是 immutable typed tree，不接受 arbitrary Python callback、control flow、comparison、`where`、general einsum 或用户自定义 VJP。`Exp`、`Log`、`Sigmoid`、`Sqrt`、`Reciprocal` 和 `Divide` 按当前实现以 FP32 计算；State algebra 和 State readout 也以 FP32 为主，Readout 最后 cast 到声明的 output dtype。

`Constant` 在构造时按 declared dtype 量化并记录 canonical bits；NaN/Inf 被拒绝。`Add`/`Multiply` 会按 semantic identity 排序 operand，但不会使用结合律、分配律或浮点重排改变 association。

### 5.3 Transition

```python
StateTransition(
    "memory",
    propagations=(...),
    injections=(...),
)
```

#### Propagation

- `AxisScale(factor, roles=())`：whole-State/scalar scale；
- `AxisScale(factor, (KEY,))`：第一 State Feature 轴的 diagonal action；
- `AxisScale(factor, (VALUE,))`：第二 State Feature 轴的 diagonal action；
- `AxisScale(factor, (KEY, VALUE))`：elementwise State scale；
- `RankOnePropagation(role, left, right, coefficient)`：在指定 Feature 轴做一般 rank-one update；`left` 和 `right` 可以不同；
- `Identity`：显式 identity；Transition 中会安全地删除 identity node。

Propagation 严格按 authored order 执行。非交换的 row/column scale 和 rank-one update 不会被编译器凭直觉重排。

#### Product Injection

一个 `ProductInjection` 至少包含一个 `ProductFactor`：

```python
ProductInjection((
    ProductFactor(Input("key"), (KEY,)),
    ProductFactor(Input("value"), (VALUE,)),
))
```

所有 factor 的 declared roles union 必须覆盖该 State 的 Feature roles。缺失 role 表示沿该轴 broadcast；feature-free factor 可以声明一个 role 但由 scalar/token value broadcast 到该轴。多个 Product Injection 相加形成 additive update。共享 Feature role 的乘法必须在 Input Expression 中显式写出，Injection 不会猜测隐式 contraction/transpose。

### 5.4 Readout

简单 contraction：

```python
StateContraction(
    "output", "memory", Input("query") * scale, KEY, torch.bfloat16
)
```

一般表达式：

```python
Readout(
    "output",
    StateTerm("numerator", query_pair, SECOND_ORDER)
    / (StateTerm("denominator", query_pair, SECOND_ORDER) + Constant(1e-5)),
    torch.bfloat16,
)
```

Readout 支持多个 named outputs。`StateTerm` 默认读取当前 token Transition 之后的 State；`prior=True` 读取 Transition 之前的 State。后者用于 RWKV6 的 bonus/pre-transition readout。rank-1 State 使用 `role=None` 进行 identity readout。

当前 readout 仍然是受限 typed expression，不是 arbitrary tensor contraction：通常是单 Feature contraction、StateTerm 之间的 arithmetic、Feature reduction 和 typed softmax。一般 readout dependency graph、multi-axis State contraction、任意 einsum 不在当前生产契约内。

### 5.5 HeadMapping

Head role 不绑定 `query/key/value` 这些固定名称。通过 `HeadMapping` 明确 source head 到 State head 的映射：

```python
HeadMapping({
    QUERY_HEADS: STATE_HEADS,
    KEY_HEADS: STATE_HEADS,
    VALUE_HEADS: STATE_HEADS,
})
```

当前校验要求 State head 数是 source head 数的整数倍，支持 identity 和 contiguous group broadcast；任意非整除映射会在 compile/runtime 失败。Backward 会做相应 head reduction。

## 6. 已表达的 Linear Attention / FLA profiles

每个 profile 在独立 `examples/<family>.py` 文件中构造 Stateful Operator，`examples/fla_coverage.py` 记录对应关系。代表性程序如下：

| Profile | DSL 结构 | 示例 |
|---|---|---|
| plain linear attention | 一个 `[K,V]` matrix State；`key × value` injection；query contraction | `examples/linear_attention.py:linear_attention` |
| simple GLA / Lightning / forgetting / retention | scalar log gate 的 `Exp + AxisScale` | `examples/linear_attention.py`、`retention_attention.py` |
| feature-wise GLA | KEY-axis AxisScale | `examples/linear_attention.py:gla_attention` |
| dual-axis GLA | KEY/VALUE 两个 AxisScale | `examples/linear_attention.py:dual_gate_gla_attention` |
| delta rule / GDN | AxisScale + `RankOnePropagation` + beta-scaled injection | `examples/linear_attention.py`、`gated_delta_rule.py` |
| generalized delta rule / RWKV7 | `left != right` 的 rank-one/DPLR State algebra | `examples/gated_delta_rule.py`、`rwkv7_state.py` |
| normalized linear attention | matrix numerator State + rank-1 key-sum denominator State + division Readout | `examples/normalized_linear_attention.py` |
| HGRN | rank-1 State + identity Readout | `examples/hgrn.py` |
| RWKV6 | prior-State readout + Feature `Contract` bonus term | `examples/rwkv6_attention.py` |
| GSA / ABC shape | 两个 State + slot-axis `Softmax` readout | `examples/gated_slot_attention.py` |
| Based | 多个 numerator/denominator State + `K×K` ProductFeature + normalized Readout | `examples/based_attention.py` |
| ReBased | `[K×K,V]` numerator + `[K×K]` denominator ProductFeature State | `examples/rebased_attention.py` |
| Mamba2-style selective State | typed Input、FP32 gate/State、AxisScale/affine recurrence | `examples/mamba2.py` |
| RetNet recurrent | generic State transition profile | `examples/retnet_recurrent.py` |

这张表说明“已有程序表达和测试覆盖”，不等于每个 profile 都走同一个专用 kernel。当前实现会根据 typed structure 选择 scalar/vector/diagonal/factored/product-feature/chunked/reference 等 candidate。

## 7. 能力边界

### 7.1 运行时和硬件边界

| 项目 | 当前约束 |
|---|---|
| 生产设备 | CUDA NVIDIA H20；产品识别和 capability `(9,0)` 都参与校验 |
| 拒绝设备 | CPU、ROCm、H100、其他 CUDA 产品、未知同 capability 产品、无法识别的 MIG |
| Input/Output dtype | BF16、FP32 |
| State dtype | FP32；不支持 BF16 State storage 作为生产契约 |
| 布局 | 输入必须 contiguous，并符合声明的 role 顺序；不提供隐式 transpose、device move、dtype conversion 或 padding fallback |
| dense Sequence | 每次 invocation 内所有带 Sequence input 必须使用同一个正长度 |
| packed offsets | device-local contiguous int32、一维、非降、首尾与 token 数一致 |
| role sizes | 编译固定 Head/Feature role；运行时 Batch/Sequence/token count 可变 |
| matrix specialized Feature pair | H20 specialized path 支持 `{64,128} × {64,128}`；超出者可能是 specialization rejection |
| resource | Target 默认 workspace envelope 为 1 GiB；workspace policy 在 launch 前检查 |

单个 `StatefulOperator` instance 内部不保存 mutable recurrent State。State 必须由 caller 通过 `initial_state` 显式传入，并用 `return_final_state=True` 显式取回。

### 7.2 语言边界

当前不支持或不保证：

- Transition/injection 中读取 State 的 State-dependent recurrence；Transition 和 Injection expression 必须由 Inputs/Constants 驱动；
- arbitrary Python transition、injection、readout 或 backward callback；
- general rank-r propagation（公共节点只有 single Rank-One）；
- arbitrary multi-axis State contraction、general einsum、任意 readout DAG；
- TTT/Titans 一类需要 chunk-local inner loop、State 重初始化、momentum/bias State 的程序；
- RWKV4 一类 State 随历史增长、不是固定大小 State 的程序；
- ordinary softmax/sparse attention 作为 Stateful Operator 递推。它们继续属于 `AttentionEngine`；
- `LinearAttentionEngine` 的 modifier-based API 与 Stateful IR 之间的透明互换。

`fla_coverage.py` 中明确标为 `UNSUPPORTED` 的 TTT/Titans，是语义边界，不是“换一个 lowering 就一定可以”。

### 7.3 编译/执行语义边界

- `StatefulOperator` 本身不能直接调用；未编译调用会抛出 runtime error。只有 `operator.compile(...)` 返回的 handle 可执行。
- compile 不接收真实 Tensor；把 `query=...` 等 runtime Tensor 传给 compile 是错误用法。
- 每次 invocation 必须提供且只能提供声明的 input names；缺失/多余 name、dtype、device、shape、layout 和 alias 都会结构化失败。
- 两个 input name 可以绑定同一 Tensor object，Backward 会累加语义贡献；不同 Tensor object 的 overlapping storage 会拒绝。
- final State 不请求时不应依赖其存在，也不应把它当作隐式 cache；实现会避免 materialize 未请求 final State。
- packed empty sequence 保持对应 initial/zero State；输出 token 只对应非空 token，逻辑 sequence order 由 offsets 保持。
- 浮点结果只要求在声明 tolerance 内一致，不承诺跨 chunk、跨优化表示或跨 continuation boundary bitwise identical。
- `optimized=False` 是 generic baseline/诊断路径，不应被理解为生产性能配置。

### 7.4 专用 lowering 的适用条件

当前计划/候选包括：

- `tilelang-generic`：generic dense affine fallback；
- scalar/vector propagation；
- axis-scale / elementwise chunked；
- product-injection GEMM；
- rank-one factorized / factored recurrence；
- shared State contraction；
- ProductFeature chunked；
- generic typed chunked Stateful；
- reference Stateful lowering。

专用路径有结构前提。例如：

- chunked diagonal inference 通常要求 dense、无梯度、一个或多个合法 diagonal State、可识别的 typed readout；
- vector propagation 要求一个 rank-2 State、一个 first-Feature AxisScale、一个 rank-one injection、一个 first-Feature readout；
- factored recurrence 要求有序 AxisScale/RankOne propagation、disjoint single-feature injection、first-Feature contractions；
- native raw binding 要求输入表达式可以直接证明为受支持的 Input/scale/Exp 形态；
- packed、gradient 或不满足结构证明的程序可能回到 reference/generic path。

专用 candidate miss 不应改变合法程序的语义；选择失败时 generic path 是语义 fallback。反过来，workspace、Target、dtype 或实际 kernel resource 不满足时，程序会显式失败，而不是偷偷改成另一种数学语义。

## 8. 性能

### 8.1 测量口径

现有记录主要来自 otherwise-idle NVIDIA H20。除非另有说明：

- 输入 BF16，gate/State 为 FP32；
- public layout 多为 `[B,H,T,features]`；
- `public`/`end-to-end` 包含 Python dispatch、runtime validation、内部 input normalization、kernel launches 和返回 output/State；
- CUDA Graph replay 只表示 graph replay，不包含每次 public call 的 host dispatch；
- compilation time 排除在 latency 外；
- baseline FLA 只作为速度参考，不是 Meta 输出的 correctness oracle；
- 正确性对照使用独立 authored-order FP32 token recurrence。

### 8.2 Profile 级别 H20 结果

来自 `docs/API.md` 已记录的 H20 测量：`B=1,H=8,T=2048,K=V=64`，BF16，5 warmups，20 个 CUDA-event samples，最终 State requested。单位为 ms：

| Structure | 当前/after public | 当前/after Graph | stock FLA Graph | 备注 |
|---|---:|---:|---:|---|
| Based (`K=16`) | 1.931 | 0.816 | 0.468 | FLA 更快 |
| ReBased (`K=16`) | 1.084 | 0.609 | 0.078 | FLA 更快 |
| Normalized linear | 0.874 | 0.203 | 0.068 | public latency 相比历史 before 0.766 有回退 |
| RWKV6 | 0.908 | 0.365 | 0.131 | FLA 更快 |
| GSA (`slots=16`) | 0.988 | 0.312 | 0.142 | FLA 更快 |
| HGRN (`H=1`) | 0.519 | 0.060 | 0.065 | Graph 略优；public 不代表所有 H |
| GDN | 1.074 | 0.604 | 0.190 | FLA 更快 |
| RWKV7 State | 1.031 | 0.601 | 0.351 | FLA 更快 |
| ordered composition pressure | 1.372 | 0.924 | unavailable | 没有 legacy FLA baseline |

同一记录中的历史 `Before` public 数字为：Based 11.384、ReBased 2.814、Normalized linear 0.766、RWKV6 2.796、GSA 2.119、HGRN 0.570、GDN 4.661、RWKV7 4.968、pressure 82.628。它们用于说明当前分支演进，不是 `main` 的统一 benchmark baseline，也不应用来宣称跨版本固定 speedup。

### 8.3 Native diagonal/readout 结果

较新的 `benchmark_results/diagonal_native_final.json` 记录：H20、BF16、`B=1`、output-only、5 warmups、20 samples；public medians 和 Graph medians：

| Structure | H | T | Meta public | Meta Graph | FLA public |
|---|---:|---:|---:|---:|---:|
| RWKV6 | 1 | 64 | 0.263 | 0.064 | 0.303 |
| RWKV6 | 1 | 2048 | 0.355 | 0.084 | 0.312 |
| RWKV6 | 8 | 64 | 0.266 | 0.065 | 0.362 |
| RWKV6 | 8 | 2048 | 0.397 | 0.159 | 0.378 |
| GSA | 1 | 64 | 0.332 | 0.123 | 0.562 |
| GSA | 1 | 2048 | 0.501 | 0.137 | 0.505 |
| GSA | 8 | 64 | 0.333 | 0.130 | 0.568 |
| GSA | 8 | 2048 | 0.637 | 0.325 | 0.576 |
| HGRN | 1 | 64 | 0.119 | 0.014 | 0.113 |
| HGRN | 1 | 2048 | 0.175 | 0.020 | 0.115 |

结论不是“DSL 总是快”：HGRN 长序列和 RWKV6/GSA 的部分 public case 仍由 FLA 获胜；HGRN 的 H>1 记录没有可比 FLA adapter。Profiler 在该最终记录中不可用时，launch/kernel 字段保持 null，没有从缺失数据推断。
### 8.4 同数学 workload：softmax attention 与 FlashAttention-4 CuTe

`testing/benchmark_flash_attention.py` 对同一个 fixed-length softmax attention IR 同时调用 MetaAttention 与 `../flash-attention` 的 `flash_attn.cute.interface.flash_attn_func`；两侧使用相同的 BSHD FP16 输入、同一 H20、同一随机种子，先以 PyTorch `scaled_dot_product_attention` 做 correctness check，再用 CUDA event 计时，编译时间排除。完整记录：`benchmark_results/flash_attention_comparison_full.json`。

截至 `../flash-attention` revision `616b0e8a` 的完整矩阵（warmup=10、samples=30，median）：

| Case | Shape | Meta ms / TFLOP/s | FlashAttention-4 ms / TFLOP/s | Meta/Flash latency | Correctness |
|---|---|---:|---:|---:|---|
| MHA causal | `B=1,H=16,HKV=16,S=512,D=64` | 0.033 / 16.18 | 0.073 / 7.42 | 0.458x | 全部通过 |
| MHA noncausal | `B=1,H=16,HKV=16,S=512,D=64` | 0.037 / 29.11 | 0.073 / 14.69 | 0.505x | 全部通过 |
| GQA causal | `B=1,H=16,HKV=4,S=512,D=64` | 0.035 / 15.30 | 0.073 / 7.42 | 0.485x | 全部通过 |

三组 case 均满足 Meta、Flash 对 PyTorch reference 以及 Meta 对 Flash 的 `rtol=0.03, atol=0.03` 对照。两侧均由 CUPTI CUDA activity capture 到 1 个 kernel launch；warm invocation 的增量 allocator 峰值分别约为 Meta `1.03 MiB`、Flash `1.00 MiB`（GQA 两侧相同量级）。

这里的 TFLOP/s 按 `4 * B * query_heads * attended_pairs * D` 计算，multiply-add 计作 2 FLOPs；它是 attention work estimate，不是硬件峰值利用率。结果表明在这组 H20/shape 上 MetaAttention latency 约为 FlashAttention-4 的 0.46–0.50 倍，但不得外推到其他 shape、dtype、GPU 或 training/backward workload。

该 benchmark 比较的是数学等价的 softmax attention，不是 Stateful linear-attention；后者的 FLA 对照仍见 8.2/8.3。
### 8.5 Stateful DSL 与 FLA 完整矩阵

完整 runner 为 `testing/benchmark_fla_meta_stateful.py`，入口包装为 `../bench.sh`。它沿用 `testing/benchmark_stateful_plans.py` 的 13 个可比 Stateful family：`linear`、`simple-gla`、`gla`、`normalized-linear`、`retention`、`rwkv7`、`rwkv6`、`gsa`、`hgrn`、`based`、`rebased`、`gdn`、`retnet`。`examples/fla_coverage.py` 中标记为 `AttentionEngine`、`UNSUPPORTED` 或没有等价 FLA adapter 的条目不被伪造为 FLA 对照。

本轮完整矩阵命令和输出目录：

```text
../bench.sh --families linear,simple-gla,gla,normalized-linear,retention,rwkv7,rwkv6,gsa,hgrn,based,rebased,gdn,retnet --seqlens 64 2048 --batches 1 --heads 1 8 --contract all --timing e2e --warmup 3 --repetitions 10 --output-dir benchmark_results/stateful_fla_meta_all_correctness
```

输出记录按 `output-only`、`final-state`、`continuation` 分开；FLA 没有完整等价 State contract 的 family 明确标记 `fla_contract_comparable=false`，只提供 output-only 速度诊断。Meta correctness 使用独立 authored-order FP32 recurrence；FLA 只作为执行性能 baseline，不作为 correctness oracle，也不以 FLA 输出替代 Meta oracle。结果 JSON/CSV 中同时保留 selected lowerer、mathematical representation、workspace、public CUDA event、CPU wall、allocator peak、CUPTI activity/kernel launches 和失败原因。
本次实际运行选择空闲 NVIDIA H20（GPU 3；显存使用 4 MiB、无 compute PID、utilization 0%），结果为 156/156 条 Meta 记录 `status=OK`，且 156/156 条 `meta_correctness.status=VALID`；独立 oracle 为 authored-order FP32 CPU token recurrence。FLA baseline 成功 150 条；HGRN 的 `H=8` 六条记录因 adapter 输入解包错误 `too many values to unpack (expected 3)` 失败，HGRN `H=1` 仍成功。失败只影响对应 FLA baseline，不影响 Meta correctness 或其测量。

结果目录为 `benchmark_results/stateful_fla_meta_all_correctness/`，包含 `manifest.json`（`expected_records=actual_records=156`）、`records.json`、`summary.csv`、总览图及三个 contract speedup 图。该矩阵的 public CUDA event 是 warmed invocation median；同时记录 CPU wall、allocator peak 和可用的 CUPTI launches。


### 8.5 同一 IR：generic 对结构化优化

`benchmark_results/composition_profiles_final.json` 使用同一 Algorithm IR、`B=1,T=129`、matrix Feature `(row,column)=(64,128)`、BF16 input/FP32 gate+State、一个 public call with final State；每侧 10 个交替 batch，编译时间排除：

| Composition | selected optimized lowerer | Generic CUDA ms | Optimized CUDA ms | Generic/optimized | Optimized peak / generic peak |
|---|---|---:|---:|---:|---:|
| Axis scales | `tilelang-chunked-stateful` | 5.106 | 0.233 | 21.95x | 0.28% |
| Axis + Rank-One | `tilelang-factored-recurrence` | 5.319 | 0.778 | 6.83x | 1.88% |
| Multiple injections（早期记录） | `tilelang-product-gemm` | 0.744 | 0.761 | 0.98x | 100.42% |
| Rank-One | `tilelang-factored-recurrence` | 5.131 | 0.543 | 9.46x | 1.41% |

Multiple-injection 早期候选只有语义覆盖价值：优化侧比 generic 慢约 2.3%，paired bootstrap 95% interval 为 `[1.006, 1.050]`（optimized/generic），因此不能声称普遍 speedup。后续同一结构的独立记录 `benchmark_results/multiple_injections_after.json` 为：

| Path | CUDA-event ms | wall ms | peak incremental allocated |
|---|---:|---:|---:|
| Generic (`optimized=False`) | 0.750024 | 0.774290 | 93,401,088 bytes |
| Multiple-injection chunked | 0.286528 | 0.304401 | 926,720 bytes |

这次记录对应 generic/chunked CUDA ratio `2.62x`，chunked peak allocation 约为 generic 的 `0.992%`。这些结果依赖具体 IR、shape、编译选项和同一进程的 allocator 状态；不能直接外推到所有 injection count 或 all packed/training cases。

### 8.5 影响性能的主要因素

1. **Plan structure**：scalar factorized 和 raw dense binding 可以减少中间 Tensor；ordered rank-one 使用 factored recurrence；diagonal State 使用 64-token chunk carry/replay；ProductFeature 在 tile 内合成 outer product，避免保存 token-sized product。
2. **Sequence length**：长序列更能摊薄 host launch/dispatch，但 workspace 与 chunk carry 会增长；非对齐 tail 有 identity/zero-injection 语义，不等价于免费 padding。
3. **Batch/head/Feature size**：candidate estimate 按 State elements、summary elements、bytes/token、FLOPs/token 和 launch count 定价；H=8、`64×128` 与 H=1、`64×64` 不是同一个性能点。
4. **Mode**：dense inference 才能使用当前最激进的 chunked/native paths；packed 保留 segmented scan，但部分 dense-only schedule 不可用。
5. **Autograd**：training/backward 需要保存相应 context 或使用 generic/reference path，通常不会等于 inference 的 workspace/launch profile。
6. **return_final_state**：请求 final State 会增加 materialization 和 workspace allowance；不请求时，计划可以省去不需要的 final-State buffer。
7. **public vs graph**：短序列和小 batch 的 public latency 很容易由 Python/validation/launch overhead 主导；Graph replay 数字不能直接作为普通调用 latency。

## 9. 如何运行测试和 benchmark

### 9.1 安装

仓库已有的安装方式：

```bash
uv venv --python 3.12
. .venv/bin/activate
uv pip install torch
uv pip install "tilelang==0.1.15"
uv pip install --no-deps -e .
```

FLA 对照 benchmark 需要 bench extra：

```bash
uv pip install -e '.[bench]'
```

实际 CUDA/Torch/TileLang 版本必须与目标 GPU 环境匹配；`pyproject.toml` 固定 TileLang `0.1.15`，并通过 `uv.lock` 锁定其运行时依赖。

### 9.2 CPU-safe unit/import checks

```bash
uv run --extra test pytest -q tests/unit
uv run --extra test pytest -q tests/unit/test_example_imports.py
```

Stateful unit tests 会通过 `STATEFUL_OPERATOR_TEST_CPU=1` 绕过生产 H20 gate，验证 IR、cache identity、plan selection、workspace/error contract；这不是 CPU production support 的证据。

### 9.3 Stateful profile benchmark

基础 profile benchmark：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench \
  python testing/benchmark_stateful_plans.py \
  --families linear,simple-gla,gla,normalized-linear,retention,rwkv6,rwkv7,gsa,hgrn,gdn \
  --seqlen 512 2048 \
  --contract final-state \
  --fla \
  --json /tmp/stateful-plans.json
```

脚本默认 `warmup=3`、`repetitions=10`，并可用：

```bash
# output-only/final-state/continuation 全测
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench \
  python testing/benchmark_stateful_plans.py \
  --families linear,simple-gla,gla \
  --seqlen 512 2048 \
  --contract all \
  --fla

# 固定 B/H/T 组合矩阵
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench \
  python testing/benchmark_stateful_plans.py \
  --matrix \
  --matrix-families linear,gla,gdn,hgrn \
  --matrix-batches 1 2 \
  --matrix-heads 1 8 16 \
  --matrix-seqlens 63 64 65 130 2048 8192 \
  --json /tmp/stateful-matrix.json
```

每条记录包含 selected lowerer、mathematical representation、static estimate、workspace、public CUDA-event median、CPU wall median、Graph replay（如果启用）、allocator peak、correctness diagnostics 和可选 FLA baseline。FLA 的 layout preparation 在其 public-call 口径中计时；另有 native-layout 记录供拆解。

### 9.4 Linear Attention 专用 benchmark

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench \
  python testing/benchmark_linear_attention.py \
  --seqlen 512 \
  --heads 8 \
  --timing both \
  --contract all \
  --json /tmp/linear-attention.json
```

该入口只测 `linear`、`simple-gla`、`gla` 三个 `examples/linear_attention.py` factory。它要求 CUDA；没有 GPU 时应预期命令退出并提示 `CUDA is required`，不要把跳过/失败当作性能数字。

### 9.5 H20 correctness scope

针对 H20 的功能测试：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test pytest -q --run-gpu \
  -m 'functional and h20' tests/functional/test_gdn_h20.py

CUDA_VISIBLE_DEVICES=0 uv run --extra test pytest -q --run-gpu \
  tests/functional/test_linear_attention_examples.py \
  tests/functional/test_fla_coverage.py
```

这些测试覆盖 independent FP32 recurrence parity、非对齐长度、非零 initial State、final State、continuation、部分 gradient 和 profile coverage。性能 benchmark 本身不能替代 correctness test。

## 10. 证据索引与维护边界

- 公开入口和数据契约：`attention_engine/attn_engine/stateful_operator.py`、`docs/API.md`；
- normative schema/compiler contract：`docs/stateful-operator-ir-spec.md`；
- plain/Gated Linear Attention：`examples/linear_attention.py`、`examples/gated_delta_rule.py`；
- multi-State/normalized/product/readout 示例：`examples/normalized_linear_attention.py`、`gated_slot_attention.py`、`rwkv6_attention.py`、`based_attention.py`、`rebased_attention.py`；
- FLA coverage：`examples/fla_coverage.py`；
- independent correctness：`tests/functional/test_linear_attention_examples.py`、`test_fla_coverage.py` 及对应 profile tests；
- plan/candidate/workspace：`attention_engine/attn_engine/stateful/compile/plan.py`；
- benchmark implementation：`testing/benchmark_stateful_plans.py`、`testing/benchmark_linear_attention.py`；
- recorded H20 measurements：`docs/API.md`、`benchmark_results/diagonal_native_final.json`、`benchmark_results/composition_profiles_final.json`、`benchmark_results/multiple_injections_after.json`。

本文将“已表达/已测”“当前代码契约”和“设计 specification”分开描述。若三者不一致，应优先以当前源码和对应 functional test 判断可执行行为，再更新本文或 specification；尤其不要把 `docs/API.md` 中的历史 H20 数字当作跨设备性能保证。
