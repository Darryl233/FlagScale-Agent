<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# 算子热点、调用链与实现判断依据

本文解释热点证据、NPU 公共接口接入、实现归属和正确性/计时判断。
完整更新与 A/B 测量口径见 [共享测量依据](../ascend_training/measurement-and-records.md)。
纯通信、并行、数据或调度问题可查 [训练优化知识](../ascend_training/production-optimization.md)。

## 热点分桶与关键路径

比较单位为同一次运行、已确认 step/rank 下的
`op / public callable × shape × dtype × layout × rank/stage`。
不同采集开关、运行或基线/候选的数据具有不同来源，不能直接合并。缺失字段保留未知。

| 字段 | 判断用途 |
| --- | --- |
| 次数、累计 device 时间、单次分布 | 区分频繁小调用与单次慢调用；样本不足时列原值 |
| step/rank/stage、stream、前后依赖及重叠 | 判断关键路径上的可移除时间上限 |
| shape/dtype/stride/layout、可选参数、前后向 | 选择代表性用例；避免平均同 OP Type 的不同负载 |
| 设备算子、公共 callable 与实际后端绑定 | 将 profile 行映射到实现，不能仅凭名称推断 fallback |
| Cube/Vector/搬运指标与实际分母 | 形成计算、访存或下发假设；缺字段不是零 |

MatMul 的 M/N/K 取决于真实张量语义和转置。MoE kernel 的尺寸取决于本 rank 的 token/专家分布及展开后输入，
不能仅由模型配置推断。shape 可由公共调用参数或有界探针确认，不一定需要整场开启高开销采集。

累计 device 时间是筛选依据，重叠事件之和不是 step 耗时；聚合表也不能证明关键路径。
同名算子在不同 shape、dtype、layout 或 rank 下可能对应不同瓶颈，分桶后仍需根据调用链和时间线确定归属。

## 三仓调用链与绑定证据

常见调用层次如下，具体工作负载可能跳过某些层：

```text
FlagScale entry/config
  -> Megatron model or training utility
  -> TransformerEngine public API or FLA API
  -> FL plugin dispatch
  -> vendor adapter
  -> transformer_engine_npu / fla_npu / torch_npu
  -> device kernel
```

每个边界的契约包括公共 callable、实际绑定对象、参数语义、shape/dtype/layout、
可微输入、输出顺序以及 unsupported 输入的处理方式。
compiled-extension facade、公共 Python API 与内部 Triton 函数可能走不同的分派路径。

### TransformerEngine-FL dispatch

算子名和后端 ID 来自当前 registry，以下只是版本相关的选择日志示例：

```text
Op 'get_permutation_class' using 'vendor.npu'
Op 'get_attention_backend' using 'vendor.npu'
```

| 绑定证据缺失处 | 对应检查点 |
| --- | --- |
| 没有公共定义 | 当前 op definition 是否有该算子 |
| 后端未加载 | NPU backend 是否导入 adapter |
| 实现未暴露 | getter 或实现是否由 backend 导出 |
| registry 不含算子 | `register_ops.py` 的确切名称与实现 |
| 注册有但调用未切换 | public facade 是否在 import 阶段安装返回的 adapter |
| 表面配置正确 | 进程实际绑定 callable 的身份与来源 |

某些 Python/Triton 路径不经过 compiled-extension facade。它们适合在稳定公共 API 边界接入，
而不是为了一个私有实现扩大全局 op manager。实现文件存在、成功 import 或直接 vendor 调用通过，均不证明训练接入。

## 竞争性解释及其证据

| 假设 | 能区分假设的证据 | 判断边界 |
| --- | --- | --- |
| 专家负载不均 | 各 rank 专家 token 数与完整 step 时长 | 强制均匀路由改变语义，只可用于独立诊断 |
| 分配或同步代价 | 分配寿命、流/数据依赖与实际同步位置 | `empty_cache()` 或同步只有证明冗余后才可移除 |
| kernel 慢或 launch 多 | 设备工作与任务间隔、公共调用成本 | 调用次数本身不能判定 launch 瓶颈 |
| 计算或通信主导 | 重叠区域、rank 关键路径 | 累计算子/通信表不能直接还原暴露时间 |
| 融合路径收益 | 相同语义、数据和配置下的完整调用 | 辅助转换和临时内存仍属于候选成本 |

否定假设的结果也属于证据，可以避免重复尝试相同方向。

## 实现选择与仓库归属

现有且实际测量过的兼容实现是首选对照。vendor 标签本身不证明优于可工作的 FlagOS 路径。
缺少 Ascend 接入时，可考虑匹配的 `transformer_engine_npu`/`fla_npu` 公共接口、
局部签名/dtype/返回顺序 adapter，或受支持的 `torch_npu` 算子；unsupported 输入保留框架的参考路径。
仅包装普通 PyTorch 组合不能证明获得了硬件专用实现。

| 修改类型 | 通常所属层 |
| --- | --- |
| 配置、launcher、数据热路径 | FlagScale |
| 模型/FLA override、并行与模型集成 | Megatron-LM-FL plugin |
| TE 算子、attention、permutation adapter 及注册 | TransformerEngine-FL plugin |
| 设备 kernel | `transformer_engine_npu`、`fla_npu` 或 `torch_npu` |

已有 plugin override/backend adapter 通常能限制修改范围。必须让框架 import 可替换时，
可使用窄的 overridable/facade 安装点，把路径选择保留在基础抽象；模型语义和通用控制不宜移入 vendor 子类。
集中分派负责 reference fallback，NPU backend 不必复制整段 reference 代码。
逐次 `try/except` 静默重试会隐藏实现错误和成本，不适合作为正常热路径路由。

## 正确性与公共接口契约

| 检查层 | 需要保留的行为 |
| --- | --- |
| 静态和分派 | 可导入、无重复注册、目标后端实际被调用、unsupported 输入遵循既有路径 |
| 前向 | 所有输出的 dtype/device/shape、数值误差与返回顺序 |
| 反向 | 全部可微输入的梯度、计算图和必要依赖 |
| 离散输出 | row/index map、排列顺序及 probability 张量语义，不能仅做浮点近似 |
| 边界 | 空/退化输入、可选参数、真实最大展开 token、布局及索引边界 |
| 公共集成 | 训练使用的同一公共 API 可执行，避免私有函数成功掩盖 facade 或适配错误 |

误差标准取决于项目精度和累加语义，须在比较前确定。“无异常”不是正确性测试。
边界例子如 65535/65536 仅在相关索引或形状约束成立时使用，不是所有算子的固定用例。
热路径 `.cpu()`、`.item()`、Python 列表、日志或 shape 整理可能引入 host 往返或隐藏同步；
若接口确实需要，完整调用成本必须包含它们。

## Kernel 与 public-call 计时口径

| 层次 | 包含的工作 | 不能据此声称 |
| --- | --- | --- |
| Device kernel | 当前版本支持的 profiler、设备 event 或可靠基准观测到的设备执行 | 训练公共接口已更快 |
| 公共 API 完整调用 | 输入整理、临时分配、布局转换、辅助张量、launch、输出恢复和必要同步 | 完整训练已更快 |
| 端到端训练 | 同工作负载的完整前后向、通信与优化器更新 | 尚未覆盖的模型/长度/布局也有收益 |

壁钟计时需要按已确认的设备/流 API 等待所测工作完成；异步提交时间不等于设备执行时间。
边界同步由基准 harness 控制，不能为计时改变生产热路径依赖或给每个内部算子添加同步。
编译、初始化和预热单列；采集运行与正常性能测量分开。计时结果需要保留
输入桶、次数/分布、额外内存、代码修订、后端和边界，独立候选使用可辨识名称及输出。

kernel 更快而完整调用更慢，常提示新增转换、分配或同步抵销了收益；这些成本不能从候选中排除。
吞吐目标需要解释关键路径上的收益来源，容量或可靠性目标也可能预先允许一定耗时代价。
微基准仅决定候选是否值得继续；训练收益按共享测量依据判断，不能由单一快 kernel 推导。

## MoE 与 GDN 的条件化诊断

以下判断取决于模型结构、安装修订和实际调用链，不能仅凭模型名称推断。

| 现象或条件 | 机制与判断依据 |
| --- | --- |
| 数据热路径逐批调用 `empty_cache()` | 释放缓存可能增加后续分配或同步成本；是否冗余取决于内存生命周期及调用前后的设备依赖，不能仅凭调用存在就删除 |
| MoE dispatch 中出现 AICPU `argsort` | 可能来自 token 排列和索引构造；需关联公共 permutation 调用，比较后端融合实现及 NPU mask-map 的语义、适用输入和完整调用成本 |
| 大 token 排序 kernel 更快，但训练未变快 | chunk-sort 等局部算法可能受输入整理、辅助张量、同步或关键路径份额限制；算子正确性不等于端到端收益 |
| 不同专家的 token 数相差较大 | 路由分布可能影响负载与通信；强制均匀路由会改变训练语义，只能用于隔离负载不均假设 |
| GDN 使用 `l2norm`、`chunk_gated_delta_rule` | 接入层取决于实际 FLA override；Megatron-LM-FL Ascend override 与其下游实现的签名、梯度和绑定需一致 |
| TE permutation 走 Python/Triton 路径 | 该路径可能绕过 compiled-extension，公共 facade adapter 是否安装决定训练能否调用目标实现 |
| 实现文件可导入，但仍选中参考后端 | 注册、backend 导入或 facade 安装可能缺失；选择日志与公共 API 的实际绑定比文件存在更有判别力 |
| 配置启用 `moe_router_fusion` | 当前后端必须提供已注册且兼容的 fused top-k；还需满足返回值、dtype、索引与梯度契约 |

更底层资源、tile 和极端输入依据见 [kernel 资源与正确性依据](kernel-experiments.md)。
