# 重计算、通信重叠与图模式的机制与约束

重计算用重放代价交换保存张量的内存，通信重叠改变计算与通信的时序，图模式减少重复调度开销。三者共享张量生命周期、RNG 和异步执行依赖，因而收益与兼容性不能独立相加。指标含义见 [训练测量与正确性](measurement-and-records.md)，系统代价见 [系统优化机制](production-optimization.md)。

参数名表示 Megatron-LM-FL/TE-FL 中可能存在的实现边界，不构成可直接复制的 FlagScale YAML；具体支持由当前 validator、模型结构和 NPU 后端共同决定。跨平台参数与实现差异见 [平台适配边界](sources-and-adaptation.md)。

## 1. 内存来源与重计算边界

训练内存包括常驻模型/optimizer 状态、保存 activation、重放/算子 workspace、通信 buffer 与 allocator/图缓存。模型初始化、首个完整 optimizer step 和稳态真实数据可能分别出现峰值；容量由最重 rank 的实际峰值约束，缺失的 API 或数据不能解释为零占用。

高 reserved 可能只是健康缓存；单 rank OOM 可能来自 stage 或 token 偏斜，二者都不能直接证明碎片。仅在 allocator 证据与目标 NPU API 支持成立时比较相关选项，不能引用 CUDA allocator 的“零开销”结论。

首个 step 未返回而 allocated 异常增长时，checkpoint backward 反复重入是需要与正常重放区分的机制。`CheckpointFunction.backward` 的递归与 `CheckpointWithoutOutput` 是不同的定位入口，实际栈中的类和函数决定问题归属。递归耗尽显存属于后向执行故障，不能仅因最终异常是 OOM 就解释为普通容量不足或 allocator 碎片；降低 MBS 可能延后耗尽，却不能消除递归。某个 selective 模块路径的问题也不代表所有 MLP 重计算均不受支持。

重计算的边际收益由相同 backend、layout、MBS/GBS、精度和输入下的保存张量差异决定，同时受到重放时间与峰值迁移影响。最小有效边界由实际峰值张量决定，而不是固定模块顺序；无重计算配置若超出容量，只能说明该条件下的容量差异，不能提供有效的稳态时间对照。

| 结构/证据 | 可检查的边界与源码关键词 | 必须核实的代价或无效情形 |
| --- | --- | --- |
| 普通 attention 保留 score/probability 等中间量 | attention core，可能名为 `core_attn` | 融合 attention 可能已内部重物化；额外 checkpoint 的边际节省可能很小，CP 下还可能重放通信 |
| MLA 展开后的 Q/K/V 占据峰值 | up-projection/RoPE 区域，可能名为 `mla_up_proj` | 与 attention core 是不同边界；核实投影是否实际展开、dtype 与本模型实现 |
| grouped MoE expert 中间 activation 占据峰值 | expert FC1/FC2 间的输出丢弃/重放，可能名为 `moe_act` | 必须确认 grouped 实现支持，具体保留/重放哪些张量；不能假设重跑完整 GEMM |
| norm 输出占据峰值 | input/pre-MLP norm，可能名为 `layernorm` | 范围通常窄，但收益受外层 checkpoint 和图捕获影响，不保证与其他模块可加 |
| dense FFN 保存张量占据峰值 | dense MLP，可能名为 `mlp` | whole-MLP 重放代价较高；纯 MoE 层可能没有这个路径，混合模型需计有效层数 |
| 整 MoE 区域必须释放才可运行 | outer MoE，可能名为 `moe` | 可能重跑 router、dispatch/combine、expert 与 shared-expert；计入通信和 transient backward 峰值 |
| shared expert 是独立峰值 | shared-expert MLP，可能名为 `shared_experts` | 检查是否已被外层 checkpoint 覆盖、是否与 shared-expert overlap 冲突 |

模块标签只有被当前 validator 接受并在实际路径执行，才代表对应重计算边界。标签存在于文档、输出配置或另一分支并不够。GDN/其他混合架构的有效边界取决于自身保存张量与实现，不能由 attention/MoE 模块表推断覆盖范围。

更宽或 full-layer 边界通常释放更多保存张量，也会重放更多计算和可能的通信；状态容量则主要受分片与布局影响。full granularity 的 method/层数与 selective 模块列表有不同语义，未启用相应 granularity 的字段可能无效或冲突。uniform/block 在 PP/VPP 下的实际层分布由 schedule 和层划分共同决定。

OOM 从 forward 迁移到梯度同步或 optimizer 只表示峰值阶段改变，仍未满足完整更新的容量要求。首次 optimizer 状态分配与真实数据波动可能高于 forward 峰值。

## 2. 跨特性兼容证据

组合兼容性由三仓/运行时版本、配置 validator、实际调用边界、分组和 fallback 路径共同决定。静态允许、功能可运行和数值等价属于不同层次，不能互相替代。

| 组合 | 实现依赖 | 正确性与性能边界 |
| --- | --- | --- |
| whole-MoE 重计算 + EP overlap | backward replay 是否重入已重排的 dispatch/combine 区域，validator 是否拒绝 | 通信顺序、重复/遗漏梯度、hang；必要时比较关闭 overlap 与窄边界两条合法路径 |
| shared-expert 重计算 + shared-expert overlap | shared expert 是否被移出原 forward 顺序、保存/重放状态是否一致 | 本版是否互斥，不能复制 upstream 禁用规则后宣称 NPU 验证完成 |
| selective/full 重计算 + graph | checkpoint 是否完全处于图内或图外，capture 是否跳过 wrapper | RNG、dropout、hook 与梯度；跨图边界或不同 graph 实现要求须追源码 |
| CP + attention 重计算 | 哪些 collective 会在 backward replay 再执行 | 拓扑、临时 buffer 与暴露通信；只测 attention kernel 不足以判断代价 |
| delayed wgrad + DP overlap/累积融合 | 延后的权重梯度是否在 reduce/optimizer 前按正确顺序完成 | 多 microbatch 的 main-grad 累积与多次真实更新，不只一次 forward |
| EP overlap + PP/VPP/MTP | 实际 schedule 是否提供需要的交错与依赖，MTP 路径是否兼容 | 层分布、最小 microbatch 数、最慢 stage；不通用固化某版本 VPP 必需条件 |
| offload + PP/重计算/graph | 当前 offload 实现的限制、传输 stream、buffer 生命周期 | 只有本版支持且传输/主机容量合适才加入候选；不直接沿用 NeMo 的 PP=1 门槛 |
| 精度/backend + 任一边界 | 对应 dtype、保存/重放与累积实现是否支持 | 保持已授权精度与容差；NVIDIA TE 版本号不能证明 TE-FL 能力 |

无 hang、有限 loss 只说明基本运行状态，不能证明梯度与更新等价。初始化成功也不证明 overlap 或 graph 已执行；实际绑定、调用边界、replay 和时间线分别说明不同执行事实。

调用计数可证明对应边界被命中，但不证明实际通信重叠时长；TE adapter 调用也不保证没有后端 fallback。此类 hook 用于功能、质量或 profile 诊断，带 hook 的运行不进入稳定性能排名，即使采集结束时 hook 已恢复。

## 3. overlap 的依赖与归因

overlap 的价值取决于通信是否暴露在最慢 rank 的关键路径上。集体操作类型、消息规模、生产/消费位置、等待时刻、stream 与额外 buffer 共同决定可隐藏程度；集体操作总时长不等于未覆盖时间。

DP reduce/gather、PP P2P、expert dispatch/combine 和 shared expert 具有不同的依赖与内存开销。布局或 dispatcher 同时变化时，通信差异不能只归因于 overlap；有效收益是完整 step 的缩短，并需考虑额外内存与计算竞争。

expert 路径的主要依赖为：

| 机制 | 关键依赖 |
| --- | --- |
| dispatch/combine overlap | routing、dispatcher、重计算和图模式决定 token 流与重放顺序 |
| delayed wgrad | 权重梯度必须在归约与 optimizer 消费之前完成，并保持跨 microbatch 累积语义 |
| dispatcher/shared-expert 变化 | 可能改变通信算法、forward 顺序和临时张量生命周期 |
| 多机制组合 | 单项合法不保证组合合法；共享通信与计算资源可能增加等待和峰值内存 |

便捷开关可能同时启用 EP overlap、delayed wgrad、关闭 shared-expert overlap，或替换 dispatcher，具体联动由 helper 实现决定。此时完整有效差异是一组联合变更，不能把收益归因于某一个开关。

bucket/prefetch 参数可能按字节、元素或参数数量计量，dtype 转换还可能改变实际通信大小。不同单位、作用域或平台的数值不能直接比较；开关被 runtime 静默关闭时，输入配置变化不构成有效的机制消融。

## 4. NPU 图模式的收益条件

图模式主要减少重复 host/launch/调度开销；只有这些成本位于关键路径且 FlagScale→Megatron-LM-FL→TE-FL/torch_npu 路径具有图实现，才存在相应收益空间。CUDA 配置名或 NVTE/NCCL 环境变量本身不代表 NPU 图支持。

图执行包含预热、compile/capture 与 replay，不同阶段的时间含义不同。capture 成本需要由后续 replay 摊销；持续 graph break、重编译或静默回退可能抵消调度节省。图还可能增加各 rank 的常驻内存。

可捕获范围由 shape/RNG 稳定性、collective、重计算与 hook 的兼容性决定。最小静态子区域与受支持的 bounded shape 集可能比整模型更适合捕获；扩大范围或加入 overlap 会改变依赖和内存边界。replay 与 eager 的性能可比性要求输入、layout、dispatcher、运行环境和数学语义一致。

dropless MoE 的 expert token 数可能动态变化，因而全图捕获不一定适用。改变 routing、capacity/drop-token 或有效 token 口径会改变训练问题；padding/packing 也可能改变张量、mask 与比较语义，不能只视为图执行优化。

图模式下的 RNG/dropout、gradient accumulation、recompute 和 optimizer 更新必须保持相应语义。在线诊断被图实现限制时，未观察到异常不等于数值正确；关闭 NaN 检查也不会增加正确性证据。

图内存增量、可节省的 host 开销和 capture 的摊销都由目标工作负载测量；不承诺固定 GB 或加速百分比。持续重编译、重复 capture 与回退是生产成本，不能全部事后剔除来制造稳态收益。

## 5. 结果解释的边界

重计算、overlap 和图模式的效果取决于实际模块边界、helper 联动、峰值阶段以及通信/计算竞争。单项有效不保证组合有效，模块加速比不能相加；没有效果也可能来自未执行、静默禁用或 fallback，需与性能退化区分。

容量目标衡量内存需求与可接受的时间代价，速度目标衡量完整更新耗时。短跑可行不能证明生产稳定，不同版本或 layout 的单个数字也不能直接形成模块排名。
