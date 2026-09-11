# 重计算、通信重叠与图模式的组合调优

本篇解释重计算、通信重叠与图模式的收益条件、相互约束和验证依据。训练语义、计时、状态与正确性门禁共用 [测量与实验记录](measurement-and-records.md)；系统代价见 [生产优化手册](production-optimization.md)。

这里列的是待验证边界与实验方法。参数名用于定位 Megatron-LM-FL/TE-FL 源码，不构成可直接复制的 FlagScale YAML。NVIDIA 研究依据与适配说明集中维护在 [来源与适配](sources-and-adaptation.md)。

## 1. 先归因内存，再选择重计算

分开观察常驻模型/optimizer 状态、保存 activation、重放/算子 workspace、通信 buffer 与 allocator/图缓存。记录所有 rank 在模型初始化、首个完整 optimizer step、稳态真实数据阶段的 allocated/reserved 峰值；数值或 API 缺失保持未知。

高 reserved 可能只是健康缓存；单 rank OOM 可能来自 stage 或 token 偏斜，二者都不能直接证明碎片。仅在 allocator 证据与目标 NPU API 支持成立时比较相关选项，不能引用 CUDA allocator 的“零开销”结论。

若首个 step 尚未返回，allocated 已异常增长，先检查原始异常栈中 checkpoint backward 是否反复重入，并区分真正的算子重试与递归调用。按栈中的类和函数定位，不能将所有 selective 重计算问题归到 `CheckpointWithoutOutput`。[本机版本例证](hardware-validation.md)中，小 dense 的 selective `mlp` 在普通 `CheckpointFunction.backward` 出现约 246 次重复帧并耗尽显存，而 core/full 对照完成更新；这不支持将所有 MLP 重计算判为不支持，也不支持直接归因于 allocator 碎片。保留失败阶段、调用栈与显存证据；只有调用路径变化或新增观测能回答未解问题时再考虑有界复现，不用盲调 MBS 或 allocator 反复触发同一故障。

容量和预算允许时保留明确无重计算 control，固定 backend、layout、MBS/GBS、精度和输入；同条件已证实的 OOM 可直接引用，不为对照重复确定性失败。根据真实峰值选择最小有效边界，逐项比较时间与峰值；不是按模块名称固定顺序叠加。

| 结构/证据 | 可检查的边界与源码关键词 | 必须核实的代价或无效情形 |
| --- | --- | --- |
| 普通 attention 保留 score/probability 等中间量 | attention core，可能名为 `core_attn` | 融合 attention 可能已内部重物化；额外 checkpoint 的边际节省可能很小，CP 下还可能重放通信 |
| MLA 展开后的 Q/K/V 占据峰值 | up-projection/RoPE 区域，可能名为 `mla_up_proj` | 与 attention core 是不同边界；核实投影是否实际展开、dtype 与本模型实现 |
| grouped MoE expert 中间 activation 占据峰值 | expert FC1/FC2 间的输出丢弃/重放，可能名为 `moe_act` | 必须确认 grouped 实现支持，具体保留/重放哪些张量；不能假设重跑完整 GEMM |
| norm 输出占据峰值 | input/pre-MLP norm，可能名为 `layernorm` | 范围通常窄，但收益受外层 checkpoint 和图捕获影响，不保证与其他模块可加 |
| dense FFN 保存张量占据峰值 | dense MLP，可能名为 `mlp` | whole-MLP 重放代价较高；纯 MoE 层可能没有这个路径，混合模型需计有效层数 |
| 整 MoE 区域必须释放才可运行 | outer MoE，可能名为 `moe` | 可能重跑 router、dispatch/combine、expert 与 shared-expert；计入通信和 transient backward 峰值 |
| shared expert 是独立峰值 | shared-expert MLP，可能名为 `shared_experts` | 检查是否已被外层 checkpoint 覆盖、是否与 shared-expert overlap 冲突 |

只在当前 validator 接受且实际执行路径证明生效时加入这些标签。标签存在于文档、输出配置或另一分支并不够。GDN/其他混合架构从本版保存张量和实现推导，不借用 attention/MoE 模块表猜测覆盖范围。

细粒度候选仍不足时再比较更宽或 full-layer 边界；也可因状态容量问题转向分片/布局。full granularity 的 method/层数和 selective 模块列表不应误当作叠加参数，按本版解析器清理失效字段。uniform/block 在 PP/VPP 下的实际层分布也须核对。

若 OOM 从 forward 迁移到梯度同步或 optimizer，记录新失败阶段与峰值，但状态仍是 oom。必须覆盖首次 optimizer 状态分配和多个代表性更新，不能以“比 baseline 多走一步”验收容量。

## 2. 跨特性兼容证据

为候选维护一张小表：组合、精确三仓/运行时版本、配置 validator、实际调用边界、支持证据、probe 结果和 fallback。没有验证的组合保留 `evidence_level=unknown`；静态不兼容记录原因后停止该分支。

| 组合 | 在目标实现上追踪的问题 | 实验关注点 |
| --- | --- | --- |
| whole-MoE 重计算 + EP overlap | backward replay 是否重入已重排的 dispatch/combine 区域，validator 是否拒绝 | 通信顺序、重复/遗漏梯度、hang；必要时比较关闭 overlap 与窄边界两条合法路径 |
| shared-expert 重计算 + shared-expert overlap | shared expert 是否被移出原 forward 顺序、保存/重放状态是否一致 | 本版是否互斥，不能复制 upstream 禁用规则后宣称 NPU 验证完成 |
| selective/full 重计算 + graph | checkpoint 是否完全处于图内或图外，capture 是否跳过 wrapper | RNG、dropout、hook 与梯度；跨图边界或不同 graph 实现要求须追源码 |
| CP + attention 重计算 | 哪些 collective 会在 backward replay 再执行 | 拓扑、临时 buffer 与暴露通信；只测 attention kernel 不足以判断代价 |
| delayed wgrad + DP overlap/累积融合 | 延后的权重梯度是否在 reduce/optimizer 前按正确顺序完成 | 多 microbatch 的 main-grad 累积与多次真实更新，不只一次 forward |
| EP overlap + PP/VPP/MTP | 实际 schedule 是否提供需要的交错与依赖，MTP 路径是否兼容 | 层分布、最小 microbatch 数、最慢 stage；不通用固化某版本 VPP 必需条件 |
| offload + PP/重计算/graph | 当前 offload 实现的限制、传输 stream、buffer 生命周期 | 只有本版支持且传输/主机容量合适才加入候选；不直接沿用 NeMo 的 PP=1 门槛 |
| 精度/backend + 任一边界 | 对应 dtype、保存/重放与累积实现是否支持 | 保持已授权精度与容差；NVIDIA TE 版本号不能证明 TE-FL 能力 |

先静态检查，再按任务范围做最小有界功能验证；无新增 hang、有限 loss 只是初筛，梯度/更新等价仍按共享契约检查。初始化成功不证明 overlap 或 graph 已执行；补真实绑定、hook/replay 或时间线证据。

调用计数可证明对应边界被命中，但不证明实际通信重叠时长；TE adapter 调用也不保证没有后端 fallback。此类 hook 用于功能、质量或 profile 诊断，带 hook 的运行不进入稳定性能排名，即使采集结束时 hook 已恢复。

## 3. overlap 分步消融

只调当前关键路径上的通信。记录集体操作、消息规模、生产/消费位置、等待时刻、所用 stream、额外 buffer 和最慢 rank；集体操作总时长不是未覆盖时间。

DP reduce/gather、PP P2P、expert dispatch/combine、shared expert 是不同候选。先以当前稳定布局和 dispatcher 为 control，单独加入一个受支持 overlap；比较内存增量、暴露通信与完整 step，再决定组合。

对 expert 路径采用下列隔离方式；不存在的能力不创建实验：

1. 固定 routing、dispatcher、重计算和图设置，比较普通 dispatch/combine overlap。
2. 保持普通 overlap 已生效，独立加入 delayed wgrad；检查梯度累积和 optimizer 消费时序。
3. 确有通信瓶颈且目标 NPU 后端受支持时，再比较 dispatcher 或 shared-expert 方案。
4. 保留单项有效候选后，重新验证可兼容组合，检查通信/算力资源竞争和内存峰值。

便捷开关可能同时启用 EP overlap、delayed wgrad、关闭 shared-expert overlap，或替换 dispatcher；以实际 helper 源码为准。保存调用前后完整有效 diff。若无法拆开，在卡片中把它描述为一组联合变更，不能据此给单项收益归因。

bucket/prefetch 参数先查单位与作用域，再选相邻值。确认是按字节、元素还是参数数量，以及 dtype 转换后的通信大小；不可直接搬 GPU 示例中的数值。开关被 runtime 静默关闭时记录有效值与原因，不能将其计为有效消融。

## 4. NPU 图模式的有条件实验

只有 host/launch/调度开销位于关键路径，且当前 FlagScale→Megatron-LM-FL→TE-FL/torch_npu 路径确有图能力时进入本节。没有实现时记录缺口，不用 CUDA 配置名、NVTE/NCCL 环境变量模拟支持。

1. 保留当前 eager control，先完成真实数据、正确性与所需 shape 的稳定运行。
2. 查明具体图实现、可捕获范围、shape/RNG 限制、collective、重计算与 hook 的兼容性。
3. 从最小有用范围或受支持的 bounded shape 集开始；记录 warmup、compile/capture、replay 三个阶段。
4. 验证 replay 实际发生，观测 graph break、重编译、静默回退与各 rank 常驻内存。
5. 在匹配输入、layout、dispatcher 与容器下比较 replay 稳态和 eager；capture 成本单独报告。
6. 确有目标收益且内存可接受后，才扩大范围或加入 overlap，并重新测该组合。

dropless MoE 的 expert token 数可能动态变化。先找现有实现可捕获的静态子区域；不能为了全图擅改路由、capacity/drop-token 或训练有效 token 口径。padding/packing 改动须先按共享不变量判断语义与比较组。

必须检查图模式下 RNG/dropout、gradient accumulation、recompute 与多次 optimizer 更新。若图实现限制在线诊断，先明确缺失的观测并以任务要求的等价检查补足；不能照抄上游“关闭 NaN 检查”来宣称通过正确性门禁。

图内存增量、可节省的 host 开销和 capture 的摊销都由目标工作负载测量；不承诺固定 GB 或加速百分比。持续重编译、重复 capture 与回退是生产成本，不能全部事后剔除来制造稳态收益。

## 5. 结果解释的边界

重计算、overlap 和图模式的效果取决于实际模块边界、helper 联动、峰值阶段以及通信/计算竞争。单项有效不保证组合有效，模块加速比不能相加；没有效果也可能来自未执行、静默禁用或 fallback，需与性能退化区分。

容量目标按约定的时间代价衡量，速度目标按完整更新衡量。短跑可行不能证明生产稳定，跨版本、跨 layout 的单行历史数字也不能作为模块排名。具体执行、状态更新与下一步选择由 [主调优工作流](../../../skills/train-ascend-performance-tuning/SKILL.md) 负责。
