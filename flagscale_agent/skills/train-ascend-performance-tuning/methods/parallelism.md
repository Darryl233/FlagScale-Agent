<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 并行布局、调度与状态分片

## 适用条件

已有配置、日志或 profile 支持容量、通信、pipeline 空泡或 stage/expert 偏斜的假设，或用户明确要求比较布局时进入；廉价配置对照不要求先采完整 profile。
从父配方和已有日志取得与候选相关的布局、MBS/GBS、层分布与分组信息。约束不清时，按需读 `know-ascend-training` 的 `ascend_training/parallelism.md` 对应章节；状态分片按需读 `ascend_training/memory-and-sharding.md`。

## 生成候选

复制父配方，从已生效布局出发选择下列操作，写明预期减少什么内存、通信或空泡。下表未写全路径的字段默认放在合成配置的 `train.system`；已有同名字段时原位修改，避免在 system/model 中重复定义。若编辑被挂载为 `train` 的子 YAML，则省略 `train` 前缀。每行是独立方向，数值从当前布局附近的合法值选择，不同时开启整表。

| 方向 | 具体操作 |
| --- | --- |
| 优化器状态分片 | 当前 optimizer 支持且实际分片组大于 1 时，设置 `train.system.use_distributed_optimizer: true`；保持 optimizer family、精度与起始状态。无需同时新增通信 overlap 或改 checkpoint 格式，只有实际兼容性要求才联动。 |
| TP：容量 / 通信与计算粒度 | 修改 `tensor_model_parallel_size`：通信突出且有显存余量时比较更小值，分片容量受限时比较更大值。固定 PP/CP；降至 TP=1 时将 `sequence_parallel` 设为 `false`。MoE 中若 ETP 原来随 TP 默认继承，显式保留原有效 `expert_tensor_parallel_size`，或将 ETP 变化记录为联合候选。 |
| SP：激活容量 | 固定已验证的 TP>1，对照 `sequence_parallel: false/true`；若父配方启用了 `distribute_saved_activations`，开启 SP 时联动关闭它。当前 MoE 组合若要求 SP，不构造不合法的关闭对照。 |
| PP：层状态容量 / pipeline 空泡 | 修改 `pipeline_model_parallel_size`，固定 TP/CP：按容量或空泡假设比较更大或更小的合法值。已有 VPP 或显式层分配时，同步调整其 stage/chunk 划分；检查派生 DP 与 microbatch 数，必要时联合修改 `train.model.micro_batch_size`，保持 `train.model.global_batch_size`。 |
| VPP：流水线调度 | 普通均匀布局下，固定 PP>1，用 `num_virtual_stages_per_pipeline_rank` 比较不同 chunk 数；删除旧的 `num_layers_per_virtual_pipeline_stage`，不直接设置派生字段 `virtual_pipeline_model_parallel_size`。回到非 interleaved 对照时移除 VPP 定义字段；使用显式 layout 或 hybrid pattern 的配方按下一行调整，不叠加该字段。 |
| stage 层分配：负载偏斜 | 固定 PP 与模型层序列，首尾 stage 较慢时调整 `decoder_first_pipeline_num_layers` / `decoder_last_pipeline_num_layers`，将部分层移给有余量的 stage。父配方使用 `pipeline_model_parallel_layout` 时，调整其中的 stage 边界，保留层类型、总数和顺序；使用 `hybrid_layer_pattern` 时沿用该字段调整分段，不叠加另一套 layout/VPP 定义。 |
| CP：长序列容量 | 修改 `context_parallel_size`，在当前 attention 已支持的范围内比较更大值；通信代价突出且容量允许时比较更小值。固定序列长度、TP/PP 及当前 `cp_comm_type`，核对派生 DP/累积和序列切分要求，不靠缩短序列使候选通过。 |
| EP / ETP：专家容量与通信 | 固定 dense TP/PP/CP，分别修改 `expert_model_parallel_size` 或 `expert_tensor_parallel_size`；比较 EP 时显式固定有效 ETP，比较 ETP 时固定 EP。保持专家数、routing、top-k、容量/丢 token 语义和 dispatcher，核对 expert 分组及专家分片；dispatcher 对照另按 [通信方法](communication.md)生成。 |

VPP 与 P2P overlap 联动时使用 [通信方法](communication.md)中的配置入口，并核对实际调度；联动项计入完整差异。
保持授权资源规模和训练数学定义；默认保留 MBS，只在明确的容量、计算或调度权衡下联合调整。布局改变后重新核对派生 DP、累积数与有效 GBS，参数约束复用已有结论。

以上是常用起点，不限定搜索范围或顺序。有证据且预算允许时，可深入调查拓扑映射、层分配、调度和 dense/expert 布局的关系，生成更广的布局或联合候选，无需先试完表中方向。
拓扑映射等未列出固定字段的方向，先定位当前已支持的配置入口；确需代码调整时，明确修改点和预期分组，源码修改须在任务授权范围内。

## 额外检查

- **新分片路径或续训**：检查实际主参数/optimizer 状态的分片与更新；任务要求恢复时补保存/加载验证。改变 optimizer family、精度或起始状态的结果不能当作原条件下的等价比较。
- **启动前**：复用已确认的配置、日志和实现条件，只补核本次变化影响的分组、形状与调度；涉及 TP 或词表分片时，保持实际词表及 embedding/output 形状可比。参数被拒绝、改写或缺少生效证据时，再定位对应 parser、初始化或 Ascend override。
- **短跑生效**：核对本次改变的布局、stage/chunk 或实际 rank groups；涉及 CP 时补查 attention、序列切分、真实位置与 mask/packing，涉及 MoE 时补查 expert mesh 和 dispatcher。未验证组合在候选短跑中补足兼容性证据，记录静默改写或 fallback，初始化成功不代替生效检查。
- **质量比较**：相同初态、逻辑 batch 下按真实分片对齐 loss、梯度与参数更新，不能直接比较同号 rank 的局部张量。检查实际使用的状态加载路径；任务涉及续训或要求完整恢复时，再补齐 optimizer、scheduler、RNG 与数据进度的恢复检查。涉及状态对齐或恢复时按需读 `ascend_training/state-and-resume.md`。

布局改变后重新记录峰值；仅在容量目标或下一候选需要时探索边界。质量差异按真实分片、位置或恢复状态解释，冷启动成功不能覆盖恢复失败。
