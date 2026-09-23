<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 保持 GBS 的 micro-batch 调整

## 适用条件

已有配置、日志或 profile 支持计算粒度、下发开销、流水线空泡或激活容量的优化假设时进入；廉价 MBS 对照不要求先采完整 profile。
从父配方和已有日志取得有效 MBS、GBS、DP 与调度。批量、累积和可比性不清时，按需读 `know-ascend-training` 的 `ascend_training/batch-and-accumulation.md`。

## 生成候选

固定 GBS 和现有并行布局，从 `GBS / DP` 的正整数因子中选择合法 MBS，再检查当前模型和 pipeline schedule 的限制；不限定为 2 的幂，也不默认寻找最大可运行值。
下表路径指合成配置；已有同名字段时原位修改，避免重复定义。编辑被挂载为 `train` 的子 YAML 时省略 `train` 前缀。

| 方向 | 具体操作 |
| --- | --- |
| 增大计算粒度、减少 microbatch 次数 | 将 `train.model.micro_batch_size` 改为较大的合法值，固定 `train.model.global_batch_size`；比较完整更新耗时，检验是否抵消激活增加及 microbatch 数减少的代价。 |
| 降低激活峰值或改善流水线填充 | 将 `train.model.micro_batch_size` 改为较小的合法值，固定 GBS；PP 空泡突出时也可尝试，不仅用于 OOM。先区分压力是否来自激活，权重/optimizer 常驻量过大时转 [并行与状态分片](parallelism.md)。 |
| MBS 与重计算联合 | 在有依据的组合中同时调整 MBS 与 `train.model.recompute.recompute_modules` 或 `recompute_num_layers`，按 [重计算方法](recompute.md)匹配当前模式；例如省下激活后减少重放，或用重计算容纳更大 MBS。记录两项差异，不要求各单项先独立加速。 |
| MBS 与流水线调度联合 | 固定 GBS，普通均匀 VPP 布局可联合调整 MBS 与 `train.system.num_virtual_stages_per_pipeline_rank`；其余布局按 [并行方法](parallelism.md)处理。重算 microbatch 数和 chunk 约束，不通过增大 GBS 使候选合法。 |

常量批量配置若省略 GBS，先将父配置的实际 GBS 显式写入 `train.model.global_batch_size`，再改 MBS，避免默认值随之变化。Megatron 通常由 GBS/MBS/DP 派生累积次数，不另加未经该入口支持的 `gradient_accumulation_steps`。
已有动态 batch schedule 时保留其语义，在相同进度和 batch 阶段比较；不要额外叠加与 schedule 冲突的固定 GBS。不能用 `decrease_batch_size_if_needed` 的自动取整掩盖 GBS 变化。

例如 PP=1、GBS=64、DP=16、MBS=1 时，每次更新有 4 个 microbatch；MBS=2/4 分别对应 2/1 个，MBS=3 不整除。此例只说明批量算术，不证明模型支持或性能更优。
以上是常用起点，不限定探索范围或顺序。有证据且预算允许时，可深入调查实际 GEMM/attention 形状、每次更新的下发和通信次数、在途激活与调度空泡，形成更广的合法 MBS 或联合候选；动态 microbatch 等扩展先确认实现接入，源码修改须在任务授权范围内。
复用已确认的字段映射，仅在被拒绝、改写或生效证据缺失时查消费者。

## 额外检查

- **普通对照**：核对实际 MBS、累积次数及运行 GBS，保持每次更新的逻辑样本、序列/packing、精度与 optimizer 不变；覆盖完整更新和峰值显存。开启图执行时，对新 shape 完成必要预热后再测稳态。
- **动态批量或分阶段配置**：确认比较窗口没有跨 batch 阶段；父配方按 microbatch 索引定义重计算/调度时检查其覆盖，不能只改 MBS 后沿用失配的表。短跑只覆盖一个阶段时不声称整个 schedule 已验证。
- **质量**：复用主循环的相同初态、逻辑 batch 与质量口径；变长 mask、packing 或 MoE 批内统计可能使固定 GBS 仍不等价，出现差异时按需核对 loss 权重、累积缩放、梯度及更新。仅 loss 有限不证明等价。

更大 MBS 无收益不等于整个方向无效；计算粒度、流水线空泡和显存可能产生不同最优点。实际 GBS 改变的结果不能进入原工作负载排名；其他重计算、图或布局下的 MBS 边界需要重新判断。
