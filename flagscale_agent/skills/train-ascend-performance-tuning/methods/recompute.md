<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 激活保存、重计算与卸载

## 适用条件

已有配置、日志或 profile 支持调整激活保存、重放或卸载的假设时进入；可减少激活峰值，也可用更多保存量换取更少重放。廉价配置对照不要求先完成全部内存归因。
重计算边界按需读 `know-ascend-training` 的 `ascend_training/recompute.md`；卸载机制读 `ascend_training/memory-and-sharding.md`。权重/optimizer 常驻量过大时转 [并行与状态分片](parallelism.md)。

## 生成候选

从父配方的已生效配置出发，指出预计减少的内存来源或重放开销，选择对应操作。下表重计算字段放在 `train.model.recompute`：

| 方向 | 具体操作 |
| --- | --- |
| 选择性重计算 | 设置 `recompute_granularity: selective`，用 `recompute_modules: [mlp]` 等当前模型实际支持的标签选定边界；已有 selective 时增减一个目标模块。删除 `recompute_method`、`recompute_num_layers`；融合 attention 不默认优先重算 `core_attn`。 |
| 部分层重计算 | 设置 `recompute_granularity: full`、`recompute_method: block`、`recompute_num_layers: 1`，或对已有有效层数取相邻值。层数须适合实际 PP/VPP 本地块；删除遗留 `recompute_modules`。 |
| 重计算分组 | 在 full 下对照 `recompute_method: uniform`，调整 `recompute_num_layers` 为当前实现允许的分组大小，例如 `1`。它控制每组层数，不是只重算这些层；删除遗留 `recompute_modules`。 |
| 减少已有重计算 | 显存有余量时，减少 selective 模块或 block 层数；uniform 可与部分层 block 对照。完全关闭时删除重计算配置，并清理父配方中会重新启用它的旧别名及 `*_per_stage_micro_batch` 覆盖；不要用空模块列表或 YAML `null` 代替关闭。 |
| checkpoint 输入分片 | 当前 TP > 1、full 重计算、未启用 SP，且实现支持时，设置 `train.model.recompute.distribute_saved_activations: true`，比较保存输入减少与恢复通信代价；不要为启用它而默认关闭已有 SP。 |

以上是常用起点，不限定范围或顺序。有证据表明激活保存或重放值得优化时，可按预算深入调查峰值 rank、阶段与张量存活期，探索更合适的保存边界、输出丢弃式重计算、重放调度或激活卸载；实现修改须在任务授权范围内。
卸载等社区能力先确认当前模型和 NPU 路径已接入，再形成配置或代码候选，不直接移植 MindSpeed 开关。
MBS 调整衔接 [micro-batch 方法](micro-batch.md)，TP/PP/CP/EP 调整衔接 [并行方法](parallelism.md)，通信缓冲及 overlap 调整衔接 [通信方法](communication.md)，融合与临时张量优化衔接 [算子方法](operator.md)。

普通对照固定布局、MBS/GBS、精度和 backend；有依据时也可提交联合候选，明确各项差异与依赖，不把联合收益归于一个开关。
复用已确认的参数映射；只有字段被拒绝、改写或生效证据缺失时再查对应消费者。切换模式时检查父配置已有的 `distribute_saved_activations` 和分阶段覆盖是否仍合法。

## 额外检查

- **普通配置对照**：核对实际生效参数和目标模块，覆盖首次 optimizer 状态分配与完整更新；记录可取得的峰值 rank、阶段及显存口径。通用质量检查复用主流程，不逐次重新审计全部参数组与 checkpoint。
- **新重计算边界或效果异常**：按需区分原始前向与反向重放，检查 RNG/dropout、梯度和更新；checkpoint 入口被调用不证明实际重放。新增卸载路径时检查搬运完成后再消费张量、Host 内存和传输代价。

OOM 转移到其他阶段仍表示容量不足。无收益时区分重放已被外层覆盖、峰值来自其他来源或搬运抵消；异常重入需定位实现原因。
