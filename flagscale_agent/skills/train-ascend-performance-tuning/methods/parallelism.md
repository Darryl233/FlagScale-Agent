<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 并行布局与调度调整

## 适用条件

已有父配置和容量、通信、pipeline 空泡或 stage/expert 偏斜的证据，或用户明确要求比较布局。
取得当前 TP/PP/CP/SP/VPP、MoE 的 EP/ETP、实际 rank groups、MBS/GBS 及层分布。
通过 `know-megatron-parallel` 查询分组/调度机制，按当前参数消费者和并行初始化核对；Ascend 差异再查 `know-ascend-training` 的
`ascend_training/search-space.md` 与 `ascend_training/stack-capabilities.md`。缺少后端证据时返回待核对项。

## 生成候选

1. 从 TP/SP、PP/VPP、CP、EP/ETP 中选与瓶颈相关的一组，生成一个或少量相邻布局，
   核对 dense/expert mesh、stage/chunk、MBS/累积与词表形状，再复制父配方落成候选。
2. 保持总进程数和训练数学定义；保留 MBS，确需调度联动时完整记录原因和派生差异，保持有效 GBS。
3. 补核本版 NPU attention、dispatcher 和 Ascend override 的实际路径；未验证组合附最小兼容性检查。
   返回 `parent / hypothesis / change / checks`，不自动生成全部维度组合，也不在此运行训练。

### FlagScale dense 配方起点

以下各行是独立候选，在父配方的 `train.system` 下修改，不把整表同时开启；通过已加载的 `train-run` 启动。

| 目标 | 最小改动示例 | 保持或核对 |
| --- | --- | --- |
| TP | `tensor_model_parallel_size: 2` | 其余布局不变，核对派生 DP/累积 |
| SP | `sequence_parallel: true` | 固定已验证的 TP>1，比较 SP 关闭/开启 |
| PP | `pipeline_model_parallel_size: 2` | 固定 TP，层数与 microbatch 数满足调度 |
| VPP | `num_virtual_stages_per_pipeline_rank: 2` | 固定 PP；四层、PP=2 时每个 chunk 一层，还须满足当前 P2P 调度约束 |

VPP 不直接设置派生的 `virtual_pipeline_model_parallel_size`，也不同时设置多个 VPP 定义字段。
若本版要求联动 P2P overlap，记录整个变更组；不能满足层分配或调度条件就暂缓该项。
先核对启动日志中的实际参数；只在不支持、改写或分组证据缺失时查询对应实现，不重查整个 launcher。

## 额外检查

- **启动前**：按当前初始化代码核对 dense/expert mesh、rank ordering、分片形状、有效 GBS 和 PP/VPP 调度，
  保持实际词表/embedding 形状可比；核对 NPU override 是否覆写请求的分组、调度或 backend。
- **短跑**：核对实际分组、stage/chunk、dispatcher 和 attention 后端；CP 额外核对序列切分、
  真实位置、mask/packing 与前后向路径。记录静默改写或 fallback，不能以初始化成功代替生效检查。
- **质量比较**：按真实分片对齐相同初态、逻辑 batch、梯度与参数更新，不能直接比较同号 rank 的局部张量；
  测量与恢复边界查 `ascend_training/measurement-and-records.md`，分别记录模型、optimizer、RNG 的恢复覆盖。

## 结果反馈

主循环返回有效结果后，仅沿改善目标的布局方向提出下一个邻居；新布局重新取得内存边界。
调度或分组不合法则修正对应派生项；后端未生效则返回调用路径缺口，不将其记为该布局的性能结果。
质量失败时保留失败候选与对齐证据，先定位分片/位置/恢复差异；不以冷启动成功覆盖恢复失败。
无收益则结束当前方向或根据新的瓶颈证据选择另一组调整；未测试的布局保持未验证。
