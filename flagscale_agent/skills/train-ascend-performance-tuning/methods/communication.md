<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 通信与重叠配置

## 适用条件

框架分组/调度先查询 `know-megatron-parallel`，overlap 机制查询 `know-megatron-model`；
TE 通信路径需要时再查 `know-te-comm`，其中 CUDA/NCCL 专属实现不自动代表 NPU 支持。

已有证据表明 DP 同步、PP P2P 或 MoE dispatch/combine 的等待影响完整更新，或某项 overlap 引入额外显存/退化。
取得相关 rank/stage、通信与等待位置、当前布局和实际后端；只有通信总耗时时先返回需补的时间线问题。
原理与兼容约束按需查 `know-ascend-training` 索引中的 `ascend_training/recompute-overlap-graphs.md`；
字段消费者与 helper 联动查 `ascend_training/stack-capabilities.md`。

## 生成候选

1. 固定当前布局、MBS/GBS、精度、重计算及 routing，从一条实际等待路径选择变更。
2. DP 路径选择一次 reduce/gather overlap 消融或一个 bucket 相邻值；PP 路径选择已有 P2P overlap；
   MoE 路径先比较一个 dispatch/combine overlap，再按证据考虑 dispatcher、shared-expert overlap 或 delayed wgrad。
   dispatcher 改变时单独比较，不同时叠加其他 overlap。
3. 从 parser、helper 到运行对象核实参数、默认值、单位、必要依赖与最终有效 diff。
   需要优化器分片等前置条件时先单独验证它；无法拆开的联动作为一个变更组，不归因于单一开关。
4. 返回 `parent`、`hypothesis`、`change`、`checks`。已验证的单项合并时也生成新候选，交主循环重新比较。

## 额外检查

- **启动前**：检查当前实现的分组、optimizer、调度、重计算/图模式约束；bucket 明确按字节还是元素计。
- **短跑**：确认有效开关及目标执行路径，检查所有 rank 的通信顺序、完成状态与额外 buffer 峰值。
  无效果时按需补采时间线；开关为 true 或 helper 被调用不能单独证明发生重叠。
- **采纳前**：核对同一起点多次更新的梯度累积和参数更新；delayed wgrad 要检查 optimizer 前梯度已就绪，
  MoE 保留 routing/top-k/容量/丢 token 语义。额外观测作业只提供诊断证据。

## 结果反馈

等待减少且完整更新目标改善时，返回保留建议及尚未测试的相关邻居。
时间线重叠增加但训练更慢或显存不可接受时，保留原配置，检查资源竞争后结束或缩小该变更。
hang/数值失败拒绝候选；静默禁用记录实际原因，不能当成有效消融。若真正问题是分组或负载偏斜，
向主流程返回证据，由其选择并行方法；不在本方法中启动另一套搜索。
