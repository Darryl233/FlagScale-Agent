<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 通信调度与重叠

## 适用条件

已有配置、日志或 profile 支持通信等待、额外 buffer 或 overlap 退化的假设时进入；廉价配置对照不要求先采完整 profile。
参数依赖按需读 `know-ascend-training` 的 `ascend_training/communication.md`，涉及重放时读 `ascend_training/recompute.md`，涉及图执行时读 `ascend_training/graph-execution.md`。

## 生成候选

从父配方的已生效配置出发，选择与证据对应的操作，写明预期减少什么等待或内存：

| 方向 | 具体操作 |
| --- | --- |
| DP 梯度同步 / 参数 gather | 在已满足本版 optimizer 依赖的配置上，对照 `overlap_grad_reduce` 或 `overlap_param_gather` 的开/关状态。单独比较 gather 时保留梯度同步重叠；关闭梯度同步重叠时，联动关闭依赖它的 gather 重叠，记录联合变更。 |
| 梯度 bucket | 取得当前有效字段、值和单位，按假设生成一个更小或更大的邻近值；固定其余通信配置，比较完整更新耗时与峰值显存。 |
| PP P2P | 当前 PP/VPP 调度允许时，用 `train.system.no_overlap_p2p_communication: true` 关闭；启用对照则移除该禁用项或设为 `false`。内部生效字段是 `overlap_p2p_comm`，不要把它直接添加为 YAML 开关。某一状态不合法时不直接切换，布局联动按 [并行方法](parallelism.md)。 |
| MoE dispatcher / overlap | 从当前已接入的实现中，选择一个 dispatcher 替换或 dispatch/combine、shared-expert overlap 对照；保持 routing、top-k、容量和丢 token 语义，记录必要联动。 |

以上是常用起点，不限定搜索范围或顺序。有证据表明通信是瓶颈或值得优化时，可以按预算开展更深入、全面的调查：
沿相关通信链路检查消息大小与调用次数、rank 负载与拓扑、计算通信依赖和缓冲生命周期，必要时扩展到跨模块、跨 rank 的关联分析。
据此探索通信合并/分块、TP/CP/EP 布局与映射、dispatch/combine 数据流，以及更深的通信调度或后端实现优化，无需先试完表中方向。
布局变更衔接 [并行方法](parallelism.md)，算子实现变更衔接 [算子方法](operator.md)；源码修改仍须在任务授权范围内。

复用已确认的参数映射与依赖，只补查当前候选缺少的条件；字段被拒绝、改写或生效证据缺失时再定位对应消费者。

## 额外检查

- **普通配置调整**：核对目标参数实际生效、bucket 单位及该变更的必要依赖；复用已验证的分组、optimizer 与调度条件。
- **改变分组、dispatcher 或调度**：补查受影响 rank 的通信顺序、梯度归约/累积及缓冲生命周期；涉及 delayed wgrad 时，确认梯度在归约和 optimizer 消费前完成。仅调整 bucket 不要求重新调查全部调度与图模式。
- **效果异常或机制未明**：按需采集能区分当前假设的时间线；开关为 true 或入口被调用不证明实际重叠。诊断运行与无 profiler 性能测量分开。

无时间线证据时可以报告端到端收益，但不能声称已证明通信等待减少。布局、负载或实现问题的证据用于下一候选。
