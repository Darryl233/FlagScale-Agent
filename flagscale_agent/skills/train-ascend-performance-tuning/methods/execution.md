<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 数据供给与执行路径

## 适用条件

数据 reader/样本契约有缺口时检查受影响的读取与分发链路，可参考已有数据准备流程；Energon 机制按需查 `know-energon`。
已有有效数据不重复转换或换成 mock 数据，配置与质量检查继续沿用主流程。

日志或 profile 指向数据等待、host 下发开销，或一个已实现的后端/融合配置值得比较。
取得等待位置、输入 shape、当前数据路径及实际执行实现；选取下面与证据对应的一条分支。
按需查 `know-ascend-training` 索引中的 `ascend_training/production-optimization.md`；
图模式约束查 `ascend_training/recompute-overlap-graphs.md`，配置接入查 `ascend_training/stack-capabilities.md`。

## 生成候选

1. **数据等待**：确认数据读取/预处理是等待来源，选择当前已实现的 workers 或 prefetch 参数的一个相邻值。
   保留真实数据、采样顺序与处理语义；已有 CPU/NUMA 绑定可作为独立候选，不与 workers 同时批量修改。
2. **已有后端/融合配置**：追踪开关到训练实际公共接口，确认目标 NPU 实现，再只切换该路径及必要联动。
   如果需要实现新的算子或 adapter，返回代码缺口，由主流程按任务范围选择算子方法。
3. **图/编译模式**：先确认本版真实 NPU capture/replay 或编译能力，保留 eager 对照。
   从支持的最小有用范围/shape 集生成候选；当前实现缺失则返回缺口，不用 CUDA 参数名代替接入。
4. 每次只选择一条分支，返回 `parent`、`hypothesis`、`change`、`checks`，由主循环运行。

## 额外检查

- **数据分支**：比较实际样本标识/顺序、有效 token 和更新进度，确认没有遗漏或重复；检查主存、共享内存及数据等待变化。
- **后端分支**：检查训练实际绑定、shape/dtype/mask 支持及 fallback；核对公共接口前后向和数次训练更新质量。
- **图分支**：区分 warmup、compile/capture、replay，确认 replay 实际发生，记录 graph break、回退和每 rank 常驻显存。
  检查动态 shape、RNG/dropout、梯度累积、重计算与多个 optimizer 更新；初始化成功不足以验收。
  首次编译成本单列，持续重编译或重复 capture 纳入实际运行成本。
- 保留任务要求的数据与 I/O 工作；不能通过去掉数据处理、评估或 checkpoint 工作改变比较口径。
  涉及 checkpoint 行为的修改另列恢复检查，不能只凭吞吐采纳。

## 结果反馈

完整更新改善且对应质量检查通过时，返回保留建议；等待转移或无收益时返回当前瓶颈和结束/邻居建议。
数据次序改变、后端质量失败或持续回退时拒绝当前候选；缩小图范围或回到原执行路径后，作为新候选交主循环。
不要从 mock 数据证明真实数据收益，也不要由图捕获成功宣称训练加速。
