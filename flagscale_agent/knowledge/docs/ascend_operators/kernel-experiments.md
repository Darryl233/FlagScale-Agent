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

# 有界 kernel 实验

仅在时间线证明设备 kernel 是可改善的热点、现有兼容实现不足且用户允许修改其所属扩展时读取。适用于 Triton 或 Ascend C 候选，不承担环境安装、整机测试或训练配置搜索。公共接口与三仓接入仍按[算子与源码优化](operator-optimization.md)，预算、计时和状态沿用[共享测量规范](../ascend_training/measurement-and-records.md)。

## 1. 固定比较对象与实验边界

从热点分桶选实际工作负载覆盖的小、中、大形状及关键尾部；保存 dtype、stride/layout、输入分布、梯度需求和基线 callable。比较当前可用的兼容实现，不将任意 PyTorch 组合当作唯一基线。以关键路径份额估计训练收益上限，收益不足以抵偿维护成本时结束该方向。

在 `operator-candidate.md` 写入 `kernel_scope`、依赖版本、可用设备、候选次数和每次编译/运行时限，并预留正确性与最终验证预算。一个候选对应一个可解释假设；不默认开启无限 autotune 或把候选扩大到无关算子。遇到资源冲突、编译超时、设备错误或预算不足，保存已有证据并停止该 attempt，不重置共享设备或清理其他作业。

## 2. 核实目标能力

在已分配环境中记录实际 SoC/设备资源、CANN、PyTorch/torch_npu、Triton-Ascend 或 Ascend C 编译器及扩展修订；查当前安装文档、编译器目标信息或已有最小示例，确认使用的接口和可用资源。区分 AI Core/Vector/Cube 资源口径及 kernel 执行类型。

UB/L1/L0 容量、对齐要求、grid 映射、并行核数、累加精度、编译提示和双缓冲接口均按目标版本确认。不要把其他型号的固定容量、核数上限或某段示例 API 写成通用约束；也不因上游 Triton 存在同名接口就假设 Ascend 支持。没有事实支持的能力先记 `unverified`，用预算内最小编译/调用探针核对后再使用。

## 3. 根据证据选择改动

| 观察 | 最小实验 | 必须同时检查 |
| --- | --- | --- |
| 访存或搬运占关键时间 | 调整连续 tile、合并搬运、复用已载入数据 | 实际 stride、尾块、有效带宽；新增布局转换和分配的完整调用代价 |
| 计算有效工作不足或冗余高 | 先改算法/重复计算，再比较合法 block 形状与映射 | 累加误差、目标编译器支持、真实计算引擎指标 |
| 小任务 launch/同步主导 | 比较融合、每次任务工作量和依赖组织 | 是否扩大临时内存、减少有效并发或破坏流/事件顺序 |
| 编译资源超限或执行落差大 | 缩小 tile、缩短活跃区间、核对生成代码/编译报告 | 附加 index/mask、中间张量、精度转换及实际搬运粒度 |

先核对字段单位、采样范围和分母；缺失字段不是零。按核内/核间的数据独立性安排工作，记录尾核负载与同步需求。融合、缓存复用、多缓冲或调度重排只有在对应瓶颈及版本能力成立时成为候选，不能由固定利用率阈值自动触发。

## 4. 核算活跃量与极端输入

为每个 tile 列一张资源表：对象、形状、实际存储 dtype、对齐后大小、所在存储层、创建/最后使用位置、是否与其他对象同时活跃。至少包括输入/输出、中间结果、升精度量、累加器、offset/index/mask 和多缓冲副本。按每个执行阶段的同时活跃量求峰值，不把逻辑张量总大小误当物理资源占用，也不把所有存储层简单累加为 UB。

核算只是估计；用目标编译器的资源分配报告、溢出诊断和生成代码校准，记录估计与实际差异。向上对齐、tile 扩张或向上取整都会增加占用；选择候选后重新核算，不能沿用原尺寸结论。资源余量由实测确定，不套用统一百分比。

按算子语义覆盖：

- 零元素、单元素、非整除尾块、边界前后尺寸、最大真实展开 token 及跨整数索引边界的用例。
- 支持的非连续 stride/布局、合法别名或原地行为、可选输入缺省与存在、mask 全空/全满和有效索引范围。
- 真实 dtype 的极小/极大幅值、抵消、归约长轴；NaN/Inf、重复索引等仅在 API 契约涉及其行为时验证。
- 全部可微输入的前后向；mask、index、排列映射等离散输出按语义检查，不能仅用浮点近似比较。

attention/CP 候选还需固定布局、位置与 mask 的对应关系：记录每个 rank 的 Q/K/V 实际全局位置、分片及 gather 顺序，以及 causal/window/显式 mask 的行列含义。只有确认连续分片时，才能用 `rank × local_seq_len` 作为 query 起点；不能把连续位置 mask 直接套到 zigzag 双块分片。

先用自有小序列位置编号做纯 CPU 对照：按实际分片与 gather 顺序排列 Q/K，再以真实位置生成参考 mask（causal 时为 `k_pos <= q_pos`，其他约束按原契约叠加），逐项统计错放行和错屏蔽；保留 CP1/连续分片对照及源 hash、函数位置。helper 反例不能替代真实 backend 绑定和最终 mask 证据，也不能据此假定 K/V backward 缺少归约。[本机版本例证](../ascend_training/hardware-validation.md)说明了这类组合检查的必要性，不代表所有 CP 不支持或修复已验证。

先对齐参考数学行为再比较性能。精度与梯度容差沿用预先确定的项目标准；需要更高精度累加时核对其资源代价和基线语义。不支持的形状/dtype 在既有集中分派层确定性回退或按契约报错，不让越界或未知输入进入 kernel。

## 5. 分层验收并返回公共路径

通过正确性后，用相同输入和测量方法比较 kernel device 时间、完整 public-call 时间以及显存/临时分配。编译与 warmup 另记；布局转换、辅助张量展开和 copy 的代价纳入 public-call。不同候选保留可辨识的 kernel 名、配置和独立产物，防止 profiler 同名聚合混淆。

用实际公共 API 验证注册、参数转换和 autograd，确认训练绑定到目标实现。需要训练收益结论时，在剩余预算内完成独立训练 A/B；若只改善 kernel、完整调用变慢或训练收益未超过噪声，不报告训练加速。组合已验证 kernel 与其他补丁后仍需覆盖其交互的整体复测。

返回候选补丁、编译/资源报告、正确性及两层计时证据、未验证项与撤回条件。保持公共接口 fallback 可用；失败或超预算不自动转向更大重写。

## 参考经验与本地事实

以上流程为本项目独立编写的实验约束。[Ascend Triton 优化技能](https://github.com/Ascend/agent-skills/blob/master/skills/triton-operator-performance-optim/SKILL.md)及其[优化案例](https://github.com/Ascend/agent-skills/blob/master/skills/triton-operator-performance-optim/references/optimization-patterns.md)提供了分层计时、访存布局与隐含资源开销的研究线索；这里不继承其具体参数、固定容差或收益数值。网页案例不能证明本地三仓、设备型号或编译器具有相同能力，结论必须由本次工件支持。
