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

# Ascend kernel 的资源、正确性与性能依据

Triton 或 Ascend C 实现的训练收益受热点占比、硬件资源、输入语义和公共调用成本共同约束。公共接口与三仓接入见[算子调用链与实现判断依据](operator-optimization.md)，训练计时口径见[共享测量依据](../ascend_training/measurement-and-records.md)。

## 比较对象与收益上限

代表性输入包括实际工作负载的小、中、大形状及关键尾部，并由 dtype、stride/layout、输入分布和梯度需求共同定义。相同数学函数的不同公共 callable 可能有不同的输入整理与调度成本，当前可用的兼容实现比任意 PyTorch 组合更适合作为性能参照。

关键路径中可移除的 kernel 时间约束训练收益上限；非关键路径上的提速可能被已有重叠完全隐藏。自动搜索的编译和测量开销也属于优化成本，搜索空间由目标瓶颈、合法输入和设备能力决定。

## 目标能力与版本边界

实际 SoC/设备资源、CANN、PyTorch/torch_npu、Triton-Ascend 或 Ascend C 编译器及扩展修订共同决定可用接口。安装文档、编译器目标信息和最小调用结果的适用范围均受版本限制。AI Core、Vector、Cube 的资源口径与 kernel 执行类型需要分别解释。

UB/L1/L0 容量、对齐要求、grid 映射、并行核数、累加精度、编译提示和双缓冲接口均依赖目标版本。其他型号的固定容量、核数上限或示例 API 不是通用约束；上游 Triton 存在同名接口也不等于 Ascend 支持。文档描述、编译通过和实际调用可用分别证明不同层次的能力。

## 瓶颈与优化机制

| 观察 | 可能有效的改动 | 同时受哪些条件约束 |
| --- | --- | --- |
| 访存或搬运占关键时间 | 调整连续 tile、合并搬运、复用已载入数据 | 实际 stride、尾块、有效带宽；新增布局转换和分配的完整调用代价 |
| 计算有效工作不足或冗余高 | 先改算法/重复计算，再比较合法 block 形状与映射 | 累加误差、目标编译器支持、真实计算引擎指标 |
| 小任务 launch/同步主导 | 比较融合、每次任务工作量和依赖组织 | 是否扩大临时内存、减少有效并发或破坏流/事件顺序 |
| 编译资源超限或执行落差大 | 缩小 tile、缩短活跃区间、核对生成代码/编译报告 | 附加 index/mask、中间张量、精度转换及实际搬运粒度 |

字段单位、采样范围和分母决定指标含义，缺失字段不是零。核内/核间的数据独立性决定可并行程度，尾核负载与同步需求可能限制扩展。融合、缓存复用、多缓冲或调度重排的收益依赖对应瓶颈及版本能力，不能由固定利用率阈值推断。

## 资源活跃量与极端输入

每个 tile 的资源占用由对象形状、实际存储 dtype、对齐后大小、所在存储层及创建到最后使用的活跃区间决定。输入/输出、中间结果、升精度量、累加器、offset/index/mask 和多缓冲副本都可能计入峰值。峰值是每个执行阶段同时活跃对象的占用，既不等于逻辑张量总大小，也不能把所有存储层简单累加为 UB。

核算只是估计；用目标编译器的资源分配报告、溢出诊断和生成代码校准，记录估计与实际差异。向上对齐、tile 扩张或向上取整都会增加占用；选择候选后重新核算，不能沿用原尺寸结论。资源余量由实测确定，不套用统一百分比。

容易改变正确性或资源占用的输入边界包括：

- 零元素、单元素、非整除尾块、边界前后尺寸、最大真实展开 token 及跨整数索引边界的用例。
- 支持的非连续 stride/布局、合法别名或原地行为、可选输入缺省与存在、mask 全空/全满和有效索引范围。
- 真实 dtype 的极小/极大幅值、抵消、归约长轴；NaN/Inf、重复索引等仅在 API 契约涉及其行为时验证。
- 全部可微输入的前后向；mask、index、排列映射等离散输出按语义检查，不能仅用浮点近似比较。

attention/CP 的正确性依赖布局、位置与 mask 的对应关系，包括每个 rank 的 Q/K/V 实际全局位置、分片及 gather 顺序，以及 causal/window/显式 mask 的行列含义。只有确认连续分片时，才能用 `rank × local_seq_len` 作为 query 起点；连续位置 mask 不能直接套到 zigzag 双块分片。

小序列位置编号可构造独立的 CPU 数学参照：按实际分片与 gather 顺序排列 Q/K，再以全局位置定义参考 mask（causal 时为 `k_pos <= q_pos`，其他约束按原契约叠加）。逐项比较错放行和错屏蔽，并与 CP1/连续分片对照，可区分位置映射和掩码语义问题。简化 helper 的反例不证明真实 backend 使用了相同布局或最终 mask，也不能据此推断 K/V backward 缺少归约；这些结论分别依赖实际绑定、传入张量和梯度通信路径。

性能可比的前提是参考数学行为一致。精度与梯度容差由模型数值要求决定，更高精度累加同时改变资源代价。不支持的形状/dtype 需要由集中分派层确定性回退或按契约报错，越界或未知输入不能进入不具备相应处理能力的 kernel。

## 分层性能与集成正确性

相同输入和测量方法下，kernel device 时间、完整 public-call 时间以及显存/临时分配反映不同层次的成本。编译与 warmup 不属于稳态执行；布局转换、辅助张量展开和 copy 属于 public-call。Profiler 按同名 kernel 聚合时可能混淆不同配置，调用身份与输入范围决定统计是否可比。

注册、参数转换和 autograd 的正确性取决于训练实际使用的公共 API，而不只是私有 kernel 调用。Kernel 变快但完整调用变慢，或训练差异未超过测量噪声，都不足以证明训练加速。多个独立有效的改动可能在布局、分派、内存生命周期或流依赖上相互影响，组合收益和正确性不能简单相加。

公共接口的 fallback 保证未覆盖输入遵守原有语义。编译资源估计、数值误差、分层耗时和目标版本共同界定实现的适用范围，单一收益数字不能推广到其他设备型号、编译器或工作负载。
