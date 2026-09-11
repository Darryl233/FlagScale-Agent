# 训练测量与正确性证据

本章说明吞吐、显存、分布式计时、数值对齐和 checkpoint 恢复的证据语义。实验执行、记录和分支由调优 SKILL 组织；这里的指标定义用于判断两次运行是否可比。历史案例只对注明的版本和工作负载成立。

## 目标与受控差异

实验前指定目标类别：吞吐改善、容量满足或可靠性改善。吞吐目标需收益超过预设最小值和噪声；容量目标需在真实配置与显存约束下完成，并满足预先声明的吞吐代价上限；可靠性目标需有对应长尾/故障/恢复改善证据且性能损失可接受。不能在看到结果后把吞吐失败改名为容量胜利，也不能要求所有容量优化都提高 token/s。

A/B 允许实验卡声明的配置、环境或源码变量及必要派生值变化，其余条件固定。配置候选保存 resolved/argv/env diff；源码候选额外保存仓库、接口、补丁及 hash。TP/MBS 改变时可同步调整 DP/累积数以保持 GBS，但不能顺带更改训练数学定义或数据起点。

比较指纹使用运行时实际形状，包括 tokenizer/special token 与填充后的词表，不能只比较输入 YAML 的 `vocab_size`。TP 可能改变词表填充与输出分片；即使 token 输入相同，也须先证明 logits、初始状态和更新可比。派生参数与目标版本例证见[候选约束](search-space.md)。

## 计时与吞吐口径

一个样本对应完整 optimizer step，包含该 step 的全部梯度累积、必要 collective 和 optimizer update。先检查日志计时器是否等待设备工作完成、是否只测 host enqueue、是否按多个 step 平均。不能把 microbatch 的 forward/backward 时间当作完整 step。

使用框架已有的低扰动计时；需要独立计时时，核对目标 NPU event/synchronize API，只在测量边界等待完成，不往每个 op 内加入同步以破坏 overlap。相同测量方法用于所有候选。

设有效窗口包含 N 个完整 step，每步耗时 `t_i` 秒、实际训练 token 数 `k_i`：

```text
window_tokens_per_second = sum(k_i) / sum(t_i)
tokens_per_second_per_npu = window_tokens_per_second / W
step_p50 = median(t_i)
step_p90 = p90(t_i)，固定分位数算法并记录
speedup = candidate_tokens_per_second / baseline_tokens_per_second - 1
step_time_reduction = 1 - candidate_step_p50 / baseline_step_p50
```

区分加速比例与耗时下降比例。不要先取每步 token/s 再做算术平均作为窗口吞吐，也不要混用 seconds/ms。

- 固定长度且每个 token 都计入训练工作时，可用 `k_i = GBS × seq_length`。GBS 已是全局批量，不能再乘 DP/TP/PP/EP/CP。
- packing、padding、变长文本、多模态或 loss mask 存在时，同时报告实际有效训练 token 和名义容量口径；从数据/训练日志获得计数并避免跨模型并行 rank 重复累计。
- 存在 padding/drop-token 变化的候选，先判断是否改变训练语义；不能只靠更高的名义 token/s 入选。
- 只知 p50 时可给出 `GBS×seq_length / p50` 的吞吐估计，但明确标注估计，不替代窗口真实吞吐。
- W 变化的实验另外列明资源规模和每 NPU 吞吐；没有同工作负载小规模参照时，不编造扩展效率。MFU 只有算量模型和本精度硬件峰值均可靠时才报告，MoE 需区分激活参数和总参数。

## 稳定窗口与 rank

固定采样规则，剔除编译、数据/优化器初始化、显式 warmup、checkpoint、evaluation 和 profiler step。记录每条剔除原因，不因某条计时慢而事后删掉它。重复编译、数据卡顿或通信长尾属于实际问题，应另报告或在生产窗口中保留。

启用了功能调用计数、执行 hook 或状态快照复制的诊断作业不参与稳定性能排名；在退出时移除 hook 不会消除运行期间的扰动。保留这类作业作为执行或质量证据，性能复测使用相同口径且关闭这些观测的独立作业。

普通训练计时排除周期性 I/O 以比较计算性能；最终生产验证另报包含真实 I/O 的 wall-clock/有效 token 吞吐。所有候选的窗口与筛选规则必须一致。

框架只由一个指定 rank 打印全局 step 时间时，用该权威日志并记录口径；不要拼接多个 rank 的重复行。各 rank 独立计时且覆盖同一 step 时，按 step id 取最大完整 step 耗时表示同步关键路径，再算统计量。不能对不同 rank 时钟的绝对时间戳直接求差。

核对 step id 单调、没有混入重启前日志、有效样本数达标、每个 worker 的完成信息可追溯。只收到 rank 0 的 loss 不是分布式任务完成证明。

普通 PP 的非末物理 stage 可合法返回空 loss；由实际 PP group、stage 位置和训练入口的 loss provider 语义确定哪些 rank 应有有限 loss，VPP 或特殊 schedule 不能只按全局 rank 猜测。仍须核对全部 rank 的成功更新、有限梯度证据、绝对 iteration/consumed progress 和最终退出；不能因非末 stage 无 loss 判整个作业失败，也不能因末 stage 有 loss 补判其他 rank 完成。

显存需明确 MiB/GiB 或 MB/GB，记录跨 rank 的 peak allocated、peak reserved 及可用容量来源。`npu-smi` 是设备级占用观察，不能直接当作某个训练进程的 PyTorch 分配峰值。精确 API 名称按安装版本确认；缺失不能填 0。显存峰值至少覆盖首次 optimizer step 与代表性训练阶段，不只测稳态计时窗口。

## 独立 A/B 与正确性

候选短跑阶段检查正常结束、有限 loss/grad、没有意外 skipped updates 或 backend 回退；这些是可行性检查，不是数值等价证明。

最终复测采用固定 checkpoint、数据起点与种子，交错运行 B0/C1/B0/C1，预算允许至少三对独立作业。以每次作业的窗口吞吐为统计单位，不将同一作业的 30 个相邻 step 当成 30 次独立重复。
容量任务的基线若已确定 OOM，引用既有失败证据并独立复测可行候选，不为凑 A/B 次数重复确定性失败；吞吐代价只能相对于明确的可运行参照计算，不能替 OOM 基线填写吞吐。

吞吐目标可先取成对吞吐比的中位数，并报告各次运行的范围。验收门槛同时参考用户规定的最小收益和重复基线波动；结果接近噪声时优先保持基线或按预算补测。容量/可靠性目标按上述约束判断，同时报告吞吐代价。三对结果只提供有限置信度，不能声称统计显著，除非采用了适当检验和足够样本。

对应的正确性证据：

| 变化 | 必要检查 |
| --- | --- |
| 仅配置、保持数学定义 | 相同逻辑 batch/初始状态的 loss、grad norm 和更新结果，按项目容差比较；布局改变需正确重分片对齐 |
| 通信 overlap/调度 | 多个真实更新的完整性、梯度归约/累积结果，无新增 hang/丢更新 |
| TE/Ascend backend 或融合算子 | 实际公共接口的前向和全部可微输入梯度；dtype/shape/layout/mask/空 tensor 等受影响边界 |
| 重计算 | RNG/dropout 语义、梯度一致性、优化器更新和额外数值误差 |
| 跨布局 `model_only` 质量 | 初始与更新后模型参数按真实分片对齐，同逻辑 batch 的 loss、grad norm 和更新差量符合预设门槛；不代表 optimizer/RNG 恢复 |
| 完整续训或重分片恢复 | 模型/optimizer/scheduler 状态、RNG、consumed samples 与绝对更新进度的加载、保存和恢复 |

短跑恢复时同时核对 LR 和 weight decay 调度状态。部分 Megatron 版本即使采用 constant weight decay，
仍由 `train_iters × GBS` 派生 `wd_incr_steps`；只固定 `lr_decay_iters`，修改结束步数后仍可能导致 checkpoint 加载断言。
若契约要求沿用保存的调度，核实后显式使用当前版本的 `use_checkpoint_opt_param_scheduler`，
并检查加载后的 LR、weight decay、调度进度；不能用 override 静默重置它们。

记录 checkpoint loader 实际恢复了哪些组件，不能只读请求中的 `no_load_rng`。在[本机版本验证](hardware-validation.md)中，跨 TP/PP 的加载路径会忽略 RNG；这类候选的冷启动或加载成功可作为兼容性结果，模型对齐检查通过后可单独标注 `model_only` 质量通过，不能标记完整续训成功。`model_only` 未检查或未恢复的 optimizer、scheduler、数据进度和 RNG 均保持未验证；若本次目标要求完整恢复，局部质量通过不能取消该门禁。

需要验证完整 checkpoint 恢复时，比较“连续训练到 N”与“训练到 K、保存退出、恢复到 N”的同一逻辑边界。
核对每个 rank 的绝对更新序列、consumed samples 和 optimizer 步数，而不只比较两次运行的计数是否相等。
按组件检查 Python、NumPy、torch CPU、当前 NPU 与实际使用的并行 RNG tracker；没有采集到的状态不能视为相等。
浮点权重的数值容差不适用于要求精确恢复的 RNG 状态、离散计数或配置。任一必需 RNG 组件不一致时，
即使短跑权重完全相同或 dropout 为 0，恢复门禁仍失败；沿保存、加载、数据迭代器初始化和首个更新定位差异，不能事后取消门禁。

容差与检查方法须在看结果前定好，参考项目 BF16/FP16 测试。对存在合理非确定性的路径，比较统计量与误差界限；不能对整段训练随意要求 bitwise 相等，也不能用“loss 大致下降”代替验证。
模型参数的绝对容差若大于一次更新量，仅比较更新后权重可能漏掉优化器错误；同时检查初始状态一致、参数实际发生更新、
更新差量及 optimizer 主参数/动量、步数与 scheduler 状态。用实验前约定的差量误差门槛补充参数 allclose，不事后放宽。
optimizer 比较同时覆盖状态结构、非 tensor 标量、计数和超参数；只遍历 tensor 会漏掉部分实现中的整数 step 或学习率。

采集 `ChainedOptimizer` 等组合优化器时，先遍历全部子优化器，包含各自的本地状态、FP32 主参数和动量；
不要读取只适用于单个子优化器的便利属性。只有明确记录 `params=[]`、确认没有对应状态的空组，
才可要求其计数保持不变；所有活动组仍须满足绝对更新进度。缺失状态不能推断为空组。
观测器出错时单独记录工具故障；修复后使用新 attempt 和快照 schema，不覆盖原失败或混用不兼容 schema。
离线比较优先受限 CPU 反序列化；遇到实际 RNG 类型时只允许已检查的固定重建类型，
未知对象应报错，不能自动回退为无限制加载。

profiling 先用短窗口的 CPU/NPU 时间线、operator 耗时和通信/内存数据，必要时再单独开 stack/shapes 等高开销项。在实际训练循环每个选定 step 调用 profiler 的推进接口，并确保 profile 覆盖梯度累积和 optimizer；调用参数按本版 API 核实。输出独立保存，最多覆盖能定位关键路径的 rank/step，避免占满磁盘。
分别记录全部训练 worker 的正常退出和所选 rank 的有效采集产物；前者不能补齐未采集 rank 的时间线。
例如 16 个 worker 均完成、仅 rank 0/15 有 profile，只证明该训练作业完成及这两个 rank 的采集有效，不能据此判定全部 rank 的通信重叠或最慢阶段。

