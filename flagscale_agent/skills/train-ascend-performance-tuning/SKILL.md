---
name: train-ascend-performance-tuning
description: 在 FlagScale、Megatron-LM-FL、TransformerEngine-FL 的昇腾训练栈上建立基线、生成候选、运行比较并复测交付。用于训练自动调优、配置性能优化或接续已有调优实验；以一个主循环接入不同调优方法。
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 昇腾训练调优

## 输入与完成条件

从实际训练配方、启动命令和已有日志开始。读清目标指标、可改范围、训练语义、设备与时间预算，
沿用用户已有约定，只补齐阻塞当前步骤的信息。记录实际加载的三仓路径、版本及原有修改。
完成条件是交付可复现的候选与复测结论；没有有效改进时保留基线。只要求建议或分析时不启动训练。

## 执行流程

`准备比较 → 建立基线 → 生成一个候选 → 运行并比较 → 继续或复测交付`

### 1. 准备比较

先 `plan_status()` 接续当前计划；没有相关计划且需要多步执行时才 `plan_create`，方法和采集共用同一计划。
通过 `memory_list / memory_read` 按当前环境、模型查已有事实与经验，并核对适用版本。
使用 `read_file` 读取原配方/已有证据，`write_file` 保存本轮配置、命令、环境、目标和预算，约定计时窗口、质量检查及容差。
固定模型、数据、GBS、精度、优化器和可比的训练起点；本次允许改变的语义另建比较组。
复用已有实验记录，或用 `write_file(mode="append")` 维护一份 `experiments.jsonl`；后续方法共用它。
用 `plan_update` 的 notes 记录文件位置、当前阶段和下一步，verification 指向验收证据；
具体字段、恢复与工具调用见 [Agent 工具复用](references/agent-tools.md)。
按比较条件准备质量检查，已有有效对齐证据直接复用。
**得到**可执行的比较条件。仅静态建议可直接进入步骤 3，明确基线尚未实测。

### 2. 建立基线

使用当前已验证的 launcher 完成原配方短跑，再取得稳定计时窗口；初始化/预热单列。
运行和日志检查复用下方统一约定中的 `shell / shell_jobs / flagscale_train_monitor`。
保存完整更新耗时、吞吐口径、显存、质量检查、全部 worker 完成状态和原始日志。
同工作负载、环境与测量口径的有效基线可复用。基线失败先定位原因；容量任务的 OOM 可带证据进入显存方法，
取得首个可行配置后再建立计时基线。其他失败或计时不稳时先解决问题，暂不排名。
**得到** baseline 及其测量规则，后续候选使用相同规则。运行约定见下文。

### 3. 生成一个候选

按已有证据选择下表的一种方法，仅阅读选中的方法；基础批量调优从 [micro-batch 方法](methods/micro-batch.md) 开始。
方法返回 `parent / hypothesis / change / checks`（父配置、假设、完整变更、额外检查），写入同一实验记录。
复制父配方生成独立候选，解析 defaults/覆盖到实际 argv，保存完整差异；
确认变化符合范围且参数被当前 NPU 路径消费，静态无效则记录原因并换候选。
**得到**独立候选与检查状态。仅建议任务在此交付差异和待验证项；框架/入口缺失时明确尚未解析，
只给已核实的运行方式或具体缺口，不补造 argv。

### 4. 运行并比较

先短跑检查所有 worker、完整更新、有限 loss/梯度与方法要求的质量证据；
按既定条件评估质量，通过后再做独立性能测量。
比较同一规则下的完整训练更新，记录显存代价；profiler 或状态快照作业不进入性能排名。
每次用 `write_file` 追加候选变更、命令、日志、运行/质量状态、指标与保留/回退理由，失败结果也保留。
用 `plan_update(action="step_doing", ...)` 追加当前尝试的结论与下一步；完整原始结果留在实验文件中。
**通过且有改善**：暂留候选，按方法反馈选择下一个候选；**失败或无收益**：保留当前有效配置，换候选或结束该方法。
缺少质量证据或有效计时只记待验证，不以最快数字直接采纳。下一轮仍复用本循环。

### 5. 复测并交付

没有可测试候选、达到目标或预算不足以继续并收尾时结束搜索。
对最终候选和原基线做独立、交错复测，核对同一起点的更新质量与性能波动；按步骤 1 的验收条件判断。
源码候选按各自版本/补丁分别运行并核对加载路径，基线复测使用原代码状态，保留用户原有修改。
任务没有既定次数时以三对为起点；预算不足或结论受波动影响时交付候选及验证缺口。
通过本次质量与收益要求才推荐采用；失败、无收益或证据不足时保留基线，说明原因。
完整续训、跨布局状态对齐或生产验证在任务/方法需要时追加，不把短跑结果写成这些结论。

**统一运行约定**：每次启动前核对已分配设备可用、剩余预算和退出方式，使用独立运行目录。
用 `shell(command=..., background=True)` 启动作业并取得实际 job id，紧接着调用
`flagscale_train_monitor(output_dir=..., mode="check")`，再保存 job 信息并用 `shell_jobs` 接续状态。
远程日志不可达、实际 worker 和全 rank 完成按 [工具边界](references/agent-tools.md) 核对。
单次期限由实际 launcher 控制，等待超时不等于训练停止；结束后确认全部自有 worker 退出，重试新增记录。
复用 native tuner 时先核对生成/停止行为并关闭自动 `run_best`；它的结果仍经过步骤 4–5。

## 按需扩展

已有 `train-config`、`train-run`、`train-monitor`、`train-parallel-strategy`、`train-precision-alignment`
保持原样，仅按当前问题参考适用内容；不假设它们已提供统一 candidate/attempt 接口或完成 NPU 适配。
实际执行以本流程、当前工具 schema 和目标源码为准，旧工具名或 CUDA 示例不能直接套用。
环境、数据、拓扑、模型移植与复现只在任务需要时接入；原有 SKILL 的修订等实际测试发现问题后再针对处理。

| 触发条件 | 接入口 | 返回给主循环 |
| --- | --- | --- |
| 基础批量调优，有可比较基线 | [批量](methods/micro-batch.md) | MBS 候选与批量/数值检查 |
| 分片、拓扑映射、stage 或 expert 负载需要调整 | [并行布局](methods/parallelism.md) | TP/PP/CP/SP/VPP/EP/ETP 候选及对齐检查 |
| 已定位显存压力，或重计算代价过高 | [显存与重计算](methods/memory.md) | 重计算/状态分片候选及容量、质量检查 |
| 同步等待突出，或 overlap 引入退化 | [通信与重叠](methods/communication.md) | overlap/bucket/dispatcher 候选及依赖检查 |
| 数据等待、host 开销或已有执行路径可优化 | [数据与执行路径](methods/execution.md) | 数据供给/后端/图模式候选及专项检查 |
| 瓶颈不明，需要采集或分析 profile | `load_skill(name="train-ascend-profiling")` | 原始证据与待验证假设，返回步骤 3 |
| 热点已定位且任务包含源码优化 | [算子方法](methods/operator.md) | 补丁和公共接口测试，返回步骤 4 |

并行布局可作为调优变量；首个 micro-batch 方法保持布局不变，只是该方法的范围。
这些入口组织候选生成方式，不表示当前环境已支持全部选项。方法不足时返回缺口；不自动遍历整张表。
扩展方法不另建预算、运行器、排名或验收流程。新增方法使用 [四段模板](methods/TEMPLATE.md)，再补此表入口。
需要领域依据时先用 `load_knowledge(name="know-ascend-training")` 查询索引，再用同一 name 和索引中的 doc 路径按段读取。
参数约束、栈接入和测量分别查 `ascend_training/search-space.md`、`ascend_training/stack-capabilities.md`、
`ascend_training/measurement-and-records.md`；不默认加载全部优化知识或历史实验。

## 交付

按实际范围交付配置/补丁、可取得的复现命令、一份实验记录和 `report.md`；静态任务缺少完整入口时说明命令生成条件。
报告回答：与原基线相比改了什么、实际结果与质量证据、是否建议采用、还没验证什么。
导出时去掉采集/搜索专用覆盖并重新解析，保留回退配置；未通过验收的候选明确标注待验证或失败。
使用 `write_file` 落盘，`plan_update` 完结本次负责的步骤并附 verification；整个当前计划完成后才将其 complete。
有跨会话价值且已经核实的结论再经 `memory_list / memory_read` 查重后 `memory_write`；
带版本、证据与复核方式，每次 attempt 和待测状态继续保留在计划/实验文件。
