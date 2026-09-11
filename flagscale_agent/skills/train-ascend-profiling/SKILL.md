---
name: train-ascend-profiling
description: >-
  按工作流生成 torch_npu profiler wrapper、采集 FlagScale 昇腾训练 profile，
  或分析已有 NPU 性能数据，交付带原始证据的瓶颈报告与下一步实验。
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 昇腾训练 profile 采集与分析

## 输入与完成条件

记录要回答的问题、输入位置及已知的 run/step/rank 和版本。复用已有实验目录与资源约定，按本次任务选择终点：

| 任务 | 必要输入 | 执行范围与完成条件 |
| --- | --- | --- |
| 只生成 wrapper | 可读取的原训练入口、目标输出位置 | 步骤 1；交付独立入口、配方改动及命令，不启动训练 |
| 采集 profile | 可运行配方/命令、目标窗口与 rank、已有设备和预算约定 | 步骤 1–3；真实 NPU 产物验收通过，全部训练 worker 已收尾 |
| 分析已有数据 | CSV、trace、数据库或 PROF 目录 | 步骤 3–4；交付带原始依据的观察、假设及缺口，不启动训练 |

采集且分析时继续完成步骤 4。只有日志时先标出慢 iteration 与建议采集范围；任务不含采集则交付方案。
缺少当前步骤必需的材料时交付具体缺口，不要求补齐其他分支的输入。
用 `plan_status` 接续当前任务，沿用主调优计划；独立多步任务没有活动计划时才用 `plan_create` 建立计划。
进度、文件读写和实验记录复用 [Agent 原生工具约定](../train-ascend-performance-tuning/references/agent-tools.md)。
只生成或分析已有数据不创建训练 attempt；实际采集才向同一实验记录追加 `stage=profile` 的 attempt。

## 执行流程

### 1. 生成独立入口

按照 [wrapper 操作说明](references/wrapper-generation.md) 核对入口和生成参数。
`SKILL_DIR` 指向本技能目录；以下路径替换为本次实际位置：

```bash
python "$SKILL_DIR/scripts/generate_profile_wrapper.py" \
  --entrypoint "$TRAIN_ENTRYPOINT" --output "$RUN_DIR/train_npu_profile.py" \
  --profile-output "$RUN_DIR/npu-profile" \
  --wait 3 --warmup 1 --active 1 --ranks 0 --level Level1
```

生成器必须能读取原入口；只有不可访问的远端路径时，交付目标环境的待执行命令，明确 wrapper 尚未生成，不用占位文件绕过检查。
已核实实际 `megatron.training.training` 路径时，用 `--training-file` 加入运行时保护，不按仓库名称猜路径。
**得到** wrapper 和采集配方改动。只生成任务在此交付；采集任务继续步骤 2。

### 2. 启动有界采集

核对目标 Python、实际训练模块、设备和剩余预算；记录 global ranks、wait/warmup/active 与训练退出条件。
把 wrapper 配置到本次专用配方，沿实际 resolved/argv 关闭内置 profiler。
保留模型、数据和分布式策略，只修改本次采集与短跑需要的项。
使用原 launcher 执行这一次 `stage=profile` 运行，保存配方、窗口、设备、硬期限、退出条件和共享记录位置。
按原生工具约定用 `shell(background=True)` 启动，紧接着 `flagscale_train_monitor(mode="check")`，
再记录 job/目标端身份并用 `shell_jobs` 跟踪；远程日志按工具约定读取，另核对全部所属 worker 收尾。
取得实际命令、全 worker 日志与退出证据后继续验收。
wrapper 按正常返回的 `train_step` 推进窗口，**窗口结束不会结束训练**；用配方和 launcher 控制训练退出。
**得到**采集产物及全部 worker 日志，进入步骤 3。失败或窗口不足时保留证据并只收尾本作业；
父调优调用时返回缺口，由主循环决定新 attempt；独立采集任务才按既定预算和查明的原因安排重试。

### 3. 盘点并验收产物

对输入目录执行只读盘点：

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" inventory /path/to/profile
```

用 `read_file` 读取可访问的文本产物，用 `write_file` 保存验收结果、文件清单、实际 headers 和读取限制；
大文件按段读取，目录深度或数量截断时继续盘点相关子目录。二进制产物仍由上述脚本和匹配的导出工具读取。
对新采集作业分别检查 **全部训练 worker** 的退出和更新记录，以及 **所选采样 rank** 的
`wrapper-status.json`、入口终态、schedule/返回次数、导出回调和真实 NPU 事件。
已有数据按可取得的元信息核对来源与范围，不要求补造 wrapper 状态。记录 rank/device 映射、时间戳、目标窗口及原始路径；
采集通信问题时另查通信明细。目录存在、回调返回或 rank 0 完成均不能单独证明采集成功。
**得到** `profile-validation.json`，列清有效范围与缺失项。只采集任务在此交付；需要分析且数据可用时进入步骤 4。
缺字段不补零；只有 db/raw 或字段不支持时走下方导出入口，无法转换则交付现有证据和缺口。

### 4. 提取窗口事实并提出验证实验

从 step 标记或已有时间范围确定实际窗口，核对单设备时钟域后执行；下列数值仅示意参数用法：

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" window /path/to/kernel_details.csv \
  --start-us 1000000 --end-us 1100000 --device-id 0
```

检查窗口 JSON 的 `status`、设备范围、坏行和截断；超限时导出目标范围并保留跨边界任务。
从未覆盖窗口、算子分桶或 rank/stage 差异选择最需要解释的项，回到原始行和相关时间线核对。
每项写明“观察事实 → 候选原因 → 尚缺证据 → 最小验证”；host 归因先核对 host/device 对应关系，缺关联时保留未知。
`uncovered` 仅表示记录任务未覆盖；累计算子时长不能直接当关键路径收益。完整更新、microbatch 与重计算边界按实际标记核实。
**得到**窗口 JSON 和有证据出处的瓶颈报告；不能确定原因时交付假设及下一步所需证据。

## 按需扩展

| 当前缺口 | 接入口 | 取得的结果 |
| --- | --- | --- |
| wrapper 参数、入口绑定或状态验收细节 | [wrapper 操作说明](references/wrapper-generation.md) | 可执行生成/采集命令或具体兼容性缺口 |
| 采集配方或实际运行 | 当前配方/launcher 与 [原生工具约定](../train-ascend-performance-tuning/references/agent-tools.md) | 有效配置、一次有界运行及全部 worker 证据 |
| 格式、时钟或归因不明；需要转换 db/raw | `load_knowledge(name="know-ascend-profiling")` 查询索引，再按段读取 `ascend_profiling/collection-and-analysis.md` | 输入契约、已安装且版本匹配的导出工具及分析依据 |
| 完整更新、吞吐或质量口径不明 | `load_knowledge(name="know-ascend-training")` 查询索引，再按段读取 `ascend_training/measurement-and-records.md` | 本次窗口与指标口径 |

由调优任务调用时，把原始证据、假设和缺口返回 [主调优流程](../train-ascend-performance-tuning/SKILL.md) 的候选生成步骤。
主流程选择方法并验证收益；profile 耗时不进入性能排名。本技能不另建候选运行和排名循环。

## 交付

按实际执行范围交付，不要求所有任务生成同一套文件：

- **只生成**：wrapper、配方差异和启动命令；入口不可访问时交付待执行生成命令，注明文件尚未生成。
- **采集**：原始产物位置、复现命令、文件清单和 `profile-validation.json`，分别说明全部 worker 收尾及所选 rank 的验收结果。
- **分析**：窗口 JSON（数据可解析时）与 `bottleneck-report.md`，包含原始路径/行号、版本、run/rank/窗口、观察与假设、缺口和最小验证实验。

采集失败或只能分析部分数据时仍交付已有产物及限制，不将待执行命令、目录存在或局部证据写成采集成功。
有活动计划时，将结果、证据路径和下一步写入 `plan_update` 的 `notes`，完成相应步骤时填写 `verification`；
由主调优流程调用时只更新对应步骤，不结束整个计划。只有值得跨会话复用的已核实结论才经 `memory_list`/`memory_read` 查重后用 `memory_write` 保存，逐次采集记录留在实验文件。
