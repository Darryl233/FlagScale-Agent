---
name: train-ascend-profiling
description: >-
  按工作流生成 torch_npu profiler wrapper、采集 FlagScale 昇腾训练 profile，
  或分析已有 NPU 性能数据，交付带原始证据的瓶颈报告与下一步实验。
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 昇腾训练 profile 采集与分析

复用当前配方、运行记录和资源约定，先明确本次要回答的问题：仅生成 wrapper、采集 profile，或分析已有产物。只执行对应步骤；调优调用时沿用主计划，不另建调优循环。

## 1. 生成 wrapper

按 [wrapper 操作说明](references/wrapper-generation.md) 使用目标环境的原训练入口；`SKILL_DIR` 是本技能目录。

```bash
python "$SKILL_DIR/scripts/generate_profile_wrapper.py" \
  --entrypoint "$TRAIN_ENTRYPOINT" --output "$RUN_DIR/train_npu_profile.py" \
  --profile-output "$RUN_DIR/npu-profile" \
  --wait 3 --warmup 1 --active 1 --ranks 0 --level Level1
```

默认窗口需要 5 次正常 `train_step` 返回；实际 wait 依据已知预热情况设置。已核实训练模块文件时可加 `--training-file`，不要按仓库名猜测。
仅生成任务交付 wrapper、配方差异和命令后结束。原入口不可访问时交付待执行命令，明确文件尚未生成。

## 2. 采集

复制独立配方，将 `experiment.task.entrypoint` 指向 wrapper，按操作说明关闭内置 profiler，保留工作负载及分布式策略。
通过已加载的 `train-run` 启动；支持的单机短跑使用其 [有界执行](../train-run/references/single-run.md)。在同一实验记录中保存本次 `stage=profile`、配置、global ranks、窗口、日志与退出证据。

wrapper 结束采集窗口后训练仍继续，由配方迭代数和 launcher 期限结束训练。确认全部自有 worker 收尾；失败返回原始错误与产物，由调用方决定重试。

## 3. 盘点与验收

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" inventory "$PROFILE_DIR" > "$RUN_DIR/profile-inventory.json"
```

检查摘要和读取限制；若目录盘点截断，只对相关子目录继续。新采集任务核对所选 rank 的 `wrapper-status.json`、完整入口终态、采集窗口与真实 NPU 事件；目录存在或回调返回不是完整采集证据。已有产物只使用其实际元信息，不补造 wrapper 状态。

`inventory` 只识别文件和 CSV 表头，不转换 DB、raw PROF 或 trace。直接窗口分析支持逐任务 CSV 的 `Start Time(us)`/`Duration(us)` 或 `Task Start Time(us)`/`Task Duration(us)`，以及非空 device 列。只有其他格式时，复用已有验证的导出方法；缺少方法则返回格式缺口，不临时编造命令。

仅采集任务交付原始产物、命令及已验收范围后结束。

## 4. 分析目标窗口

从实际 step/时间标记取得同一设备的窗口，替换下面的示意数值：

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" window "$KERNEL_CSV" \
  --start-us 1000000 --end-us 1100000 --device-id 0 > "$RUN_DIR/profile-window.json"
```

默认最多读取 8 MiB / 100000 行；输入超限时可在资源预算内显式设置 `--max-bytes`、`--max-rows`，或使用已有的完整窗口导出。不能截取 CSV 开头后宣称覆盖了整个目标窗口。
检查 JSON 的 `status`、设备范围、坏行与截断，再对照原始任务解释重点差异。`uncovered` 表示没有记录任务覆盖，算子累计时长也不是关键路径贡献。

每项只记录“观察事实、候选原因、缺少的证据、最小验证”。统计或时钟关系不明时，按需读取 `know-ascend-profiling` 的 `ascend_profiling/collection-and-analysis.md`；训练指标定义读 `know-ascend-training` 的 `ascend_training/measurement-and-records.md`。

交付原始路径、窗口 JSON、观察与待验证假设；返回主调优流程决定候选。带 profiler 的运行时间不进入无 profiler 的性能排名；部分产物或未知原因保持其实际证据范围。
