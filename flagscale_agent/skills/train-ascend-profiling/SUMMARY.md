<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 昇腾训练 profile 采集与分析

**加载时机**：需要生成 torch_npu profiler wrapper、采集 FlagScale 训练 profile，或分析已有 NPU 数据。

**工作流**：根据任务选择起点和终点：只生成完成入口后交付；采集完成有界短跑与产物验收后交付；分析已有数据从盘点开始，再提取窗口事实和提出验证实验。

**产物**：只生成交付 wrapper、配方差异和命令；采集交付原始产物与 `profile-validation.json`；分析交付可解析的窗口结果、瓶颈报告及缺口。由调优任务调用时，将证据与假设返回主流程，不另建实验循环。

**原生能力复用**：沿用当前计划，以 `plan_update` 回写进度和验收；文本产物用 `read_file`/`write_file`，采集共享实验记录并标 `stage=profile`，通过 `train-run` 选择有界单次运行或直接 CLI 路径，沿其启动与监测方式执行，另验全部实际 worker 收尾。记录与工具边界见 [Agent 原生工具约定](../train-ascend-performance-tuning/references/agent-tools.md)，不依赖改写后的通用 SKILL。

**按需知识**：`load_knowledge` 先加载 `know-ascend-profiling` 索引，再按章节读取格式、兼容性与归因依据；测量口径来自 `know-ascend-training`。脚本用法保留在 [wrapper 操作说明](references/wrapper-generation.md)。

已有数据分析不启动训练；真实 NPU 产物通过验收后才能报告采集成功，profile 耗时不参与训练性能排名。
