<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# FlagScale 启动命令与参数

在已有环境和原配方的工作目录执行，配置文件名任意。Agent 需要等待训练完成时使用前台运行：

```bash
PYTHONUNBUFFERED=1 flagscale train -c /absolute/path/recipe.yaml --test
```

已有可用命令就复用。CLI 要求 MODEL 时，在 `train` 后添加该配方的模型名称；遇缺少参数或参数不支持的错误，再查 `flagscale train --help`。

| 参数或设置 | 作用 |
| --- | --- |
| `-c <yaml>` / `--config <yaml>` | 完整 FlagScale 配方，含 `experiment`、`train` |
| `--test` | 前台运行真实训练，不缩短迭代数；省略时可能后台提交并提前返回 |
| `train.model.train_iters` | 训练迭代数；短跑在独立 YAML 中设置 |
| `experiment.runner.nproc_per_node` | 每节点训练进程数，与本次设备分配一致 |
| `experiment.exp_dir` | 本次独立输出目录，用于定位日志 |

Agent 用 `shell(background=True)` 执行，保存 job id，再用 `shell_jobs(action="wait", job_id=..., timeout=60)` 等待。
结束后通过 `flagscale_train_monitor(output_dir=<exp_dir>, mode="check")` 检查本次训练日志，并确认 worker 已退出后再复用设备。

按需将 `--test` 替换为 `--dryrun`（只生成脚本）或 `--stop`（停止该配方作业，先确认属于本任务）；这些不是正常启动的附加步骤。

自动调优需要单次超时和测量摘要时，使用 [单次运行说明](single-run.md)。
