<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 单次 FlagScale 训练

用于单机 Megatron、SSH runner 的本地执行（`nnodes: 1`、`hostfile: null`、`backend: torchrun`）。
先确认当前 FlagScale 的 `--test` 映射到 `runner.run(background=False)`，且本地 launcher 等待 worker。
这里的 `--test` 是前台运行方式；迭代数、数据、训练参数仍来自 YAML。其他版本或多机任务不要直接套用。

每次准备独立完整 YAML，保留原配置的位置关系与训练起点；只有 MBS 和运行产物路径允许随试验改变。
将 `experiment.exp_dir` 设为明确的绝对新目录，例如 `/workspace/experiments/mbs-baseline-01`，不要用时间插值或复用旧目录。
使用默认日志目录，不覆盖 `train.system.logging` 的路径字段。运行前确认 checkpoint 保存路径也独立，加载路径仍为同一起点。
辅助模块不查设备空闲、不分配 NPU；使用前必须核对已授权设备、环境与剩余预算。

准备请求 JSON（以下数值是示例，按当前测量约定替换）：

```json
{
  "config_path": "/workspace/configs/baseline.yaml",
  "cwd": "/workspace/FlagScale",
  "model": "gpt",
  "timeout_seconds": 300,
  "expected_ranks": 16,
  "run_id": "baseline-01",
  "role": "baseline",
  "first_iteration": 1,
  "end_iteration": 40,
  "warmup_steps": 10,
  "global_batch_size": 64,
  "sequence_length": 512
}
```

`model` 仅在 CLI 要求 MODEL 时提供已确认名称。YAML 文件名任意；配置、cwd 都用绝对路径。
在已安装 FlagScale-Agent、配置好三仓 PYTHONPATH/CANN 的环境中，用现有 shell 工具后台执行：

```bash
python -m flagscale_agent.training_run --request /workspace/configs/baseline-01.json
```

辅助模块内部执行 `flagscale train [model] -c <原始配置> --test`，不搬动启动配置，不改训练入口。
记下 shell 返回的 job id，随后用 `shell_jobs(action="wait", job_id=..., timeout=60)` 等待。
它会把 launcher 输出留在磁盘，结束后只返回摘要。确需中途诊断时读取本次 `agent_run/run.json`、launcher 日志尾部，
或在日志出现后对明确的 exp_dir 调用 `flagscale_train_monitor(mode="check", filter="progress", lines=3)`。

每次的 `experiment.exp_dir/agent_run/` 保存：

- `run.json`：本次状态、命令、路径和检查结果。
- `launcher.log`、`launcher.exit_code`：实际 launcher 输出与退出码。
- `config.yaml`：启动配置的证据副本。
- `measurement.json`：该次准确 loss rank 日志的完整测量；最终比较仍用 `analyze_training_results`。

`status="measured"` 仅表示本次日志可测量，不能视为质量验收或全部 worker 已成功退出。
运行摘要失败或超时时停止后续候选，先处理本次问题；不能把 iteration 到达终点、CLI 提交成功或等待超时当成成功退出。
超时仅清理辅助模块启动的进程组，不全局杀训练进程。
有些 FlagScale 版本的脚本以 `$cmd; sync` 结束，会掩盖训练失败码；CLI 退出 0、迭代完整和日志 rank 数
均不能单独证明所有 worker 成功退出。遇到明确异常应拒绝本次结果，缺少独立退出证据继续标记未验证。
不要修改缺失退出码为 0。对比时直接使用 run.json 给出的 loss 日志和退出码路径；输出缺失应修复路径/分析请求，不重跑有效训练。
