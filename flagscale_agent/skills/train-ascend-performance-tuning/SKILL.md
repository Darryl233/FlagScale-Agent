---
name: train-ascend-performance-tuning
description: 在 FlagScale 昇腾训练中执行基础调优闭环：固定工作负载，调整 micro-batch，测量、比较并复测交付。
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 昇腾训练基础调优

当前阶段只启用 micro-batch（MBS）调整，先跑通 `基线 → 一个候选 → 比较 → 复测交付`。
并行策略、重计算、通信重叠、图模式、profiling 和算子优化暂不接入本流程。
默认只读本文件；实际启动训练时加载一次 `train-run`，复用其中的昇腾执行分支。不预加载其他 SKILL、Knowledge 或 methods 目录。遇到阻塞当前步骤的具体问题，
只查询对应源码或 Knowledge 文档小节，得到答案即返回当前步骤；没有收益也不自动展开进阶搜索。

## 1. 确认比较条件

读取用户给定的配置、FlagScale CLI 命令和已有日志，沿用已确认的设备、时间预算与目标。
记录实际加载的 FlagScale / Megatron-LM-FL / TransformerEngine-FL 路径与版本，保留已有修改。
固定模型、数据及顺序、序列长度、GBS、并行布局、精度、优化器和训练起点；约定预热、计时窗口、数值容差与复测波动上限。
最低收益门槛另行约定；复测波动上限不是最低收益门槛，也不能用来把单次小幅加速判为噪声。
复用当前 plan 和一份实验记录，保存原配置、启动 cwd/命令、每次运行的日志与结果位置。
恢复会话时先 `plan_status`、读记录并核对运行中的作业。仅分析配置或旧日志时不启动训练。
已完成任务的进度询问直接读取已保存的具体计划文件与结果；`No active plan` 本身不是完成证据。
没有新工作时不重建计划或重跑验收。

## 2. 建立基线

首次训练先调用 `load_skill(name="train-run")`，选择 Ascend 分支。
符合条件的单机短跑接着用 `read_file` 读取该技能目录的 [references/single-run.md](../train-run/references/single-run.md)，照其请求示例、启动和等待命令执行；其他情形走该技能的 FlagScale CLI 路径。
基线、候选和复测复用这一说明及已确认的环境、设备、布局与测量口径；基线短跑就是本次验证，不额外插入单卡 smoke。
同条件、同口径的有效基线可以复用。仅分析已有结果时不加载启动流程、不启动训练。
核对完整迭代、有限 loss/梯度、跳步/NaN 和退出证据，取得预热后的稳定计时窗口；缺失指标保持未知。
基线失败或计时不稳时先定位当前问题，取得有效基线后再比较；本阶段不转入其他优化方法。
单次期限由实际 launcher 控制；等待超时不等于训练停止，结束后确认全部自有 worker 退出。

## 3. 生成一个 MBS 候选

复制当前有效配置为独立 YAML，通过 `-c` 选择，并保持配置依赖和相对路径有效。
只改 MBS 及框架需要的梯度累积派生值。显存有余量时先试相邻更大的合法值。
核对实际 DP，保证 `GBS / (DP × MBS)` 为正整数并满足当前 pipeline 调度约束；GBS 和并行布局保持不变。
保存完整配置差异，核对 CLI 解析后的参数确实生效。一次只运行一个候选，不自动调用整套 tuner。

## 4. 运行并比较

按已加载的 `train-run` 路径及基线相同的训练起点和测量规则短跑候选，核对实际 MBS、样本数、loss/梯度及异常计数。
`analyze_training_results` 是可直接调用的 Agent 工具。对支持的 Megatron 日志通过工具接口调用：每次运行提供一个准确的 loss rank 日志及已有的 `exit_code_path`，
使用 `log_interval=1`，明确首末 iteration、warmup_steps、GBS、sequence_length 和已约定的 loss 容差，设置 `output_path` 保存 JSON。
每个比较请求只含原基线和一个固定候选；参数或结果字段不清楚时再读 [结果分析流程](references/result-analysis.md)。
默认使用分析工具返回的摘要和已落盘 JSON；只在异常时读取对应详细证据，避免把全量 JSON/日志带回上下文。
从生成的指标比较完整更新耗时、吞吐与 loss 差值，另记显存和 worker 退出证据；`status="ok"` 只表示解析成功。
在同一记录追加配置差异、命令、日志、结果路径和保留/回退理由，更新 plan 的进度与下一步。
有收益且质量检查通过时，仅在优于此前最佳时更新暂留候选；按预算尝试下一个合法 MBS。
OOM 时回退并结束增大方向；质量失败时拒绝候选；无收益时按预算测试剩余合法值或结束。
没有可试候选或预算不足时进入交付，不扩大搜索范围。

## 5. 复测并交付

没有更优候选时直接交付原基线和已测结果。
确定最终候选后，默认做三对：先准备 `pair1-baseline.json`、`pair1-candidate.json` 至 `pair3-candidate.json` 共六份请求，每份使用新的 YAML、run_id 和 exp_dir；初测不计入。
单机有界执行路径用 `shell(background=True)` 执行以下循环，按 `train-run` 的方式等待。次数另有约定时调整请求列表：

```bash
RETEST_REQUEST_DIR=/absolute/path/to/retest-requests
for run in pair1-baseline pair1-candidate pair2-candidate pair2-baseline pair3-baseline pair3-candidate; do
  PYTHONUNBUFFERED=1 python -m flagscale_agent.training_run \
    --request "${RETEST_REQUEST_DIR}/${run}.json" || exit $?
done
```

最终只比较清单中的新运行（默认六次），按实际执行顺序传入日志及退出码路径。其他启动后端通过 `train-run` 按相同次数准备新配置并逐次运行。
按既定容差和波动上限判断；预算不足或证据缺失时标记候选待验证。
交付配置、FlagScale CLI 复现命令、实验记录和简短报告：改了什么、收益与质量证据、是否采用、尚未验证什么。
报告直接引用同一份最终分析摘要中的耗时、tokens/s、吞吐增长与耗时下降，不混用均值与中位数或手算替代。
mock data 结果注明模型、固定序列长度和合成数据范围。
短跑 loss 检查不代表参数更新等价或长期收敛。未通过本轮验收则保留原基线，并附上实际缺口。
保留原配置以便回退，完成负责的计划步骤并附证据；整个当前计划完成后才标记 complete。
把报告与必要的记录一次核对完再结束；无新证据时不反复改写总结、重复保存同一结论。
