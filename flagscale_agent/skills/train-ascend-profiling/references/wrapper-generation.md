<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 生成 torch_npu profiler 训练入口

原训练入口可以正常运行，且需要独立采集 NPU profile 时使用。适用于通过
`megatron.training.training.train/train_step` 执行的 FlagScale Megatron 入口，包括 Qwen3.5。
实现由 [生成器](../scripts/generate_profile_wrapper.py) 和 [独立模板](../assets/npu_profile_wrapper.py) 组成。
生成器只需 Python 标准库，不启动训练；产物运行时使用目标训练环境已有的 torch、torch_npu 和三仓依赖。

生成器会检查 `--entrypoint` 是本机存在的原始训练文件。在已有权限可访问目标环境时，可在那里只运行生成命令；
若本次无法访问该环境且只有远端路径，交付目标环境的生成命令和配方改动，明确 `.py` 尚未生成。
不要创建空的同名入口来通过检查，也不要把待执行命令写成采集成功。

## 1. 核对入口并生成

在目标容器中复用原作业已确认的 cwd、Python 与模块路径。FlagScale launcher 可将自身
`flagscale/train` 放入 `PYTHONPATH`，因此 `megatron.training.training` 不一定来自 Megatron-LM-FL 仓库。
只在绑定路径尚未确认或发生错误时核对实际模块文件；已有记录不重复调查。
`--training-file` 可将核实的路径作为运行时保护；它不是按仓库名称猜出的路径。

以下变量需指向本次实际位置；`RUN_DIR` 使用独立 attempt 目录。
`SKILL_DIR` 是安装后或源码中的本技能目录，生成器需要其相邻 assets 模板。

```bash
python "$SKILL_DIR/scripts/generate_profile_wrapper.py" \
  --entrypoint "$FS_ROOT/flagscale/train/megatron/train_qwen35.py" \
  --output "$RUN_DIR/train_qwen35_npu_profile.py" \
  --profile-output "$RUN_DIR/npu-profile" \
  --training-file "$VERIFIED_TRAINING_FILE" \
  --python-path "$FS_ROOT" \
  --python-path "$FS_ROOT/flagscale/train" \
  --wait 3 --warmup 1 --active 1 --ranks 0 --level Level1
```

生成结果为 JSON，含 wrapper 路径、完整配置和 `training_started: false`。
生成器拒绝覆盖已有 wrapper 或原入口。生成后的 `.py` 自包含，无需再安装本技能；
它引用的原入口、依赖和输出均为目标机绝对路径。多节点须保证每个 worker 可访问相同 wrapper 及对应路径。
生成后可搬移 wrapper 文件，但跨机器目录不同时应在目标环境重新生成配置。

支持 `--ranks 0,7` 或 `--ranks all`，这里是 **global rank**，不是物理 NPU 编号。
默认仅 rank 0；PP、EP 或慢 rank 诊断应按证据选择代表 rank。
设备分配、可见设备与 world size 仍由原 launcher 管理，生成器不选择或占用设备。
`--record-shapes`、`--with-stack`、`--profile-memory` 按证据缺口启用，默认全部关闭；
level 默认 Level1。核对当前 torch_npu 支持的 level、导出格式和开销，不为此自动升级依赖。

## 2. 使用专用采集配方启动

复制原配方为本次 profiling attempt，替换其入口。当前 FlagScale runner 读取的配置路径是：

```yaml
experiment:
  task:
    entrypoint: /absolute/run-dir/train_qwen35_npu_profile.py
```

这是配置片段，应合并到现有配方，保留 task 其他字段。
沿当前 schema 关闭内置 profiler：核对 `use_nsys_profiler` 映射的最终 `args.profile` 为 false、
`use_pytorch_profiler` 为 false、`pytorch_profiler_collect_chakra` 为 false。
不同版本的 YAML 嵌套可能不同，以 resolved 配置与最终 argv 为准；不要附加 `--profile`。
wrapper 会拒绝内置 `args.profile`、Chakra 请求及 `skip_train`，不会偷偷改变这些参数。

使用原 FlagScale launcher、分布式参数和模型配置启动，只更换入口及本次明确的采集/短跑范围。
wrapper 用 `runpy.run_path(..., run_name="__main__")` 执行原入口一次，保留训练参数、
Qwen3.5 的 provider、额外参数解析、online eval 和 tensorboard 回调；不复制它的 `pretrain(...)` 调用。
绑定不到实际训练函数、原入口重绑 hook 或从未进入 train 时，报告错误并撤销自身补丁。
不同训练循环、自定义 train_step 查找或一个进程多次 train 调用，需要先适配，不能宣称通用兼容。

启动前沿 [共享测量契约](../../../knowledge/docs/ascend_training/measurement-and-records.md)
核对已分配设备、时长、磁盘和本次进程的停止办法。wrapper 只限制 **采集窗口**，不限制训练总时长；
原训练可以在窗口后继续，需由短跑配方与 launcher 保证作业在现有额度内退出。
不因生成或首次失败而重新计算用户给定的总预算。

## 3. 理解窗口和返回计数

profiler 在完成训练初始化并进入 `train` 后启动，配置 `repeat=1`。
每次原 `train_step` **正常返回后**推进一次 `prof.step()`，保留原返回值，不改变 rerun 或跳过逻辑。
默认 wait=3、warmup=1、active=1 对应本次 train 调用的第 1–3 次返回等待、第 4 次预热、第 5 次 active。
至少需要 5 次正常返回才能走完默认窗口；实际采集前还应根据编译、初始化和训练稳定性选择 wait，不能把默认值当稳定保证。

`train_step` 通常包含全部 microbatch、反向、优化器和学习率更新；但 overflow、rerun 或退出条件可能使
正常返回不等于成功 optimizer update。计数相对于本次 train 调用，与恢复后的绝对 iteration 不同。
状态中 `iteration_argument` 只记录调用实际传入的整数关键字 `iteration`；缺失时为 null，不推测编号。
完整更新边界仍需结合训练实现、日志和 trace 验证。

profiler 包围整个 train，因此相邻返回边界间可能含外层日志、评估、保存或调度。
模板用 `ascend_train_step/N` 标记采集期间的函数体；N 为上述相对调用序号。
不要把整个 profiler step 当作纯计算耗时，也不要把这些带 profiler 的时间用于性能选优。

到达 wait+warmup+active 后立即停止 profiler，训练继续；异常路径也执行清理并撤销自身 hook。
`SystemExit(0/None)` 保留正常退出语义，窗口不足仍记 incomplete；非零退出、KeyboardInterrupt 或训练错误记 failed。
原入口在 train 之后的保存/评估失败也回写入口状态。清理失败不掩盖原始训练错误。
SIGKILL、进程崩溃或断电无法保证 finally/flush，残留 running/pending 状态必须视作未验证。

## 4. 验收产物并交给分析

每个所选 rank 新建 `$RUN_DIR/npu-profile/rank-00000/` 等目录；已存在时拒绝复用。
其中 `trace/` 交给 `tensorboard_trace_handler`，具体下级目录、CSV、JSON 或 db 由安装版本决定。
`wrapper-status.json` 记录实际模块/函数文件、rank/world size、schedule、返回计数、回调和入口状态。

| 状态或证据 | 含义 |
| --- | --- |
| `collecting` / `window_complete_pending_training_end` | 仍在运行或未正常收尾，不能交付成功结论 |
| `incomplete` | 返回次数不足或没有导出回调；即使 worker 返回 0 也未完成采集闭环 |
| `failed` | 训练、入口或采集清理失败；已有部分文件可保留诊断 |
| `window_complete_needs_artifact_validation` | 调度窗口结束且回调返回；仍需查真实 NPU 事件 |
| `entrypoint_status: completed` | 整个原入口正常返回或退出码为 0；不能单独证明采集完整 |

对全部训练 worker 检查正常退出，对所选采集 rank 另检查上述状态和实际产物；不能只看 rank 0 打印或目录存在。
先执行只读盘点：

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" inventory "$RUN_DIR/npu-profile/rank-00000/trace"
```

出现深度/数量截断时，对实际 worker 子目录继续有界盘点。
确认有当前 attempt 的 NPU 事件、设备映射、有效时间戳及目标完整更新；
再从 trace/step 标记取得实际时间窗，将逐任务 CSV 交给 `profile_inspect.py window`。
具体字段和区间口径见 [采集与分析参考](../../../knowledge/docs/ascend_profiling/collection-and-analysis.md)。缺 NPU 事件、空文件、只有汇总或窗口未命中时，
记录限制并修正具体原因后有界重试，不能将回调次数当作产物有效性证明。
