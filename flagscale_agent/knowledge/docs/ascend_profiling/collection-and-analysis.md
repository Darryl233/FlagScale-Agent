<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 采集入口、输入适配与证据契约

采集或解释本技能工具输出时读取。完整 optimizer step、A/B 与实验 ledger 使用
[共享测量契约](../ascend_training/measurement-and-records.md)。

## 采集前确认实际能力

记录 Python、torch、torch_npu、CANN、驱动、三仓 commit/dirty diff、容器和分配设备。
离线 CSV 工具仅需 Python 标准库，不需要 NPU、torch_npu 或 pandas；不要为已有文件分析安装训练依赖。
已安装的 `msprof`、`msprof-analyze`、MindStudio Insight 可用于当前格式的导出、集群分析和人工时间线核对。
先检查真实可执行路径、版本、help 和输入契约；不自行安装或运行下载来的脚本，不把最新版接口假定成当前能力。
数据库、压缩 trace 和原始 PROF 数据由匹配版本的工具处理，本技能脚本不会偷偷降级为不完整 CSV 分析。

CANN 8.1.RC1 文档中 Level1 才增加通信明细等数据；不同版本及芯片的选项与字段仍须现场核对。
先采集低开销 CPU/NPU 时间线；缺调用归属才短开 stack，缺 shape 才补 shapes，缺通信细节再用相应 level。
给出采样 rank、窗口、输出和额度，不默认 rank 0 代表所有阶段。采集保留完整更新，并与无 profiler 性能测量分离。
[Ascend PyTorch Profiler 文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0033.html)

## FlagScale 训练入口核验

以下是对 FlagScale `3444b573474104065144fadba63841d41bb0756f` 的本地源码阅读锚点，不是硬件实测兼容声明：

| 层次 | 当前观察与要核对的证据 |
| --- | --- |
| 配置生成 | `flagscale/train/megatron/training/config/common_config.py` 中 `use_nsys_profiler` 映射 `--profile`，运行时名为 `profile` |
| 独立参数 | `use_pytorch_profiler`、`profile_step_start/end`、`profile_ranks` 分别定义，不能只设置一个布尔值就认定命中 |
| 消费条件 | `training.py` 约 3350 行：`args.profile`、所选 rank 与 `args.use_pytorch_profiler` 同时满足才进入该分支 |
| schedule | 此分支使用 `wait=max(start-1,0)`、起点大于零时 `warmup=1`、`active=end-start`、`repeat=1`；需检查起止合法且覆盖所需更新 |
| 推进 | 查 `start`、循环内 `prof.step`、`stop` 和恢复 iteration 的关系；配置中的 step 数不当然等于绝对训练 iteration |
| handler | 保存 `rank-X.json.gz`、`_cuda_kernel_non_comm.csv`、`_torch_aten_op.csv`；部分汇总按 `cuda_time_total` 和 NCCL 名称过滤 |
| NPU 绑定 | `platform_npu` 等路径可能通过 `transfer_to_npu` 改写绑定；应核实运行时 profiler 对象与实际 NPU 事件 |

不要直接复制 MindSpeed 参数或原版 Megatron handler。修改前核对当前版本的配置生成与实际消费者。
以实际导出的 NPU 时间戳、任务、设备、通信与 step 标记证明采集有效；文件名带 CUDA 既不是有效证明，也不是无效证明。
必要时沿已安装 torch_npu API 增加一个有界采集入口，验证后再采集目标窗口。
本技能提供 [独立 wrapper 生成流程](../../../skills/train-ascend-profiling/references/wrapper-generation.md)，保留原训练入口并直接使用 NPU handler；
使用时关闭内置 profiler，验证 hook、返回计数、原入口终态和真实产物。

### 实测兼容检查：构造成功只是第一步

2026-09-08 容器功能验证使用 FlagScale `b49ce7a77b3177072d307c73086088e379d5461e`、
Megatron-LM-FL `e03018df34119b2401da4b1606aed32d57cfff8c`（含用户已有未提交修改）、
TE-FL `137b344e6f30ffefd9bb2f69e9d489afd13c30d6`、PyTorch `2.7.1+cpu`、
torch_npu `2.7.1.post6`，CANN 路径标识为 `9.0.0`。该记录是具体环境证据，不是版本兼容性承诺。

- 独立 `torch_npu.profiler` 包裹 TE 公共 RMSNorm 的 4 次 optimizer 更新成功，导出含 34 条任务记录的 `kernel_details.csv`。
- FlagScale `train_gpt.py` 小型 mock 配方启用 `--profile --use-pytorch-profiler --profile-step-start 1 --profile-step-end 3`，
  在首个训练 step 前因 `TypeError: profile.__init__() got an unexpected keyword argument 'execution_trace_observer'` 失败；worker 已退出。
- 仅导入公开 TE 后，`torch.profiler.profile.__module__` 仍为 `torch.profiler.profiler`，但训练实际调用因上述参数失败。
  独立 import 结果不能证明平台初始化后的绑定兼容；必须在完整初始化后的消费点核对 `inspect.signature`、绑定类型和对象属性。

为当前绑定建立以下兼容矩阵，并用一个有界采集闭环验证：

| 接口层次 | 必须核对 |
| --- | --- |
| 构造 | 消费者实际传入的全部关键字；`execution_trace_observer=None` 仍是传参，不等于省略 |
| 生命周期 | `start/step/stop` 及停止后的属性访问；本地参考消费者还访问 `prof.execution_trace_observer` 并调用 observer 的 `unregister_callback` |
| handler/export | 回调时机、导出方法、统计字段和输出格式；平台替换后不假设继承原 PyTorch 接口 |
| 实际产物 | NPU 事件、设备、逐任务时间戳、step 覆盖与 flush 完成；构造成功不证明这些成立 |

只删除不支持的构造参数不能证明兼容，还必须核对结束与导出路径。任务明确要求 execution trace 时，不能静默丢弃该功能；
记录能力缺口并选择满足要求的已支持入口，或在授权范围内实施最小适配并验证完整生命周期。不据此固定推荐升级依赖。
独立采集成功可作为其公共接口和产物分析的功能证据；必须单列 FlagScale 内置采集的阻塞，不能宣称该训练采集入口已打通。
随后对本技能生成的独立 wrapper 完成了单机 16 NPU 合成 GPT 训练采集，具体边界见
[独立 wrapper 的兼容边界](#独立-wrapper-的兼容边界与早期验证)。这不改变上述内置 profiler 失败的结论。

## 先判定输入能做什么

`inventory ROOT` 仅记录来源、文件大小、CSV 首行、trace/db/raw 目录存在性，不读取大 trace 或大 CSV 正文。
限制可以调整：`--max-files` 限所有访问目录项（含目录），`--max-depth` 限子目录层数，
`--max-read-bytes` 限单个首行，`--max-total-read-bytes` 限累计读取。
输出有 `entry_limit/depth_limit` 时，未发现文件是未知。符号链接（包括输入路径中的链接组件）不跟随。
大型 `operator_details.csv` 常含多行调用栈；不要拿前几行样本外推全文件统计。

`window` 当前支持以下实际列组合，而非固定文件名：

| 信息 | 识别列 |
| --- | --- |
| 起点、时长（us） | `Start Time(us)`、`Duration(us)`；或 `Task Start Time(us)`、`Task Duration(us)` |
| 设备 | `Device_id`、`Device ID`、`Device Id`、`device_id` |
| 名称和类型 | `Name/Op Name`；`Type/OP Type`，缺失时保留 task type 作为分桶信息 |
| 加速核 | `Task Type/Accelerator Core`，与 operator type 分开保存 |
| 其他证据 | `Stream ID`、`Input Shapes`、`Input Data Types`；缺失在 `missing_fields` 中列出 |

有 Count/平均值但无逐次起止的聚合表不能定位空洞。`step_trace_time.csv` 中的 Computing、Free、Stage 等是汇总，
不能伪造出 `[start,end)`。另一方面，CANN `op_summary_*.csv` 可能有 `Task Start Time(us)` 和 `Task Duration(us)`，
不能像某些技能参考一样仅凭文件名断言没有时间戳。
该官方文档把 `Task Wait Time(us)` 定义为前后任务间隔，而非 host enqueue 延迟；时间语义须跟随 producer。
[CANN op_summary 字段说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0067.html)

只有已知一个设备时钟域时才能合并任务。无设备列或设备值缺失返回 `unknown_scope`；多设备表须显式选择。
多个 rank 文件不能直接拼接，跨 rank 时间戳差值需可靠的时钟对齐；stage 比较先用各自完整 step 时长。
导出小窗口时保留所有与窗口相交的记录，而非只保留起点位于窗口内的任务；保留来源映射和筛选条件。

将同次导出的 host trace 标注映射到 kernel CSV 时，核对实际任务关联，例如名称、精确起点与
Model/Stream/Task ID；时长字段可能采用不同的导出精度，不能用任意偏移或放宽误差来制造时钟对齐。
同一线程中 `ascend_train_step/<call>` 被唯一 `ProfilerStep#` 包含，可关联训练调用与采样 step；
这仍是 host 标注的时间窗，异步设备任务可能跨越其边界，不能声称整次更新的异步工作都被覆盖。
窗口相交任务数与通信报告按 step 归属的计数可不同，应分别标明口径。
通信识别依实际 producer 的 task type 和通信报告；本机导出使用 `COMMUNICATION`、`hcom_` 等标记，
不能仅因 kernel 名称不含 `HCCL` 就断言没有通信。

## window 输出契约

示例：`python scripts/profile_inspect.py window INPUT.csv --start-us 1000000 --end-us 1100000 --device-id 0`

| 字段 | 含义 |
| --- | --- |
| `schema_version/source/source_bytes/headers/limits` | 版本、来源、实际列和读取上限 |
| `window` | 请求范围；绝对微秒时间以十进制字符串保存，避免大时间戳丢失亚微秒精度；时长是有限数值 |
| `scope/devices_observed/missing_fields` | 设备范围、观察到的设备、缺失证据；不声称跨文件时钟已对齐 |
| `status` | `ok`、`partial`、`unknown_scope`、`unsupported_schema`、`no_window_events` 或输入错误 `error` |
| `rows_read/invalid_row_count/invalid_rows/truncated` | 扫描范围、最多 20 条错误位置与是否触及行数或解析边界 |
| `metrics.observed_task_union_us` | 所选设备、窗口中有效记录的并集，包含跨窗口裁剪与多流重叠 |
| `metrics.uncovered_us/uncovered_ratio` | 窗口减记录并集；是记录未覆盖量，不自动等于全设备 idle |
| `metrics.interpretation` | 完整读取时对有效记录精确；坏行/截断时忙时间下界、未覆盖时间上界 |
| `uncovered_windows` | 按时长列出的前 top 项，包含起止、前/后原始行号、名称、核类型、stream；总数和省略数单列 |
| `top_ops` | `Name+Type+shape+dtype` 分桶的裁剪后时长和；保留次数和示例行号，不能当关键路径贡献 |
| `limitations` | 采集覆盖、窗口边缘和根因结论的限制 |

默认只读取不超过 8 MiB 的 CSV，最多 100000 行；输入过大返回错误，不先加载再截断。
所有数字必须有限；NaN/Inf、负时长、列数不符会列为坏行。无任务或空文件时不输出 100% 空闲。
`ok` 表示 CSV 记录可计算，并不证明采集完整、窗口是完整 step、NPU 已同步或根因已确定。
标准输入错误返回 JSON 和退出码 2；可报告的能力不足或部分分析返回对应 status，调用方必须检查该字段。
工具不解析 host trace、通信 JSON、模型结构或 PMU，也不自动转换格式；这些留给有明确输入适配能力的官方工具。

## 从区间到原因

先把完整更新、microbatch、FWD/BWD、重计算、PP stage 和层注释对齐，再讨论局部贡献。
缺注释时重复 kernel 模板只能作为结构候选；MoE 动态路由、融合、重计算和调度都可能改变调用次数。
报告 wall span、区间并集、算子时长和不同口径，不能相加为全 step 延迟。
AI CPU、HCCL 都计入全部任务并集时，AI CPU 活动不能同时被解释成该定义下的空洞。
要讨论计算空闲，应另算计算任务并集；要讨论通信暴露，应测通信并集与计算并集的交集，避免按大算子与小任务重复计数。

空洞附近保留前后任务和时间窗；同一时钟域下再查 host、ACL、copy/sync、通信等待证据。
host 覆盖率须合并区间，不能叠加嵌套 Python/CPU 事件。用关联 ID/flow 和线程关系核对异步启动，不能只凭 host 时间包围。
“未覆盖窗口反复出现”可成为事实；“Python 锁、数据等待、通信同步”仍需相应证据。
低 host 覆盖不能证明 host-bound。采集起止附近空洞可能是截断；没有 PMU 不能由 busy 比例断言算力饱和。
阈值如 30%、1 ms、95% wait ratio 只可作记录在案的筛选参数，不是芯片无关诊断结论。

多卡瓶颈可按已安装 `msprof-analyze cluster` 的当前 help 接入；核对它所需的 L1 数据、rank 元数据、
step_trace_time、通信明细与矩阵，或它支持的 db 组合。部分版本不支持混放 text/db，不能随意拼接产物。
不机械采用旧文档中的带宽阈值或“差 5% 就是慢卡”；结合分配拓扑、并行策略、负载与重复基线。
[Ascend mstt 集群分析输入契约](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze/cluster_analyse/README.md)

## 方法来源与改编边界

技能组统一来源与取舍见 [来源与适配说明](../ascend_training/sources-and-adaptation.md)。
参考研究固定到 Ascend/agent-skills 提交 `155ac37bd169ddb89479af528297cfb2237400aa`。
本技能与脚本独立编写，未复制其代码或文档；本项目文件按 Apache-2.0 标识。
上游仓库许可为 [MulanPSL-2.0](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/LICENSE)，
不是本项目的 Apache-2.0；如后续直接引入上游代码，需随分发保留相应许可与声明。

参考链接：

- [官方 profiling 技能](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascend-profiling-anomaly/SKILL.md)：区间事实与软归因分开；不移植其推理架构报告强制要求。
- [输入与区间方法](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascend-profiling-anomaly/references/kernel_data_guide.md)：借鉴多流并集和原始证据，字段以 producer 官方文档与实际 header 校正。
- [参考实现](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascend-profiling-anomaly/scripts/reference_host_gap_branch.py)：上游是依赖 pandas 的 helper，并非带输入发现的完整分析 CLI；本工具使用标准库独立实现。
- [判定参考](https://github.com/Ascend/agent-skills/blob/155ac37bd169ddb89479af528297cfb2237400aa/skills/ascend-profiling-anomaly/references/rulebook.md)：保留不确定性表达；不采用 AICPU 一律不允许、等待即全设备空闲或固定阈值等绝对规则。

## 独立 wrapper 的兼容边界与早期验证

模板直接调用安装版本的 `torch_npu.profiler.profile` 和 NPU trace handler，不传
`execution_trace_observer`，也不调用原 Megatron 的 profiler 导出逻辑或全局替换 `torch.profiler`。
这提供独立采集路径；不是对 FlagScale 内置 profiler 的源码修复，也不支持 Chakra 联合采集。
生成器成功和离线行为测试通过都不能代替目标容器中的三仓训练采集验证。

2026-09-08，本实现的 36 项离线行为测试通过；随后在同日记录的三仓环境、torch_npu 2.7.1.post6 上，
生成 wrapper 包裹 FlagScale `train_gpt.py`，单机 16 个逻辑 NPU 从独立合成 GPT checkpoint 恢复，
全部 rank 完成 8 次更新并正常退出。所选 rank 0 和 15 各导出 3786 条任务记录，含计算与 HCCL 事件，
设备 ID、时间戳及本技能区间工具检查通过。首次尝试在进入 train 前因 weight decay scheduler 恢复冲突失败，
明确沿用 checkpoint scheduler 后重试通过；未修改远端框架源码。
这是当时合成 GPT 采集的验证范围。后续 PP、MoE 和 Qwen 短序列验证见 [实机验证记录](../ascend_training/hardware-validation.md)；任何单次记录均不证明所有 profiler 选项兼容。
16 个 worker 的结束证据与 rank 0/15 的采集证据分别成立；这里没有取得其余 14 个 rank 的 profile，也未证明 checkpoint 的全部 RNG 状态能等价恢复。
安装版本的签名和实际产物优先于 [Ascend profiler 源码参考](https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/profiler/profiler.py)
或最新文档；一些实现会捕获内部错误，故生命周期调用未抛异常仍不足以证明有效。
