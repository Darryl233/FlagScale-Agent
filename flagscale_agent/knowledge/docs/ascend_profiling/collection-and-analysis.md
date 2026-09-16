<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# NPU profiling 的采集兼容性与分析依据

Profiling 的结论受采集接口、时钟域、事件覆盖和统计口径约束。完整 optimizer step 与 A/B 的比较依据见
[共享测量依据](../ascend_training/measurement-and-records.md)。

## 采集能力与依赖

Python、torch、torch_npu、CANN、驱动、三仓修订、容器和设备共同决定采集兼容性。
已有 CSV 的离线区间分析不依赖 NPU 执行环境；它与原始数据解码、数据库解析及训练采集的依赖不同。
已安装的 `msprof`、`msprof-analyze`、MindStudio Insight 可用于当前格式的导出、集群分析和人工时间线核对。
可用格式由实际可执行路径、版本和输入契约决定，最新版接口不代表当前环境能力。
数据库、压缩 trace 和原始 PROF 数据需要匹配版本的解析工具，转换后仅保留的 CSV 字段可能不足以支持原有结论。

CANN 8.1.RC1 文档中 Level1 才增加通信明细等数据；不同版本及芯片的选项与字段仍须现场核对。
先采集低开销 CPU/NPU 时间线；缺调用归属才短开 stack，缺 shape 才补 shapes，缺通信细节再用相应 level。
给出采样 rank、窗口、输出和额度，不默认 rank 0 代表所有阶段。采集保留完整更新，并与无 profiler 性能测量分离。
[Ascend PyTorch Profiler 文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0033.html)

## FlagScale 训练入口的配置与绑定

FlagScale 的配置到 profiler 消费点存在多层映射。以下源码字段适用于修订 `3444b573474104065144fadba63841d41bb0756f`，其他版本需按实际消费者确认：

| 层次 | 映射关系与兼容条件 |
| --- | --- |
| 配置生成 | `flagscale/train/megatron/training/config/common_config.py` 中 `use_nsys_profiler` 映射 `--profile`，运行时名为 `profile` |
| 独立参数 | `use_pytorch_profiler`、`profile_step_start/end`、`profile_ranks` 分别定义，不能只设置一个布尔值就认定命中 |
| 消费条件 | 训练循环中 `args.profile`、所选 rank 与 `args.use_pytorch_profiler` 同时满足才进入该分支 |
| schedule | 此分支使用 `wait=max(start-1,0)`、起点大于零时 `warmup=1`、`active=end-start`、`repeat=1`；需检查起止合法且覆盖所需更新 |
| 推进 | 查 `start`、循环内 `prof.step`、`stop` 和恢复 iteration 的关系；配置中的 step 数不当然等于绝对训练 iteration |
| handler | 保存 `rank-X.json.gz`、`_cuda_kernel_non_comm.csv`、`_torch_aten_op.csv`；部分汇总按 `cuda_time_total` 和 NCCL 名称过滤 |
| NPU 绑定 | `platform_npu` 等路径可能通过 `transfer_to_npu` 改写绑定；应核实运行时 profiler 对象与实际 NPU 事件 |

不要直接复制 MindSpeed 参数或原版 Megatron handler。修改前核对当前版本的配置生成与实际消费者。
以实际导出的 NPU 时间戳、任务、设备、通信与 step 标记证明采集有效；文件名带 CUDA 既不是有效证明，也不是无效证明。
必要时沿已安装 torch_npu API 增加一个有界采集入口，验证后再采集目标窗口。
独立 wrapper 可在保留原训练入口的前提下直接调用 NPU handler；与内置 profiler 同时启动可能重复采集或产生生命周期冲突。
hook 绑定、调用返回计数、原入口终态和真实产物共同决定该采集路径是否有效。

### 接口兼容检查：构造成功只是第一步

平台初始化可能替换 profiler 绑定；独立 import 的模块名或单算子采集成功不能证明训练消费者与该接口兼容。
在完整初始化后的消费点核对 `inspect.signature`、绑定类型和对象属性。
若出现 `unexpected keyword argument 'execution_trace_observer'`，应定位消费者与实际绑定的签名差异，
该错误表明当前消费者与绑定的接口不匹配，不能据此认定 NPU 采集整体不可用。

Profiler 兼容性包含以下相互独立的接口层次：

| 接口层次 | 必须核对 |
| --- | --- |
| 构造 | 消费者实际传入的全部关键字；`execution_trace_observer=None` 仍是传参，不等于省略 |
| 生命周期 | `start/step/stop` 及停止后的属性访问；使用 execution trace 的消费者可能访问 `prof.execution_trace_observer` 并调用 observer 的 `unregister_callback` |
| handler/export | 回调时机、导出方法、统计字段和输出格式；平台替换后不假设继承原 PyTorch 接口 |
| 实际产物 | NPU 事件、设备、逐任务时间戳、step 覆盖与 flush 完成；构造成功不证明这些成立 |

只删除不支持的构造参数不能证明兼容，还必须核对结束与导出路径。任务明确要求 execution trace 时，不能静默丢弃该功能；
替代入口必须满足相同功能需求，接口适配的有效性取决于完整生命周期；版本升级本身也不保证兼容。
独立采集成功可作为该入口的功能证据；内置入口若仍有阻塞，需分别记录，不能由 wrapper 成功推断内置入口已修复。
独立 wrapper 的适用条件见 [兼容边界与有效性证据](#独立-wrapper-的兼容边界与有效性证据)。

## 输入字段与分析能力

来源、文件大小、CSV 表头以及 trace/db/raw 目录结构可以用于判断输入类型，而无需先加载完整大文件。
目录深度、访问目录项数、单文件及累计读取字节数会限制发现范围；扫描截断时，未发现文件只能记为未知。
符号链接可能使数据离开预期输入范围，去重和范围判断还需考虑链接目标。
大型 `operator_details.csv` 常含多行调用栈；不要拿前几行样本外推全文件统计。

逐任务时间窗分析依赖实际列语义，而非固定文件名。常见列组合包括：

| 信息 | 识别列 |
| --- | --- |
| 起点、时长（us） | `Start Time(us)`、`Duration(us)`；或 `Task Start Time(us)`、`Task Duration(us)` |
| 设备 | `Device_id`、`Device ID`、`Device Id`、`device_id` |
| 名称和类型 | `Name/Op Name`；`Type/OP Type`，缺失时保留 task type 作为分桶信息 |
| 加速核 | `Task Type/Accelerator Core`，与 operator type 分开保存 |
| 其他证据 | `Stream ID`、`Input Shapes`、`Input Data Types`；缺失限制流重叠、形状和精度归因 |

有 Count/平均值但无逐次起止的聚合表不能定位空洞。`step_trace_time.csv` 中的 Computing、Free、Stage 等是汇总，
不能伪造出 `[start,end)`。另一方面，CANN `op_summary_*.csv` 可能有 `Task Start Time(us)` 和 `Task Duration(us)`，
因此不能仅凭文件名断言没有时间戳。
该官方文档把 `Task Wait Time(us)` 定义为前后任务间隔，而非 host enqueue 延迟；时间语义须跟随 producer。
[CANN op_summary 字段说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0067.html)

只有已知一个设备时钟域时才能合并任务。无设备列或设备值缺失意味着范围未知；多设备表须按设备分别分析。
多个 rank 文件不能直接拼接，跨 rank 时间戳差值需可靠的时钟对齐；stage 比较先用各自完整 step 时长。
导出小窗口时保留所有与窗口相交的记录，而非只保留起点位于窗口内的任务；保留来源映射和筛选条件。

将同次导出的 host trace 标注映射到 kernel CSV 时，核对实际任务关联，例如名称、精确起点与
Model/Stream/Task ID；时长字段可能采用不同的导出精度，不能用任意偏移或放宽误差来制造时钟对齐。
同一线程中的训练调用标注被唯一 `ProfilerStep#` 包含时，可关联训练调用与采样 step；
这仍是 host 标注的时间窗，异步设备任务可能跨越其边界，不能声称整次更新的异步工作都被覆盖。
窗口相交任务数与通信报告按 step 归属的计数可不同，应分别标明口径。
通信识别依实际 producer 的 task type 和通信报告；部分导出使用 `COMMUNICATION`、`hcom_` 等标记，
不能仅因 kernel 名称不含 `HCCL` 就断言没有通信。

## 时间窗统计与完整性

设分析窗口为 `W=[start,end)`，同一设备内的任务区间为 `I_i`。记录覆盖时长是
`|⋃(I_i ∩ W)|`，未覆盖时长是 `|W| - |⋃(I_i ∩ W)|`；多流并发区间只计一次。
两者描述采集记录的覆盖，不直接描述设备利用率或全设备空闲。

| 信息 | 解释边界 |
| --- | --- |
| 版本、来源、实际表头和读取范围 | 决定字段是否可解释、输入是否完整，不能由文件名替代 |
| 窗口与时间精度 | 大绝对微秒时间戳用低精度浮点保存会丢失亚微秒差值；十进制或等效精度表示可避免此问题 |
| 设备范围、缺失字段 | 设备归属不明或跨文件时钟未对齐时，不能合并为单一设备时间线 |
| 有效行、坏行与截断位置 | NaN/Inf、负时长、列数不符及读取截断都会削弱完整性；有限正常数值只是可计算的必要条件 |
| 任务区间并集与未覆盖量 | 完整读取时对有效记录精确；遗漏事件或截断时，并集为真实任务覆盖的下界，未覆盖量为上界 |
| 最大未覆盖窗口 | 起止、前后原始行号、名称、核类型和 stream 可用于定位上下文；只列 top 项时需区分总数与省略数 |
| `Name+Type+shape+dtype` 分桶 | 裁剪后的累计时长、次数与示例行用于热点筛选，重叠时长和不能当关键路径贡献 |

空文件或没有可解析任务不能证明 100% 空闲。CSV 数值可计算也不证明采集完整、窗口覆盖完整 step、NPU 已同步或根因已确定。
Host trace、通信 JSON、模型结构和 PMU 提供不同证据，无法从缺少相应字段的 kernel CSV 中还原。

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

## 独立 wrapper 的兼容边界与有效性证据

独立 wrapper 可直接调用安装版本的 `torch_npu.profiler.profile` 和 NPU trace handler，
避免复用与当前绑定不兼容的 Megatron 导出逻辑或全局替换 `torch.profiler`。
不支持 `execution_trace_observer` 的接口不能仅靠省略参数提供 execution trace 或 Chakra 联合采集能力。
独立采集路径可用与内置 profiler 兼容是两个不同结论，离线调用测试也不能代替真实训练循环的采集证据。

有效采集需要同时确认 wrapper 命中了真实训练循环、调度窗口覆盖所需更新、trace 回调产生可解析的 CPU/NPU 产物，
并检查所选 rank 的设备映射、时间戳及实际计算/通信事件。任务条数用于检查产物，不能成为跨模型的通过阈值。
全部训练 worker 正常结束与所选 rank 采集有效是两类证据；局部采集不代表其他 rank 的关键路径，也不证明 checkpoint 完整恢复。
安装版本的签名和实际产物优先于 [Ascend profiler 源码参考](https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/profiler/profiler.py)
或最新文档；一些实现会捕获内部错误，故生命周期调用未抛异常仍不足以证明有效。
