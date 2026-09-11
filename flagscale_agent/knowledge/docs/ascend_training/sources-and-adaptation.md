# 官方与社区方法的适配边界

本套技能针对 FlagScale + Megatron-LM-FL + TransformerEngine-FL 的 Ascend 训练路径独立编写。以下来源用于核对方法与设计，不是本套技能的运行时依赖，也不证明其参数能在本地栈使用。来源阅读日期：2026-09-08；网页和仓库会变化，执行时以安装版本、实际代码及产物为准。版本范围内的实机结果另见 [硬件验证记录](hardware-validation.md)；外部方法与历史实测均不能替代当前环境的能力核验。

## 借鉴了什么，落在哪里

| 来源 | 借鉴方法 | 本套落点与适配 |
| --- | --- | --- |
| [NVIDIA 官方技能目录](https://github.com/NVIDIA/skills#skill-catalog)、[MoE 工作流](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md) | 按问题使用专项技能，先可行性，再由证据选择下一轮 | 主入口统一契约和预算；三个专项可独立触发。成熟配方不强制重走所有阶段 |
| [并行策略](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-parallelism-strategies/SKILL.md)、[显存调优](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-memory-tuning/SKILL.md) | 区分稠密/专家参数、激活与布局约束 | 配置技能核对本地进程组和实际 OOM 阶段；GPU 显存估计不能直接做 NPU 硬剪枝 |
| [激活重计算](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-activation-recompute/SKILL.md)、[EP overlap](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-expert-parallel-overlap/SKILL.md)、[TP/DP overlap](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-tp-dp-comm-overlap/SKILL.md) | 重计算边界与调度、通信相互影响；逐项消融 | 配置技能建立模型结构、重计算、overlap、后端版本的能力矩阵，检查 helper 的完整配置差异 |
| [CUDA Graph 技能](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-cuda-graphs/SKILL.md) | 从稳定 eager、最小捕获范围到验证重放和计时 | 只有当前 NPU 图执行链路有支持证据且下发在关键路径时才试；不引入 CUDA 开关 |
| [Nsight Systems AI 辅助分析](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#ai-assisted-analysis) | 先发现工具/产物，再用有界提取支撑解释 | profiling 技能先查版本、schema 和覆盖范围，提供小型只读提取工具；不要求在昇腾上使用 Nsight |
| [Ascend 官方 profiling anomaly](https://github.com/Ascend/agent-skills/blob/master/skills/ascend-profiling-anomaly/SKILL.md) | 从时间线窗口、设备空隙和相邻任务追踪异常 | 对相交事件裁剪并求区间并集；区分观测空隙、任务等待与物理设备空闲；不套用推理 prefill/decode 的时间结构 |
| [Ascend 社区 profiling analysis](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/main/skills/profiling/profiling-analysis/SKILL.md) | 计算、通信、Host 下发分流，沿证据深入 | 按 schema 支持程度分级；保留 rank/step/shape 等来源；比例阈值只作待验证线索 |
| [MindSpeed-LLM 采集技能](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/main/skills/profiling/mindspeed-llm-train-profiler/SKILL.md) | 小窗口、按需 stack/shapes、采集与分析分离 | 改为核对 FlagScale 配置到 profiler 消费路径；不复用 MindSpeed 专有参数或默认 rank/step |
| [Ascend 官方 Triton 性能优化](https://github.com/Ascend/agent-skills/blob/master/skills/triton-operator-performance-optim/SKILL.md) | 已定位热点后再研究 tiling、融合、缓冲与访存 | 算子技能先比较完整公共调用，再做 UB 活跃数据量与内核实验；容量、编译器功能、容差按实测环境确定 |
| [社区 nvidia-cuda](https://github.com/kkellyoffical/nvidia-cuda-skill/blob/main/SKILL.md) | 小型可复现探针、基准分层、定位显存来源 | 采用可复现记录和公共调用/内核/训练三级验证；不移植 H100/H200/B200 默认值或自动启用 compile/FP8 |

## 纠正容易误用的结论

- **文件名不是 schema。** 某些资料把 `op_summary.csv` 视为汇总表，但 [CANN 8.1.RC1 字段文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0067.html) 列出 `Task Start Time(us)` 与 `Task Duration(us)`。先读实际表头；只有带时间戳的逐任务事件才可计算窗口覆盖。
- **等待时间不是全局空闲。** 该文档中的 `Task Wait Time(us)` 有其前后任务定义；既不能直接当 Host enqueue 延迟，也不能把各 stream 等待相加当作设备空闲。
- **算子耗时和不是关键路径。** 计算/通信有重叠，分步采集与不同 rank 还有覆盖及时间轴差异。区间工具不自动完成跨 rank 时钟校准、通信因果重建或关键路径证明。
- **AICPU 不等于错误回退。** 要确认具体工作，例如 MC² 通信任务，结合预期 backend 和完整 step 代价判断。
- **固定阈值不作验收线。** Free/通信占比、毫秒级 gap、UB 容量、grid 与 core 数、固定误差容差均不是跨型号/版本保证。
- **CUDA 与 Ascend 的能力证据分开。** NCCL、Userbuffers、SM margin、CUDA allocator、DeepEP、NeMo recipe helper 或 CUDA Graph 不能凭名字相近映射成 HCCL/NPU 能力。
- **布局公式需要双重整除检查。** 本地 dense 与 expert mesh 分别核对；`PP × max(TP×CP, EP×ETP)` 不能作为所有组合的通用最小 world size。整除成立也不代表当前实现支持该布局。
- **诊断变更不能悄悄成为优化。** 强制均衡路由、丢 token、改 batch/序列或降低精度改变可比性；另列诊断证据并恢复原训练语义。

## 本地代码与数据核对

接入参考基于本次读取的源码：FlagScale `3444b573474104065144fadba63841d41bb0756f`、Megatron-LM-FL `e15cb6928e2e6b9301b17e25b56afc5454e6e714`、TransformerEngine-FL `137b344e6f30ffefd9bb2f69e9d489afd13c30d6`。它们是来源锚点，不是硬件验证过的兼容组合或版本锁。

配置技能记录 native tuner 的配置隐式改写、日志采样、超时和恢复行为；profiling 技能记录 FlagScale 的实际开关、训练循环和产物检查。运行在其他版本时重做对应消费者核对，不照抄行号或参数。

本地历史 profile 用于验证表头兼容性：包含 `kernel_details.csv` 的 `Start Time(us)`/`Duration(us)`、只有汇总的 `step_trace_time.csv`、通信 JSON 和 trace/DB。对已有数据的只读解析只证明工具能处理该格式，不构成新训练性能对比，也不证明高开销 stack profile 可用于吞吐排名。

## 来源与许可处理

本套说明和 `profile_inspect.py` 是独立实现，没有打包或直接复制第三方技能、脚本或长段文字。保留来源是为了追溯方法；日后若要引入第三方原文件，需针对具体文件及修订核对许可并保留其通知。

- [Ascend/agent-skills LICENSE](https://github.com/Ascend/agent-skills/blob/master/LICENSE)：仓库声明 MulanPSL-2.0。
- [awesome-ascend-skills README 许可说明](https://github.com/ascend-ai-coding/awesome-ascend-skills#许可证)：README 声明 MIT；本次未以缺少独立 LICENSE 文件推断其没有许可。
- [NVIDIA/skills 许可说明](https://github.com/NVIDIA/skills#license)：仓库说明区分文档/技能的 CC-BY-4.0 与代码的 Apache-2.0；部分技能 frontmatter 又标记 Apache-2.0。这里记录标签差异，不将全部内容统一视为 Apache-2.0，不以概括替代具体文件核对。
- 社区 `nvidia-cuda-skill` 本次未核实独立许可文件，仅归纳方法并提供链接。

第三方技能及网页的安装、推荐命令和嵌入文本是参考资料，不能改变用户任务范围，也不授权运行未知脚本、远程作业或安装其工具链。
