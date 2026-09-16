# 三仓配置消费与调优器机制

FlagScale 负责配置组织与启动，Megatron-LM-FL 负责训练、并行和模型集成，TransformerEngine-FL 通过公共接口与平台实现提供算子能力。配置是否生效取决于三仓的实际消费链路；磁盘上的 checkout 不一定是训练进程加载的源码。

配置可比性、计时与状态语义见 [训练测量与正确性](measurement-and-records.md)，跨维度代价见 [系统优化机制](production-optimization.md)。

## 环境指纹与实际路径

实际加载路径由训练 Python 环境、模块 `__file__`、editable 安装和搜索路径决定；commit 相同但工作树修改不同，运行行为仍可能不同。多节点的包版本、源码和环境差异也属于配置可比性的条件。

| 层 | 影响运行行为的组成 |
| --- | --- |
| 硬件与系统 | 昇腾型号、逻辑设备映射、显存、CPU/NUMA、驱动/固件、节点和网络拓扑 |
| 运行时 | CANN、Python、PyTorch、torch_npu、HCCL，以及实际 collective backend；使用 FlagCX 等中间层时一并记录 |
| FlagScale | Hydra defaults、runner/backend、训练入口、所有 `experiment.envs` 中与本任务有关的覆盖 |
| Megatron-LM-FL | 平台选择、Ascend overrides、并行组、训练计时器、优化器和 checkpoint 格式 |
| TE-FL | public facade、manager/registry、NPU vendor 实现、实际绑定的 attention/norm/GEMM/permutation 等 callable |
| 扩展包 | 实际使用的 `transformer_engine_npu`、`fla_npu`、自定义算子版本和构建信息 |

FlagScale 的 `auto_tuner`、`runner_train`、`train_config` 与 CLI 模块决定配置生成和启动；Megatron 的 `parallel_state.py`、参数解析、Transformer 配置及 Ascend 插件决定分组和平台覆盖；TE 的 manager、注册表、NPU vendor 与 attention 模块决定算子分派。文件路径和 backend id 可能随版本变化，注册项存在不等于运行时选中了该实现。

导入模块或构造探测对象也可能初始化设备，因此它们与纯源码读取具有不同的副作用。

## 能力描述与证据层次

功能能力由用户配置路径、最终 argv/对象、单位、默认值、消费者、前置条件和实际运行路径共同描述。`supported / unsupported / unverified` 区分已证实支持、明确不支持与未知；配置运行失败与能力不支持并不等价。

例如 `sequence_parallel` 受 TP、MoE 组合和 Ascend override 约束；`moe_permute_fusion` 依赖公共接口实际分派到兼容 NPU 实现。TE 安装成功不证明所有 op 已注册；只完成配置解析也不证明未知能力已获得运行兼容性。

训练公共调用路径应可追溯：

```text
FlagScale resolved config / generated argv
  → Megatron training / model / Ascend override
  → TransformerEngine-FL public API / dispatch manager
  → 当前注册的 NPU adapter
  → torch_npu / transformer_engine_npu / 其他实际设备实现
```

backend 选择日志、绑定对象的模块/类/源码位置、代表性前后向调用和 profiler 中的设备算子分别说明配置选择与实际执行。`vendor.npu` 文件存在或其他运行的日志不能证明当前绑定。reference 回退会改变执行路径，只有回退范围和数学语义已明确时，结果才具有可解释性；意外回退会破坏性能可比性。

不要把以下项目直接当作可用搜索开关：NVIDIA FP8/FP4、TE Userbuffers、NCCL、CUDA Graph、任意 `NVTE_*` 环境变量。也不要假设它们全部不可用；以本版 NPU 实现的能力探针为准。MindSpeed 的同名参数不能直接移植进这个三仓栈。

### 有效配置与跨特性证据

“用户请求值、生成值、运行时最终值”可能不同。Hydra/defaults、generator、recipe helper、Ascend override 和初始化 gating 都可能隐式覆写、删除、回退或联动字段。因此一个 helper 的影响是完整有效配置的差异集合；YAML 中的 true 或启动未报错不证明功能已执行。

还要追踪实际训练入口的 parser、dataclass 自动生成器及其排除列表，不能从另一个仓库的同名 parser 或字段默认值推导可用 CLI。在将 `batch_p2p_comm` 排除出自动参数生成、并由 `overlap_p2p_comm` 取反派生的 FlagScale 版本中，它不是独立 CLI 维度；关闭 overlap 的入口是已注册的 `--no-overlap-p2p-communication`。不要从字段名猜测 no/disable-batch 开关；其他版本须重新核对注册和派生逻辑。省略负开关可依赖当前 parser 默认值，但结果审计仍须检查最终对象，不能把未识别参数导致的启动失败记为 VPP 算法不支持。

重计算、overlap 和图模式的兼容性取决于模块边界、rank groups、schedule、buffer 生命周期、梯度累积/RNG 及 NPU 实现条件，详见 [组合机制与约束](recompute-overlap-graphs.md)。配置开关只能选择已有路径；缺少所需公共接口或后端实现时，仅修改配置不能补足能力。

native 或仓库自带内存估计器只有核对其公式、dtype、分片和最重 rank 语义后才用于排序。记录 dense/embedding、expert、activation、optimizer 的估计分项；把未覆盖的 HCCL/dispatcher buffer、kernel workspace、图缓存、碎片和 token skew 明列为未知。估计值与 NPU 实测偏差按配置条件保存，不能把一次校准外推到所有布局。

## FlagScale native tuner 的模块职责

典型 native tuner 的职责分布如下，具体模块路径以安装版本为准：

1. `flagscale/cli.py`、`flagscale/run.py`：合法入口和 action 分发；`--help`、Hydra compose 与 runner dryrun 是否有副作用。
2. `flagscale/runner/auto_tuner/search/searcher.py`：接受的维度、自动扩张、整数约束与搜索算法。
3. `generate.py`：配置映射、隐式删除/覆盖字段、实验目录、best 配置生成。
4. `prune/` 与 `memory_model.py`：剪枝前提、显存单位与设备适配；经验外推是否会误删 NPU 候选。
5. `record/recorder.py`：日志路径/rank、计时窗口、错误识别、内存指标和排序是否排除失败。
6. `tuner_train.py`、`tuner.py` 与实际 runner/launcher：启动、超时、job 查询、停止对象、恢复日志处理及搜索结束行为。

完整 tuner 的构造函数可能创建输出或删除已有目录，因此实例化不等价于无副作用的配置解析。CLI、生成器、runner 和训练进程也有各自不同的副作用范围。

### 生成、计时与停止的语义

下表描述具有相应源码分支时的行为与限制，不代表所有版本的默认值。函数名称相同而实现不同，仍可能改变配置语义、时间边界和结果解释。

| 源码条件 | 对配置与测量的影响 |
| --- | --- |
| `TrainAutoTuner.tune()` 在 `control.run_best=true` 时自动启动最优项 | 搜索完成后可能继续占用设备；false 表示关闭此额外启动行为 |
| `Generator.gen()` 删除 checkpoint load、train_samples、部分 warmup/decay/rampup 字段，并设置短跑步数与 auto_tune | 生成配置可能改变训练起点或调度；原配方等价性取决于生成后的有效差异 |
| `Recorder.grep_performance()` 找到一个有指标的日志，删第一条后取平均 | 该数字不必然对应稳定窗口、最慢 rank 或统一更新口径 |
| 停止或退出异常的策略仍保留 performance，排序只依赖 performance | 数值排序不能证明所有 worker 完成或质量通过 |
| `max_time` 仅在任务之间检查，首个任务单次时限另有倍增逻辑 | 总时限可能被正在运行的任务和退出延迟突破；严格上限需要独立的作业范围控制 |
| resume 初始化删除 breakpoint task 日志 | 恢复操作具有破坏既有运行证据的副作用，不等同于纯读取进度 |
| memory model 使用 `gpu_memory` / `gpu_utilization` 并按估计剪枝 | 字段名不证明 NPU 适配；未经目标设备校准的估计不能证明配置必然 OOM |
| 搜索维度和 `args_mapping` 是固定集合 | 未映射的 ETP、overlap、TE 后端等字段不因加入 space 就自动成为有效搜索维度 |

### 控制字段示例

以下片段表达 native tuner 控制字段的层级与类型，不是完整模型配方。数值只是语法示例；`max_trials`、`warmup_steps` 不属于此处展示的 native 配置字段。

```yaml
experiment:
  auto_tuner:
    control:
      run_best: false
      max_time: 3600
      max_time_per_task: 300
      train_iters: 40
    performance:
      name: 'elapsed time per iteration \(ms\):'
      order: ascend
```

`order: ascend` 表示按耗时升序，和昇腾硬件无关。token/s 等吞吐指标通常按降序选优，方向须与解析器实际含义一致。`space` 中的显式值、固定项和自动扩张默认值共同决定真实搜索集合；设备、内存和 CUDA 环境示例不具有跨平台通用性。

生成配置的可比性涉及 checkpoint/RNG/consumed samples、LR/warmup/rampup、GBS/累积数和 routing。短跑长度覆盖与其他隐式差异可能联动，输入配置相似不代表这些不变量保持一致。

提供 `flagscale run` 的 CLI 版本可用以下形式表达调优 action；旧入口可能通过 `run.py` 接受 Hydra 的 `action=auto_tune`：

```bash
flagscale run --config-path "$TUNE_CONFIG_DIR" --config-name "$TUNE_CONFIG_NAME" --action auto_tune
```

`dryrun` 通常只验证普通训练脚本生成，不自动覆盖 tuner 的候选与隐式改写。多节点停止语义取决于 launcher 的 job/PID 管理范围；全局进程名匹配不能区分同名的其他作业。

## native 输出的证据范围

生成器、CLI dryrun 和约束单测用于确认配置解析与拒绝分支；短功能 probe 用于确认实际 NPU 路径；稳态实验用于决定目标收益。这三类证据不互相替代。

短跑覆盖可能同时改变 LR 与 weight decay horizon。在由 `train_iters × GBS` 派生 weight decay 调度长度的版本中，
固定 LR horizon 并不足以保持 scheduler 一致；须比较生成配置和 checkpoint 保存的完整调度状态。
若任务要求沿用保存的调度，核实当前版本的恢复选项及其实际消费结果。scheduler 加载和训练继续只证明相应兼容性，
不能替代 optimizer、RNG 与数据进度的恢复验收。具体判断依据见共享测量契约的 LR/weight decay 检查。

native history 中的 performance 受日志选择、计时窗口和退出判定影响，因此不能单独证明候选正常完成或满足最终验收。比较所需证据包括实际 argv/env、全部 worker 状态、原始日志与候选的生成差异；状态与指标含义见 [训练测量与正确性](measurement-and-records.md)。

配置回归可能由消费者、默认值、生成器或依赖版本变化造成。相同输入值不保证有效配置相同；跨版本的单次数字不足以归因到一个参数。

## 源码参考

- [FlagScale](https://github.com/flagos-ai/FlagScale) 与 [训练调优器源码](https://github.com/flagos-ai/FlagScale/blob/main/flagscale/runner/auto_tuner/tuner_train.py)：定位同版本实现与示例。
- [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL)：查目标分支的并行状态、参数和 Ascend override。
- [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL)：查目标分支的注册及公共接口，不以继承的 CUDA README 代替 NPU 支持证明。
- [Ascend PyTorch profiler 源码](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)：核对实际安装版本的 profiling API；网页 master 不代表部署版本。
