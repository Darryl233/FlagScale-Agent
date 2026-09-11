# 三仓能力与调优器接入

本篇提供配置消费链路、能力证据字段和 native tuner 的已核对行为，供工作流按需查询。以下路径是查找入口，实际安装源码优先；不要假设本地 checkout 就是远程进程加载的代码。

记录与验收共用 [测量与实验记录](measurement-and-records.md)。本文件只补充配置接入证据，跨维度收益判断参考 [生产优化手册](production-optimization.md)，不另建一套计时或状态协议。

## 环境指纹与实际路径

在目标容器的训练 Python 中记录包版本和模块 `__file__`，并核对三仓 commit、dirty diff 与 editable 安装位置。对多节点抽查所有节点一致性，差异必须记录或解决后再比较。

| 层 | 需记录 / 需核对 |
| --- | --- |
| 硬件与系统 | 昇腾型号、逻辑设备映射、显存、CPU/NUMA、驱动/固件、节点和网络拓扑 |
| 运行时 | CANN、Python、PyTorch、torch_npu、HCCL，以及实际 collective backend；使用 FlagCX 等中间层时一并记录 |
| FlagScale | Hydra defaults、runner/backend、训练入口、所有 `experiment.envs` 中与本任务有关的覆盖 |
| Megatron-LM-FL | 平台选择、Ascend overrides、并行组、训练计时器、优化器和 checkpoint 格式 |
| TE-FL | public facade、manager/registry、NPU vendor 实现、实际绑定的 attention/norm/GEMM/permutation 等 callable |
| 扩展包 | 实际使用的 `transformer_engine_npu`、`fla_npu`、自定义算子版本和构建信息 |

先看设备占用与分配；导入/探测可能初始化设备，运行探针也要限于已分配设备。环境快照只保留版本与相关配置，不能整包输出凭据、token 或其他用户数据。

从工作区分别查找：

```bash
rg --files "$FS_ROOT" | rg 'auto_tuner|runner_train|train_config|(^|/)run.py|cli.py'
rg --files "$MG_ROOT" | rg 'parallel_state|arguments.py|transformer_config|plugin/.+(Ascend|ascend|npu)'
rg --files "$TE_ROOT" | rg 'plugin/.+(manager|ops|register_ops|npu|ascend)|attention'
```

`FS_ROOT`、`MG_ROOT`、`TE_ROOT` 是在执行环境中已确认的路径，不是需要写入用户全局 shell 配置的变量。文件路径、backend id 大小写可能变化；查到注册文件后仍要验证运行时选择。

## 能力矩阵

每行记录：功能、用户配置路径、最终 argv/对象、单位、默认值、消费者文件/函数、前置条件、runtime probe、结论 `supported / unsupported / unverified`。该结论描述能力；未核实另标 `evidence_level=unknown`，不作为候选运行状态。

例：`sequence_parallel` 必须同时检查 TP、MoE 组合和 Ascend override；`moe_permute_fusion` 必须验证调用的公共接口确实落到兼容 NPU 实现；TE 安装成功不证明所有 op 已注册。`unverified` 不直接放进性能搜索，先做有界兼容性探针。

训练公共调用路径应可追溯：

```text
FlagScale resolved config / generated argv
  → Megatron training / model / Ascend override
  → TransformerEngine-FL public API / dispatch manager
  → 当前注册的 NPU adapter
  → torch_npu / transformer_engine_npu / 其他实际设备实现
```

必要证据包括 backend 选择日志、绑定对象的模块/类/源码位置、代表性前后向调用和 profiler 中的设备算子。不要仅凭 `vendor.npu` 文件存在或某条历史日志认定接入成功。允许受控 reference 回退，但必须记录；意外回退使本轮比较失效。

不要把以下项目直接当作可用搜索开关：NVIDIA FP8/FP4、TE Userbuffers、NCCL、CUDA Graph、任意 `NVTE_*` 环境变量。也不要假设它们全部不可用；以本版 NPU 实现的能力探针为准。MindSpeed 的同名参数不能直接移植进这个三仓栈。

### 有效配置与跨特性证据

同时保存“用户请求值、生成值、运行时最终值”。检查 Hydra/defaults、generator、recipe helper、Ascend override 和初始化 gating 是否隐式覆写、删除、回退或联动其他字段。能力矩阵记录每项 helper 的完整差异集合；仅看 YAML 中的 true 或启动未报错不证明功能生效。

还要追踪实际训练入口的 parser、dataclass 自动生成器及其排除列表，不能从另一个仓库的同名 parser 或字段默认值推导可用 CLI。[本机版本例证](hardware-validation.md)中，FlagScale 将 `batch_p2p_comm` 排除出自动参数生成，并由验证后的 `overlap_p2p_comm` 取反派生；只有 `--no-overlap-p2p-communication` 关闭 overlap，猜测的 no/disable-batch 开关都未注册。省略负开关可依赖 parser 默认值，但结果审计仍须检查最终对象，不能把未识别参数导致的启动失败记为 VPP 算法不支持。

对重计算/overlap/图组合记录模块边界、rank groups、schedule、buffer 生命周期、梯度累积/RNG 和 NPU 实现条件。详见 [组合调优参考](recompute-overlap-graphs.md)。若需要源码才能实现候选，先交接精确缺口与调用路径给算子/源码专项，不在配置搜索中隐式扩成实现项目。

native 或仓库自带内存估计器只有核对其公式、dtype、分片和最重 rank 语义后才用于排序。记录 dense/embedding、expert、activation、optimizer 的估计分项；把未覆盖的 HCCL/dispatcher buffer、kernel workspace、图缓存、碎片和 token skew 明列为未知。估计值与 NPU 实测偏差按配置条件保存，不能把一次校准外推到所有布局。

## FlagScale native tuner 的源码审查

按顺序读取当前版本对应模块中的相关完整函数：

1. `flagscale/cli.py`、`flagscale/run.py`：合法入口和 action 分发；`--help`、Hydra compose 与 runner dryrun 是否有副作用。
2. `flagscale/runner/auto_tuner/search/searcher.py`：接受的维度、自动扩张、整数约束与搜索算法。
3. `generate.py`：配置映射、隐式删除/覆盖字段、实验目录、best 配置生成。
4. `prune/` 与 `memory_model.py`：剪枝前提、显存单位与设备适配；经验外推是否会误删 NPU 候选。
5. `record/recorder.py`：日志路径/rank、计时窗口、错误识别、内存指标和排序是否排除失败。
6. `tuner_train.py`、`tuner.py` 与实际 runner/launcher：启动、超时、job 查询、停止对象、恢复日志处理及搜索结束行为。

不要通过实例化完整 tuner 做“只读验证”：构造函数也可能创建或删除历史输出。先静态检查，在新的实验目录测试配置生成，再进行单候选有界试运行。

### 已核对的源码行为

2026-09-08 本地参考快照：FlagScale `3444b573474104065144fadba63841d41bb0756f`；Megatron-LM-FL `e15cb6928e2e6b9301b17e25b56afc5454e6e714`；TransformerEngine-FL `137b344e6f30ffefd9bb2f69e9d489afd13c30d6`。这些是编写时的阅读锚点，不是兼容性版本锁，也不代表做过 NPU 实测。目标版本可能不同，以下行为需复查。

| 观察到的行为 | 接入决策 |
| --- | --- |
| `TrainAutoTuner.tune()` 的 `control.run_best` 默认为 true | 显式设 false；确认关闭生效，再由外层做受预算约束的复测 |
| `Generator.gen()` 删除 checkpoint load、train_samples、部分 warmup/decay/rampup 字段，并设置短跑步数与 auto_tune | 每个生成配置都做语义 diff；续训/严格 LR 等价任务若无法保留不变量，采用外层搜索 |
| `Recorder.grep_performance()` 找到一个有指标的日志，删第一条后取平均 | native 数字仅作粗筛；重新计算稳定窗口，补齐 rank 与指标口径 |
| 部分被 tuner 停止或退出异常的策略仍可含 performance；排序主要看 performance | 以所有 worker 完整状态、数值与样本校验重新过滤，不能直接接受 history 第一行 |
| `max_time` 在任务之间检查，首个任务单次时限可加倍 | 外层预算必须覆盖首任务与退出延迟；严格总时限依赖有作业范围的 watchdog |
| resume 初始化路径可能删除 breakpoint task 日志 | 先保留历史，在新目录继续未完成队列；不盲目复用原 tuner 输出目录 |
| memory model 使用 `gpu_memory` / `gpu_utilization` 并按模型估计剪枝 | 字段名不证明 NPU 适配；未校准时不启用模型硬剪枝，用实测显存约束 |
| 搜索维度和 `args_mapping` 有明确集合 | ETP、overlap、TE 后端等额外维度不能只塞进 space；使用外层分组配置 |

### 接入配置片段

这是合并进现有可运行配置的控制字段示例，不是可独立启动的模型配方。数值仅示意预算，须按基线实测成本调整；`max_trials`、`warmup_steps` 等外层契约字段不属于此 native 配置。

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

`order: ascend` 表示按耗时升序，和昇腾硬件无关。使用 token/s 等吞吐指标时应按降序并核对解析器。复制本地 native 示例的 `space` 后，把所有支持的维度显式限定为本轮白名单，包括固定项；不要留下会自动展开未知组合的默认值。别复制 CUDA envs 或示例的设备/内存容量。

先生成一个候选，逐项核对 checkpoint/RNG/consumed samples、LR/warmup/rampup、GBS/累积数和 routing 等不变量以及允许的短跑覆盖。将 native 输出路径与本次候选/attempt 映射后，再生成其余队列；无法解释的隐式差异使该候选不具可比性。

经源码和帮助核实的 CLI 可采用下列形状，参数均需指向已生成配置：

```bash
flagscale run --config-path "$TUNE_CONFIG_DIR" --config-name "$TUNE_CONFIG_NAME" --action auto_tune
```

如该安装只有旧入口，则依实际 `run.py` 使用 Hydra 的 `action=auto_tune`。`dryrun` 通常验证普通训练脚本生成，不自动证明 tuner 的候选与隐式改写正确。多节点启动器必须能根据本次 job/PID 文件停止本作业；含无范围 `pkill` 的停止逻辑不适合共享环境自动搜索，先选择已有的按 job 控制方式。

## native 输出的证据范围

生成器、CLI dryrun 和约束单测用于确认配置解析与拒绝分支；短功能 probe 用于确认实际 NPU 路径；稳态实验用于决定目标收益。这三类证据不互相替代。

2026-09-08 的单机 16 NPU 恢复验证还发现：固定 LR horizon 后，缩短/延长 `train_iters` 仍会改变
派生的 weight decay horizon，导致已保存 scheduler 与新构造值不一致。按契约显式恢复 checkpoint 中完整 scheduler 后，
同一合成配方可以继续训练。这里只证明 scheduler 加载后的运行兼容性；完整 optimizer、RNG 和
数据进度恢复仍须独立验收，已经记录的 RNG 失败不会因此取消。见共享测量契约的 LR/weight decay 检查，不能只核对学习率。

native history 中的 performance 受日志选择、计时窗口和退出判定影响，因此不能单独证明候选正常完成或满足最终验收。比较所需证据包括实际 argv/env、全部 worker 状态、原始日志与候选的生成差异；状态与指标口径由 [测量与实验记录](measurement-and-records.md) 定义。

配置回归可能由消费者、默认值、生成器或依赖版本变化造成。相同输入值不保证有效配置相同；跨版本的单次数字不足以归因到一个参数。

## 当前源码缺失时的官方入口

- [FlagScale](https://github.com/flagos-ai/FlagScale) 与 [训练调优器源码](https://github.com/flagos-ai/FlagScale/blob/main/flagscale/runner/auto_tuner/tuner_train.py)：定位同版本实现与示例。
- [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL)：查目标分支的并行状态、参数和 Ascend override。
- [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL)：查目标分支的注册及公共接口，不以继承的 CUDA README 代替 NPU 支持证明。
- [Ascend PyTorch profiler 源码](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)：核对实际安装版本的 profiling API；网页 master 不代表部署版本。
