<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 数据供给与 Host 调度

## 适用条件

已有配置、日志或 profile 支持数据等待、CPU 争用或 Host 下发开销的假设时进入；廉价配置对照不要求先采完整 profile。
复用当前数据路径与等待证据，原理按需读 `know-ascend-training` 的 `ascend_training/data-and-host.md`；仅缺 Energon reader/恢复语义时再读相关 `know-energon` 章节。
图捕获与编译边界调整按 [图与编译方法](graph-execution.md)，算子内部 Host 往返与同步按 [算子方法](operator.md)处理。

## 生成候选

从父配方的有效配置出发，选择对应操作，说明减少的是读取等待、CPU 争用还是下发开销。路径指合成配置；编辑被挂载为 `train` 的子 YAML 时省略 `train` 前缀，已有字段原位修改。

| 方向 | 具体操作 |
| --- | --- |
| 数据读取/预处理并发 | 调整 `train.system.num_workers` 为相邻较大或较小值；解码慢且 CPU 有余量时增大，CPU 争抢或 IPC 开销突出时减小，必要时比较 `0`。保持 sampler、数据与处理语义；Energon 仍走原 `WorkerConfig`/loader，不替换为普通 DataLoader。 |
| 数据预取深度 | 对已有配置入口调整有效 prefetch 值；普通 PyTorch loader 未暴露配置且任务允许改源码时，在实际 `DataLoader(...)` 构造处仅对 `num_workers > 0` 显式传入 `prefetch_factor`，如对照 2/4。不要直接添加不存在的 YAML 字段，也不对 Energon 套用该参数。 |
| CPU 亲和性 | CPU 迁移、抢占或 NUMA 访问值得优化时，在 `experiment.envs.CPU_AFFINITY_CONF` 比较当前模式与本版支持的 `"1"`/`"2"`；已有自定义核区间则保留映射。结合容器允许的 CPU 核及 NPU 拓扑选区间，不照抄其他机器的核号，不默认强制覆盖已有绑定。 |
| NPU 任务下发队列 | 当前二进制执行路径支持 Level 2 且下发突出时，在 `experiment.envs.TASK_QUEUE_ENABLE` 对照 `"1"`/`"2"`，同时观察显存。若仍有 `ASCEND_LAUNCH_BLOCKING=1`，先区分诊断运行与性能运行；完成诊断后恢复 `"0"` 的异步基线，不把阻塞诊断结果混入正常排名。 |

以上是常用起点，不限定范围或顺序。有证据且预算允许时，可深入检查读取—预处理—H2D—执行的依赖、CPU 热点与隐式同步，探索批量预处理或搬运流水。
代码候选写明目标函数和变更，主循环选中且任务允许源码修改后实施；H2D 异步化必须维护源缓冲生命周期和消费依赖，不能仅加 `non_blocking=True` 就宣称重叠。
复用已确认的支持与映射，仅补查候选缺口；已有有效数据不重复转换，也不换成 mock 数据。保持布局、MBS/GBS、精度、模型及任务要求的数据、评估和 checkpoint 工作。

## 额外检查

- **数据/Host 配置**：核对实际 worker、预取或线程绑定，检查样本顺序、有效 token 与更新进度，以及 Host 内存/共享内存和 NPU 峰值。只在涉及数据状态恢复时补续训验证；不能用队列短暂预填充代替持续供给收益。

供给收益需要持续运行的证据；局部等待减少不自动等于训练加速，mock 数据不能证明真实数据路径收益。
