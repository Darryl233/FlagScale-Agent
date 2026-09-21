<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 重计算与优化器状态分片

## 适用条件

框架机制先按需查询 `know-megatron-model` 的重计算与内存章节、`know-megatron-training` 的 optimizer/checkpoint 章节；
候选落盘和通用质量检查沿用主流程，下文补候选选择和专项检查。

已有 OOM 阶段或完整更新的跨 rank 峰值证据，需要降低内存，或回收过度重计算带来的时间开销。
取得父配置、首先失败/峰值最大的 rank 和阶段，以及模型、optimizer、activation、临时 buffer 的归因。
来源不明时返回最小缺口：初始化、首次 optimizer 更新与代表性更新各自的峰值/异常信息。
通过 `know-ascend-training` 按需查询 `ascend_training/recompute-overlap-graphs.md`、
`ascend_training/search-space.md` 和 `ascend_training/stack-capabilities.md`；模块标签与分片支持须核对当前实际消费者。

## 生成候选

1. 从父配置复制完整配方，按已定位的来源选一个方向：
   - activation：选择实际保存张量最多且当前实现支持的最小重计算边界；仍不足再扩大范围。
     已有重计算且内存有余量时，先缩小一处覆盖范围，检验能否减少重放成本。
   - optimizer 状态：检查当前优化器 family 与可用状态分片，生成仅调整分片的候选。
     若主要是权重常驻量，则返回并行布局方法所需证据；不默认用 activation 重计算处理。
   - 通信/重放临时 buffer：返回已定位的 overlap 或模块边界，先消融该来源，避免叠加更宽重计算。
2. 重计算保留当前 layout、MBS/GBS、精度和 backend；切换 selective/full 时，
   按 parser 清理互斥或失效的 modules、method、层数，逐项列出有效差异。
   分布式优化器不同时强制开启 grad-reduce/param-gather overlap；保留已有开关，除非当前实现要求联动。
   必须联动时引用对应 validator/消费者及原因，把整组变化作为候选，不归因为一个开关。
3. 返回 `parent`、`hypothesis`、`change`、`checks`，注明预期减少的内存来源与可能增加的重放/通信。
   若分片需要改变 optimizer family 或起始状态，先返回比较条件缺口，不能混入原状态的等价比较。

### FlagScale 配方示例

对支持 full/uniform 的 Megatron 模型，将以下片段合并到父配方；其他字段沿用父配置。
无重计算基线不设置这一组字段。通过 `train-run` 启动，核对日志中三个同名参数的实际值；
字段被拒绝或改写时再查当前版本的映射，不需要重新调查整个 launcher。

```yaml
train:
  model:
    recompute:
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
```

分布式优化器使用 `train.system.use_distributed_optimizer`。以下是含分布式 checkpoint 的候选片段；
仅在当前版本要求或任务需要时联动保存格式，并记录这项差异。不要同时开启 overlap。

```yaml
train:
  system:
    use_distributed_optimizer: true
    checkpoint:
      ckpt_format: torch_dist
```

配方能运行只代表配置可用，分片和保存/恢复仍按下文单独验收。

## 额外检查

- **启动前**：核对目标层/模块实际存在、保存与重放边界、PP/VPP 层分配及当前 overlap/graph 兼容性；
  核对状态分片的参数组、dtype、checkpoint schema 和实际加载范围。
- **短跑**：重计算诊断区分目标层的原始前向与反向阶段重放；只统计 checkpoint 入口次数不能证明重放。
  分片诊断检查实际 optimizer 状态；两者均测量首次状态分配与完整更新峰值。
  保留 original/replay 的异常栈，区分峰值迁移、重复 backward 重入和真正的内存节省。
- **质量比较**：重计算检查 RNG/dropout、梯度与参数更新；分片检查各活动参数组的主参数、
  optimizer 状态与更新步数。比较方法按需查 `ascend_training/measurement-and-records.md`。
  比较 checkpoint 时先核对完整键集、形状、dtype 和有限性，再比较张量；最终权重一致不替代逐步梯度检查。
  任务要求续训时补齐保存/恢复检查；缺失状态与空参数组不能等同。

## 结果反馈

主循环返回有效结果后，按既定容量目标及允许的时间代价判断下一步，只扩展尚未解决的内存来源。
OOM 转移到后续阶段仍返回 OOM，并注明新阶段；重复 backward 重入先返回实现诊断，不继续盲扫边界。
无收益时先区分未命中、外层已覆盖、峰值来自其他来源和重放代价，再缩小范围或结束该方向。
质量失败则拒绝当前候选并返回具体差异；单项有效后需要组合其他方法时，交主循环生成新的组合候选。
