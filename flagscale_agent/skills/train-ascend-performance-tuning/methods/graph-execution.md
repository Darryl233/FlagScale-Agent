<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 图捕获与编译

## 适用条件

已有配置、日志或 profile 支持图捕获、编译边界或重编译成本值得优化的假设时进入；廉价配置对照不要求先采完整 profile。
复用已确认的目标调用与 NPU 支持范围，机制按需读 `know-ascend-training` 的 `ascend_training/graph-execution.md`。数据供给与运行时任务队列调整按 [数据与 Host 方法](data-and-host.md)，融合/后端选择按 [算子方法](operator.md)。

## 生成候选

从父配方与实际调用出发，选择已支持的路径，写明目标模块、输入条件和拟调整边界：

| 方向 | 具体操作 |
| --- | --- |
| 局部训练图捕获 | 已确认当前 TorchNPU 支持时，在目标静态模块调用处接入 `graphed = torch_npu.npu.make_graphed_callables(module, sample_args)`，用返回 callable 替换该处调用；样例张量匹配实际 shape/dtype/`requires_grad`，保留原参数对象及外围训练循环。已有图路径则对照捕获范围/shape 集，不重复包裹。 |
| 已支持的训练编译 | 当前 NPU backend 已证明支持目标前反向时，在目标 callable 处使用 `torch.compile(target, backend=npu_backend, ...)`；`npu_backend` 指已确认的后端对象/名称，不使用默认后端猜测支持。根据已有 graph break/重编译证据调整编译边界或 `dynamic`，保留 eager 对照；不把推理用 `npugraph_ex` 当作通用训练入口。 |

以上是常用起点，不限定范围或顺序。有证据且预算允许时，可深入检查 graph break、重复准备和图间开销，探索更合适的编译/捕获边界。
保持布局、MBS/GBS、精度、模型与数据语义；代码候选写明目标函数和变更，主循环选中且任务允许源码修改后实施。复用已确认的支持，仅补查当前候选缺口。

## 额外检查

- **图/编译**：区分 warmup、compile/capture、replay；核对实际执行、图断裂/回退、输入更新和图资源。接入新边界时覆盖变化的输入、多个更新及相关梯度累积、RNG/dropout、重计算；准备阶段产生的梯度、RNG 或状态副作用不得污染对照初态。一次性准备成本单列，反复编译/捕获计入实际成本。

准备成本与稳态收益分开解释；合法局部回退的收益只归于实际执行范围。持续回退或反复捕获时，结合成本判断是否缩小或修复边界，capture 成功不代表训练已加速。
