<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 已定位热点的算子方法

## 适用条件

已有明确公共接口失败或 profile 热点，且任务包含源码优化；只要求诊断时返回诊断即可。
输入实际调用、shape/dtype/layout、相关 rank/stage 和原始证据。
调用链或实现归属不清楚时，查 `know-ascend-operators` 索引及 `ascend_operators/operator-optimization.md`；
涉及设备内核实现时再查 `ascend_operators/kernel-experiments.md`。

## 生成候选

1. 追踪实际加载的 FlagScale → Megatron → TE-FL/FLA 公共 API → NPU 实现，确认热点执行路径。
2. 保存父版本与原有差异，围绕一个可验证假设在正确仓库实现最小补丁，保留用户原有修改。
3. 返回 `parent`、`hypothesis`、`change`、`checks`，其中 change 为实际补丁，checks 为公共接口测试。
   缺少足够证据时返回缺口；不要由一个失败调用自动扩大为全局分派重写或依赖升级。

## 额外检查

训练测量前，对照参考实现测试生产形状和受影响边界的公共 API 前向及所有可微输入的反向。
核对 dtype/device/shape、布局/索引/返回语义、实际后端，以及 unsupported 输入的原有行为；
按既定容差保存测试命令和结果，正确性失败不进入性能候选集合。
需要微基准时把 kernel 与公共调用耗时分开；端到端收益交回主循环步骤 4–5 验证。
微基准通过或私有函数可用不能代替训练调用与完整更新结果。

## 结果反馈

接口失败则修复或撤回本次补丁；接口通过但训练无收益时重新核对关键路径，或结束该方法。
把补丁、测试证据和待检查项返回主循环；不在此重复建立训练 A/B、排名和最终验收。
仅算子任务完成后可直接交付接口结果，并将训练收益标为未验证。
