<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# NPU 分配器设置

仅在变尺寸分配或碎片有证据时读取；原理按需读 `know-ascend-training` 的 `ascend_training/memory-and-sharding.md`。

有变尺寸分配或碎片证据时，确认当前 TorchNPU/芯片支持后，在 `experiment.envs` 的 `PYTORCH_NPU_ALLOC_CONF` 中合并 `expandable_segments:True` 做对照。保留无关设置，按本版约束处理冲突；已有等效设置时不重复生成候选。

固定工作负载与其他配置，覆盖完整更新；比较完整更新耗时、最重 rank 的 allocated/reserved 峰值与原 OOM 阶段。reserved 大于 allocated 本身不能证明碎片。
