<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 实机验证范围与限制

本记录说明已观察到的能力与失败边界，不是跨版本兼容承诺或生产性能基准。

## 2026-09-08 首轮：单机 16 逻辑 NPU 批量调优

在既有容器中使用真实 FlagScale `train_gpt.py`、Megatron-LM-FL 和 TE-FL NPU 路径，
通过 Agent 外层有界搜索执行。配置经过真实 FlagScale 参数转换函数，采用独立 torchrun 作业，
未执行 native tuner 的完整搜索/停止路径。全部实验文件写入独立目录，未修改三个远端仓库。

环境为 `Ascend910_9382`、PyTorch `2.7.1+cpu`、torch_npu `2.7.1.post6`，CANN 路径标识 `9.0.0`。
代码提交：FlagScale `b49ce7a77b3177072d307c73086088e379d5461e`，
Megatron-LM-FL `e03018df34119b2401da4b1606aed32d57cfff8c`，
TE-FL `137b344e6f30ffefd9bb2f69e9d489afd13c30d6`。
前两仓带原有修改且前后 diff hash 一致；TE-FL 原有 submodule 元数据问题使 diff 检查失败，不能据此声明其工作树 clean。

固定工作负载为 4 层、hidden 512、FFN 2048、8 heads、sequence 512、vocab 4096、BF16、
GBS 64、DP 16、TP/PP/CP 1、dropout 0、固定种子 mock GPT。验证目标是自动调优流程；
这些数字不代表 Qwen3.5、MoE 或真实数据上的训练收益。

| 验证项 | 实机结果 |
| --- | --- |
| 合法候选生成及静态拒绝 | MBS 1/2/4 可运行；MBS 3 和 TP 3 在训练启动前被拒绝 |
| 自动选择与重复性能比较 | 选择 MBS 4，累积次数从 4 降为 1，保持 GBS 64；三对独立运行的成对吞吐提升中位数 145.30%，超过基线波动 3.97% |
| 测量完整性 | 每次丢弃前 10 次更新，测量 30 次；核对全部 16 rank 的同一绝对更新、框架同步计时、有限 loss/grad、无跳步和正常退出；profile/状态快照作业不参与排名 |
| 显存 | 重复运行跨 rank reserved 峰值从 534 MiB 到 684 MiB；实际运行时可用容量约 61.27 GiB/逻辑 NPU；allocator 数字不含全部设备运行时开销 |
| 数值对齐 | 相同起点的 3 次更新通过预设 BF16 loss/grad/参数、更新差量及 optimizer/scheduler 检查；不能证明长期收敛等价 |
| 生成 profiler wrapper | 基线及候选各完成 8 次更新，全部 16 worker 正常退出；rank 0/15 各取得真实计算和 HCCL 记录，分别为 3786/2140 条任务 |
| checkpoint 完整恢复 | **失败**：模型权重完全一致、optimizer 数值在容差内、调度/样本进度一致，但全部 16 rank 的 torch CPU RNG 不一致 |

三对性能结果只支持这一固定合成工作负载上的有限重复证据，不作统计显著性声明。
profile 的记录区间未覆盖部分不等同于设备空闲；只有两个 rank 的采集，不能外推全部 rank 的通信关键路径。
候选通过性能门禁仍不能标记 `verified` 或进入 `best/`，因为必需的恢复门禁失败。

## 从实际失败补充的检查

首次短跑恢复在进入训练前因 weight decay 调度长度与 checkpoint 不一致而失败。
在契约明确要求沿用保存的调度后，显式使用本版本的 `use_checkpoint_opt_param_scheduler`，
新 attempt 加载成功，保留原失败记录。详见 [测量契约](measurement-and-records.md)。

恢复 RNG 的差异来自本次 mock 数据路径：checkpoint 恢复全局 CPU RNG 后，重新创建 train/valid/test
三路 DataLoader iterator，各消耗一个 int64 `base_seed`。在全部 16 rank 上，从连续训练结束的状态
创建三个 CPU DataLoader iterator，可逐字节复现恢复运行的 CPU RNG 状态；两次运行各自的 train 前后该状态均未变化。
调用顺序由目标版本的函数结构和本轮日志确认，诊断未修改快照或 RNG 来使门禁通过。
这支持在修复时检查数据迭代器专用 generator 与状态恢复的生命周期；任何修复仍需新的等价续训验证，不能直接推广到其他数据路径。

首轮尚未验证：生产 Qwen3.5/真实数据、TP/PP/CP/EP 布局搜索与重分片、MoE、重计算/overlap/NPU 图组合、
OOM 边界、故障注入与超时停止、多机通信、算子源码补丁、长期训练和完整 native tuner。
四个技能中的 profiling 采集分析与配置搜索有上述实机覆盖；算子专项尚无本轮源码优化实测。

原始验证材料按 [实验记录](measurement-and-records.md) 保存契约、attempt、全部 rank 记录、profile、
CPU 状态快照、独立审计及恢复诊断。完整审计结论为 `not_verified`；不能将文档/离线测试通过替代实机门禁。

## 2026-09-08 扩展轮：分布式策略与后端边界

继续在同一环境、单机 16 逻辑 NPU 上进行独立实验，用户将本轮总预算延长到四小时。
Dense、八层 VPP 模型、小型 MoE、含共享专家的 MoE、真实 Qwen 短序列各自形成独立工作负载组；
不同组不互相排名。并行策略允许变化；进入同组质量/性能比较的候选必须保持数学模型、实际 padding 后词表、
GBS、序列、精度和比较起点。仅兼容性与冷启动记录单列，不据此声称同起点等价。
“运行完成”“数值对齐”“稳定性能复测”“完整恢复”是不同门禁。

扩展轮共记录 103 个 attempt（含失败与工具修复后的新 attempt）：52 个兼容性 probe、7 个种子
checkpoint、26 个质量/恢复作业、15 个性能作业、3 个 profile。逐 rank 审计确认 84 个完成各自
要求的更新与退出；其中 59 个达到至少六次更新，其余 25 个是有意缩短的种子、质量、恢复或采集作业。
19 个未完成或配置不成立；观测器错误单独归因，不能将其全部称为后端失败。

实际完成的配置覆盖 TP 1/2/4/8、PP 1/2/4、TP2+PP2、CP1/2、SP 开关、PP4+VPP2，
以及 MoE EP1/2/4/8/16、ETP1/2；ETP4/8 只在非 MoE 模型的派生配置中出现，不算 MoE ETP 验证。
本轮 MBS 为 1/4，首轮另覆盖 2；数据 workers 对照为 0/2，Qwen 沿用配方的 8。
它们是代表性配置集合，没有穷举这些维度的所有组合。

已经确认的限制如下；失败记录保留，不用后续成功覆盖：

- TP 会改变默认词表 padding。此 NullTokenizer 配置的原始词表为 4096，加 EOD 后为 4097，
  TP1/2 的默认 padding 分别为 4224/4352。先前 TP2 对比不满足同模型条件；
  显式调整 divisor 后，TP2 实际 padding 为 4224，并通过初始参数逐元素相同检查。
- 同布局的 full uniform/block 重计算、selective core attention 和梯度重叠通过三次更新的
  模型、optimizer、调度、起始 RNG 及 loss/grad 检查。分布式优化器梯度/参数重叠也通过。
  这不证明长程收敛或完整续训。
- PP2、PP4 和上述 TP2 对比通过模型参数及更新差量容差，但只属于模型参数对齐：
  当前跨 TP/PP 加载路径会忽略 RNG，不能标为完整 optimizer/RNG 恢复验证。
- 八层模型的 PP4→VPP2 开启/关闭两路通过模型参数比较，83 个参数的初始值相同，
  更新差量相对 L2 约 0.013699、cosine 约 0.999906。固定 VPP2 布局后，P2P overlap
  开启与关闭的六次更新完全一致，16 rank 的 optimizer、scheduler、初末 RNG 和进度均通过。
  这说明该具体配置的兼容性和数值结果，尚未测得实际通信被隐藏的时间比例。
- CP2 完成训练，初始模型一致、loss/grad 在容差内，但更新差量相对 L2 为 0.168420、
  cosine 为 0.985778，未通过预设 0.05/0.99 门禁。实际调用计数确认参考 attention 回退；
  源码与 CPU 位置反例提示 zigzag CP 切片与参考实现连续位置假设不一致。
  未采集最终实际 mask，未验证修复，不能把该推断写成已解决根因。
- selective MLP 重计算在首次 backward 出现 CheckpointFunction 递归并最终 OOM；
  按后向执行失败记录，不能当作普通容量不足或碎片问题。
- MoE 从 EP1 普通优化器 checkpoint 加载到 EP2/4/8/16 在训练前出现 `param_state` 缺失。
  独立 EP8 checkpoint 可运行 allgather/alltoall；冷启动 EP2/4/16、TP2+EP8+ETP1、
  TP2+EP4+ETP2 也可运行。冷启动成功不证明 EP 重分片后等价续训。
- 同 EP8、同 checkpoint 的 allgather→alltoall 三次更新通过严格 16 rank 状态比较：
  模型更新、loss/grad 和初末 RNG 一致，optimizer 在原容差内。另建 DOpt
  `fully_reshardable` checkpoint 后，EP1→EP8 加载并完成六次更新；该跨 EP 项仅验证运行兼容性，
  未验证跨 EP 完整状态等价。
- 共享专家 overlap 的六次更新虽能运行，14/16 rank 的更新差量相对 L2 为
  0.050564–0.054907，超过预设 0.05，严格质量检查失败；普通参数/optimizer allclose、
  loss/grad、RNG 和进度检查通过。没有因接近阈值而调整容差，也不直接认定其数学实现错误。
- 在已能运行的 grouped GEMM + expert overlap 配置上开启延迟权重梯度计算，
  首步在 `te_general_grouped_gemm` 发生 no-grad 创建的 view 被 grad-enabled 路径原地修改的错误；
  未取得完整更新后的状态，不能给出数值对齐通过结论。
- `local` graph 在初始化 TE RNG tracker 时访问空的 `torch.cuda.default_generators` 而失败。
  当前 TE 模型的另一 `transformer_engine` graph 候选共享这一前置路径，按依赖未满足而不再启动；
  未取得训练中的 capture/replay 证据，不能从 NPU graph API 存在推断训练图已支持。
- 真实 Qwen3.5 配方保持架构、路由、数据和 tokenizer，仅在独立组中缩短序列为 1024；
  原 permute fusion 路径发生 Triton kernel 失败，关闭该融合后全部 16 rank 完成六次更新。
  峰值 reserved 约 59.19 GB。此结果不验证 64K 生产长度、完整数值对齐或性能收益。

分布式优化器审计还遇到显式空参数组：`params=[]` 的组不执行 step，计数保持不变；
非空组仍须准确推进，跨运行状态仍须一致。依实际参数归属修正审计后，原误拒绝的质量对通过，
旧报告仍保留；不能将任何缺失 optimizer state 解释成空组。

质量采样还曾错误读取多子优化器 `ChainedOptimizer.optimizer` 的单优化器属性，
使两次 MoE 实验在首步前中止。修复观测器为先遍历全部子优化器，使用新 attempt 和新快照 schema；
这是验证工具故障，不作为 MoE 后端不支持的证据。后续快照比较改为受限 CPU 加载，
仅为实际 NumPy uint32 RNG 状态允许固定重建类型，未知对象不自动放行。

本轮重复出现三次首步 HCCL `hcclGetRootInfo` / `EI0019` 错误，日志中的端口为 65536。
该数字超出合法端口范围；设备空闲和端口汇总未证明端口耗尽，也未查明其生成机制。
经核对 [CANN 9.0 官方说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0143.html)
与已安装库标识，只对新实验 worker 设置 `HCCL_HOST_SOCKET_PORT_RANGE=auto`；
隔离 DOpt 短跑完成六次更新，16 rank 正常退出。此兼容结果不证明已定位根因，
不等于 HCCL 性能调优，也不外推到文档明确限制的 A2 MC² 场景；未修改全局网络或其他作业。

性能初筛与复测分开保留。普通优化器与 DOpt 使用各自固定 checkpoint 起点，
初筛观察到的梯度重叠增益未成为最终收益：DOpt 三对交错复测中，第一对基线漂移超过 10%，
第三对两次作业均首步建连失败；唯一有效对的候选吞吐比为 0.988263。
因此本轮性能复测门禁未通过，不舍弃失败对来计算三对中位收益，也不混入新端口环境的短跑计时。

扩展轮还对 DOpt+梯度/参数重叠执行连续训练、保存退出、恢复三段验证，均完成所需更新。
最终模型在 16 rank 上逐位相同，optimizer 浮点状态在容差内，scheduler 和绝对更新进度通过；
但全部 16 rank 的 CPU RNG 不一致。保存/加载边界还发现 rank 14/15 的显式空参数组计数
从 2 变成 4，连续运行最终仍为 2、恢复运行最终仍为 4；各运行内部空组保持不变，
并不能取消跨 checkpoint 的精确状态差异。边界和终点的严格恢复检查均失败，未改变 RNG 或放宽门禁。
本轮用私有 CPU generator 从保存边界状态执行三次 int64 draw，或创建三次带该 generator 的合成
DataLoader iterator，均在 16/16 rank 上复现加载边界的 CPU RNG；全局 RNG 和原快照保持不变。
这验证了状态偏移等价，尚未验证本轮真实数据路径的三次调用因果，不能直接套用首轮的根因结论。

PP2、MoE EP8 alltoall 和 Qwen permute-off 各完成独立 profile，前两组八步、Qwen 五步，
全部 16 worker 正常退出，rank 0/15 的 wrapper、回调和实际 CPU/NPU/通信产物均通过审计。

| Profile 组 | rank 0 导出设备任务 / 通信任务 | rank 15 导出设备任务 / 通信任务 |
| --- | --- | --- |
| Dense PP2 | 1330 / 40 | 1522 / 50 |
| MoE EP8 alltoall | 1434 / 70 | 1434 / 70 |
| Qwen3.5 seq1024 permute-off | 20326 / 780 | 19725 / 781 |

这些是导出表计数，不是完整训练的 kernel 总数或通信暴露时长。用同一任务的名称、精确起点和
Model/Stream/Task ID 关联 trace/CSV，未拟合时间偏移；每个采样 rank 都取得超过三条精确身份匹配。
PP/MoE 对应训练调用 5/6，Qwen 对应调用 5。host ProfilerStep 窗口可与异步任务相交，
不能据此声称完整更新的所有异步设备工作都在窗口内；通信报告的 step 归属计数另行保存。
采集/状态快照运行不进入性能排名，两个采样 rank 不代表全部 16 rank 的关键路径。

尚未覆盖 FP16/FP8、CPU/NVMe offload、多机和跨节点 HCCL 算法、非零 dropout 的随机路径、
长序列/完整生产模型、真实数据长程续训、evaluation 与数据切换稳定性、训练中的 NPU graph replay，
以及算子源码补丁的性能闭环。native tuner 仍仅有接入与停止边界审查；本轮执行的是 Agent 外层搜索。
没有通过全部门禁的新最佳配置，保留原配置与失败证据，不发布生产加速结论。
最终检查无本轮 worker 残留、无未完成 attempt，8 个设备组均报告空闲，未超过四小时和 104 次内部上限。
FlagScale/Megatron 的原有 diff/status hash 与前次基线相同；TE 提交一致，完整工作树认证仍受原有 submodule 问题限制。
