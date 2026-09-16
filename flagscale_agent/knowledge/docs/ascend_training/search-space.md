# 昇腾训练配置搜索空间与约束

训练配置搜索同时受整数约束、模型语义、后端能力和性能代价限制。小型离散空间适合确定性枚举与约束剪枝；已有可行配置附近适合局部邻域搜索；随机或贝叶斯方法的价值取决于空间规模、测量噪声和可用样本数。可比性与指标定义见 [训练测量与正确性](measurement-and-records.md)，跨维度代价见 [系统优化机制](production-optimization.md)。

实现约束由当前 parser、并行初始化和实际后端共同决定，不能从参数名推导所有 Ascend 环境的支持范围。重计算、overlap 与图模式的相互作用见 [组合机制与约束](recompute-overlap-graphs.md)；配置消费和 native tuner 行为见 [三仓与调优器](stack-capabilities.md)。

## 并行布局与批量的整数约束

以下算式适用于常见同构 decoder-only 布局。encoder-decoder、异构并行、动态 CP 或自定义 rank mapping 需要从当前初始化代码推导，不能套用简式。

```text
W = 本次分配的总训练进程数
D = W / (TP × PP × CP)
G = GBS / (MBS × D)             # 每个 optimizer step 的 microbatch 数

要求：所有并行度、MBS、GBS 为正整数
      W % (TP × PP × CP) == 0
      GBS % (MBS × D) == 0，G >= 1
```

SP 是张量分区模式，不另乘进 W；EP 也不能在 dense DP 公式中再次相乘。MBS/布局变化后，核对框架实际使用的 G 与有效 GBS，而不只验证手算值。batch rampup 存在时，要在同一已固定的 batch 阶段比较。

布局变化还要核对 tokenizer 的实际词表、额外 special token、`padded_vocab_size` 和 embedding/output 分片形状。配置中的 `vocab_size` 不一定包含 EOD，也不一定等于填充后的词表；新增的 logits 不能自动视为不影响 loss 的空位。在 NullTokenizer 额外增加 EOD、且词表按 `make_vocab_size_divisible_by × TP` 向上填充的实现中，改变 TP 可能同时改变实际词表大小。这是特定实现的派生关系，须核对当前 tokenizer 与 padding 函数，不能作为通用 tokenizer 公式。若为保持同一实际形状调整 divisor，先核对 parser 与 checkpoint 的比较条件，将其列为必要派生差异；若更改原始词表，则另建工作负载组与 seed，不能混入原组排名。

| 约束 | 校验方式与例外 |
| --- | --- |
| TP | hidden/FFN/head/group 分片按当前实现检查；GQA/MQA 可存在 KV 复制等路径，不能一律要求 KV heads 被 TP 整除 |
| SP | 通常依赖 TP > 1；确认 Ascend collective 和形状要求，不能将 TP=1+SP 当有效组合 |
| PP/VPP | 简单均匀布局检查层数与 stage/chunk 整除；自定义 layout、首尾不均匀层、embedding/loss/MTP 必须按实际解析器计数 |
| microbatch 调度 | G 必须满足实际 pipeline schedule 的最小值/整除要求；不要把某个版本的 VPP 限制固化到所有版本 |
| CP | attention 的具体通信模式、序列切分、mask、packing/THD、GQA/MLA、dropout backward 都须有支持证据；某些算法需要长度被 2×CP 整除，但以该后端为准 |
| EP/ETP | expert 数与分片、dense/expert rank groups、dispatcher、grouped GEMM 和 TP/SP 组合需同时合法 |
| 优化器 | distributed optimizer、overlap、梯度累积、checkpoint 格式和状态分片需要兼容 |
| 重计算 | granularity/method/modules/层数需相互匹配；未开启 full recompute 时不要遗留只适用于 full 的参数 |

在 Megatron-LM-FL `parallel_state.py` 采用独立 expert rank generator、其 CP 固定为 1，且 ETP 默认继承 TP 的实现中，expert 分组关系为：

```text
ETP = 显式 expert_tensor_parallel_size；未给定时继承 TP
EDP = W / (ETP × EP × PP)
expert rank generator 的 CP = 1
```

这是该实现的分组规则，不意味着 CP 在所有 MoE 分支都可用。还需核对 `W % (ETP×EP×PP)==0`、PP groups 对齐和 rank ordering 断言。不要擅自再除 CP，也不要通用断言 EP 一定整除 dense DP；其他版本可能使用不同规则。

两组 mesh 在同一 PP stage 上复用进程，不能将 TP、CP、EP、ETP 全相乘。也不能无条件用 `PP×max(TP×CP, EP×ETP)` 宣称最小规模：max 只有同时满足两边整除时才有效。对上述分组公式，一般算术下界为 `PP×lcm(TP×CP, EP×ETP)`，实际分配仍须通过模型、分组、拓扑与 rank ordering 限制。此下界仅用于静态检查，不证明框架支持任意两种 mesh。

**算术检查示例**：W=8、GBS=64、TP=2、PP=2、CP=1 得 D=2；MBS=4 得 G=8，MBS=3 不合法。W=16、TP=2、PP=2、CP=2 得 D=2；若该版本支持 ETP=1、EP=4，则 EDP=2，但还须验证实际 MoE/CP 与 rank ordering，不能仅凭整数结果判定可运行。

## 参数类别与联动

下面列的是查找源码的参数名称，不是可直接复制的完整 YAML。FlagScale 常把 `micro_batch_size` 放在 `train.model`，并行/重计算放在 `train.system`，精度/日志/优化器还可能嵌套。以当前 generator 和实际 argv 确认路径、类型、默认值和冲突。

| 优化对象 | 相关维度 | 依赖与筛选依据 |
| --- | --- | --- |
| 可行性 | MBS、distributed optimizer、重计算 granularity/method/modules/层数 | 实测峰值显存、模型与优化器状态、activation；避免低估通信 buffer 和临时 workspace |
| 批量与重计算 | `micro_batch_size`、selective/full recompute 及覆盖范围 | 保持 GBS；增大 MBS 可能减少 G 和 pipeline 调度效率，不能只追求填满显存 |
| 并行映射 | TP、PP、CP、EP、ETP、SP、VPP/显式 layer layout | 先核对合法性、设备拓扑和 checkpoint 可恢复性；比较完整 step |
| DP 通信 | `overlap_grad_reduce`、`overlap_param_gather`、bucket 大小等本版参数 | optimizer/DP 支持、内存额外开销、HCCL 实际未被覆盖时间 |
| PP 通信 | VPP、P2P overlap、stage 分配 | schedule 的 microbatch 要求、阶段负载、激活峰值、通信竞争 |
| MoE | EP/ETP、dispatcher、grouped GEMM、permute fusion、已实现的通信重叠 | token 分布、padding、容量设置、公共接口与 backward；固定 top-k/aux-loss/丢 token 策略 |
| TE-FL/算子 | 已注册的 attention、norm、RoPE、SwiGLU、GEMM、permutation 实现 | shape/dtype/layout、mask/概率输出、前后向数值；backend 选择必须可观测 |
| 主机与数据 | dataloader workers/prefetch、CPU affinity、既有 NPU graph/compile 路径 | 数据顺序保持一致，检查 host 瓶颈与动态 shape，改变图模式先验证正确性 |

FP8、FP4、精度降级、optimizer/LR/GBS 变化涉及训练语义，不能与保持原数学定义的实现优化混为同类。MoE 强制均衡、容量因子或 token dropping 变化会改变路由或有效 token；假数据、减层和减序列长度会改变工作负载。此类结果只在各自明确的语义和质量要求下可比。

HCCL 环境参数的作用域、默认值和硬件限制取决于运行时版本与目标通信后端。提高超时只能改变等待期限，不能修复进程组、成员配置或通信初始化错误；通信等待时间也不直接等于链路传输成本。

训练 parser 的环境前置条件与设备算子支持是两层约束。若当前 TP/CP 参数验证路径要求 `CUDA_DEVICE_MAX_CONNECTIONS=1`，须核对该断言及平台分支；不能仅因名称包含 CUDA 就认为它对 NPU 无效，也不能未经核对将它设为所有 NPU 任务的默认值。前置断言失败不能证明 TP/CP 算子不支持；某个未被选中的 attention backend 含 CP 断言，也不足以剪掉实际执行的其他路径。支持结论仍须限定到已观测的 backend、shape、并行组合与版本。

fallback 必须同时核对实际执行入口与数值质量：框架捕获某个 backend 的不支持异常后继续训练，只证明存在后续执行路径，不说明回退与基线等价。保留异常、实际 fallback 调用及覆盖范围，并按原门禁比较对齐初态后的更新、loss/梯度。CP 还须检查分片顺序、真实位置和 mask 的一致性：采用 zigzag 切片时，不能默认按连续分片构造位置的参考 attention 与其兼容；须检查实际绑定和最终 mask。短跑完成或参数最终值 allclose 不能替代更新质量检查，局部 fallback 的质量问题也不能推广为全 CP 不支持。

参数维度由单位、默认值和 helper/runner 的联动覆盖共同定义；例如 bucket 可能按元素、参数数量或字节计，具体含义由消费者决定。输入只改一个开关不等于实际只改一个变量，resolved 配置、argv 和生效环境的差异才描述真实变化。能力未知与运行失败是不同状态，未运行的配置不能证明后端支持。

## 搜索方法的适用条件

尚不可行的配置主要受容量和兼容约束支配；已有稳定配置的优化更适合围绕瓶颈做局部搜索。配置回归的解释依赖有效配置和环境差异，相同输入 YAML 不保证比较对象相同。

### 可行域与内存边界

可行域是模型、拓扑、后端兼容和显存约束的交集。固定布局且其他条件相同时，若显存随 MBS 近似单调，可用倍增和区间收缩估计容量边界；单调性只在已确认的条件范围内成立。

不同内存来源对应不同优化变量：更小 MBS 主要影响 activation 和部分临时空间，optimizer 分片影响常驻状态，重计算减少特定保存张量。attention 重计算不直接解决 optimizer 状态分配过大的问题。

可用显存余量取决于最重 rank、首步状态分配、真实数据峰值和运行时额外开销，不能由平均 rank 占用或固定芯片无关百分比决定。

理论显存只用于排序与初筛建议。未校准 NPU 的模型不能按“预计 OOM”彻底删掉候选，也不能因利用率低而认为性能必差。MoE token 偏斜、CP buffer、kernel workspace 或编译会破坏单调性，发生时收窄剪枝范围。

| 失败阶段或来源 | 优先确认 | 候选方向与限制 |
| --- | --- | --- |
| 模型加载、权重或 optimizer 状态初始化 | 最重 stage/expert shard、状态 dtype、分片是否生效 | 调整状态分片或合法布局；activation 重计算不直接减少权重/optimizer 常驻状态 |
| forward 保存 activation | 层类型、attention backend、真实 tokens 与保存张量 | 比较相邻 MBS 和最小有效重计算边界，不默认 `core_attn` |
| backward 或重计算重放 | 嵌套 checkpoint、临时 workspace、重复通信、峰值迁移 | 比较完整更新的峰值，保存 original/replay 阶段；模块收益不能相加 |
| 梯度同步、参数 gather 或 overlap | bucket/prefetch 数量、buffer 生命周期、等待点 | 消融对应 overlap，测 HBM 与暴露通信的交换关系 |
| compile/capture 或后续 replay | 图/编译缓存、shape 集与私有/常驻 buffer | 缩小受支持范围或返回 eager 对照；重新测该组合内存边界 |
| 个别 rank 或真实数据后期 OOM | token/expert 分布、stage 差异、长样本、checkpoint 峰值 | 先定位差异；单 rank OOM 不等于 allocator 碎片，不改变 routing 语义造均衡 |

OOM 的解释需要失败 rank、执行阶段、allocated/reserved 峰值及申请失败信息。OOM 从 forward 转移到首次 optimizer 更新表示峰值阶段发生变化，仍未达到完整更新的容量要求。没有内存证据的通信超时不能判为 OOM。

### MBS、重计算和布局

速度和显存之间可能存在多个互不支配的配置，不能只按单一吞吐数字剪枝。PP/VPP、CP 或 EP 布局改变后，张量归属、临时通信空间和峰值阶段均可能变化，因此其他布局的 OOM 阈值不能直接沿用。

跨 TP/PP 候选先声明比较范围并检查 loader 的实际恢复策略。某些版本会在布局变化时忽略 checkpoint RNG，即使没有显式传入 `no_load_rng`。这样的加载成功只提供兼容性证据；模型初始状态与更新按真实分片对齐后，可按契约单独给出 `model_only` 质量结果，不能据此通过完整续训恢复门禁。两类检查按[共享测量契约](measurement-and-records.md)分别记录。

MoE 跨 EP 还须核对 optimizer 的实际保存/加载 schema、dense/expert optimizer 分组和分片 metadata，不能从模型权重可重分片推断状态兼容。EP 布局变化可能改变子优化器的组织；在 `convert_to_ep` 多子优化器加载分支要求 `param_state` 的版本中，必须确认源 optimizer 的保存 schema 确实提供该字段，不能假设 Float16 与 distributed optimizer 状态格式相同。字段缺失属于 checkpoint/schema 兼容问题，与 distributed optimizer 空参数组审计不同，也不能直接归为 dispatcher、EP 算子或硬件不支持。改变 optimizer family 必须另建 seed 和比较组。独立冷启动或目标 EP 的新 seed 可检验功能，但不能补足跨 EP 恢复证据，也不能未经状态对齐进入同组续训排名。

重计算候选按实际架构与保存张量生成：融合 attention 先与无重计算对照；MLA 看扩展投影；grouped MoE 看 expert 中间 activation；dense FFN 看整体 MLP。模块标签、是否生效、与重放通信的关系见 [组合调优参考](recompute-overlap-graphs.md)。变更 overlap buffer 或图范围后同样重测内存，VPP 的主要目的为调度，不能当作普遍的容量补救。

TP 增大可能提高单算子通信量，PP 增大引入 bubble，CP 改变 attention 通信与临时内存，EP 受 token 分布和网络影响。按实测最慢 rank 与未覆盖通信时间决定扩展方向，不能硬套“TP 优先跨某条总线”等 GPU 规则。

### 关键路径与局部变量

| 观测 | 相关变量 | 必须防止的误判 |
| --- | --- | --- |
| HBM 压力/重计算耗时大 | MBS、selective recompute、optimizer 分片 | reserved 不等于 allocated；显存省下来不必然更快 |
| HCCL 位于关键路径 | 拓扑映射、TP/EP/CP、受支持的 overlap | 算子时间求和会重复计算已经重叠的通信 |
| PP 空泡或 stage 偏斜 | 合法 G/VPP、层划分、P2P overlap | 层数相等不代表 MoE/混合层开销相等 |
| AICPU/大量 Vector 小算子 | 确认 TE/Ascend dispatch、融合、layout 转换 | AICPU 不一定是错误；只处理已确认的关键路径热点 |
| Cube compute 主导 | 合法矩阵 shape、grouped/fused GEMM、attention 实现 | 更快的微基准不等于端到端收益 |
| host 空洞或数据等待 | workers/prefetch、host sync、编译/launch 路径 | 不能靠删除真实数据处理工作来提升基准 |
| MoE 个别 rank 慢 | token 分布、dispatcher、EP/ETP | 不改变路由分布或 token dropping 以制造速度提升 |

单变量消融有助于区分直接效应与联动效应。组合可能因内存和通信竞争而退化，单项改善既不能证明组合有效，也不能相加为组合加速比。

MoE dispatch/combine overlap 的效果受 dispatcher、routing 和 recompute 影响；delayed wgrad 还引入权重梯度完成时序与累积约束。会同时改多个字段的 helper 只能表征联合变更，无法直接归因到某一个开关。NPU 图的收益来自实际 replay，相应 shape、RNG 和 eager 等价性是必要条件，capture 成功不表示已获得稳态加速。

### 搜索成本与结论边界

搜索成本包括配置生成、初始化、编译、预热、有效测量和退出，以及独立复测的成本。候选数相同的搜索可能有不同总耗时，尤其图编译、长序列和故障退出会使单次成本差异很大。

“最佳项”只相对于已完成且可比较的配置。连续邻域无改善只说明已覆盖范围内未观察到收益；unsupported、观测失败和外部环境失败不能充当性能劣化证据。理论显存、未知实现或不同环境与工作负载的结果也不能独立支持剪枝。

容量、吞吐和可靠性是不同目标。容量改善可以接受事先约定的吞吐代价，不额外要求所有目标同时变好。短跑可行、数值通过、独立性能复测和完整续训恢复分别回答不同问题，其证据不能互相替代；最终结论受本次约定的验收条件限制。
