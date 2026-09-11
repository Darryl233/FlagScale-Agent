# 昇腾训练配置搜索空间与约束

本篇提供候选合法性、参数联动与搜索方法的知识。具体任务的执行顺序由 [主调优 SKILL](../../../skills/train-ascend-performance-tuning/SKILL.md) 决定。使用确定性枚举、约束剪枝和逐阶段邻域探索即可，已有 native tuner 可用时复用它。大空间采用随机/贝叶斯等策略只有在已有实现与足够实验预算下才有价值。目标、计时和验收统一采用 [测量与实验记录](measurement-and-records.md)，跨维度代价参考 [生产优化手册](production-optimization.md)。

本文的实现例证限定于 [三仓能力](stack-capabilities.md) 与 [实机验证记录](hardware-validation.md) 标注的版本，不构成所有 Ascend 环境的支持清单。涉及模块重计算、overlap 或图模式时按需读 [组合调优参考](recompute-overlap-graphs.md)；参数来源与 native tuner 接入读 [三仓与调优器](stack-capabilities.md)。

## 先校验约束，再启动训练

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

布局变化还要核对 tokenizer 的实际词表、额外 special token、`padded_vocab_size` 和 embedding/output 分片形状。配置中的 `vocab_size` 不一定包含 EOD，也不一定等于填充后的词表；新增的 logits 不能自动视为不影响 loss 的空位。目标版本的 NullTokenizer 会增加一个 EOD，再按 `make_vocab_size_divisible_by × TP` 向上填充：输入 4096、默认 divisor 128 时，TP1/2/4/8 的实际词表分别为 4224/4352/4608/5120。这些是[本机版本例证](hardware-validation.md)，不是通用 tokenizer 公式。若为保持同一实际形状调整 divisor，先核对 parser 与 checkpoint 的比较条件，将其列为必要派生差异；若更改原始词表，则另建工作负载组与 seed，不能混入原组排名。

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

在编写时核对的 Megatron-LM-FL `parallel_state.py` 中，expert 分支使用：

```text
ETP = 显式 expert_tensor_parallel_size；未给定时继承 TP
EDP = W / (ETP × EP × PP)
expert rank generator 的 CP = 1
```

这是该实现的分组规则，不意味着 CP 在所有 MoE 分支都可用。还需核对 `W % (ETP×EP×PP)==0`、PP groups 对齐和 rank ordering 断言。不要擅自再除 CP，也不要通用断言 EP 一定整除 dense DP；其他版本可能使用不同规则。

两组 mesh 在同一 PP stage 上复用进程，不能将 TP、CP、EP、ETP 全相乘。也不能无条件用 `PP×max(TP×CP, EP×ETP)` 宣称最小规模：max 只有同时满足两边整除时才有效。对上述分组公式，一般算术下界为 `PP×lcm(TP×CP, EP×ETP)`，实际分配仍须通过模型、分组、拓扑与 rank ordering 限制。此下界仅用于静态检查，不证明框架支持任意两种 mesh。

**算术检查示例**：W=8、GBS=64、TP=2、PP=2、CP=1 得 D=2；MBS=4 得 G=8，MBS=3 不合法。W=16、TP=2、PP=2、CP=2 得 D=2；若该版本支持 ETP=1、EP=4，则 EDP=2，但还须验证实际 MoE/CP 与 rank ordering，不能仅凭整数结果判定可运行。

## 形成参数白名单

下面列的是查找源码的参数名称，不是可直接复制的完整 YAML。FlagScale 常把 `micro_batch_size` 放在 `train.model`，并行/重计算放在 `train.system`，精度/日志/优化器还可能嵌套。以当前 generator 和实际 argv 确认路径、类型、默认值和冲突。

| 阶段 | 可考虑的维度 | 依赖与筛选依据 |
| --- | --- | --- |
| 可行性 | MBS、distributed optimizer、重计算 granularity/method/modules/层数 | 实测峰值显存、模型与优化器状态、activation；避免低估通信 buffer 和临时 workspace |
| 批量与重计算 | `micro_batch_size`、selective/full recompute 及覆盖范围 | 保持 GBS；增大 MBS 可能减少 G 和 pipeline 调度效率，不能只追求填满显存 |
| 并行映射 | TP、PP、CP、EP、ETP、SP、VPP/显式 layer layout | 先核对合法性、设备拓扑和 checkpoint 可恢复性；比较完整 step |
| DP 通信 | `overlap_grad_reduce`、`overlap_param_gather`、bucket 大小等本版参数 | optimizer/DP 支持、内存额外开销、HCCL 实际未被覆盖时间 |
| PP 通信 | VPP、P2P overlap、stage 分配 | schedule 的 microbatch 要求、阶段负载、激活峰值、通信竞争 |
| MoE | EP/ETP、dispatcher、grouped GEMM、permute fusion、已实现的通信重叠 | token 分布、padding、容量设置、公共接口与 backward；固定 top-k/aux-loss/丢 token 策略 |
| TE-FL/算子 | 已注册的 attention、norm、RoPE、SwiGLU、GEMM、permutation 实现 | shape/dtype/layout、mask/概率输出、前后向数值；backend 选择必须可观测 |
| 主机与数据 | dataloader workers/prefetch、CPU affinity、既有 NPU graph/compile 路径 | 数据顺序保持一致，检查 host 瓶颈与动态 shape，改变图模式先验证正确性 |

FP8、FP4、精度降级、optimizer/LR/GBS 变化属于训练语义变化；只有用户将其纳入任务并规定验收时才能单列实验组。MoE 强制均衡、容量因子或 token dropping 变化、假数据、减层和减序列长度属于诊断，不能混入等价性能排名。

HCCL 环境设置从当前版本官方文档及目标通信后端核对，只在相关瓶颈明确时逐项验证。不更改系统网络/驱动/设备状态来完成普通扫参，不用提高超时掩盖进程组错误。

同时核对当前训练 parser 的环境前置条件，保留失败 attempt，再按已授权预算验证修正。参数名称中的 CUDA 不足以判断它在 NPU 分支无效：[本机版本](hardware-validation.md)的 TP/CP probe 曾被 `CUDA_DEVICE_MAX_CONNECTIONS=1` 前置检查拦截，满足该检查后 CP2 完成了 16 rank 短跑。此类初始化失败不能记为 TP/CP 不支持；某个未被选中的 attention backend 含 CP 断言，也不足以剪掉实际执行的其他路径。支持结论仍须限定到已观测的 backend、shape、并行组合与版本。

fallback 必须同时核对实际执行入口与数值质量：框架捕获某个 backend 的不支持异常后继续训练，只证明存在后续执行路径，不说明回退与基线等价。保留异常、实际 fallback 调用及覆盖范围，并按原门禁比较对齐初态后的更新、loss/梯度；CP 还须检查分片顺序、真实位置和 mask 的一致性。[本机版本例证](hardware-validation.md)中的 CP2 回退执行与质量失败应分别记录，不能因短跑完成或参数最终值 allclose 就验收，也不能推广为全 CP 不支持。

为每个维度记录单位、默认值和 helper/runner 的联动覆盖；例如 bucket 可能按元素/参数数量或字节计，必须查消费者。输入只改一个开关不等于实际只改一个变量，以 resolved/argv/env 的有效 diff 为实验定义。未知能力的 `evidence_level=unknown` 与候选运行状态分开记录，不能用未运行的候选证明支持。

## 分阶段自适应搜索

新配置尚未可行时从 A 开始；已有稳定配置和瓶颈证据时直接进入对应 B/C 邻域，不重跑无关阶段。配置回归先核对已知正常与当前的有效配置和环境指纹；复用可比较的已有结果，只为尚未回答的差异启动实验。

### A. 可行域与内存边界

1. 把可运行 baseline 保留为首个候选；只加入拓扑和模型允许的布局。
2. 固定布局，从当前 MBS 的相邻合法值扩展。仅在其他条件相同且内存近似单调的区间，用倍增后收缩定位边界。
3. 显存不足时先定位来源，再考虑更小 MBS、已支持的 optimizer 分片或针对性重计算；仍不足再探索布局。optimizer 状态分配失败不应首先扫描 attention 重计算；具体顺序可依瓶颈调整。
4. 显存余量以全部 rank 的实测最大值为准，计入首步状态分配和真实数据峰值。若无项目目标，可提出 10% 余量作为初始工程假设；生产验证前按实际波动修订，不能当作芯片规格。

理论显存只用于排序与初筛建议。未校准 NPU 的模型不能按“预计 OOM”彻底删掉候选，也不能因利用率低而认为性能必差。MoE token 偏斜、CP buffer、kernel workspace 或编译会破坏单调性，发生时收窄剪枝范围。

| 失败阶段或来源 | 优先确认 | 候选方向与限制 |
| --- | --- | --- |
| 模型加载、权重或 optimizer 状态初始化 | 最重 stage/expert shard、状态 dtype、分片是否生效 | 调整状态分片或合法布局；activation 重计算不直接减少权重/optimizer 常驻状态 |
| forward 保存 activation | 层类型、attention backend、真实 tokens 与保存张量 | 比较相邻 MBS 和最小有效重计算边界，不默认 `core_attn` |
| backward 或重计算重放 | 嵌套 checkpoint、临时 workspace、重复通信、峰值迁移 | 比较完整更新的峰值，保存 original/replay 阶段；模块收益不能相加 |
| 梯度同步、参数 gather 或 overlap | bucket/prefetch 数量、buffer 生命周期、等待点 | 消融对应 overlap，测 HBM 与暴露通信的交换关系 |
| compile/capture 或后续 replay | 图/编译缓存、shape 集与私有/常驻 buffer | 缩小受支持范围或返回 eager 对照；重新测该组合内存边界 |
| 个别 rank 或真实数据后期 OOM | token/expert 分布、stage 差异、长样本、checkpoint 峰值 | 先定位差异；单 rank OOM 不等于 allocator 碎片，不改变 routing 语义造均衡 |

记录最后完成的逻辑 step、首先失败的 rank/阶段、allocated/reserved 峰值及申请失败证据。OOM 从 forward 转移到首次 optimizer 更新是诊断进展，候选仍为 `oom`。没有内存证据的通信超时不能补判成 OOM。

### B. MBS、重计算和布局

先在若干可行布局上比较批量/重计算邻居，保留 2–3 个速度/显存不同的候选。给 PP/VPP、CP 或 EP 新布局重新测内存和批量边界；不能复用其他布局的 OOM 阈值。

跨 TP/PP 候选先声明比较范围并检查 loader 的实际恢复策略。某些版本会在布局变化时忽略 checkpoint RNG，即使没有显式传入 `no_load_rng`。这样的加载成功只提供兼容性证据；模型初始状态与更新按真实分片对齐后，可按契约单独给出 `model_only` 质量结果，不能据此通过完整续训恢复门禁。两类检查按[共享测量契约](measurement-and-records.md)分别记录。

MoE 跨 EP 还须核对 optimizer 的实际保存/加载 schema、dense/expert optimizer 分组和分片 metadata，不能从模型权重可重分片推断状态兼容。[本机版本例证](hardware-validation.md)中，EP1 seed 向 EP2/4/8/16 加载在 optimizer 的 `param_state` 查找阶段失败：EP1/EP8 成功 seed 分别观察到一个/两个 Float16 子优化器，该版本 `convert_to_ep` 多子优化器加载分支读取 `param_state`，而 Float16 状态不提供此字段。这与 distributed optimizer 空参数组审计是两个问题；改变 optimizer family 必须另建 seed 和比较组。保留原恢复失败，不直接归为 dispatcher、EP 算子或硬件不支持。独立冷启动或目标 EP 的新 seed 可继续检验功能，但不补足原跨 EP 恢复证据，也不能未经状态对齐进入同组续训排名。

重计算候选按实际架构与保存张量生成：融合 attention 先与无重计算对照；MLA 看扩展投影；grouped MoE 看 expert 中间 activation；dense FFN 看整体 MLP。模块标签、是否生效、与重放通信的关系见 [组合调优参考](recompute-overlap-graphs.md)。变更 overlap buffer 或图范围后同样重测内存，VPP 的主要目的为调度，不能当作普遍的容量补救。

TP 增大可能提高单算子通信量，PP 增大引入 bubble，CP 改变 attention 通信与临时内存，EP 受 token 分布和网络影响。按实测最慢 rank 与未覆盖通信时间决定扩展方向，不能硬套“TP 优先跨某条总线”等 GPU 规则。

### C. 关键路径驱动的局部开关

| 观测 | 下一轮假设 | 必须防止的误判 |
| --- | --- | --- |
| HBM 压力/重计算耗时大 | MBS、selective recompute、optimizer 分片 | reserved 不等于 allocated；显存省下来不必然更快 |
| HCCL 位于关键路径 | 拓扑映射、TP/EP/CP、受支持的 overlap | 算子时间求和会重复计算已经重叠的通信 |
| PP 空泡或 stage 偏斜 | 合法 G/VPP、层划分、P2P overlap | 层数相等不代表 MoE/混合层开销相等 |
| AICPU/大量 Vector 小算子 | 确认 TE/Ascend dispatch、融合、layout 转换 | AICPU 不一定是错误；只处理已确认的关键路径热点 |
| Cube compute 主导 | 合法矩阵 shape、grouped/fused GEMM、attention 实现 | 更快的微基准不等于端到端收益 |
| host 空洞或数据等待 | workers/prefetch、host sync、编译/launch 路径 | 不能靠删除真实数据处理工作来提升基准 |
| MoE 个别 rank 慢 | token 分布、dispatcher、EP/ETP | 不改变路由分布或 token dropping 以制造速度提升 |

运行一次消融验证主要假设后再合并已有效的变更，并对组合重新测试。组合可能因内存和通信竞争而退化，不能累加单项加速比。

MoE overlap 先固定 dispatcher/routing/recompute，独立比较 dispatch/combine overlap；再加入 delayed wgrad，最后比较其他 dispatcher/图组合。不要用会同时改多个字段的 helper 结果证明单项收益。对于 NPU 图实验，先确认 eager、可捕获 shape 与 RNG，最小范围生效后才扩展；验证 replay 的实际运行和稳定窗口，不以 capture 成功判胜。

### D. 搜索成本与结论边界

最终复测、编译和退出都占用预算。按基线实际成本估计候选数量，比只限制搜索器的策略数更可靠；可用约三分之一总预算规划复测，再按候选成本修订，这只是起始工程假设。

“最佳项”只相对于已完成且可比较的实验。连续邻域无改善可以作为停止依据，但邻域须包含有意义的候选与有效测量；unsupported、观测失败和外部环境失败不能充当性能劣化证据。理论显存、未知实现或不同指纹的历史结果也不能独立支持剪枝。

容量、吞吐和可靠性是不同目标。容量改善可以接受事先约定的吞吐代价，不额外要求所有目标同时变好。短跑可行、数值通过、独立性能复测和完整续训恢复分别回答不同问题，其证据不能互相替代；最终结论受本次约定的验收条件限制。
