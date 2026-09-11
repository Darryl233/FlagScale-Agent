<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Ascend Training Performance: Mechanisms and Trade-offs

Use this reference for system-wide performance work after training is functionally correct. It organizes optimization into seven interacting dimensions: Compute, Memory, Communication, Parallelism, Scheduling, Load Balance, and Runtime.

## Contents

1. Critical-path model and production metrics
2. Compute
3. Memory
4. Communication
5. Parallelism
6. Scheduling
7. Load Balance
8. Runtime and system engineering
9. Cross-dimensional trade-offs
10. Evidence for production claims

## 1. Critical-Path Model and Production Metrics

### Optimize the slowest rank, not summed tables

Distributed step time is determined by the longest dependency path on the slowest participating rank. Operator totals can overlap and therefore cannot simply be added:

```text
step wall time
  = critical path through compute, communication, data, synchronization, and optimizer
  = max(per-rank critical path), not average rank time
```

Start every analysis with four views:

1. End-to-end stable iteration time and throughput.
2. Per-rank timeline and max/median rank skew.
3. Device active/idle intervals and overlap.
4. Operator/kernel aggregation for the critical-path ranks.

### Primary production metrics

Record a small stable set rather than every profiler counter:

- Samples/s and tokens/s for the complete job.
- Stable iteration p50/p90/p99 or median plus range.
- Per-NPU and whole-job throughput.
- Model FLOP utilization only when the model-FLOP formula and peak hardware basis are documented.
- Strong-scaling efficiency: `throughput_N / ((N / N0) × throughput_N0)` for the same model and fixed global work. Record the runnable baseline size `N0`; use `N0 = 1` only when the full workload fits on one NPU.
- Peak allocated/reserved HBM and margin to OOM.
- Exposed HCCL/P2P time on the critical path.
- Pipeline bubble/idle percentage.
- Max/median rank time and coefficient of variation.
- Data wait, compilation, checkpoint, and recovery time separately from steady-state training.
- Loss/gradient health and skipped/NaN iterations.

Never optimize only AI Core utilization, kernel count, reserved memory, or communication bandwidth. Each is diagnostic, not the product metric.

### Recommended optimization order

1. Fix correctness, fallback, and measurement problems.
2. Remove global stalls and large exposed idle regions.
3. Correct persistent load imbalance and topology mistakes.
4. Improve communication/data/compute overlap and pipeline scheduling.
5. Reduce memory traffic and allocation pressure.
6. Optimize dominant kernels and repeated operator sequences.
7. Re-tune parallelism and scale only after the single-node/scale-unit path is understood.

This order is a heuristic. Follow measured critical-path impact when evidence clearly points elsewhere.

## 2. Compute

### Questions

- Is time dominated by Cube matrix computation, Vector work, AICPU work, or idle gaps?
- Are GEMM/attention/expert shapes large and aligned enough for efficient tiling?
- Is useful math fragmented into repeated layout, cast, normalization, or elementwise kernels?
- Is recomputation intentional, and is its compute cost lower than the memory/communication benefit?

### Evidence to collect

- Total and critical-path duration by Cube, Vector, AICPU, HCCL, memcpy, and idle.
- Top kernels by total time, average time, call count, input shape, dtype, and stack.
- Achieved FLOP/s where the operation count is known.
- Shape alignment, padding, transpose/contiguous/cast frequency, and dynamic-shape recompilation.
- Fused versus unfused public paths using identical model semantics.

AICPU activity is not by itself evidence of fallback: it can also support runtime or communication work. Identify the operator, its call path, and its role in the installed backend before classifying it as an unsupported model operator or choosing a replacement.

### Optimization methods

- Select the hardware backend explicitly and verify runtime dispatch.
- Prefer proven `transformer_engine_npu`, `fla_npu`, or `torch_npu` kernels over generic compositions.
- Fuse repeated elementwise/reduction sequences when cumulative critical-path time justifies maintenance cost.
- Remove redundant casts, transposes, contiguous copies, and repeated normalization.
- Keep hot shapes static or from a small bounded set when compilation/tiling benefits are measurable.
- Adjust microbatch/token packing to produce efficient kernel shapes without changing global-batch semantics.
- Use BF16 or other approved lower precision only after numerical validation.
- Choose recompute granularity by measuring step time and peak HBM together.
- For MoE, group expert GEMMs and avoid dispatch paths that fragment experts into tiny matmuls.
- Benchmark boundary and production shapes; a kernel optimized only for a toy shape is not ready.

### Diagnostic experiments

- Replace only the suspected operator with a reference or vendor variant.
- Aggregate a repeated short-kernel sequence by call stack, not name alone.
- Compare reduced and real sequence/token shapes to expose shape-sensitive tiling.
- Run with blocking only for correctness localization; never use `ASCEND_LAUNCH_BLOCKING=1` for performance conclusions.

### Anti-patterns

- Maximizing Cube utilization while whole-step time or memory worsens.
- Treating high Vector percentage as launch overhead without timeline evidence.
- Keeping a faster microbenchmark that does not improve training.
- Hiding an unsupported call behind exception-driven fallback on every iteration.

## 3. Memory

Separate four problems: **capacity**, **bandwidth**, **copies**, and **allocator/runtime behavior**.

### Evidence to collect

- Peak allocated and reserved HBM by rank and training phase.
- Largest live tensors and activation/optimizer/model-state estimates.
- HBM read/write bandwidth or memory-bound kernel evidence.
- Device-to-device and host-device memcpy duration and frequency.
- Allocation/free frequency, fragmentation indicators, and explicit `empty_cache`/synchronize calls.
- Memory added by communication overlap, buckets, prefetch, or async checkpointing.

### Capacity methods

- Activation recompute/checkpointing with measured granularity.
- Sequence/context parallelism for long-sequence activations when communication cost is acceptable.
- Sequence parallelism with TP where supported.
- Distributed optimizer or state sharding.
- Appropriate TP/PP/EP placement to shard weights and expert state.
- Reduce unnecessary saved tensors in custom autograd; recompute only when cheaper.
- Use mixed precision and master-weight policy consistent with accuracy requirements.
- Offload only when transfer can be hidden and storage/host memory is reliable.

### Bandwidth and copy methods

- Fuse producer-consumer chains to avoid writing intermediate tensors to HBM.
- Preserve the layout expected by the next kernel; eliminate ping-pong transposes.
- Avoid `.cpu()`, `.item()`, Python lists, and host decisions on the training hot path.
- Reuse workspaces and buffers when lifetime and stream safety are clear.
- Pack MoE tokens/probabilities once and reuse metadata where semantics allow.
- Avoid materializing masks or expanded tensors when a kernel accepts compact metadata.

### Allocator/runtime methods

- Do not call `empty_cache()` per batch. Cached reserved memory is normally healthy.
- Pre-size stable workspaces and warm caches before measurement.
- Bound dynamic shapes that generate incompatible allocation patterns.
- Tune allocator or memory-reuse settings only with peak-memory and step-time A/B evidence.
- Measure checkpoint and optimizer peak phases; steady-state forward peak alone can miss OOM.

### Acceptance

Report both step-time change and peak-HBM change. A memory optimization may be accepted without a speedup when it enables the required model/sequence/batch, but state that goal explicitly.

## 4. Communication

Optimize **exposed** communication, not total collective duration. A long collective fully hidden under compute may not be urgent.

### Evidence to collect

- HCCL AllReduce, ReduceScatter, AllGather, AllToAll, Send/Recv duration by rank.
- Message-size distribution and effective bandwidth.
- Wait placement: when the consumer actually blocks on async work.
- Compute/communication overlap on streams.
- Rank skew entering collectives.
- Intra-node versus inter-node topology, NIC/HCCS/PCIe affinity, and link health.
- Collective retries, timeouts, retransmission, or hardware errors.

### Optimization methods

- Place high-volume TP groups within the fastest local connectivity domain.
- Choose DP/EP/PP groups to minimize expensive cross-node traffic for the dominant collectives.
- Tune gradient and parameter bucket sizes to balance overlap against memory and launch overhead.
- Enable overlap features one at a time: gradient reduce, parameter gather, P2P, expert communication.
- Move waits to the latest correct consumer and remove accidental barriers/synchronizations.
- Coalesce small collectives or metadata exchanges when ordering remains correct.
- Pack MoE AllToAll payloads efficiently and avoid sending dropped/padded data unnecessarily.
- Balance tokens before AllToAll; communication cannot hide a straggling sender.
- Evaluate hierarchical algorithms and HCCL settings against the actual topology and message sizes.
- Pin process/CPU/NIC affinity consistently across nodes.

### Diagnostics

- Compare one node with multiple nodes to isolate fabric cost.
- Compare collective time after equalizing rank work to separate imbalance from network limits.
- Disable one overlap feature to expose whether it truly hides time or only raises memory.
- Check whether a “communication” gap is actually a late producer on one rank.

### Anti-patterns

- Increasing timeout instead of finding a rank mismatch or deadlock.
- Reporting average collective time when max rank controls completion.
- Changing HCCL environment variables in batches without a topology hypothesis.
- Reducing communication precision without explicit numerical approval.

## 5. Parallelism

Parallelism maps model state, activations, tokens, and communication onto hardware. Treat it as a constrained search, not a fixed recipe.

### Validate arithmetic first

- `world_size` must satisfy the framework's TP/PP/CP/DP decomposition.
- EP partitions experts: `num_experts % EP == 0`. ETP partitions supported expert tensor dimensions; it does not require the expert count to be divisible by ETP. Validate the actual expert-layer implementation's tensor-shape constraints separately.
- Validate dense and expert process-group decompositions against the installed Megatron-LM-FL version, including ETP defaults and expert DP. Use [search-policy.md](search-space.md) for the shared legality checks before pruning or launching candidates.
- Preserve the intended global batch: `micro_batch × DP × gradient_accumulation`.
- Confirm layer/stage assignment and local expert counts from runtime logs.

### Dimension trade-offs

- **TP** shards dense tensor math and state but adds frequent collectives. A small TP within fast links is a useful initial candidate. Kernel shapes, memory headroom, and end-to-end measurements can favor larger TP; this heuristic must not prune other legal layouts.
- **PP** shards layers and state but introduces bubbles and P2P. Use balanced stages and enough microbatches.
- **DP** improves aggregate throughput but replicates model state unless optimizer/state sharding is used.
- **EP** reduces local expert state but adds token AllToAll and load sensitivity.
- **Expert TP** can make large experts fit but adds expert-internal collectives.
- **CP** reduces long-sequence activation pressure but adds attention/KV communication and implementation constraints.
- **SP** reduces activation memory with TP and is usually favorable when supported.

### Search method

1. Enumerate only valid configurations that fit memory.
2. Keep model, global batch, data, and optimizer semantics fixed.
3. Test a small matrix around the current strategy, not every combination.
4. Measure stable throughput, peak HBM, exposed communication, bubble, and rank skew.
5. Re-profile the best candidate because the bottleneck often changes after remapping.

Suggested initial experiment order; expand the neighborhood when kernel shapes, stage balance, recomputation, or measured critical paths support other layouts:

1. TP within a single high-bandwidth domain.
2. PP when capacity or TP communication suggests a benefit, while retaining other legal PP layouts for measured comparison.
3. EP sized for both expert memory and AllToAll behavior.
4. CP for long sequences after attention backend support is confirmed.
5. DP from remaining resources, then tune microbatch/accumulation.

Do not infer full-model scaling from a reduced-layer smoke model: PP balance, activation memory, communication-to-compute ratio, and optimizer cost differ.

## 6. Scheduling

Scheduling covers pipeline bubbles, microbatch order, stream dependencies, communication overlap, data readiness, optimizer timing, and checkpoint work.

### Evidence to collect

- Per-stage forward/backward/weight-gradient durations.
- Pipeline idle/bubble regions and P2P waits.
- Number of microbatches and the actual 1F1B/interleaved timeline.
- Compute, communication, memcpy, and host work overlap.
- Queue/task dependencies, barriers, stream synchronizations, and `.item()` waits.
- Dataloader wait before each microbatch.
- Optimizer, gradient clipping, logging, evaluation, and checkpoint placement.

For a balanced non-interleaved pipeline, the idealized bubble fraction is approximately:

```text
(pipeline_stages - 1) / (microbatches + pipeline_stages - 1)
```

Use this only as a bound; unequal stages, interleaving, communication, and recompute change the real timeline.

### Optimization methods

- Increase microbatch count while preserving global-batch semantics and memory limits.
- Balance layer cost, not only layer count, across PP stages; account for embedding, vision, MTP, and MoE layers.
- Use virtual/interleaved pipeline stages when supported and beneficial.
- Enable P2P, gradient-reduce, parameter-gather, and expert overlap independently and verify the timeline.
- Delay waits until data is consumed; remove redundant global barriers.
- Prefetch data and metadata so the next microbatch is ready before device demand.
- Keep logging reductions and `.item()` calls off the per-microbatch critical path.
- Move checkpoint serialization and durable I/O out of steady-state timing; use async checkpointing only after recovery validation.
- Warm compilation and operator caches before measuring.

### Risks

- More overlap usually consumes more HBM and can worsen allocator pressure.
- More microbatches may change kernel shapes and reduce compute efficiency.
- Interleaving increases schedule complexity and can expose framework/backend incompatibilities.
- Asynchronous work without explicit ownership can corrupt buffers or make failures nondeterministic.

## 7. Load Balance

The slowest rank, stage, expert, host, or sample determines synchronous progress.

### Sources of imbalance

- MoE router sends uneven tokens to experts or EP ranks.
- PP stages contain unequal compute, communication, or recompute.
- Variable-length samples create different token/vision workloads by DP rank.
- Data workers, CPU affinity, storage, or network differ across ranks/nodes.
- Thermal throttling, hardware health, or background workloads create persistent stragglers.
- Checkpoint/logging responsibility is concentrated on one rank.

### Evidence to collect

- Per-rank p50/p90/p99 step and component times.
- `max / median`, `(max - min) / median`, and coefficient of variation.
- Per-expert and per-EP-rank token counts before and after capacity/drop/padding.
- Per-stage forward/backward/communication time.
- Input token/image/video sizes by DP rank.
- Correlation of slow ranks across steps, without assigning the cause from persistence alone. Fixed expert placement with sustained router skew, uneven PP stages, or rank-specific data work can make the same rank persistently slow, just as hardware or placement can. Correlate token counts, stage costs, data wait, and device health; a controlled rank-to-device remapping can help distinguish logical work imbalance from a device-specific problem.

### Optimization methods

- Use router auxiliary loss, expert bias, group routing, or capacity policy only when compatible with model training objectives.
- Map experts and EP groups to topology; avoid placing a communication-heavy group across weak links unnecessarily.
- Use grouped GEMM and efficient token packing so small experts do not create excessive launch fragmentation.
- Balance PP stages using measured layer costs and special-stage overhead.
- Bucket or pack variable-length data to equalize work while preserving data semantics.
- Distribute checkpoint and data responsibilities or move them asynchronously when correctness permits.
- Fix CPU/NUMA/NIC affinity within the allocated resources. Use available healthy devices and report external contention or device faults; do not interrupt unrelated jobs to obtain a cleaner measurement.

### Diagnostic-only controls

A forced-uniform router can determine whether imbalance explains a bottleneck, but it changes model semantics. Never keep it enabled in production or use it for accuracy claims. Similarly, padding every expert to equal size trades imbalance for extra compute and memory; accept only with end-to-end evidence.

## 8. Runtime and System Engineering

Runtime optimization ensures the host, framework, compiler, allocator, data system, and storage continuously feed the NPUs and remain stable in production.

### Host and launch path

- Measure CPU thread utilization, launch gaps, Python overhead, dispatch/graph breaks, and host synchronization.
- Use one correctly bound process per NPU and align CPU/NUMA affinity with device/NIC locality.
- Avoid per-step imports, exception-driven dispatch, verbose logging, tensor-to-host scalar reads, and repeated shape construction.
- Keep operator availability and backend selection outside the hot path.
- Use task queues, multi-stream settings, and ACL/operator caches only after controlled A/B; do not assume environment defaults are optimal.
- `ASCEND_LAUNCH_BLOCKING=1` is a debugging tool, never a throughput setting.

### Data path

- Measure dataloader wait and storage bandwidth separately from model time.
- Tune workers, prefetch depth, persistent workers, CPU affinity, and pinned/shared memory against actual CPU and storage limits.
- Cache tokenizer/templates and avoid Python work repeated for every token/sample.
- For multimodal data, monitor decode and preprocessing skew, not only raw read throughput.
- Validate that prefetch does not exhaust host memory or create unstable tail latency.

### Compilation and caches

- Separate cold-start compile time from warm steady state.
- Bound dynamic shapes to prevent compilation/cache churn.
- Persist supported caches across runs only when version/config keys make them safe.
- Record CANN, PyTorch, `torch_npu`, and custom-op versions because cache validity and kernel choice depend on them.

### Allocator, logging, and checkpointing

- Remove per-step `empty_cache()` and unnecessary synchronizations.
- Reduce logging frequency and aggregate metrics without forcing device waits each microbatch.
- Measure checkpoint serialization, network/storage write, and rank barriers.
- Validate async checkpoint completion, retention, and restart before accepting its performance benefit.

### Production reliability

- Monitor NPU health, temperature/power throttling, HCCL errors, host memory, disk space, and storage latency.
- Pin the full software stack and resolved configuration for every benchmark.
- Use canary rollout for new kernels/configs before cluster-wide deployment.
- Run a soak test long enough to include evaluation, checkpoint, data rollover, and allocator steady state.
- Verify checkpoint save/restore continuity and failure recovery.
- Track p99/tail iteration time, not only average throughput; unstable fast runs are not production improvements.

## 9. Cross-Dimensional Trade-offs

| Candidate | Likely benefit | Likely cost/risk |
| --- | --- | --- |
| Activation recompute | Lower HBM capacity | More compute, possible schedule changes |
| Larger microbatch | Better GEMM efficiency, fewer launches | Higher HBM, fewer pipeline microbatches |
| More microbatches | Lower PP bubble | Smaller kernels, more launches, same-batch arithmetic constraints |
| More TP | Lower per-rank model memory | More frequent communication, smaller GEMMs |
| More PP | Lower model-state memory | Bubble, P2P, stage imbalance |
| More EP | Lower local expert state | More AllToAll exposure and load sensitivity |
| CP/SP | Lower activation pressure | Additional communication/backend constraints |
| Communication overlap | Less exposed HCCL | More HBM and stream dependency complexity |
| Operator fusion | Less launch/HBM traffic | Shape restrictions, precision and maintenance risk |
| Expert padding | Better regular shapes/balance | Extra compute and memory |
| Async checkpoint/data | Hide I/O | Host memory, consistency and recovery complexity |

For every proposal, state the expected movement in at least: step time, peak HBM, exposed communication, load skew, and correctness risk.

## 10. Evidence for Production Claims

[Measurement semantics](measurement-and-records.md) defines comparable step/token metrics, sampling, numerical alignment and recovery evidence. [Configuration constraints](search-space.md) explains legal layouts and coupled fields. Experiment execution and stopping decisions belong to the Ascend tuning workflows.

A performance claim requires an accepted end-to-end improvement with correctness and stability preserved. A capacity claim concerns the required model, sequence and batch fitting within memory with an agreed time cost. A reliability claim concerns tails, failures or recovery with an agreed performance cost. Improvements in a microbenchmark or one profile counter alone do not establish any of these claims.

Production scope includes the actual model, data, routing, precision, topology and I/O phases. Short-run evidence does not establish long-run convergence, evaluation/data-transition stability or checkpoint recovery. Those are separate claims requiring corresponding observations; a task evaluating one local configuration does not automatically require a complete production qualification campaign.
