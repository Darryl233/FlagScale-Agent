---
description: Launch, monitor, stop, and verify FlagScale training from a YAML recipe.
  Use a shared execution workflow with device-specific references for hardware
  checks, runtime dependencies, and diagnostics.
name: train-run
---

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

# FlagScale Training Launch

Launch, stop, and manage FlagScale distributed training jobs. Keep the execution workflow common; select hardware tools from the device references below.

## Critical Rules

<<<<<<< HEAD
1. **If the user says the environment/conda is already set up, reuse it.** Select its device reference (Step 2), then verify only missing or changed preflight facts. Repair packages only after diagnosing a required dependency failure.
2. **Observe the owned job and its current-run logs after starting training.** For direct CLI launches, call `flagscale_train_monitor(output_dir=..., mode="check")`; for the bounded helper below, use its job handle and built-in wait/log checks. A help query or dryrun is not a training launch.
=======
1. **If the user says the environment/conda is already set up, DO NOT install packages.** Go straight to preflight verification (Step 3). Only install if preflight imports fail.
2. **After launching training (not dryrun — dryrun only generates scripts), you MUST immediately call `flagscale_train_monitor(output_dir=...)` to observe the process.** Do not proceed to other tasks without monitoring.
>>>>>>> main
3. **Never delete experiment output directories.**
4. **Use the documented commands and parameters directly.** Reading CLI, launcher, or helper source is not a launch prerequisite. Investigate implementation only for an actual error or behavior that contradicts these instructions.

## Choose the Execution Path

- **First launch from an existing full recipe:** use the [launch commands and parameters](references/first-launch.md), preserving the supplied recipe and environment. Use applicable preflight checks below for missing or changed environment/data/entrypoint facts; do not investigate source as routine preparation.
- **A validated recipe in a tuning loop:** reuse the caller's workload, device allocation, environment evidence, plan, and experiment record. Check device availability, the configuration delta, batch arithmetic, unique output path, and remaining budget. Do not repeat environment discovery, source surveys, dataset construction, or a separate smoke run for every candidate. The baseline/candidate short run is already the validation run; preserve its parallel layout and measurement window.
- **Bounded single-host Megatron measurements:** use `read_file` to read [single-run execution](references/single-run.md) once before the first run, then fill its request example and execute its launch/wait commands. Reuse this procedure for subsequent trials. It runs the FlagScale CLI with a time limit and returns measurement/evidence paths; ordinary training can use the direct CLI path in Step 4.

When another skill calls this skill, return the command, actual output directory, job/exit evidence, exact loss-rank log, and any startup error, then resume that skill. Do not start a separate tuning or profiling workflow here.

## Prerequisites

- SSH access to training server
- Docker container with FlagScale environment (or bare metal with conda)
- FlagScale cloned and installed in the conda environment (see `train-env-setup` skill)
- Training config files ready (see `train-config` skill)

---

## Step 1: Connect to Server

SSH into training server, enter Docker container, activate conda env, cd to FlagScale project directory.

```bash
sudo docker exec -it <container_name> bash
# In non-interactive shells (agent), use: conda run --prefix <env_path> <command>
# In interactive shells (user), use: conda activate <env_name>
cd <workspace_root>/code/FlagScale
```

---

## Step 2: Select a Device Reference and Check Availability

Identify the assigned accelerator using the caller's environment evidence, host/container inventory, and active framework backend. A compatible API or an installed utility alone does not identify the hardware vendor.

Read the [device index](references/devices/index.md), then only the matching device reference. Each reference is a device branch of this workflow; callers requesting the Ascend branch select its entry there. Reuse the selected reference and verified mapping across candidates instead of reloading them.

`load_skill` loads this main file only. Use `read_file` for references, resolving paths from the active `train-run/SKILL.md` directory, not the training cwd. Reuse its known location; if unknown, locate it in the configured skill directories (later matching directories override earlier ones). The built-in location is `flagscale_agent/skills/train-run`, but a user override may supply another copy. Do not read every device file or change core tools to select a branch.

Use the selected reference for inventory, physical-to-visible device mapping, runtime checks, occupancy probes, and backend-specific diagnostics. Check every assigned host for multi-node runs. A new reference documents tool choices; it does not establish that FlagScale or the training backend supports that device.

If no reference matches, use confirmed read-only platform tools and the installed backend's documentation to establish the missing facts. Do not borrow another vendor's commands or guess visibility variables. Continue configuration/log analysis when possible; defer launch if device allocation, mapping, runtime support, or launch behavior remains unresolved.

**Go/no-go:** combine authorization, utilization, memory baseline, and visible processes. Containers may hide other namespaces' processes; no visible PID does not prove a device is free. Persistent runtime memory need not mean an active workload. A failed or unavailable probe means unknown, not idle or healthy. Do not reset devices or kill unrelated processes.

Use `shell` for device commands. `flagscale_train_monitor(mode="check")` inspects logs independently of the chip; use `watch` only when the selected reference confirms its device probes are compatible. Broad process matching is never proof of owned-worker liveness or completion; retain the actual job/PID evidence.

---

## Step 3: Preflight Check

For a first launch, verify the applicable items below. Reuse verified facts within the same tuning loop; recheck when the environment, data, checkpoint, or launch contract changes.

### 3a. Core Dependencies

Use the runtime checks in the selected device reference, in the effective worker environment. Separately confirm that the launch process can find the CLI/helper; YAML worker settings do not configure its parent process. Preserve working environment settings and use command-scoped additions only for a demonstrated missing path; do not edit shell startup files as routine preparation. Verify actual framework/module paths, backend availability, visible worker count, and required runtime libraries. Check optional packages only when this recipe consumes them.

### 3b. Device Availability

Use Step 2's selected platform probe for the assigned devices. Reuse the immediately preceding result rather than issuing the same check twice.

### 3c. Data Path Validation

Validate ALL data paths referenced in the training config. This is not just "do the files exist" — it's "will the data pipeline actually load them at runtime."
For `mock_data: true`, verify the synthetic dataset configuration; do not search for `.bin/.idx` files that the recipe does not use. Reuse a validated unchanged data pipeline during tuning.

**For Megatron binary format (FlagScale native):**
```bash
DATA_PATH="<data_path from config>"
ls -lh ${DATA_PATH}.bin ${DATA_PATH}.idx
```

**For third-party frameworks (parquet, JSONL, custom loaders):**
1. Check that data directories exist and contain expected files:
   ```bash
   ls <data_dir>/*.parquet | wc -l   # or *.jsonl, *.json, etc.
   ```
2. If the config references a metadata/index file (e.g., parquet_info JSON, dataset_info), open it and verify that the paths INSIDE the file match the actual data locations. Placeholder paths like `your_data_path/`, `/path/to/`, or paths from a different machine are the #1 cause of silent data loading failures.
   ```bash
   # Example: check if parquet_info keys match actual data_dir
   python -c "
   import json, os
   info = json.load(open('<parquet_info_path>'))
   data_dir = '<actual_data_dir>'
   actual_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
   matched = [f for f in actual_files if f in info]
   print(f'Matched: {len(matched)}/{len(actual_files)} files')
   if len(matched) == 0:
       print(f'WARNING: Zero matches! Info keys sample: {list(info.keys())[:2]}')
       print(f'Actual files sample: {actual_files[:2]}')
   "
   ```
3. If paths don't match, fix the metadata file (replace placeholder prefix with actual path) BEFORE launching.

**General rule**: any config file, JSON, or Python dict that maps dataset names to file paths is a potential source of path mismatch. After modifying data paths, always verify the FULL chain: config → dataset registry → metadata files → actual files on disk.

If files are missing or paths don't match, stop and tell the user. Suggest running `/skill train-data-prep`.

### 3d. Topology Freshness (Optional)

Compare recorded device model/count/mapping with Step 2 using the selected reference's inventory and framework count. Physical cards and framework-visible devices may differ. Refresh affected facts only when the environment or mapping changed.

### 3e. Validate a New Launch Recipe

The examples use `flagscale train -c <recipe.yaml>`; add the confirmed MODEL argument only if the installed CLI requires it.

**CRITICAL DISTINCTION:**
- `flagscale train -c <recipe.yaml> --dryrun` generates launch scripts; it does not run the training loop or prove the model/data pipeline works.
- A validation run sets `train.model.train_iters` in a copied YAML and launches actual training. Do not assume the FlagScale CLI accepts Megatron's `--train-iters` flag.

**Step 1: Generate scripts with dryrun**
```bash
flagscale train -c /absolute/path/recipe.yaml --dryrun
```

If dryrun fails, it means config has syntax errors or missing fields. Fix and retry.

After dryrun succeeds, inspect the generated launch script:
```bash
cat {exp_dir}/logs/scripts/host_*_run.sh
```
Verify: correct worker count (`--nproc_per_node`), selected device mapping/backend, entrypoint, expected CLI flags, and no placeholder paths.

Regenerate scripts when the launch schema or generated arguments change; do not launch stale scripts. For an already validated MBS-only trial, reuse launch evidence and check the delta and effective arguments. The bounded helper requires a fresh `exp_dir`: if a dryrun is needed first, use a separate validation output directory.

**Step 2: Run a short validation training**
After dryrun scripts look correct, run actual training with minimal iterations to validate the full pipeline (model loading, data loading, forward/backward pass):
```bash
# Set the agreed short train_iters in this full recipe first:
flagscale train -c /absolute/path/validation.yaml
```
Only proceed to full training after applicable validation passes. In a tuning loop, the scheduled baseline/candidate short run supplies this evidence; do not insert another 20-step run.

For third-party reproduction tasks (no FlagScale launcher): construct the full launch command, print it, and verify it manually before executing. Check: correct `--nproc_per_node`, correct `PYTHONPATH`, correct entrypoint script, all required CLI args present, no placeholder paths in any referenced config files.

### 3f. Handle Launch Errors or Version Differences

Start with the commands and parameter descriptions in [launch commands and parameters](references/first-launch.md). If the installed CLI rejects an argument, read `flagscale train --help` and correct that mismatch. For an import/config/runtime failure, inspect the actual error and current-run logs.

Only when the error or observed behavior cannot be resolved from those instructions should you inspect the relevant implementation. Do not read parser, launcher or helper source merely to reconfirm normal behavior. A new session or the absence of a prior source audit is not a reason to delay the planned run.

### 3g. Config Arithmetic Verification

Check batch arithmetic and the selected model's parallel-layout constraints; reuse the unchanged layout's validation during MBS tuning:
- `global_batch_size % (micro_batch_size × data_parallel_size) == 0`
- `num_attention_heads % tensor_model_parallel_size == 0`
- `num_key_value_heads % tensor_model_parallel_size == 0` (for GQA models)
- `num_layers % pipeline_model_parallel_size == 0` (if PP > 1)

Preserve the supplied recipe's working value types. If a new field is rejected, use its argument help or error message to correct it; do not audit parser definitions before a normal launch.

### 3g. Checkpoint Compatibility Verification

If loading a checkpoint (`--load`):
1. Verify checkpoint exists and contains `latest_checkpointed_iteration.txt`
2. Verify checkpoint TP/PP matches config TP/PP — a TP=1 checkpoint cannot be loaded with TP=4 without resharding
3. Verify `vocab_size` matches between checkpoint and config
4. Verify checkpoint format (`torch` vs `dist`) matches `ckpt_format` in config

```bash
ls <checkpoint_path>/latest_checkpointed_iteration.txt
cat <checkpoint_path>/latest_checkpointed_iteration.txt
```

### 3h. Memory Budget Estimation

Before a new workload, estimate per-device memory. This rough expression omits activations, temporary buffers, master weights and implementation-specific sharding; it is not a sufficient fit test:
```
per_device_memory = model_params × 2 (bf16) + gradients × 2 + optimizer_states × (8 / DP)
```
If the workload exceeds available device memory, resolve the memory budget before launching. During fixed-layout tuning, return OOM evidence to the caller rather than changing its parallelism or recomputation policy here.

### 3i. Test a New or Changed Data Pipeline

For an unverified custom data pipeline, check a few samples before a costly full launch. An unchanged pipeline already exercised by the baseline, including mock data, does not need a separate reconstruction for each trial.

This catches: path mismatches, format errors, infinite loops, missing files, wrong tokenization — all in seconds instead of the 10+ minutes of model loading.

```bash
python -c "
import sys, time
sys.path.insert(0, '.')
# Import the dataset class used in training config
from <dataset_module> import <DatasetClass>

# Instantiate with the SAME args as training config
ds = <DatasetClass>(<args_from_config>)
print(f'Dataset length: {len(ds)}')

# Fetch 3 batches, with timeout
for i in range(3):
    t0 = time.time()
    batch = ds[i]
    elapsed = time.time() - t0
    if isinstance(batch, dict):
        shapes = {k: v.shape if hasattr(v, 'shape') else type(v).__name__ for k, v in batch.items()}
    else:
        shapes = batch.shape if hasattr(batch, 'shape') else type(batch).__name__
    print(f'Batch {i}: {shapes} ({elapsed:.2f}s)')
    if elapsed > 10:
        print(f'WARNING: batch {i} took {elapsed:.1f}s — possible infinite loop or I/O issue')
        break
print('Data pipeline OK')
"
```

If this script hangs, crashes, or shows unexpected shapes, fix the data pipeline BEFORE launching training.

### 3j. Checkpoint Loading Verification

If loading a pretrained checkpoint, verify the requested checkpoint identity and the framework's actual load/missing-key messages. First-iteration loss is a diagnostic clue, not proof that weights did or did not load. Training from scratch and synthetic-data benchmarks can legitimately begin near `ln(vocab_size)`.

After the validation run (step 2 of 3e) produces metrics:
1. Check the first logged loss
2. Compare with `ln(vocab_size)` — the expected loss for random initialization

```bash
# Extract the first logged loss
grep -m1 "lm loss" <log_file>
python -c "import math; print(f'Random init baseline: {math.log(<vocab_size>):.2f}')"
```

If the loss is unexpected for this model and dataset, investigate the loading evidence and forward path. Do not accept or reject checkpoint loading from that number alone.

Common causes of checkpoint not loading:
- Conversion code exists but isn't called in the training script
- `--load` path is wrong or points to empty directory
- TP/PP mismatch between checkpoint and config (silent fallback to random init)
- `--finetune` flag missing (Megatron skips optimizer state but still needs the flag to load weights in some modes)

Proceed once the checks applicable to the current execution path have evidence; retain explicit unknowns instead of restarting unrelated checks.

---

## Step 4: Start / Stop Training

**ALWAYS use FlagScale Launcher** — never bypass it with raw `torchrun` or hand-written launch scripts. The launcher provides per-rank log separation, experiment directory structure, config validation, and clean shutdown (`--stop`). Without it, all ranks write to one stream (debug prints get lost or interleaved), there's no experiment directory structure, and you can't use `--stop`. If the launcher fails, fix the root cause — do not work around it.

**Exception — third-party reproduction tasks**: When reproducing a third-party model's training (e.g., LLaVA-OneVision, Qwen-VL) using their own training scripts before migrating to FlagScale, use their native launch method (typically `torchrun` + their training script). Apply the relevant preflight checks, selected hardware probes, job tracking and log checks, and reuse the same experiment record.

**IMPORTANT**: Always set `PYTHONUNBUFFERED=1` before launching training. Without it, Python buffers stdout and training logs appear delayed or empty, making health monitoring unreliable.

Use a full FlagScale YAML containing `experiment` and `train`. Its filename is arbitrary. Prefer `flagscale train -c /absolute/path/recipe.yaml`; add the confirmed `<model>` argument only when the installed CLI requires it. Preserve the caller's cwd, environment, and config-relative dependencies.

```bash
# Start (CLI)
PYTHONUNBUFFERED=1 flagscale train -c /absolute/path/recipe.yaml
# Foreground training, useful for waiting on a short run
PYTHONUNBUFFERED=1 flagscale train -c /absolute/path/recipe.yaml --test
# Start (legacy)
PYTHONUNBUFFERED=1 python run.py --config-path ./examples/<model>/conf --config-name train action=run

# Stop (CLI)
flagscale train -c /absolute/path/recipe.yaml --stop
# Stop (legacy)
python run.py --config-path ./examples/<model>/conf --config-name train action=stop

# Dry run (generate scripts, without running the training loop)
flagscale train -c /absolute/path/recipe.yaml --dryrun
```

Default CLI execution may submit a background launcher and return early. `--test` runs training in the foreground in the CLI used by this workflow; it does not shorten the YAML's iteration count. Use it directly for foreground runs, or use the bounded wrapper for supported automated trials. Track actual workers and logs: an outer shell exit is not by itself training completion. If a run returns while workers are still active, treat it as incomplete and diagnose that observed discrepancy. Some versions mask worker failure codes; retain log/error evidence and do not claim all-rank success from CLI exit 0 alone.

### Record, Launch, Observe, Return

1. Reuse the caller's plan and one experiment record. Before each actual run, append its config delta, command/cwd, platform and assigned device mapping, time limit, and unique output directory. Record the selected runtime and communication backend. Do not create a new plan or registry for every MBS candidate.
2. Confirm the preceding owned job has ended and the assigned devices remain available. Identify jobs by their returned job handle or exact PID file. Stop only the owned job; never use a global `pkill -f torchrun` or device reset. Before `--stop`, verify it targets this experiment's PID; a missing PID file must not trigger a broad fallback kill.
3. Launch through the CLI, or the bounded helper which calls the CLI. Use `shell(background=True)` for an agent-managed long command, retain its job id, and use `shell_jobs(action="wait", job_id=..., timeout=60)` for bounded waits. Waiting on the outer job is sufficient only when the launcher is actually foreground.
4. Inspect current-run logs as below. For the bounded helper, its owned-process wait and log/iteration checks already produce a compact result; inspect extra logs only for a specific missing fact or anomaly.
5. Record the terminal result, exact evidence paths and unresolved checks once. Return them to the calling tuning skill for comparison. Missing evidence, a timeout, or a submitted launcher is not success. Use the existing record file and available plan tools; no separate experiment-registry tool is required.

For the bounded helper, use its returned job handle and result for observation. Read additional logs only for a missing fact or anomaly; the following monitoring steps apply to direct CLI launches.

<<<<<<< HEAD
**Direct CLI path — within 30 seconds of launch:**
1. Observe the returned job and actual output directory. Logs may not exist immediately after background submission; an initial missing directory is a startup state, not permission to select an older run.
2. Use `flagscale_train_monitor(output_dir="<exp_dir>", mode="check", filter="progress", lines=3)` for logs. Query device state with the selected reference's probes when needed.
3. If stderr has errors → training failed at startup. Fix and retry.
4. **Check stderr FIRST, not stdout** — crash info is in stderr. A process showing "wandb initialized" in stdout may already be dead.

**Direct CLI path — after first metrics appear:**
4. **Use `flagscale_train_monitor(output_dir="<exp_dir>", mode="check", filter="progress", lines=3, vocab_size=<vocab_size>)`** for a concise snapshot; increase detail only for a concrete anomaly. Use the selected reference's compatible monitoring path and owned-job waits. For timing comparisons, pass the exact loss-rank log to `analyze_training_results` with the caller's measurement window and output path.
=======
```
workspace_experiment(action="create", name="<model>_<config>_<purpose>",
    purpose="<what you are verifying and why>",
    hypothesis="<expected outcome — e.g., loss ~ ln(vocab) and decreases>")
```

If the experiment already exists (retry after failure), skip this step.

**0b. Record this attempt — BLOCKING GATE:**

```
workspace_experiment(action="add_attempt", name="<experiment_name>",
    change="<what changed vs previous attempt, or 'initial run'>",
    config={"model": "...", "tp": N, "pp": N, "dp": N, "ep": N,
            "global_batch_size": N, "micro_batch_size": N, "seq_length": N,
            "precision": "bf16", "train_iters": N, ...},
    hardware={"gpus": N, "gpu_type": "...", "driver": "...", "cuda": "..."},
    output_dir="<unique output directory for this attempt>")
```

**If you haven't called `add_attempt`, you are NOT allowed to call `flagscale train`.** This is the single most important discipline rule. During rapid debug-fix-retry cycles, this is ESPECIALLY critical — those are exactly the attempts you'll need to reconstruct later.

**Version bumping rule — what counts as a new experiment:**
- Changed a meaningful parameter (LR, TP/PP, batch size, data, model code) → new experiment (`create`)
- Launch failed before any metrics (import error, path error, config typo) → same experiment, new attempt (`add_attempt` with change description)
- Training crashed after producing metrics, restarting with same config → same experiment, new attempt

**0c. Kill old processes and verify GPUs free:**

```bash
pgrep -fa "torchrun|train_|flagscale" | grep -v grep
# If any found:
pkill -9 -f "torchrun|train_|flagscale" 2>/dev/null; sleep 5
nvidia-smi | grep -E "MiB|%"
```

#### POST-RESULT (do IMMEDIATELY after monitor/metrics return):

**8a. Record the result — BLOCKING GATE:**

```
workspace_experiment(action="update_last_attempt", name="<experiment_name>",
    result="<SUCCESS/FAILED — key metrics, loss trajectory, throughput, or error cause>")
```

**If you haven't called `update_last_attempt`, you are NOT allowed to proceed to the next task or launch.** Do this BEFORE fixing config, BEFORE analyzing, BEFORE anything else.

**8b. Finalize (when done with this experiment line):**

```
workspace_experiment(action="finalize", name="<experiment_name>",
    status="completed|failed",
    learnings=["lesson 1", "lesson 2", ...],
    root_cause="<if failed, what was the fundamental problem>")
```

**Within 30 seconds of launch:**
1. **Wait 10-15 seconds** before checking logs — the log directory may not exist yet (race condition with nohup/background launch)
2. Use `flagscale_train_monitor(output_dir="<exp_dir>", mode="check")` to auto-discover logs AND scan stderr — NEVER use raw `find` commands (they may find old logs from previous runs)
3. If stderr has errors → training failed at startup. Fix and retry.
4. **Check stderr FIRST, not stdout** — crash info is in stderr. A process showing "wandb initialized" in stdout may already be dead.

**After first metrics appear (usually 1-3 minutes):**
4. **Use `flagscale_train_monitor(output_dir="<exp_dir>", mode="check", vocab_size=<vocab_size>)`** — do NOT use `tail -f` or `grep` to manually scan logs. The tool parses structured metrics and runs the health checks automatically (`vocab_size` enables the random-output check). For continuous supervision use `mode="watch"` with `duration`.
>>>>>>> main
5. Interpret the health check results:
   - `loss ≈ ln(vocab_size)` → check against the intended initialization/data. This can be expected for scratch or mock-data runs; inspect actual checkpoint-load evidence when pretrained weights were requested.
   - `grad_norm = 0` or `num_zeros ≈ total_params` → gradients not flowing. Check loss computation, frozen params.
   - Unexpected loss behavior → compare with the workload's declared tolerance and baseline; a short performance run does not establish convergence.
6. Report the first metrics to the user with health assessment. Include: initial loss, loss trend, grad norm, throughput (tokens/sec or samples/sec).

**After training completes or fails — close the experiment:**

8. Append the terminal result and evidence to the same record, then return to the caller. Do not repeat finalization for every intermediate log snapshot.

**If health judge killed a long-running command:**
When the agent's health judge kills a `sleep` or `tail -f` command, do NOT blindly retry with another sleep. Instead:
1. Check if the training process is still alive: `kill -0 <pid>` or check PID file
2. Check the assigned devices with the selected reference's occupancy probes.
3. Check the latest log lines directly (no sleep)
4. Then decide: wait more, or investigate a problem

**Never declare training successful based only on lack of a crash.** Check the expected iterations, loss/gradient validity, errors, and actual completion evidence against the caller's workload.

---

## Log Directory Structure

FlagScale training logs are organized as follows. Understanding this structure is CRITICAL — you MUST use the correct commands to find logs, never guess paths.

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   ├── scripts/host_0_<hostname>_run.sh      # actual launch script
│   ├── scripts/host_0_<hostname>_stop.sh     # stop script
│   └── details/host_0_<hostname>/
│       ├── 20260424_153816.588538/           # timestamp dir (YYYYMMDD_HHMMSS.us)
│       │   └── default_<hash>/attempt_0/
│       │       ├── 0/stdout.log  stderr.log  # rank 0
│       │       ├── 1/stdout.log  stderr.log  # rank 1
│       │       └── .../                      # one dir per rank
│       └── 20260424_162209.763893/           # another run (newer!)
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key points:
- `exp_dir` comes from the selected YAML's effective `experiment.exp_dir`; the file need not be named `train.yaml`
- Each training launch creates a NEW timestamp directory under `details/host_X_<hostname>/`
- Prefer one output directory per attempt. If historical runs share a directory, correlate the recorded job/start time; “latest” alone does not prove it is your run.
- Each rank has its own `stdout.log` and `stderr.log`
- Find the actual loss-reporting rank; pipeline layouts may report from the last stage rather than rank 0
- stderr.log contains errors, warnings, and import failures

### Locate This Run's Logs

**Always use the dedicated tool first** — it handles the full directory traversal, rank scanning, and health checks in one call:

```
flagscale_train_monitor(output_dir="<exp_dir>", mode="check", vocab_size=<vocab_size>)
```

<<<<<<< HEAD
Use the actual experiment directory from the caller's record or helper result. Do not search other experiments to substitute for missing current-run logs.
=======
If the experiment dir is recorded in the experiment ledger memory entry, use that path directly. NEVER use `find`, `ls -R`, or shell globbing to search for log files.
>>>>>>> main

**Manual fallback:** if the tool is unavailable, inspect only the recorded run's `logs/details/host_*/<timestamp>/<run>/attempt_*/<rank>/` directories. Match the launch identity, scan rank stderr, and locate the rank that actually reports iterations/loss. Do not assume rank 0 or pick an older run because it has logs.

---

## Quick Verification Paths

When the user wants to quickly verify a training setup works:

These reduced workloads are for a separate setup smoke test. If called from a fixed-workload tuning skill, keep that skill's device count, GBS, layout and iteration window instead.

1. **Minimal config**: `train_iters: 3-5`, `micro_batch_size: 1`, `global_batch_size: DP × 1`
2. **Small supported device count**: Use an authorized device layout that fits this smoke test; do not shrink a caller's measured workload
3. **Smallest dataset**: Use the smallest available split or demo data
4. **Dry run**: Use `flagscale train -c <recipe.yaml> --dryrun` to generate and inspect scripts
5. **Stage-by-stage**: If the recipe has stages, run one stage at a time to isolate failures

### Common Pitfalls

- `global_batch_size` must be divisible by `micro_batch_size × data_parallel_size`
- Megatron checkpoint format: `--load` path must contain `latest_checkpointed_iteration.txt`
- Multi-node: investigate the configured communication backend using the selected device reference; reuse valid connectivity evidence when unchanged
- OOM on first iteration: return the memory/error evidence to the calling tuning skill. For a standalone setup, resolve the memory budget before retrying; preserve the caller's fixed workload and layout.

---

## Error Handling

### Launch Failures

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `ModuleNotFoundError: megatron.*` | Megatron-LM-FL not installed or wrong PYTHONPATH | Check `pip list \| grep megatron`, reinstall if needed |
| Device runtime or collective communication error | Mapping, connectivity, or runtime/backend mismatch | Read the first error and rank logs; follow the selected device reference's diagnostics |
| Device out-of-memory / allocation failure | Insufficient free memory for this workload | Use the selected memory probe and owned-process evidence; return to the tuning caller without changing its fixed workload |
| `FileNotFoundError: data path` | Data files missing or wrong path in config | Verify data path with `ls`, check the selected YAML's data section |
| `Address already in use` | Port occupied or previous owned job still running | Identify the port owner and this run's PID/job; stop only an owned job or choose an allowed free port |
| `Hydra config error` | YAML syntax error or missing required field | Run `flagscale train -c <recipe.yaml> --dryrun` and inspect the selected recipe |
| Process starts but exits silently | Import error or early crash | Check launcher output and rank stderr, including nonzero ranks |

### Recovery Steps

1. Inspect the first relevant error and enough surrounding launcher/rank log context to diagnose it
2. Fix ALL identified issues before relaunching
3. Preserve the failed run's artifacts. Use a new output directory and verify the next launch reads the intended YAML and generated arguments; do not delete experiment history as a routine retry step.
4. Never retry more than once without a clear diagnosis

### Fast Isolated Verification (before relaunching)

A full training launch can take 10+ minutes just to load the model before reaching the code you're debugging. Before relaunching, ask: "can I verify this fix without a full launch?"

**Data pipeline bugs** (the most common category):
```python
# Write a quick standalone script — runs in seconds, no model loading
import sys; sys.path.insert(0, '<project_root>')
from <dataset_module> import <DatasetClass>
ds = <DatasetClass>(<args_from_config>)
batch = next(iter(ds))
print(f"Batch keys: {batch.keys()}, shapes: {[(k, v.shape) for k, v in batch.items() if hasattr(v, 'shape')]}")
```

**Import / path errors**: `python -c "import <module>; print('OK')"` — instant.

**Config errors**: run `flagscale train -c <recipe.yaml> --dryrun` to inspect generated arguments without a training loop.

**Shape / architecture errors**: instantiate model on meta device, no checkpoint needed.

Only relaunch the full training when the fix is in a component that can't be tested in isolation (e.g., distributed communication, optimizer state, checkpoint loading itself).

---

## Related Skills

- `train-config` — generate and validate training configuration YAML files
- `train-monitor` — monitor running training jobs, check health, detect anomalies
- `train-env-setup` — install FlagScale and all dependencies
- `topo-detect` — detect hardware topology for parallelism planning
- `train-data-prep` — prepare training data in Megatron binary format
