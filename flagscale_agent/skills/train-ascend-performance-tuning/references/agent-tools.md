<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Reusing Agent Plans, Experiment Records, and Execution Tools

Training, monitoring, quality checks, tuning, and profiling share this guide. It defines where to use native tools and a small set of experiment fields, without creating a new recording service.
Use the tools currently available; the interfaces below belong to FlagScale-Agent. If another host lacks a tool, state the gap
and use its existing plan/file/execution capabilities. Do not pretend a tool was called or relaunch training because recording tools are missing.

## 1. Resume Plans and Evidence

- Start with `plan_status()` and reuse the current or parent plan and step IDs. Use `plan_create(title=..., steps=...)` only when a multistep plan is needed and no relevant plan exists.
  Do not create a plan for each candidate or profiling subflow. Put targets and acceptance criteria in the steps' `acceptance`; the current bottleneck hypothesis may go in `thinking`.
- Use `plan_update(action="step_doing", step_id=..., notes=...)` to append experiment directories, record paths,
  the current candidate/attempt, and the next action. Read original configurations and evidence with `read_file`. Keep paths and summaries in notes, not copies of all raw logs.
- When a stage completes, use `plan_update(action="step_done", step_id=..., notes=..., verification=[...])` to reference evidence files.
  verification replaces the step's existing array; retain still-valid entries when adding repeat measurements. notes are appended on each call.
  A failed candidate can establish that it was evaluated, not that training passed. Do not call `action="complete"` while the main task remains unfinished.
- To retrieve verified environment facts or experience, use `memory_list(keyword=...)` for the current model/environment, then `memory_read(key=...)` for relevant entries.
  Check their versions and scope first. The Agent handles session persistence/resume; call `recall(index=...)` only with the actual index of an evicted-message placeholder.

## 2. Save Experiment Details

Reuse the current `experiments.jsonl` or equivalent record. Append one complete JSON object and newline with `write_file(path=..., mode="append", content=...)`.
Keep each content value within the tool's size limit. Save larger configurations/logs separately and reference them; do not split a JSON record.
Native file tools operate in the Agent's environment. Record the host/container and original path for remote evidence, and identify the source of local copies.
Use `edit_file` for minimal replacements in candidate copies of text recipes. For structural or type changes, parse the configuration and save the complete candidate with `write_file`.

Record actions as they actually occur:

| When | Fields to Preserve |
| --- | --- |
| Candidate generated | `candidate_id, parent, hypothesis, change, checks`; links to the current plan/step and original/effective configuration |
| Run prepared | New `attempt_id`, `candidate_id`, `stage`, separate `output_dir`, command, environment/starting state, devices, deadline, and stop method |
| Launch returns | Actual job identifier returned by the tool and target-side job identity; accurately record an unstarted or failed launch |
| Run ends or is interrupted | Run status, quality status, metrics/units/window, raw evidence, retention/rollback reason, and next action |

Use `stage` to distinguish `train / probe / quality / performance / profile` as applicable. Repeats and retries get a new attempt referencing the same candidate.
Meaningful configuration/patch changes create a new candidate with its parent retained. Advice, wrapper generation, and analysis of old files do not create an executed attempt.
Leave unknown metrics empty with a reason, rather than filling them with zero. Distinguish observations, run success, quality acceptance, and final adoption.

This example only demonstrates appending a record; actual field values must come from real output. `completed` means only that this job completed:

```python
write_file(
    path="/path/to/tuning/experiments.jsonl", mode="append",
    content='{"event":"attempt_result","candidate_id":"mbs2","attempt_id":"mbs2-probe-01","stage":"probe","run_status":"completed","quality_status":"not_run","metrics":null,"evidence":["runs/mbs2-probe-01/result.json"],"decision":"Awaiting quality and performance checks"}\n'
)
```

Use `read_file` to verify recent records as needed. Multiple executors write their own artifacts; the main workflow serializes appends to the shared record.
On resume, read the plan and records and check the actual jobs marked as running, so a disconnected session does not cause a duplicate launch.

## 3. Launch, Monitor, and Clean Up

Use native `shell` to execute a verified launcher command. Run long commands in the background and capture the actual returned job ID.
Load the required SKILLs and establish the log path before launching. The launch Guard requires an immediate `flagscale_train_monitor` call;
do not insert `load_skill`, file writes, or polling between launch and that check. Here, `output_dir` points to this run's logs accessible to the Agent:

```python
shell(command=launch_command, background=True)
flagscale_train_monitor(output_dir=output_dir, mode="check", filter="all", lines=100)
shell_jobs(action="list")
shell_jobs(action="poll", job_id=job_id)
shell_jobs(action="wait", job_id=job_id, timeout=30)
```

`shell` has no timeout argument. A controlled launcher/job manager in the actual training environment enforces the run deadline.
The timeout on `shell_jobs.wait` limits one wait; unfinished jobs keep running. poll/wait perform health checks and may terminate the registered process.
Job IDs are valid only within the Agent process. After restart, list jobs and verify commands/actual identities; do not mistake a new `job1` for the old job.
Save the target host, container, and actual job/PID/PGID identity. SSH or launcher exit does not prove that remote workers exited.
To stop an identified job, use the original launcher's stop procedure, then `shell_jobs(action="kill", job_id=...)` if needed.
kill targets only the registered process. Still check all owned workers; it is not a remote process-group cleanup mechanism.

Save the monitor's log paths, errors, progress, and JSON summary, then check expected world size and original completion records.
The current `health_ok` is a numerical heuristic and can coexist with nonempty `error_ranks`; it alone establishes neither job success nor quality.
The tool has no SSH/container arguments. Prefer a shared directory or synchronized snapshot for remote logs, preserving directory structure and recording snapshot time.
If only a remote path is available at launch, still call check in the Guard's required order and record the access failure, then read original logs through the existing authorized connection.
An inaccessible path does not prove training failed or justify relaunching. Do not fabricate local logs to bypass the check.
For unsupported formats, read original logs and record the parsing gap rather than rewriting or fabricating training output.
The monitor's check/watch, target_step, and duration do not stop training. The watch implementation's nvidia-smi and broad process checks
cannot establish whether this Ascend job is alive or has exited. Log summaries do not replace all-rank completion, stable timing, or numerical comparison.
When structural checks are needed, trusted model checkpoint files are accessible, and the interpreter has the dependencies, reuse `inspect_checkpoint(path=..., reference_path=...)`.
Its shape/sampling checks do not establish full parameter-update, optimizer, or RNG restore correctness.

## 4. Deliver and Preserve Reusable Findings

Save configurations, reports, and acceptance evidence with `write_file`; reference the delivery directory and record file from the plan's notes/verification.
Complete only the assigned steps. Use `plan_update(action="complete")` only once the entire current plan is complete.
Use `memory_write` for verified environment facts, established failure lessons, or insights worth retaining across sessions,
under `fact / pitfall / insight`, including versions, scope, evidence paths, or a recheck method. Look up existing keys first and update the same concept.
Keys use `type/domain/specific`; domain/specific must start with a lowercase letter and contain only lowercase letters, digits, and underscores. Put literal hostnames/IPs in content.
Keep individual attempts, pending queues, full configurations, and raw logs in plans/experiment files rather than creating a Memory entry for each.

The older names `workspace_experiment` and `workspace_state` are not implemented/registered native tools in the current Agent; this workflow does not call them.
