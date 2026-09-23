<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Collect and Analyze Ascend Training Profiles

**Load when:** The task requires generating a torch_npu profiler wrapper, collecting a FlagScale training profile, or analyzing existing NPU data.

**Workflow:** Choose the start and end point for the task. For generation only, deliver the wrapper once the entrypoint is ready. For collection, complete a bounded short run and validate its artifacts. For existing data, start with inventory, extract facts from a target window, and propose validation experiments.

**Outputs:** For generation, deliver the wrapper, recipe change, and command. For collection, deliver raw artifacts and the validated scope. For analysis, deliver a parseable window result, bottleneck report, and evidence gaps. When invoked by a tuning task, return evidence and hypotheses to the main workflow rather than creating another experiment loop.

**Reuse native capabilities:** Continue the current plan and use `plan_update` for progress and acceptance. Use `read_file`/`write_file` for text artifacts. Add `stage=profile` to the shared experiment record. Use `train-run` for a bounded single run or the direct CLI path, following its launch and monitoring guidance. Also verify that every actual worker has exited. Reuse existing records and read further details only when a specific gap requires them.

**Knowledge on demand:** Read relevant sections of `ascend_profiling/collection-and-analysis.md` in `know-ascend-profiling` as needed. Use `know-ascend-training` for measurement definitions. Script usage stays in the [wrapper instructions](references/wrapper-generation.md).

Analyzing existing data does not launch training. Report collection success only after validating real NPU artifacts. Do not include profiled run times in training performance rankings.
