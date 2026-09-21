<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Device references

Select from the actual assigned hardware and active training backend, then read only that file. These are instructions read with `read_file`, not separate skills or executable adapters. Paths below are relative to this index.

| Device family | Selection evidence | Reference |
| --- | --- | --- |
| NVIDIA GPU | Assigned NVIDIA hardware and the configured CUDA training backend | [NVIDIA](nvidia.md) |
| Ascend NPU | Assigned Ascend hardware and the configured CANN / torch_npu training backend | [Ascend](ascend.md) |

An API name, compatibility layer, installed CLI, or failed probe alone is insufficient to select a family. Reuse previously confirmed identity/mapping. If evidence conflicts, resolve that conflict before device-dependent execution; do not run every vendor's probes as a discovery sweep.

For an unlisted family, inspect only the actual environment's read-only inventory, framework backend and installed tool documentation. Keep unsupported probes and unverified support explicit. Configuration/log analysis may continue; launching requires a reliable authorized device mapping and supported runtime/launcher. Do not install a different vendor stack or silently fall back to CPU.

## Extending the skill

For a new family, add `references/devices/<family>.md` and one row above. Keep commands and device-specific decisions in that file; the common workflow, caller skills and core tools need no vendor branch. Adding instructions does not implement FlagScale/framework support or prove a hardware test passed.

Use the same small set of sections as the existing references:

1. **Applicability:** matching hardware/backend and known scope or version restrictions.
2. **Devices and mapping:** read-only inventory, occupancy/memory/health probes, physical-to-visible IDs, existing visibility settings, and limits of process visibility. State unsupported queries and how to check installed help.
3. **Runtime:** framework imports, availability/count API, actual module paths, required runtime and communication libraries. Do not require optional packages the recipe does not use.
4. **Monitoring:** whether existing monitor device probes work; otherwise use shared log checks plus the platform's probes and owned-job evidence.
5. **Failure diagnostics:** first checks for device/runtime, collective communication and memory failures. Return evidence to the calling tuning workflow instead of changing its workload.
6. **Optional profiling:** only an available entry point for an explicitly requested profiling task; no automatic collection during training launch.

Reuse common launch, record, wait, stop and log instructions from `SKILL.md`. Keep theoretical optimization material in Knowledge and experiment records outside the skill. No adapter class, tool registration, machine-readable schema, or new dependency is needed for a documentation-only addition.

Before publishing an addition, verify that its links resolve, its commands match the installed tools, and the agent selects only that family's instructions. Distinguish a reviewed guide from successful device probes and from an actual training run; do not mark an untested family as validated. Exercise the unchanged tuning and owned-job boundaries when checking the new path.
