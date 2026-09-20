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

# Train-Run — Summary

Launch, monitor, stop, and verify FlagScale training with a common execution workflow and device-specific references.

**Load when**: launching training from a YAML recipe, stopping a run, checking assigned devices, or debugging launch failures. Tuning skills can reuse this execution workflow.

Read the device index and only the matching reference for hardware probes, runtime checks and diagnostics. NVIDIA and Ascend instructions are provided; additional platforms extend the index and add a reference without adding vendor branches to the common workflow. Documented commands do not imply framework support or hardware validation.

Share `flagscale train -c ...`, job tracking and log inspection; reuse verified setup within a tuning loop. Bounded single-host measurements retain the helper's execution requirements. Profiling is loaded only when requested.
