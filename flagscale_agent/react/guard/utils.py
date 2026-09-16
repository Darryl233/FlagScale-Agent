# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for guards."""

from __future__ import annotations

import re


# Tools that only read state and never modify anything.
# Used by guards to distinguish exploratory actions from mutations.
READ_ONLY_TOOLS = frozenset({
    "read_file", "memory_read", "memory_list", "recall",
    "load_skill", "load_knowledge", "plan_status",
    "inspect_checkpoint", "web_fetch", "flagscale_train_monitor",
})


# ---------------------------------------------------------------------------
# Launch command detection
# ---------------------------------------------------------------------------

def _is_flagscale_launch_command(cmd: str) -> bool:
    """Detect FlagScale training launch commands.

    Supports compound commands (cd xxx && flagscale train ...).
    Strips quoted content to avoid grep "flagscale train" false positives.
    """
    if not isinstance(cmd, str):
        return False

    cmd_lower = cmd.lower()

    # Remove quoted content to avoid false positives like grep "flagscale train"
    cleaned = re.sub(r'''["'][^"']*["']''', '', cmd_lower)

    # Query flags belong to their command, not the entire shell expression:
    # `flagscale train --help && flagscale train model` still launches training.
    cleaned = cleaned.replace("\\\n", " ")
    for command in re.split(r"[;&|\n]+", cleaned):
        match = re.search(r"\bflagscale\s+(train|run)\s+(.+)", command)
        if match:
            args = match.group(2).split()
            if "--help" in args or "-h" in args:
                continue
            if match.group(1) == "train":
                non_run_flags = ("--stop", "--dryrun", "--test", "--query", "--tune")
                if not any(flag in args for flag in non_run_flags):
                    return True
            else:
                non_run_actions = ("--action dryrun", "--action stop", "--action test",
                                   "--action query", "--action auto_tune",
                                   "-a dryrun", "-a stop", "-a test")
                if not any(a in command for a in non_run_actions):
                    return True

        # Pattern 3: python[3] run.py ... action=run
        if ("python" in command and "run.py" in command
                and ("--config-name" in command or "--config-path" in command)
                and "action=run" in command):
            return True

    return False
