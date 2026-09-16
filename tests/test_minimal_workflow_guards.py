# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for help queries and duplicate delivery verification."""

import pytest

from flagscale_agent.react.guard import GuardContext, GuardRegistry
from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


@pytest.mark.parametrize("command", [
    "flagscale train --help",
    "flagscale train -h",
    "flagscale run --help",
    "cd /workspace/FlagScale && flagscale train --help",
    "flagscale train --help | head -20",
    "flagscale train \\\n        --help",
    'echo "flagscale train model --help; flagscale train model"',
])
def test_help_does_not_force_training_monitor(command):
    guard = TrainingMonitorGuard()
    guard.check_post(GuardContext(
        tool_name="shell", tool_args={"command": command},
        tool_result="Usage: flagscale train [OPTIONS] MODEL",
    ))
    assert guard.check_pre(GuardContext(tool_name="read_file")) is None


@pytest.mark.parametrize("command", [
    "flagscale train --help && flagscale train model -c config.yaml",
    "flagscale train model -c config.yaml; flagscale train --help",
    "flagscale run --help\nflagscale run --config-path /p --config-name c",
    "echo --help && flagscale train model -c config.yaml",
    "flagscale train model --dryrun && flagscale train model -c config.yaml",
    "flagscale run --action query; flagscale train model -c config.yaml",
    "flagscale train model -c /tmp/--help/config.yaml",
])
def test_compound_query_does_not_hide_real_launch(command):
    guard = TrainingMonitorGuard()
    guard.check_post(GuardContext(
        tool_name="shell", tool_args={"command": command}, tool_result="launched",
    ))
    verdict = guard.check_pre(GuardContext(tool_name="read_file"))
    assert verdict is not None
    assert verdict.reason == "must_monitor_after_launch"


def test_existing_non_training_modes_unchanged():
    for command in (
        "flagscale train model --dryrun", "flagscale train model --test",
        "flagscale run --action stop", "flagscale train model --query",
    ):
        assert not _is_flagscale_launch_command(command)


def _completion():
    return GuardContext(
        assistant_text="The verified deliverable is ready. [TASK_COMPLETE]",
        llm_responded=True,
    )


def _plan_and_guard(tmp_path, *, step_done=True):
    plan = TaskPlan(str(tmp_path))
    created = plan.create("Deliver measured configuration", ["Save verified result"])
    if step_done:
        plan.update_step(1, "done", verification=["saved config reloaded with mbs=4"])
    guard = VerificationGuard(plan)
    registry = GuardRegistry()
    registry.register(guard)
    return plan, created["id"], guard, registry


def _run_complete(plan, registry):
    """Exercise the real guard sequence and PlanUpdateTool, including failures."""
    ctx = GuardContext(tool_name="plan_update", tool_args={"action": "complete"})
    first = registry.check_pre(ctx)
    assert first.reason == "task_complete_premise_recheck"
    ctx.override_reason = "Observed reloaded config mbs=4 and measured step time=56 ms."
    # The classifier is not under test; report that the cited evidence is an
    # observation, with neither sample overfit nor a substituted requirement.
    ctx.classify_fn = lambda *args, **kwargs: False
    second = registry.check_pre(ctx)
    assert second.reason == "task_complete_delivery_hygiene"
    ctx.override_reason = "Read delivered config and report; hashes match the tested artifacts."
    assert registry.check_pre(ctx) is None
    ctx.tool_result = PlanUpdateTool(plan).execute(action="complete")
    registry.check_post(ctx)
    return ctx


def test_real_successful_complete_reuses_verification_for_final_text(tmp_path):
    plan, plan_id, guard, registry = _plan_and_guard(tmp_path)
    ctx = _run_complete(plan, registry)
    assert ctx.tool_result == "No active plan."
    assert plan.list_plans()[0]["status"] == "completed"
    assert guard._completed_plan_id == plan_id
    assert registry.check_pre(_completion()) is None


def test_failed_complete_does_not_suppress_final_verification(tmp_path):
    plan, _, _, registry = _plan_and_guard(tmp_path, step_done=False)
    ctx = _run_complete(plan, registry)
    assert ctx.tool_result.startswith("ERROR: Cannot complete plan")
    assert registry.check_pre(_completion()).reason == "text_complete_hygiene"


def test_blocked_complete_does_not_suppress_final_verification(tmp_path):
    _, _, guard, registry = _plan_and_guard(tmp_path)
    ctx = GuardContext(tool_name="plan_update", tool_args={"action": "complete"})
    assert registry.check_pre(ctx).action == "block"
    ctx.tool_result = "[BLOCKED BY GUARD] This tool call was prevented."
    registry.check_post(ctx)
    assert guard._completed_plan_id is None
    assert registry.check_pre(_completion()).reason == "text_complete_hygiene"


@pytest.mark.parametrize("tool_name", ["read_file", "plan_status"])
def test_read_only_queries_preserve_same_turn_completion(tmp_path, tool_name):
    plan, _, _, registry = _plan_and_guard(tmp_path)
    _run_complete(plan, registry)
    ctx = GuardContext(tool_name=tool_name, tool_result="Previously verified result")
    assert registry.check_pre(ctx) is None
    registry.check_post(ctx)
    assert registry.check_pre(_completion()) is None


@pytest.mark.parametrize("tool_name", ["edit_file", "write_file", "shell", "plan_create"])
@pytest.mark.parametrize("batched", [False, True])
def test_new_work_requires_verification_again(tmp_path, tool_name, batched):
    plan, _, guard, registry = _plan_and_guard(tmp_path)
    # In a batch, all calls can pass pre-check before complete executes. Its
    # later mutation must still invalidate the completed delivery in post.
    ctx = GuardContext(tool_name=tool_name, tool_args={"command": "cat report.json"})
    if batched:
        registry.check_pre(ctx)
    _run_complete(plan, registry)
    if not batched:
        registry.check_pre(ctx)
    ctx.tool_result = "Tool executed"
    registry.check_post(ctx)
    assert guard._completed_plan_id is None
    assert registry.check_pre(_completion()).reason == "text_complete_hygiene"


def test_new_user_turn_does_not_reuse_previous_delivery(tmp_path):
    plan, _, guard, registry = _plan_and_guard(tmp_path)
    _run_complete(plan, registry)
    guard.reset_turn()
    assert registry.check_pre(_completion()).reason == "text_complete_hygiene"


def test_old_completed_plan_without_observed_complete_is_not_enough(tmp_path):
    plan, _, guard, registry = _plan_and_guard(tmp_path)
    plan.complete()
    assert guard._completed_plan_id is None
    assert registry.check_pre(_completion()).reason == "text_complete_hygiene"
