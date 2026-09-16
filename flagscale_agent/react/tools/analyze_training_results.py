# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Structured training measurements derived directly from log evidence."""

import json

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.training_results import analyze_results, summarize_results


class AnalyzeTrainingResultsTool(Tool):
    name = "analyze_training_results"
    description = (
        "Parse exact Megatron training logs and compare one fixed candidate with its baseline. "
        "Computes timing, throughput, repeated-run spread and actual iteration-aligned loss differences. "
        "Requires log_interval=1 and runs in chronological order. Distinguishes missing evidence, "
        "launcher completion, short-run loss tolerance and unverified all-rank/convergence checks. "
        "Use before reporting performance or quality. Returns concise metrics by default; "
        "output_path saves the complete evidence JSON, and detail=full returns it in context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "runs": {
                "type": "array", "minItems": 1, "maxItems": 32,
                "description": "Attempts in actual chronological order; one fixed workload and candidate config. Use one original loss-reporting rank log per attempt. Hard-linked or byte-identical logs cannot establish independent runs and are rejected.",
                "items": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Unique attempt identifier."},
                        "role": {"type": "string", "enum": ["baseline", "candidate"]},
                        "log_path": {"type": "string", "description": "Exact existing log file; no latest-run guessing."},
                        "exit_code_path": {"type": "string", "description": "Optional existing file containing the actual launcher exit code as an integer. Missing means completion unknown."},
                    },
                    "required": ["run_id", "role", "log_path"],
                    "additionalProperties": False,
                },
            },
            "first_iteration": {"type": "integer", "minimum": 0, "description": "First expected logged iteration (default 1)."},
            "end_iteration": {"type": "integer", "minimum": 0, "description": "Last expected iteration, inclusive. Missing steps are errors."},
            "warmup_steps": {"type": "integer", "minimum": 0, "description": "Initial iterations excluded from timing (default 10), still checked for loss/anomalies."},
            "global_batch_size": {"type": "integer", "minimum": 1},
            "sequence_length": {"type": "integer", "minimum": 1, "description": "Fixed tokens per sample; padding/packing or real-data throughput needs separate accounting."},
            "loss_atol": {"type": "number", "minimum": 0, "description": "Predeclared absolute tolerance for logged LM loss. No implicit quality pass if omitted."},
            "loss_rtol": {"type": "number", "minimum": 0, "description": "Relative tolerance: abs(candidate-baseline) <= atol + rtol*abs(baseline)."},
            "max_run_variation_pct": {"type": "number", "minimum": 0, "description": "Predeclared range/median limit across per-run means. Requires at least two baseline and two candidate attempts in adjacent interleaved pairs."},
            "output_path": {"type": "string", "description": "Optional JSON output file in an existing directory; cannot overwrite evidence."},
            "detail": {"type": "string", "enum": ["summary", "full"], "default": "summary", "description": "Summary returns reporting metrics and unresolved checks; full additionally returns all evidence diagnostics. output_path always saves the full report."},
        },
        "required": ["runs", "end_iteration", "global_batch_size", "sequence_length"],
        "additionalProperties": False,
    }

    def execute(self, *, detail="summary", **kwargs):
        try:
            if detail not in ("summary", "full"):
                raise ValueError("detail must be summary or full")
            result = analyze_results(**kwargs)
            if detail == "summary":
                result = summarize_results(result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result = {"status": "error", "error": str(exc)}
        return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
