<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Analyze Training Results

Use `analyze_training_results` for Megatron logs with `log_interval=1`. Supply the exact loss-reporting rank log from each attempt, not a directory, concatenated rank logs, or a guessed timestamp. Keep one fixed workload and one fixed candidate in each comparison request. For other log formats, state the unsupported format and use a compatible analysis tool with explicit measurement rules.

An initial screening comparison uses the actual paths and iteration window:

```python
analyze_training_results(
    runs=[
        {"run_id": "baseline-01", "role": "baseline", "log_path": baseline_log},
        {"run_id": "candidate-01", "role": "candidate", "log_path": candidate_log},
    ],
    first_iteration=1, end_iteration=40, warmup_steps=10,
    global_batch_size=64, sequence_length=512,
    loss_atol=declared_loss_atol,
    output_path=results_json,
)
```

The numbers above illustrate a 40-step, fixed-length workload; use the current contract's values and predeclared tolerances. `output_path` must have an existing parent directory. The tool saves the complete computed JSON, including validated evidence paths and hashes, but returns a concise summary by default. Use that summary for decisions and reporting; request `detail="full"` only when a specific diagnostic needs the details. If the target has the package but the native tool is unavailable, write the analysis request without `detail` as JSON and run `python -m flagscale_agent.training.results --request request.json` in that environment; this CLI preserves its full JSON output. Do not relaunch training just to repair an analysis request.

For final repeats, supply attempts in actual chronological order as B/C, C/B, B/C pairs, and pass the agreed `max_run_variation_pct`. Each candidate attempt uses the same candidate configuration. One old baseline plus repeated candidates does not establish repeatability. The analyzer requires at least two pairs to assess a declared spread limit; use the task's repeat count (the workflow starts with three pairs when unspecified). It does not conduct a statistical significance test.

The default summary includes per-run timing, launcher status, skip/NaN counters, evidence errors, comparison metrics and unchanged acceptance checks. In `comparison.performance`, `median_run_mean_step_time_ms`, `tokens_per_second`, and `observed_speedup` use the same aggregation; copy these values directly into the final report. Do not substitute averages of individual throughputs or recalculate throughput by hand. `run_variation_pct` is the measured range divided by the median of per-run mean times. Missing evidence stays unknown and `status="ok"` does not mean acceptance.

The complete report has these additional evidence boundaries:

| Field | What It Establishes |
| --- | --- |
| Top-level `status` | Parsing/evidence validity, not acceptance |
| `runs[].measurement` | Logged iteration coverage, timing window, counters and fixed-length token throughput |
| `runs[].completion` | Iteration coverage and optional launcher exit evidence; all-rank completion remains separate |
| `comparison.performance` | Observed speedup, `time_reduction_pct` and `throughput_improvement_pct`, with different meanings |
| `comparison.loss_checks` | Actual aligned loss differences over all requested iterations, including warmup; tolerance status if declared |
| `comparison.repeatability` | This comparison's run counts, pairing and measured spread against the declared limit |
| `comparison.acceptance_checks` | Separate checks to combine with workload equivalence, all-rank status and task-specific requirements |

Missing skip/NaN counters stay unknown. Omitted loss tolerances do not pass quality. A finite, close loss trajectory does not establish parameter equivalence or long-term convergence. No input to this tool verifies configuration/seed/data equivalence; confirm those from the saved effective configurations and workload records. Use profiler evidence before making a bottleneck attribution; timing alone supports a performance observation and a hypothesis.

If the launcher captured its exit status, provide `exit_code_path` per attempt, pointing to the actual file containing its integer exit code. Do not create a zero exit file afterward based on step count or process absence. Launcher success still does not independently verify every remote worker.

When reporting mock-data measurements, state the model, fixed sequence length and synthetic-data scope. Real-data input throughput and long-term convergence require their own evidence.
