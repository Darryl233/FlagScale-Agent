# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint for existing training-results imports and commands."""

from flagscale_agent.training.results import analyze_results, summarize_results, main

__all__ = ["analyze_results", "summarize_results", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
