# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint for existing bounded-training commands."""

from flagscale_agent.training.trial import run_training, main

__all__ = ["run_training", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
