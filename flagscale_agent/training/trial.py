# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the train-run skill script."""

from importlib import import_module

_impl = import_module("flagscale_agent.skills.train-run.scripts.training_trial")
run_training = _impl.run_training
main = _impl.main

__all__ = ["run_training", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
