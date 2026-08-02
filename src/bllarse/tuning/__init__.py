"""Optuna TPE tuning for the probing arms.

Separate from bllarse.tools.run_sweep by necessity: run_sweep's contract is
`create_configs() -> list[dict]`, enumerated before submission, and TPE
generates configurations adaptively from the trials already reported.
"""
