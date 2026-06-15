from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Dict, Optional


def load_mlflow_env_defaults(credentials_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load MLflow credentials/tracking URI from ~/.mlflow/credentials (if present)
    and set environment defaults when variables are missing.
    """
    creds_path = credentials_path or (Path.home() / ".mlflow" / "credentials")
    if not creds_path.exists():
        return {}

    parser = configparser.ConfigParser()
    try:
        parser.read(creds_path)
    except Exception:
        return {}

    if not parser.has_section("mlflow"):
        return {}

    username = parser.get("mlflow", "mlflow_tracking_username", fallback="").strip()
    password = parser.get("mlflow", "mlflow_tracking_password", fallback="").strip()

    tracking_uri = ""
    for key in ("mlflow_tracking_uri", "tracking_uri"):
        tracking_uri = parser.get("mlflow", key, fallback="").strip()
        if tracking_uri:
            break

    env_defaults: Dict[str, str] = {}
    if username:
        env_defaults["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        env_defaults["MLFLOW_TRACKING_PASSWORD"] = password
    if tracking_uri:
        env_defaults["MLFLOW_TRACKING_URI"] = tracking_uri

    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    return env_defaults
