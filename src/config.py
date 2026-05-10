"""Central configuration for DSS and optional Ollama advisor."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Ollama: sidebar session state overrides these when user enables AI.
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "false").lower() in ("1", "true", "yes")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Demo maritime sample data (offline; no AIS API).
DATA_SAMPLE_DIR = project_root() / "data" / "sample"
