"""Shared helpers: project paths, configuration loading, logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# src/utils/helpers.py -> src/utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Defaults mirror the values that were previously hard-coded in each module.
# They are the single source of truth when config.yaml is absent or PyYAML is
# not installed, so the pipeline behaves identically either way.
_DEFAULTS: dict[str, Any] = {
    "chunking": {"chunk_size": 500, "overlap": 50},
    "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
    "vectordb": {"path": "chroma_db", "collection": "rag_documents", "distance": "cosine"},
    "retrieval": {"n_results": 5, "n_context_chunks": 5},
    "llm": {"model": "claude-sonnet-4-6", "max_tokens": 1024},
    "logging": {"level": "INFO", "file": "logs/app.log"},
}

_config_cache: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Overlay `override` onto `base` without dropping unspecified keys."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    """Read config.yaml once, merged over the built-in defaults."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    loaded: dict[str, Any] = {}
    try:
        import yaml  # optional; defaults are used when unavailable

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
    except ImportError:
        pass

    _config_cache = _deep_merge(_DEFAULTS, loaded)
    return _config_cache


def get(section: str, key: str) -> Any:
    """Fetch a single configuration value, e.g. get("chunking", "chunk_size")."""
    return load_config()[section][key]


def resolve_path(relative: str) -> Path:
    """Turn a config path into an absolute one anchored at the project root."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def setup_logging() -> logging.Logger:
    """Configure root logging to both the console and logs/app.log."""
    cfg = load_config()["logging"]
    log_file = resolve_path(cfg["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, str(cfg["level"]).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("rag")
