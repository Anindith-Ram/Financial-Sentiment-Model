"""Utility functions for retrieving API keys and creating authenticated clients.

This avoids hard-coding secrets in the codebase. Keys are searched in:
1. Explicit function args (highest priority)
2. Environment variables (e.g. TWITTER_BEARER_TOKEN)
3. YAML/JSON config file path specified via APP_CONFIG env var.
"""

import os
import json
from typing import Optional, Callable, Type, Tuple
from pathlib import Path

import yaml
import functools
import time


def _load_config_file() -> dict:
    cfg_path = os.environ.get("APP_CONFIG")
    if not cfg_path:
        return {}
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} declared in APP_CONFIG not found")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError("Config file must be .yaml, .yml or .json")


_CONFIG_CACHE: Optional[dict] = None


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_config_file()
    return _CONFIG_CACHE


def get_key(name: str, *, env_var: Optional[str] = None) -> str:
    """Retrieve API key named `name`.

    Search order: explicit `env_var`, then NAME in env, then config file.
    """
    env_name = env_var or name.upper()
    if env_name in os.environ:
        return os.environ[env_name]
    cfg = _config()
    if name.lower() in cfg:
        return cfg[name.lower()]
    raise KeyError(f"API key '{name}' not found in env var {env_name} or config file")


def with_retries(max_attempts: int = 3,
                 backoff_seconds: float = 1.0,
                 exceptions: Tuple[Type[BaseException], ...] = (Exception,)) -> Callable:
    """Decorator to retry a function on transient errors.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts before giving up.
    backoff_seconds : float
        Initial back-off delay. Doubles each retry (exponential).
    exceptions : tuple
        Exception types that trigger a retry.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = backoff_seconds
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
        return wrapper

    return decorator 