"""Config loading and Notion thread-persistence flags."""

from __future__ import annotations

import json
import os
from typing import Any

CONFIG_PATH = "notion_config.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def persist_notion_threads(cfg: dict) -> bool:
    env_value = os.getenv("NOTION_PROXY_PERSIST_THREADS")
    if env_value is not None:
        return _truthy(env_value)
    return _truthy(cfg.get("persist_threads", True))


def suppress_notion_thread_persistence(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    for key in (
        "createThread",
        "generateTitle",
        "saveAllThreadOperations",
        "setUnreadState",
    ):
        if key in body:
            body[key] = False
    return body
