"""Prompt-template loading and request-body rendering."""

from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from typing import Any

from .claude_io import build_error_nudge, recent_tool_errors

PROMPT_TEMPLATE_PATH = os.path.join("prompts", "proxy_to_notion.txt")
REPAIR_PROMPT_TEMPLATE_PATH = os.path.join("prompts", "repair_self_reference.txt")

NUUID_BUCKETS = {
    "thread": "00a9",
    "configStep": "00aa",
    "contextStep": "00aa",
    "userStep": "00aa",
}
DEFAULT_NUUID_BUCKET = "00aa"

_prompt_template_cache: dict[str, str] = {}


def load_prompt_template(path: str) -> str:
    if path not in _prompt_template_cache:
        with open(path, encoding="utf-8") as f:
            _prompt_template_cache[path] = f.read()
    return _prompt_template_cache[path]


def build_notion_prompt(payload: dict) -> str:
    original_request = json.dumps(payload, ensure_ascii=False, indent=2)
    nudge = build_error_nudge(recent_tool_errors(payload))
    nudge_block = f"{nudge}\n\n" if nudge else ""
    return (
        load_prompt_template(PROMPT_TEMPLATE_PATH)
        .replace("{{ERROR_NUDGE}}", nudge_block)
        .replace("{{REQUEST_JSON}}", original_request)
    )


def render_body(
    template: Any, prompt: str, history_text: str, notion_id_prefix: str = ""
) -> Any:
    uuids: dict[str, str] = {}
    nuuids: dict[str, str] = {}
    now_iso = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    def make_nuuid(name: str) -> str:
        bucket = NUUID_BUCKETS.get(name, DEFAULT_NUUID_BUCKET)
        rand = uuid.uuid4().hex
        variant = "89ab"[uuid.uuid4().int & 3]
        return f"{notion_id_prefix}-80{rand[0:2]}-{variant}{rand[2:5]}-{bucket}{rand[5:13]}"

    def sub(s: str) -> str:
        s = (
            s.replace("{{PROMPT}}", prompt)
            .replace("{{HISTORY}}", history_text)
            .replace("{{NOW}}", now_iso)
        )
        s = re.sub(
            r"\{\{NUUID:([A-Za-z0-9_-]+)\}\}",
            lambda m: nuuids.setdefault(m.group(1), make_nuuid(m.group(1))),
            s,
        )
        s = re.sub(
            r"\{\{UUID:([A-Za-z0-9_-]+)\}\}",
            lambda m: uuids.setdefault(m.group(1), str(uuid.uuid4())),
            s,
        )
        s = re.sub(r"\{\{UUID\}\}", lambda _m: str(uuid.uuid4()), s)
        return s

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return sub(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk(template)
