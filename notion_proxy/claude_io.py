"""Reading the incoming Anthropic (Claude Code) request payload."""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)


def _text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        cleaned = SYSTEM_REMINDER_RE.sub("", content).strip()
        return [cleaned] if cleaned else []
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = str(block.get("text") or "").strip()
        if not text or text.lower().startswith("<system-reminder>"):
            continue
        texts.append(text)
    return texts


def latest_user_query(payload: dict) -> str:
    for message in reversed(payload.get("messages", []) or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        texts = _text_blocks(message.get("content", ""))
        if texts:
            return texts[-1]
    return json.dumps(payload, ensure_ascii=False)


def _stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = _text_blocks(content)
        if texts:
            return "\n".join(texts).strip()
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def tool_result_texts(payload: dict) -> list[str]:
    results: list[str] = []
    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _stringify_tool_result_content(block.get("content"))
            if text:
                results.append(text)
    return results


def is_tool_result_turn(payload: dict) -> bool:
    return bool(tool_result_texts(payload))


def _tool_names_by_id(payload: dict) -> dict[str, str]:
    """Map each assistant tool_use id to its tool name across the transcript."""
    names: dict[str, str] = {}
    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tid, name = block.get("id"), block.get("name")
            if isinstance(tid, str) and isinstance(name, str):
                names[tid] = name
    return names


def recent_tool_errors(payload: dict) -> list[dict]:
    """Errors from the *most recent* tool_result turn, paired with the tool name.

    Claude Code resends the whole transcript every turn, so a failed call sits in
    the history forever. We only surface the errors from the latest turn — if the
    last message is a fresh human query (no tool_result blocks), there is nothing
    to nudge about. Each entry is ``{"name": <tool>, "error": <text>}``.
    """
    names = _tool_names_by_id(payload)
    for message in reversed(payload.get("messages", []) or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return []  # plain-text user turn -> a fresh query, not a tool round-trip
        errors: list[dict] = []
        saw_tool_result = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            saw_tool_result = True
            if not block.get("is_error"):
                continue
            errors.append(
                {
                    "name": names.get(block.get("tool_use_id"), "tool"),
                    "error": _stringify_tool_result_content(block.get("content")),
                }
            )
        # Stop at the latest user message regardless: only its errors are recent.
        return errors if saw_tool_result else []
    return []


def build_error_nudge(errors: list[dict]) -> str:
    """A focused banner telling Notion its last tool call(s) failed.

    Without this, the failed tool_result is buried in the forwarded request JSON
    and Notion tends to re-emit the identical broken call turn after turn.
    """
    if not errors:
        return ""
    lines = [
        "ATTENTION — your previous tool call(s) failed. Do NOT repeat an "
        "identical call. Read each error, fix the underlying cause (wrong path, "
        "missing file, bad arguments), or reply to the user explaining the "
        "problem.",
        "",
        "Failed tool calls:",
    ]
    for err in errors:
        detail = " ".join((err.get("error") or "").split())
        if len(detail) > 500:
            detail = detail[:500] + "…"
        lines.append(f"- {err.get('name', 'tool')} failed: {detail}")
    return "\n".join(lines)


def available_tool_names(payload: dict) -> set[str]:
    names: set[str] = set()
    for tool in payload.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def available_tools_prompt(payload: dict) -> str:
    lines: list[str] = []
    for tool in payload.get("tools", []) or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        name = tool["name"]
        input_schema = tool.get("input_schema")
        schema = input_schema if isinstance(input_schema, dict) else {}
        schema_properties = schema.get("properties")
        properties = schema_properties if isinstance(schema_properties, dict) else {}
        schema_required = schema.get("required")
        required = schema_required if isinstance(schema_required, list) else []
        args = ", ".join(properties.keys()) if properties else "no input fields"
        req = ", ".join(str(item) for item in required) if required else "none"
        lines.append(f"- {name}: args [{args}], required [{req}]")
    return "\n".join(lines) if lines else "(no tools available)"


def available_tools_by_name(payload: dict) -> dict[str, dict]:
    tools: dict[str, dict] = {}
    for tool in payload.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools[tool["name"]] = tool
    return tools
