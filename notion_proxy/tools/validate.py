"""Validate and type-coerce parsed tool calls against the payload's schemas."""

from __future__ import annotations

import json
from typing import Any

from ..claude_io import available_tools_by_name


def _coerce_field_value(value: Any, prop_schema: Any) -> Any:
    """Coerce a string value to match a JSON schema declared type.

    The delimiter-based tool format only produces strings, so once we have a
    parsed tool call we look at the tool's input_schema and convert numbers,
    booleans, arrays, and objects back to their proper types. Anything we
    can't confidently coerce is returned unchanged.
    """
    if not isinstance(value, str) or not isinstance(prop_schema, dict):
        return value
    declared = prop_schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    stripped = value.strip()
    for t in types:
        if t == "string":
            return value
        if t == "integer":
            try:
                return int(stripped)
            except ValueError:
                continue
        if t == "number":
            try:
                return float(stripped)
            except ValueError:
                continue
        if t == "boolean":
            low = stripped.lower()
            if low in {"true", "yes", "on", "1"}:
                return True
            if low in {"false", "no", "off", "0"}:
                return False
            continue
        if t in {"array", "object"}:
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue
        if t == "null":
            if stripped.lower() in {"null", "none", ""}:
                return None
            continue
    return value


def _coerce_tool_input(name: str, input_obj: dict, schemas: dict[str, dict]) -> dict:
    schema = schemas.get(name, {}).get("input_schema")
    if not isinstance(schema, dict):
        return input_obj
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return input_obj
    coerced: dict[str, Any] = {}
    for key, val in input_obj.items():
        coerced[key] = _coerce_field_value(val, properties.get(key))
    return coerced


def validate_tool_use(
    payload: dict, tool: dict | None
) -> tuple[dict | None, str | None]:
    if not tool:
        return None, None
    tools_by_name = available_tools_by_name(payload)
    name = tool.get("name")
    if not isinstance(name, str):
        return None, "Rejected malformed tool call: missing string tool name."
    if tools_by_name and name not in tools_by_name:
        return (
            None,
            f"Rejected unavailable tool call: {name!r}. "
            f"Available tools are: {', '.join(sorted(tools_by_name))}.",
        )
    input_obj = tool.get("input")
    if not isinstance(input_obj, dict):
        return (
            None,
            f"Rejected malformed tool call for {name!r}: input must be an object.",
        )
    input_obj = _coerce_tool_input(name, input_obj, tools_by_name)

    schema = tools_by_name.get(name, {}).get("input_schema")
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list):
        missing = [field for field in required if field not in input_obj]
        if missing:
            return (
                None,
                f"Rejected malformed tool call for {name!r}: missing required "
                f"input field(s): {', '.join(str(field) for field in missing)}.",
            )
    return {"name": name, "input": input_obj}, None


def validate_tool_uses(
    payload: dict, tools: list[dict]
) -> tuple[list[dict], str | None]:
    valid: list[dict] = []
    for tool in tools:
        checked, error = validate_tool_use(payload, tool)
        if error:
            return [], error
        if checked:
            valid.append(checked)
    return valid, None
