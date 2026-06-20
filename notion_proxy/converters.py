"""Convert Anthropic Messages API payloads into a unified format and build structured Notion prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedMessage:
    """API-agnostic message representation."""

    role: str
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class UnifiedTool:
    """API-agnostic tool representation."""

    name: str
    description: str | None = None
    input_schema: dict | None = None


def extract_text_content(content: Any) -> str:
    """Extract plain text from Anthropic content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def extract_system_prompt(system: Any) -> str:
    """Extract system prompt text from Anthropic system field."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(system)


def extract_tool_uses_from_anthropic_content(content: Any) -> list[dict]:
    """Extract tool_use blocks from assistant message content."""
    tool_calls: list[dict] = []
    if not isinstance(content, list):
        return tool_calls
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        tool_id = block.get("id")
        tool_name = block.get("name")
        tool_input = block.get("input", {})
        if tool_id and tool_name:
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_input
                        if isinstance(tool_input, dict)
                        else (json.loads(tool_input) if isinstance(tool_input, str) else {}),
                    },
                }
            )
    return tool_calls


def extract_tool_results_from_anthropic_content(content: Any) -> list[dict]:
    """Extract tool_result blocks from user message content."""
    tool_results: list[dict] = []
    if not isinstance(content, list):
        return tool_results
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        tool_use_id = block.get("tool_use_id")
        result_content = block.get("content", "")
        if tool_use_id:
            if isinstance(result_content, list):
                result_content = extract_text_content(result_content)
            elif not isinstance(result_content, str):
                result_content = str(result_content)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content or "(empty result)",
                }
            )
    return tool_results


def convert_anthropic_messages(messages: list[dict]) -> list[UnifiedMessage]:
    """Convert a list of Anthropic message dicts into UnifiedMessage objects."""
    unified: list[UnifiedMessage] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = extract_text_content(content)
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        if role == "assistant":
            tool_calls = extract_tool_uses_from_anthropic_content(content)
        elif role == "user":
            tool_results = extract_tool_results_from_anthropic_content(content)
        unified.append(
            UnifiedMessage(
                role=role,
                content=text,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
        )
    return unified


def convert_anthropic_tools(tools: list[dict] | None) -> list[UnifiedTool]:
    """Convert Anthropic tool definitions into UnifiedTool objects."""
    if not tools:
        return []
    unified: list[UnifiedTool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        unified.append(
            UnifiedTool(
                name=name,
                description=tool.get("description") or None,
                input_schema=tool.get("input_schema") or None,
            )
        )
    return unified


def tool_calls_to_text(tool_calls: list[dict]) -> str:
    """Convert tool calls to a human-readable text representation."""
    if not tool_calls:
        return ""
    parts: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        arguments = func.get("arguments", {})
        tool_id = tc.get("id", "")
        if isinstance(arguments, dict):
            args_text = json.dumps(arguments, ensure_ascii=False, indent=2)
        else:
            args_text = str(arguments)
        if tool_id:
            parts.append(f"[Tool: {name} ({tool_id})]\n{args_text}")
        else:
            parts.append(f"[Tool: {name}]\n{args_text}")
    return "\n\n".join(parts)


def tool_results_to_text(tool_results: list[dict]) -> str:
    """Convert tool results to a human-readable text representation."""
    if not tool_results:
        return ""
    parts: list[str] = []
    for tr in tool_results:
        content = tr.get("content", "")
        tool_use_id = tr.get("tool_use_id", "")
        if isinstance(content, str):
            content_text = content
        else:
            content_text = str(content)
        if not content_text:
            content_text = "(empty result)"
        if tool_use_id:
            parts.append(f"[Tool Result ({tool_use_id})]\n{content_text}")
        else:
            parts.append(f"[Tool Result]\n{content_text}")
    return "\n\n".join(parts)


def build_conversation_transcript(messages: list[UnifiedMessage]) -> str:
    """Build a human-readable conversation transcript for Notion.

    Tool calls and tool results are rendered inline so Notion can follow the
    conversation flow without parsing raw JSON.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        text = msg.content or ""
        if role == "assistant":
            lines.append(f"Assistant: {text}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    arguments = func.get("arguments", {})
                    lines.append("")
                    lines.append(f"<tool_use>")
                    lines.append(name)
                    if isinstance(arguments, dict):
                        for key, val in arguments.items():
                            if isinstance(val, str) and ("\n" in val or "\\" in val or '"' in val):
                                tag = f"ARG_{key.upper()}"
                                lines.append(f"{key}: <<<{tag}")
                                lines.append(val)
                                lines.append(tag)
                            else:
                                lines.append(f"{key}: {val}")
                    lines.append("</tool_use>")
        elif role == "user":
            if msg.tool_results:
                for tr in msg.tool_results:
                    tool_use_id = tr.get("tool_use_id", "")
                    content = tr.get("content", "")
                    lines.append("")
                    lines.append(f"User (tool result {tool_use_id}):")
                    lines.append(content)
                if text:
                    lines.append("")
                    lines.append(f"User: {text}")
            else:
                lines.append(f"User: {text}")
        else:
            lines.append(f"{role.capitalize()}: {text}")
            if msg.tool_calls:
                lines.append(tool_calls_to_text(msg.tool_calls))
            if msg.tool_results:
                lines.append(tool_results_to_text(msg.tool_results))
    return "\n".join(lines)


def build_tools_summary(tools: list[UnifiedTool]) -> str:
    """Build a brief summary of available tools for the prompt."""
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for tool in tools:
        name = tool.name
        description = tool.description or ""
        lines.append(f"- {name}: {description}")
        schema = tool.input_schema or {}
        if isinstance(schema, dict):
            properties = schema.get("properties")
            required = schema.get("required")
            if isinstance(properties, dict):
                args = ", ".join(properties.keys())
                lines.append(f"  args: [{args}]")
            if isinstance(required, list):
                req = ", ".join(str(r) for r in required)
                lines.append(f"  required: [{req}]")
    return "\n".join(lines)


def build_notion_prompt(payload: dict, error_nudge: str = "") -> str:
    """Build the complete Notion prompt from an Anthropic request payload.

    This replaces the raw {{REQUEST_JSON}} dump with a structured conversation
    transcript and tools summary.
    """
    from .prompt import load_prompt_template, PROMPT_TEMPLATE_PATH

    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    system = payload.get("system")
    if not isinstance(messages, list):
        messages = []
    if not isinstance(tools, list):
        tools = []

    unified_messages = convert_anthropic_messages(messages)
    unified_tools = convert_anthropic_tools(tools)
    system_prompt = extract_system_prompt(system)

    transcript = build_conversation_transcript(unified_messages)
    tools_summary = build_tools_summary(unified_tools)

    nudge_block = f"{error_nudge}\n\n" if error_nudge else ""
    system_block = f"{system_prompt}\n\n" if system_prompt else ""

    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    return (
        template
        .replace("{{ERROR_NUDGE}}", nudge_block)
        .replace("{{SYSTEM_PROMPT}}", system_block)
        .replace("{{TOOLS_SUMMARY}}", tools_summary)
        .replace("{{CONVERSATION_TRANSCRIPT}}", transcript)
    )
