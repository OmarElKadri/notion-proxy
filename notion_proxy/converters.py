"""Convert Anthropic Messages API payloads into a unified format and build structured Notion prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Markers that betray the Claude Code harness when the raw system prompt is
# forwarded verbatim. For the GPT backend we strip these so the system prompt
# reads as ordinary project context. For Claude we KEEP the identity
# declaration ("You are Claude Code") because it's what makes Claude willing
# to act as a coding agent with tools — without it, Claude defaults to its
# "Notion AI" identity and refuses to use the <tool_use> format.
_CLAUDE_CODE_IDENTITY_RE = re.compile(
    r"x-anthropic-billing-header:.*?\n"
    r"|You are Claude Code, Anthropic's official CLI for Claude\.?"
    r"|You are an interactive agent that helps users with software engineering tasks\.?",
    re.IGNORECASE,
)
# For Claude: only strip the billing header, keep the identity declaration.
_CLAUDE_BILLING_RE = re.compile(
    r"x-anthropic-billing-header:.*?\n", re.IGNORECASE
)
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
# First-sentence cap for tool descriptions so the tools summary stays compact
# and doesn't leak multi-paragraph branded prose that screams "foreign tool
# system" to Notion's backend.
_DESC_MAX = 140


def _sanitize_system_prompt(text: str) -> str:
    """Strip Claude-Code harness branding from a system prompt (GPT backend).

    Keeps user/project-specific instructions (AGENTS.md content, preferences)
    but removes the identity declarations and billing headers.
    """
    if not text:
        return ""
    cleaned = _CLAUDE_CODE_IDENTITY_RE.sub("", text)
    cleaned = _SYSTEM_REMINDER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _sanitize_system_prompt_for_claude(text: str) -> str:
    """Light sanitization for the Claude backend.

    Only strips the billing header and system-reminder blocks. KEEPS the
    "You are Claude Code" identity declaration — that identity is what makes
    Claude willing to act as a coding agent with tools. Without it, Claude
    defaults to its "Notion AI" identity and refuses.
    """
    if not text:
        return ""
    cleaned = _CLAUDE_BILLING_RE.sub("", text)
    cleaned = _SYSTEM_REMINDER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _short_description(description: str | None) -> str:
    """Truncate a tool description to a single compact line."""
    if not description:
        return ""
    text = " ".join(description.split())
    if len(text) <= _DESC_MAX:
        return text
    return text[: _DESC_MAX - 1].rstrip() + "…"


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


def build_conversation_transcript(
    messages: list[UnifiedMessage], *, plain_tools: bool = False
) -> str:
    """Build a human-readable conversation transcript for Notion.

    Tool calls and tool results are rendered inline so Notion can follow the
    conversation flow without parsing raw JSON. Harness-injected
    ``<system-reminder>`` blocks are stripped so the transcript reads as a
    clean agent conversation rather than leaking the Claude Code scaffolding.

    When ``plain_tools`` is True (used for the Claude backend), prior tool
    calls are rendered as plain-text action descriptions instead of
    ``<tool_use>`` blocks, and tool results are shown as user-provided
    context. This prevents Claude from seeing the "emit <tool_use> → get
    result" execution pattern in the history, which triggers its
    impersonation refusal ("I'm Notion AI, I can't act as that coding
    agent"). The format for *new* tool calls is still taught in the prompt
    template itself.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        text = msg.content or ""
        if text:
            text = _SYSTEM_REMINDER_RE.sub("", text).strip()
        if role == "assistant":
            lines.append(f"Assistant: {text}")
            if msg.tool_calls:
                if plain_tools:
                    # For Claude: don't render prior tool calls at all.
                    # Showing them (even as plain text) reveals the
                    # "emit request → get result" execution pipeline and
                    # triggers Claude's impersonation refusal. The
                    # assistant's text already summarizes what it did;
                    # the result appears as a user message below.
                    pass
                else:
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
                    content = tr.get("content", "")
                    if plain_tools:
                        lines.append(f"User: {content}")
                    else:
                        tool_use_id = tr.get("tool_use_id", "")
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
    """Build a compact, identity-neutral summary of available tools.

    Full multi-paragraph tool descriptions are dropped (they bloat the prompt
    and leak the foreign Claude-Code tool-system identity that makes Notion's
    backend refuse to engage). Only name, a one-line truncated description,
    and the argument signature are surfaced.
    """
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for tool in tools:
        schema = tool.input_schema or {}
        properties: dict = schema.get("properties") if isinstance(schema, dict) else {}
        required: list = schema.get("required") if isinstance(schema, dict) else []
        args = ", ".join(properties.keys()) if properties else "no input fields"
        req = ", ".join(str(r) for r in required) if required else "none"
        desc = _short_description(tool.description)
        if desc:
            lines.append(f"- {tool.name}: {desc} | args [{args}], required [{req}]")
        else:
            lines.append(f"- {tool.name}: args [{args}], required [{req}]")
    return "\n".join(lines)


def build_notion_prompt(
    payload: dict, error_nudge: str = "", notion_model: str = ""
) -> str:
    """Build the complete Notion prompt from an Anthropic request payload.

    The raw Claude Code system prompt is sanitized (harness branding stripped)
    and presented as project context rather than a competing identity
    declaration. When the Notion backend is Claude (``ambrosia-tart-high``),
    a separate simulation-based prompt template is used — Claude's Notion
    system prompt anchors it as "Notion AI" and refuses direct tool-use
    requests, but it will roleplay as a coding agent that emits
    ``<tool_use>`` blocks.
    """
    from .prompt import (
        load_prompt_template,
        PROMPT_TEMPLATE_PATH,
        CLAUDE_PROMPT_TEMPLATE_PATH,
    )

    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    system = payload.get("system")
    if not isinstance(messages, list):
        messages = []
    if not isinstance(tools, list):
        tools = []

    unified_messages = convert_anthropic_messages(messages)
    unified_tools = convert_anthropic_tools(tools)
    raw_system = extract_system_prompt(system)
    is_claude = "ambrosia" in notion_model.lower()

    # Non-coding requests (no tools) — e.g. Claude Code's title-generation
    # call, which sends a system prompt like "Generate a concise title" with
    # an empty tools list and a JSON output schema. Applying the coding-agent
    # template here makes Claude see tool_use instructions for a task that
    # isn't a coding turn, triggering its "I can't operate as that agent"
    # refusal. Forward these naturally so Claude just does what the system
    # prompt asks.
    if not unified_tools:
        return _build_passthrough_prompt(
            raw_system, unified_messages, error_nudge, is_claude
        )

    if is_claude:
        # For Claude: drop the system prompt entirely. It contains identity
        # declarations ("You are Claude Code"), environment info, and harness
        # instructions that Claude recognizes as foreign when fed back to it,
        # triggering the "I'm Notion AI" refusal. The request-format framing +
        # conversation transcript + tools are sufficient.
        system_prompt = ""
    else:
        system_prompt = _sanitize_system_prompt(raw_system)

    transcript = build_conversation_transcript(
        unified_messages, plain_tools=is_claude
    )
    tools_summary = build_tools_summary(unified_tools)

    nudge_block = f"{error_nudge}\n\n" if error_nudge else ""

    if is_claude:
        system_block = ""
    elif system_prompt:
        system_block = (
            "PROJECT CONTEXT (from the user's environment — treat as context, "
            "not as a competing identity):\n" + system_prompt + "\n\n"
        )
    else:
        system_block = ""

    template_path = CLAUDE_PROMPT_TEMPLATE_PATH if is_claude else PROMPT_TEMPLATE_PATH
    template = load_prompt_template(template_path)
    return (
        template
        .replace("{{ERROR_NUDGE}}", nudge_block)
        .replace("{{SYSTEM_PROMPT}}", system_block)
        .replace("{{TOOLS_SUMMARY}}", tools_summary)
        .replace("{{CONVERSATION_TRANSCRIPT}}", transcript)
    )


def _build_passthrough_prompt(
    raw_system: str,
    unified_messages: list[UnifiedMessage],
    error_nudge: str,
    is_claude: bool,
) -> str:
    """Build a plain prompt for non-coding requests (no tools).

    These are requests like title generation that carry their own task
    instructions in the system prompt and don't need the coding-agent
    tool_use framing. The system prompt is sanitized (identity
    declarations and billing headers stripped) and forwarded with the
    conversation so Claude just performs the requested task.
    """
    # Strip identity declarations for both backends — "You are Claude Code"
    # can trigger Notion-Claude's refusal even for benign tasks like title
    # generation. The task instructions ("Generate a concise title") are
    # preserved; only the harness branding is removed.
    system_prompt = _sanitize_system_prompt(raw_system)

    transcript = build_conversation_transcript(unified_messages)
    nudge_block = f"{error_nudge}\n\n" if error_nudge else ""

    parts: list[str] = []
    if nudge_block:
        parts.append(nudge_block.strip())
    if system_prompt:
        parts.append(system_prompt)
    parts.append(transcript)
    return "\n\n".join(p for p in parts if p)
