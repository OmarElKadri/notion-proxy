"""Repair self-referential Notion replies into useful agent responses.

After a Notion response, if the text reads as "I'm Notion AI" / "I can't access
local files" and isn't already a valid tool call, the proxy sends a second Notion
request. For Claude (which refuses vague requests), the repair modifies the
conversation to give Claude a concrete first action (e.g. "list files") so it
has a clear task to focus on instead of reflecting on the meta-situation.
"""

from __future__ import annotations

import copy
import re

from .claude_io import available_tools_prompt
from .converters import build_notion_prompt
from .logging_utils import dbg
from .notion.client import collect_notion_response
from .prompt import REPAIR_PROMPT_TEMPLATE_PATH, load_prompt_template
from .tools.parse import parse_tool_use

BAD_SELF_REFERENCE_RE = re.compile(
    r"\b(?:i\s*(?:am|'m)|as)\s+\**notion\s+ai\**\b|"
    r"\b\**notion\s+ai\**\b.*\b(?:can't|cannot|can\s+not|unable)\b|"
    r"\b(?:can't|cannot|can\s+not|unable)\b.*\b(?:local\s+filesystem|local\s+files|d:\\)|"
    r"\b(?:i\s+(?:am|'m)\s+not\s+claude\s+code)\b|"
    r"\b(?:not\s+the\s+coding\s+agent|not\s+claude\s+code)\b",
    re.IGNORECASE | re.DOTALL,
)


def has_bad_self_reference(text: str) -> bool:
    return bool(BAD_SELF_REFERENCE_RE.search(text))


def build_repair_prompt(payload: dict, original_prompt: str, bad_response: str) -> str:
    return (
        load_prompt_template(REPAIR_PROMPT_TEMPLATE_PATH)
        .replace("{{AVAILABLE_TOOLS}}", available_tools_prompt(payload))
        .replace("{{ORIGINAL_PROMPT}}", original_prompt)
        .replace("{{BAD_RESPONSE}}", bad_response)
    )


def _build_claude_repair_prompt(
    payload: dict, cfg: dict, notion_model: str
) -> str:
    """Build a repair prompt for Claude by reusing build_notion_prompt with a
    clean single-turn payload that gives Claude a direct, concrete tool-use
    request with no history or system prompt to trigger detection."""
    # Build a minimal single-turn payload: just "List the files" with tools.
    # No conversation history, no system prompt — nothing for Claude to
    # analyze and detect as foreign. Claude complies with direct, concrete
    # tool-use requests without reflecting.
    repair_payload = {
        "model": payload.get("model", ""),
        "max_tokens": payload.get("max_tokens", 4096),
        "stream": False,
        "system": "",
        "messages": [
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "tools": payload.get("tools", []),
    }

    return build_notion_prompt(repair_payload, notion_model=notion_model)


async def repair_self_referential_response(
    payload: dict, cfg: dict, original_prompt: str, full: str
) -> str:
    if parse_tool_use(full) or not has_bad_self_reference(full):
        return full
    notion_model = cfg.get("model", "")
    is_claude = "ambrosia" in notion_model.lower()
    dbg("repairing self-referential Notion response")

    if is_claude:
        # For Claude: re-send with a concrete first action to avoid the
        # vague-request refusal pattern.
        repair_prompt = _build_claude_repair_prompt(payload, cfg, notion_model)
        repaired = await collect_notion_response(
            repair_prompt, cfg, persist_threads=True
        )
    else:
        # For GPT: use the legacy repair template.
        repaired = await collect_notion_response(
            build_repair_prompt(payload, original_prompt, full), cfg,
            persist_threads=True,
        )
    return repaired or full
