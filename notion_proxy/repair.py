"""Repair self-referential Notion replies into useful agent responses.

After a Notion response, if the text reads as "I'm Notion AI" / "I can't access
local files" and isn't already a valid tool call, the proxy sends a second Notion
request with a repair prompt that rewrites the answer as a continuation of the
coding-agent conversation.
"""

from __future__ import annotations

import re

from .claude_io import available_tools_prompt
from .logging_utils import dbg
from .notion.client import collect_notion_response
from .prompt import REPAIR_PROMPT_TEMPLATE_PATH, load_prompt_template
from .tools.parse import parse_tool_use

BAD_SELF_REFERENCE_RE = re.compile(
    r"\b(?:i\s*(?:am|'m)|as)\s+notion\s+ai\b|"
    r"\bnotion\s+ai\b.*\b(?:can't|cannot|can\s+not|unable)\b|"
    r"\b(?:can't|cannot|can\s+not|unable)\b.*\b(?:local\s+filesystem|local\s+files|d:\\)",
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


async def repair_self_referential_response(
    payload: dict, cfg: dict, original_prompt: str, full: str
) -> str:
    if parse_tool_use(full) or not has_bad_self_reference(full):
        return full
    dbg("repairing self-referential Notion response")
    repaired = await collect_notion_response(
        build_repair_prompt(payload, original_prompt, full), cfg, persist_threads=False
    )
    return repaired or full
