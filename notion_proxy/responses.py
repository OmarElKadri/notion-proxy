"""Build Anthropic Messages API responses (non-streaming JSON and streaming SSE).

Both response paths are driven from one :class:`ResolvedTurn` so the streaming
and non-streaming branches can never disagree about *what* to send — they only
differ in *how* they frame it on the wire, exactly as the Anthropic API does.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from .logging_utils import log_event


_TOOL_USE_TAG_RE = re.compile(
    r"<tool_use\b[^>]*>.*?</tool_use\s*>", re.DOTALL | re.IGNORECASE
)

# Patterns matching the simulation-framing preamble that Claude emits before
# tool calls (e.g. "Here's the next response the agent would produce:"). These
# are meta-reasoning about the sample-output task, not useful assistant text.
_SIM_PREAMBLE_LINE_RE = re.compile(
    r"^(?:"
    r"here'?s\s+(?:the\s+)?(?:next\s+)?response\s+(?:the\s+)?(?:coding\s+)?agent[^:]*:?"
    r"|here'?s\s+(?:the\s+)?agent'?s\s+next\s+response[^:]*:?"
    r"|this\s+is\s+a\s+simple,?\s+benign\s+request.*"
    r"|.*\bsample\s+output\b.*"
    r"|.*\bcoding\s+agent\s+(?:would|will)\s+produce\b.*"
    r"|.*\bparser\b.*\bsample\b.*"
    r"|.*\broleplay/sample-generation\b.*"
    r"|.*\bharmless\s+roleplay\b.*"
    r"|.*\bno\s+real\s+execution\s+involved\b.*"
    r")\s*$",
    re.IGNORECASE,
)


def _strip_simulation_preamble(text: str) -> str:
    """Remove simulation-framing meta-reasoning lines from Claude's response.

    Claude's simulation prompt causes it to emit preamble like "Here's the next
    response the agent would produce:" before tool calls. These lines are
    meta-reasoning about the sample-output task, not useful to the end user.
    """
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        if _SIM_PREAMBLE_LINE_RE.match(line.strip()):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


@dataclass
class ResolvedTurn:
    """The resolved outcome of one Notion turn, ready to serialize."""

    tools: list[dict]
    text: str
    reject_reason: str | None = None
    parsed_tools: list[dict] = field(default_factory=list)

    def text_without_tools(self) -> str:
        """Return prose with ``<tool_use>…</tool_use>`` blocks removed.

        Also strips simulation-framing preamble lines (e.g. "Here's the next
        response the agent would produce:") that Claude emits before tool calls.
        """
        raw = _TOOL_USE_TAG_RE.sub("", self.text)
        return _strip_simulation_preamble(raw)


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def tool_content_blocks(tools: list[dict]) -> list[dict]:
    """Public alias for backward compatibility."""
    return _tool_content_blocks(tools)


def _tool_content_blocks(tools: list[dict]) -> list[dict]:
    """Build Anthropic tool_use content blocks with unique IDs."""
    return [
        {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex}",
            "name": tool["name"],
            "input": tool.get("input", {}),
        }
        for tool in tools
    ]


def _content_blocks(resolved: ResolvedTurn) -> list[dict]:
    """Build Anthropic content blocks in monotonic order (text first, then tools).

    If the turn contains both prose and tool calls, both are emitted so that
    the assistant can say something *and* call a tool in the same message.
    """
    blocks: list[dict] = []
    clean_text = resolved.text_without_tools()
    # If tools are present and there is clean prose, emit text first.
    if clean_text:
        blocks.append({"type": "text", "text": clean_text})
    if resolved.tools:
        blocks.extend(_tool_content_blocks(resolved.tools))
    # Fallback: no tools and no clean text → emit the raw text (even if empty).
    if not blocks:
        blocks.append({"type": "text", "text": resolved.text or ""})
    return blocks


def _estimate_output_tokens(blocks: list[dict]) -> int:
    """Crude token estimate for usage reporting."""
    total = 0
    for block in blocks:
        if block["type"] == "text":
            total += max(1, len(block.get("text", "")) // 4)
        elif block["type"] == "tool_use":
            total += max(1, len(json.dumps(block.get("input", {}), ensure_ascii=False)) // 4)
    return total or 1


def build_message_json(
    msg_id: str, model: str, resolved: ResolvedTurn, prompt_len: int
) -> dict:
    blocks = _content_blocks(resolved)
    stop_reason = "tool_use" if resolved.tools else "end_turn"
    input_tokens = 1 if resolved.tools else max(1, prompt_len // 4)
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": _estimate_output_tokens(blocks),
        },
    }


def iter_sse_events(
    msg_id: str, model: str, resolved: ResolvedTurn, prompt_len: int
) -> Iterator[bytes]:
    def emit(event: str, data: dict) -> bytes:
        log_event("SEND TO CLAUDE CODE SSE EVENT", {"event": event, "data": data})
        return sse(event, data)

    blocks = _content_blocks(resolved)
    stop_reason = "tool_use" if resolved.tools else "end_turn"
    output_tokens = _estimate_output_tokens(blocks)

    input_tokens = 1 if resolved.tools else max(1, prompt_len // 4)
    yield emit(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                },
            },
        },
    )

    for index, block in enumerate(blocks):
        if block["type"] == "text":
            yield emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if block.get("text"):
                yield emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            yield emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        elif block["type"] == "tool_use":
            yield emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            block.get("input", {}), ensure_ascii=False
                        ),
                    },
                },
            )
            yield emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )

    yield emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield emit("message_stop", {"type": "message_stop"})
