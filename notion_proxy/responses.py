"""Build Anthropic Messages API responses (non-streaming JSON and streaming SSE).

Both response paths are driven from one :class:`ResolvedTurn` so the streaming
and non-streaming branches can never disagree about *what* to send — they only
differ in *how* they frame it on the wire, exactly as the Anthropic API does.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from .logging_utils import log_event


@dataclass
class ResolvedTurn:
    """The resolved outcome of one Notion turn, ready to serialize."""

    tools: list[dict]
    text: str
    reject_reason: str | None = None
    parsed_tools: list[dict] = field(default_factory=list)


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def tool_content_blocks(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex}",
            "name": tool["name"],
            "input": tool.get("input", {}),
        }
        for tool in tools
    ]


def build_message_json(
    msg_id: str, model: str, resolved: ResolvedTurn, prompt_len: int
) -> dict:
    if resolved.tools:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": tool_content_blocks(resolved.tools),
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": max(1, len(resolved.tools))},
        }
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": resolved.text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(1, prompt_len // 4),
            "output_tokens": max(1, len(resolved.text) // 4),
        },
    }


def iter_sse_events(
    msg_id: str, model: str, resolved: ResolvedTurn, prompt_len: int
) -> Iterator[bytes]:
    def emit(event: str, data: dict) -> bytes:
        log_event("SEND TO CLAUDE CODE SSE EVENT", {"event": event, "data": data})
        return sse(event, data)

    if resolved.tools:
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
                        "input_tokens": max(1, prompt_len // 4),
                        "output_tokens": max(1, len(resolved.tools)),
                    },
                },
            },
        )
        for index, tool in enumerate(resolved.tools):
            yield emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex}",
                        "name": tool["name"],
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
                            tool.get("input", {}), ensure_ascii=False
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
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": max(1, len(resolved.tools))},
            },
        )
        yield emit("message_stop", {"type": "message_stop"})
        return

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
                    "input_tokens": max(1, prompt_len // 4),
                    "output_tokens": 0,
                },
            },
        },
    )
    yield emit(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    if resolved.text:
        yield emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": resolved.text},
            },
        )
    yield emit("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": max(1, len(resolved.text) // 4)},
        },
    )
    yield emit("message_stop", {"type": "message_stop"})
