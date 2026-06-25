"""FastAPI application: Anthropic Messages API in, Notion AI out."""

from __future__ import annotations

import uuid
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import CONFIG_PATH, load_config
from .claude_io import build_error_nudge, latest_user_query, recent_tool_errors
from .logging_utils import dbg, log_event, log_turn
from .notion.client import collect_notion_response
from .converters import build_notion_prompt
from .repair import repair_self_referential_response
from .responses import (
    ResolvedTurn,
    build_message_json,
    iter_sse_events,
)
from .tools.parse import contains_tool_use_markup, parse_tool_uses
from .tools.validate import validate_tool_uses


async def resolve_turn(payload: dict, cfg: dict, prompt: str) -> ResolvedTurn:
    """Run one Notion turn and resolve it to tools-or-text, shared by both routes."""
    full = await collect_notion_response(prompt, cfg)
    notion_response = full
    full = await repair_self_referential_response(payload, cfg, prompt, full)
    log_event("NOTION TEXT AFTER REPAIR", full)

    parsed_tools = parse_tool_uses(full)
    reject_reason = None
    if contains_tool_use_markup(full) and not parsed_tools:
        reject_reason = "Rejected malformed tool_use block from Notion."
    tools, validation_error = validate_tool_uses(payload, parsed_tools)
    reject_reason = reject_reason or validation_error

    log_event("PARSED TOOL USE", {"parsed_tools": parsed_tools})
    log_event(
        "VALIDATED TOOL USE",
        {"tools": tools, "rejected_tool_reason": reject_reason},
    )
    if reject_reason:
        dbg(reject_reason)
        full = reject_reason

    outcome = (
        "REJECTED"
        if reject_reason
        else (f"{len(tools)} TOOL CALL(S)" if tools else "TEXT")
    )
    log_turn(
        user_query=latest_user_query(payload),
        notion_prompt=prompt,
        notion_response=notion_response,
        outcome=outcome,
        tools=tools,
        reject_reason=reject_reason,
    )

    return ResolvedTurn(
        tools=tools,
        text=full,
        reject_reason=reject_reason,
        parsed_tools=parsed_tools,
    )


def create_app(config_path: str = CONFIG_PATH) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/messages")
    async def messages(req: Request):
        try:
            cfg = load_config(config_path)
        except FileNotFoundError:
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "config_error",
                        "message": f"missing {config_path}",
                    },
                },
                500,
            )

        payload = await req.json()
        log_event("RECEIVE FROM CLAUDE CODE", payload)
        nudge = build_error_nudge(recent_tool_errors(payload))
        prompt = build_notion_prompt(
            payload, error_nudge=nudge, notion_model=cfg.get("model", "")
        )
        log_event("TRANSFORM TO NOTION PROMPT", prompt)

        stream = bool(payload.get("stream"))
        model = payload.get("model") or "notion-agent"
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        prompt_len = len(prompt)

        if not stream:
            resolved = await resolve_turn(payload, cfg, prompt)
            response = build_message_json(msg_id, model, resolved, prompt_len)
            log_event("SEND TO CLAUDE CODE", response)
            return JSONResponse(response)

        async def gen() -> AsyncIterator[bytes]:
            # Collect the full response first — we need to know whether it's a
            # tool call before we can emit the right SSE event sequence.
            resolved = await resolve_turn(payload, cfg, prompt)
            for chunk in iter_sse_events(msg_id, model, resolved, prompt_len):
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
