"""Debug logging and event logging helpers."""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Any

DEBUG = os.getenv("NOTION_PROXY_DEBUG", "1") == "1"
DEBUG_FILE = "last_response.txt"
LOG_FILE = "log.txt"
TURN_LOG_FILE = "turns.log"


def dbg(*a: Any) -> None:
    if DEBUG:
        print("[notion-proxy]", *a, file=sys.stderr, flush=True)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [truncated {len(text) - limit} chars]"


def log_turn(
    *,
    user_query: str,
    notion_prompt: str,
    notion_response: str,
    outcome: str,
    tools: Any = None,
    reject_reason: str | None = None,
) -> None:
    """Append one readable record of a full round-trip to ``turns.log``.

    Unlike :func:`log_event` (a raw firehose of every internal event), this is a
    single block per turn: what the user asked, the prompt we sent Notion, what
    Notion replied, and how we resolved it. The big request JSON embedded in the
    prompt is trimmed so the file stays skimmable.
    """
    try:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        parts = [
            "\n" + "#" * 100,
            f"# TURN {ts}  ->  {outcome}",
            "#" * 100,
            "",
            "--- USER QUERY " + "-" * 85,
            _truncate(user_query, 4000),
            "",
            "--- PROMPT SENT TO NOTION " + "-" * 74,
            _truncate(notion_prompt, 6000),
            "",
            "--- NOTION RESPONSE " + "-" * 80,
            _truncate(notion_response, 8000),
        ]
        if tools:
            parts += ["", "--- RESOLVED TOOL CALLS " + "-" * 76,
                      json.dumps(tools, indent=2, ensure_ascii=False, default=str)]
        if reject_reason:
            parts += ["", "--- REJECTED " + "-" * 87, reject_reason]
        parts.append("")
        with open(TURN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(parts) + "\n")
    except Exception as e:
        dbg("could not write turn log:", e)


def log_event(title: str, data: Any = None) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(
                f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {title}\n"
            )
            if data is not None:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    f.write("\n")
                elif isinstance(data, bytes):
                    f.write(data.decode("utf-8", "replace"))
                    f.write("\n")
                else:
                    f.write(str(data))
                    f.write("\n")
    except Exception as e:
        dbg("could not write log:", e)
