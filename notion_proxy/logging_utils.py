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


def dbg(*a: Any) -> None:
    if DEBUG:
        print("[notion-proxy]", *a, file=sys.stderr, flush=True)


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
