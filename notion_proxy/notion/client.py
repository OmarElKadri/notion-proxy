"""HTTP client that calls Notion's runInferenceTranscript and yields text."""

from __future__ import annotations

import json
from typing import AsyncIterator

from curl_cffi.requests import AsyncSession

from ..config import persist_notion_threads, suppress_notion_thread_persistence
from ..logging_utils import DEBUG, DEBUG_FILE, dbg, log_event
from ..prompt import render_body
from .stream import NotionPatchStream

INCLUDE_THINKING = False


def safe_outgoing_headers(headers: dict) -> dict:
    drop = {
        "host",
        "content-length",
        "connection",
        ":authority",
        ":method",
        ":path",
        ":scheme",
    }
    return {k: v for k, v in headers.items() if k.lower() not in drop}


async def call_notion(
    prompt: str, cfg: dict, *, persist_threads: bool | None = None
) -> AsyncIterator[str]:
    body = render_body(
        cfg["body_template"],
        prompt,
        "",
        cfg.get("notion_id_prefix", ""),
        cfg.get("model", ""),
    )
    if persist_threads is None:
        persist_threads = persist_notion_threads(cfg)
    if not persist_threads:
        suppress_notion_thread_persistence(body)
    headers = safe_outgoing_headers(cfg.get("headers", {}))
    method = cfg.get("method", "POST")
    url = cfg["url"]
    log_event(
        "SEND TO NOTION",
        {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
        },
    )
    parser = NotionPatchStream(include_thinking=INCLUDE_THINKING)
    buf = b""
    raw_response = bytearray()
    if DEBUG:
        try:
            with open(DEBUG_FILE + ".request.json", "w", encoding="utf-8") as f:
                json.dump(body, f, indent=2, ensure_ascii=False)
        except Exception as e:
            dbg("could not write request dump:", e)
        dump = open(DEBUG_FILE, "wb")
    else:
        dump = None
    deltas_emitted = 0
    objs_seen = 0
    types_seen: list[str] = []
    try:
        async with AsyncSession(impersonate="chrome") as client:
            resp = await client.request(
                method, url, headers=headers, json=body, stream=True, timeout=600
            )
            dbg(
                f"upstream {resp.status_code} content-type={resp.headers.get('content-type')!r}"
            )
            if resp.status_code >= 400:
                err = await resp.acontent()
                if dump:
                    dump.write(err)
                raw_response += err
                msg = err[:2000].decode("utf-8", "replace")
                dbg(f"upstream error body: {msg}")
                yield f"[notion-proxy] upstream {resp.status_code}: {msg}"
                return
            async for chunk in resp.aiter_content():
                if dump:
                    dump.write(chunk)
                raw_response += chunk
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl].strip(), buf[nl + 1 :]
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    objs_seen += 1
                    t = obj.get("type") if isinstance(obj, dict) else None
                    if t and t not in types_seen:
                        types_seen.append(t)
                    for delta in parser.feed(obj):
                        if delta:
                            deltas_emitted += 1
                            yield delta
            tail = buf.strip()
            if tail:
                try:
                    obj = json.loads(tail)
                    objs_seen += 1
                    for delta in parser.feed(obj):
                        if delta:
                            deltas_emitted += 1
                            yield delta
                except Exception:
                    pass
    finally:
        if dump:
            dump.close()
        log_event(
            "RECEIVE FROM NOTION",
            {
                "parsed_objects": objs_seen,
                "types_seen": types_seen,
                "deltas_emitted": deltas_emitted,
                "raw_response": raw_response.decode("utf-8", "replace"),
            },
        )
        dbg(
            f"parsed {objs_seen} objects, types={types_seen}, emitted {deltas_emitted} text deltas"
        )
        if objs_seen and not deltas_emitted:
            dbg(f"no text extracted — inspect {DEBUG_FILE} to see what Notion returned")


async def collect_notion_response(
    prompt: str, cfg: dict, *, persist_threads: bool | None = None
) -> str:
    full = ""
    async for text in call_notion(prompt, cfg, persist_threads=persist_threads):
        full += text
    return full
