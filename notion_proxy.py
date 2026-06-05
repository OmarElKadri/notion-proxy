"""
Notion AI agent -> Anthropic /v1/messages streaming proxy.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import sys
import uuid
from contextlib import redirect_stdout
from html import unescape
from typing import Any, AsyncIterator, Iterable

import uvicorn
from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from extract_claude_context import extract as extract_claude_context
from extract_claude_context import print_context

CONFIG_PATH = "notion_config.json"
INCLUDE_THINKING = False
DEBUG = os.getenv("NOTION_PROXY_DEBUG", "1") == "1"
DEBUG_FILE = "last_response.txt"
LOG_FILE = "log.txt"
PROMPT_TEMPLATE_PATH = os.path.join("prompts", "proxy_to_notion.txt")
REPAIR_PROMPT_TEMPLATE_PATH = os.path.join("prompts", "repair_self_reference.txt")
_prompt_template_cache: dict[str, str] = {}

TOOL_USE_RE = re.compile(
    r"<tool_use\b[^>]*>\s*(.*?)\s*</tool_use>", re.DOTALL | re.IGNORECASE
)
TOOL_USE_MARKUP_RE = re.compile(r"</?tool_use\b[^>]*>", re.IGNORECASE)
SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
BAD_SELF_REFERENCE_RE = re.compile(
    r"\b(?:i\s*(?:am|'m)|as)\s+notion\s+ai\b|"
    r"\bnotion\s+ai\b.*\b(?:can't|cannot|can\s+not|unable)\b|"
    r"\b(?:can't|cannot|can\s+not|unable)\b.*\b(?:local\s+filesystem|local\s+files|d:\\)",
    re.IGNORECASE | re.DOTALL,
)

NUUID_BUCKETS = {
    "thread": "00a9",
    "configStep": "00aa",
    "contextStep": "00aa",
    "userStep": "00aa",
}
DEFAULT_NUUID_BUCKET = "00aa"


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


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_claude_context_extractor(payload: dict) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_context(extract_claude_context(payload))
    return buf.getvalue().strip()


def _text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        cleaned = SYSTEM_REMINDER_RE.sub("", content).strip()
        return [cleaned] if cleaned else []
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = str(block.get("text") or "").strip()
        if not text or text.lower().startswith("<system-reminder>"):
            continue
        texts.append(text)
    return texts


def latest_user_query(payload: dict) -> str:
    for message in reversed(payload.get("messages", []) or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        texts = _text_blocks(message.get("content", ""))
        if texts:
            return texts[-1]
    return json.dumps(payload, ensure_ascii=False)


def _stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = _text_blocks(content)
        if texts:
            return "\n".join(texts).strip()
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def tool_result_texts(payload: dict) -> list[str]:
    results: list[str] = []
    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _stringify_tool_result_content(block.get("content"))
            if text:
                results.append(text)
    return results


def is_tool_result_turn(payload: dict) -> bool:
    return bool(tool_result_texts(payload))


def available_tool_names(payload: dict) -> set[str]:
    names: set[str] = set()
    for tool in payload.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def available_tools_prompt(payload: dict) -> str:
    lines: list[str] = []
    for tool in payload.get("tools", []) or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        name = tool["name"]
        input_schema = tool.get("input_schema")
        schema = input_schema if isinstance(input_schema, dict) else {}
        schema_properties = schema.get("properties")
        properties = schema_properties if isinstance(schema_properties, dict) else {}
        schema_required = schema.get("required")
        required = schema_required if isinstance(schema_required, list) else []
        args = ", ".join(properties.keys()) if properties else "no input fields"
        req = ", ".join(str(item) for item in required) if required else "none"
        lines.append(f"- {name}: args [{args}], required [{req}]")
    return "\n".join(lines) if lines else "(no tools available)"


def load_prompt_template(path: str) -> str:
    if path not in _prompt_template_cache:
        with open(path, encoding="utf-8") as f:
            _prompt_template_cache[path] = f.read()
    return _prompt_template_cache[path]


def build_notion_prompt(payload: dict) -> str:
    original_request = json.dumps(payload, ensure_ascii=False, indent=2)
    return load_prompt_template(PROMPT_TEMPLATE_PATH).replace(
        "{{REQUEST_JSON}}", original_request
    )


def render_body(
    template: Any, prompt: str, history_text: str, notion_id_prefix: str = ""
) -> Any:
    uuids: dict[str, str] = {}
    nuuids: dict[str, str] = {}
    now_iso = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    def make_nuuid(name: str) -> str:
        bucket = NUUID_BUCKETS.get(name, DEFAULT_NUUID_BUCKET)
        rand = uuid.uuid4().hex
        variant = "89ab"[uuid.uuid4().int & 3]
        return f"{notion_id_prefix}-80{rand[0:2]}-{variant}{rand[2:5]}-{bucket}{rand[5:13]}"

    def sub(s: str) -> str:
        s = (
            s.replace("{{PROMPT}}", prompt)
            .replace("{{HISTORY}}", history_text)
            .replace("{{NOW}}", now_iso)
        )
        s = re.sub(
            r"\{\{NUUID:([A-Za-z0-9_-]+)\}\}",
            lambda m: nuuids.setdefault(m.group(1), make_nuuid(m.group(1))),
            s,
        )
        s = re.sub(
            r"\{\{UUID:([A-Za-z0-9_-]+)\}\}",
            lambda m: uuids.setdefault(m.group(1), str(uuid.uuid4())),
            s,
        )
        s = re.sub(r"\{\{UUID\}\}", lambda _m: str(uuid.uuid4()), s)
        return s

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return sub(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk(template)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def persist_notion_threads(cfg: dict) -> bool:
    env_value = os.getenv("NOTION_PROXY_PERSIST_THREADS")
    if env_value is not None:
        return _truthy(env_value)
    return _truthy(cfg.get("persist_threads", True))


def suppress_notion_thread_persistence(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    for key in (
        "createThread",
        "generateTitle",
        "saveAllThreadOperations",
        "setUnreadState",
    ):
        if key in body:
            body[key] = False
    return body


def _double_single_backslashes(text: str) -> str:
    parts: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            parts.append(ch)
            i += 1
            continue
        if i + 1 < len(text) and text[i + 1] == "\\":
            parts.append("\\\\")
            i += 2
            continue
        parts.append("\\\\")
        i += 1
    return "".join(parts)


def _escape_windows_path_strings(text: str) -> str:
    # Notion sometimes emits Windows paths like D:\Dev inside JSON strings.
    # If a string looks like it contains a drive path, make its backslashes valid
    # JSON before json.loads interprets sequences like \t or \r as control chars.
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != '"':
            out.append(text[i])
            i += 1
            continue
        start = i
        i += 1
        content_start = i
        backslashes = 0
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                backslashes += 1
                i += 1
                continue
            if ch == '"' and backslashes % 2 == 0:
                break
            backslashes = 0
            i += 1
        if i >= len(text):
            out.append(text[start:])
            break
        content = text[content_start:i]
        if re.search(r"[A-Za-z]:\\", content):
            content = _double_single_backslashes(content)
        out.append('"' + content + '"')
        i += 1
    return "".join(out)


def _escape_invalid_json_backslashes(text: str) -> str:
    # JSON only allows backslashes before one of these escape characters.
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _parse_loose_json_object(text: str) -> dict | None:
    text = text.strip().rstrip(";")
    text = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:", r'\1"\2":', text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    attempts = [
        text,
        _escape_windows_path_strings(text),
        _escape_invalid_json_backslashes(text),
        _escape_invalid_json_backslashes(_escape_windows_path_strings(text)),
    ]
    for candidate in attempts:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _normalize_tool_response(text: str) -> str:
    return unescape(text).replace("\ufeff", "").strip()


def contains_tool_use_markup(text: str) -> bool:
    return bool(TOOL_USE_MARKUP_RE.search(_normalize_tool_response(text)))


def _parse_tool_call(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json|xml|html)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    tool = _parse_loose_json_object(raw)
    if tool and "name" in tool:
        return tool

    call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", raw, re.DOTALL)
    if not call:
        return None
    input_obj = _parse_loose_json_object(call.group(2))
    if input_obj is None:
        return None
    return {"name": call.group(1), "input": input_obj}


def parse_tool_uses(text: str) -> list[dict]:
    """Extract all tool calls from <tool_use>...</tool_use> blocks."""
    normalized = _normalize_tool_response(text)
    tools: list[dict] = []
    for match in TOOL_USE_RE.finditer(normalized):
        tool = _parse_tool_call(match.group(1))
        if tool:
            tools.append(tool)
    return tools


def parse_tool_use(text: str) -> dict | None:
    tools = parse_tool_uses(text)
    return tools[0] if tools else None


def _available_tools_by_name(payload: dict) -> dict[str, dict]:
    tools: dict[str, dict] = {}
    for tool in payload.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools[tool["name"]] = tool
    return tools


def validate_tool_use(
    payload: dict, tool: dict | None
) -> tuple[dict | None, str | None]:
    if not tool:
        return None, None
    tools_by_name = _available_tools_by_name(payload)
    name = tool.get("name")
    if not isinstance(name, str):
        return None, "Rejected malformed tool call: missing string tool name."
    if tools_by_name and name not in tools_by_name:
        return (
            None,
            f"Rejected unavailable tool call: {name!r}. "
            f"Available tools are: {', '.join(sorted(tools_by_name))}.",
        )
    input_obj = tool.get("input")
    if not isinstance(input_obj, dict):
        return (
            None,
            f"Rejected malformed tool call for {name!r}: input must be an object.",
        )

    schema = tools_by_name.get(name, {}).get("input_schema")
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list):
        missing = [field for field in required if field not in input_obj]
        if missing:
            return (
                None,
                f"Rejected malformed tool call for {name!r}: missing required "
                f"input field(s): {', '.join(str(field) for field in missing)}.",
            )
    return {"name": name, "input": input_obj}, None


def validate_tool_uses(
    payload: dict, tools: list[dict]
) -> tuple[list[dict], str | None]:
    valid: list[dict] = []
    for tool in tools:
        checked, error = validate_tool_use(payload, tool)
        if error:
            return [], error
        if checked:
            valid.append(checked)
    return valid, None


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


def anthropic_tool_response(msg_id: str, model: str, tool: dict) -> JSONResponse:
    return JSONResponse(
        {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": tool_content_blocks([tool]),
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class NotionPatchStream:
    """Parses runInferenceTranscript NDJSON and yields text deltas."""

    def __init__(self, include_thinking: bool = False) -> None:
        self.steps: list[Any] = []
        self.include_thinking = include_thinking

    def feed(self, obj: dict) -> Iterable[str]:
        t = obj.get("type")
        if t == "patch-start":
            self.steps = list(obj.get("data", {}).get("s", []))
            for step in self.steps:
                yield from self._initial_text(step)
            return
        if t == "patch":
            for op in obj.get("v", []) or []:
                yield from self._apply_op(op)
            return
        if t == "patch-sync":
            self.steps = list(obj.get("data", {}).get("s", []))
            return

    def _initial_text(self, step: dict) -> Iterable[str]:
        if not isinstance(step, dict):
            return
        if step.get("type") != "agent-inference":
            return
        for part in step.get("value", []) or []:
            kind = part.get("type")
            content = part.get("content") or ""
            if kind == "text" and content:
                yield content
            elif kind == "thinking" and content and self.include_thinking:
                yield content

    def _apply_op(self, op: dict) -> Iterable[str]:
        o = op.get("o")
        p = op.get("p") or ""
        v = op.get("v")
        segs = [s for s in p.split("/") if s]
        if not segs or segs[0] != "s":
            return

        if o == "a":
            if len(segs) == 2 and segs[1] == "-":
                self.steps.append(v)
                yield from self._initial_text(v if isinstance(v, dict) else {})
                return
            if len(segs) >= 4 and segs[2] == "value" and segs[3] == "-":
                try:
                    n = int(segs[1])
                    step = self.steps[n]
                    step.setdefault("value", []).append(v)
                except (IndexError, ValueError, AttributeError):
                    return
                if isinstance(v, dict):
                    kind = v.get("type")
                    content = v.get("content") or ""
                    if kind == "text" and content:
                        yield content
                    elif kind == "thinking" and content and self.include_thinking:
                        yield content
                return
            self._set_path(segs[1:], v)
            return

        if o == "x":
            if len(segs) == 5 and segs[2] == "value" and segs[4] == "content":
                try:
                    n = int(segs[1])
                    i = int(segs[3])
                    part = self.steps[n]["value"][i]
                except (IndexError, KeyError, ValueError, TypeError):
                    return
                part["content"] = (part.get("content") or "") + (v or "")
                kind = part.get("type")
                if v and kind == "text":
                    yield v
                elif v and kind == "thinking" and self.include_thinking:
                    yield v
                return
            self._append_path(segs[1:], v if isinstance(v, str) else "")
            return

    def _set_path(self, rel: list[str], value: Any) -> None:
        if not rel:
            return
        try:
            node: Any = self.steps[int(rel[0])]
        except (IndexError, ValueError):
            return
        for s in rel[1:-1]:
            node = node[int(s)] if isinstance(node, list) else node.setdefault(s, {})
        last = rel[-1]
        if isinstance(node, list):
            try:
                node[int(last)] = value
            except (IndexError, ValueError):
                pass
        elif isinstance(node, dict):
            node[last] = value

    def _append_path(self, rel: list[str], value: str) -> None:
        if not rel:
            return
        try:
            node: Any = self.steps[int(rel[0])]
        except (IndexError, ValueError):
            return
        for s in rel[1:-1]:
            node = node[int(s)] if isinstance(node, list) else node.get(s, {})
        last = rel[-1]
        if isinstance(node, dict) and isinstance(node.get(last), str):
            node[last] = node[last] + value


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


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
        cfg["body_template"], prompt, "", cfg.get("notion_id_prefix", "")
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


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/messages")
async def messages(req: Request):
    try:
        cfg = load_config()
    except FileNotFoundError:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "config_error", "message": f"missing {CONFIG_PATH}"},
            },
            500,
        )

    payload = await req.json()
    log_event("RECEIVE FROM CLAUDE CODE", payload)
    prompt = build_notion_prompt(payload)
    log_event("TRANSFORM TO NOTION PROMPT", prompt)

    stream = bool(payload.get("stream"))
    model = payload.get("model") or "notion-agent"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if not stream:
        full = await collect_notion_response(prompt, cfg)
        full = await repair_self_referential_response(payload, cfg, prompt, full)
        log_event("NOTION TEXT AFTER REPAIR", full)
        parsed_tools = parse_tool_uses(full)
        rejected_tool_reason = None
        if contains_tool_use_markup(full) and not parsed_tools:
            rejected_tool_reason = "Rejected malformed tool_use block from Notion."
        tools, validation_error = validate_tool_uses(payload, parsed_tools)
        rejected_tool_reason = rejected_tool_reason or validation_error
        log_event("PARSED TOOL USE", {"parsed_tools": parsed_tools})
        log_event(
            "VALIDATED TOOL USE",
            {"tools": tools, "rejected_tool_reason": rejected_tool_reason},
        )
        if tools:
            response = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": tool_content_blocks(tools),
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": max(1, len(tools))},
            }
            log_event("SEND TO CLAUDE CODE", response)
            return JSONResponse(response)
        if rejected_tool_reason:
            dbg(rejected_tool_reason)
            full = rejected_tool_reason
        response = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": full}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": max(1, len(prompt) // 4),
                "output_tokens": max(1, len(full) // 4),
            },
        }
        log_event("SEND TO CLAUDE CODE", response)
        return JSONResponse(response)

    async def gen() -> AsyncIterator[bytes]:
        # Collect full response first — we need to know if it's a tool call
        full = await collect_notion_response(prompt, cfg)
        full = await repair_self_referential_response(payload, cfg, prompt, full)
        log_event("NOTION TEXT AFTER REPAIR", full)
        parsed_tools = parse_tool_uses(full)
        rejected_tool_reason = None
        if contains_tool_use_markup(full) and not parsed_tools:
            rejected_tool_reason = "Rejected malformed tool_use block from Notion."
        tools, validation_error = validate_tool_uses(payload, parsed_tools)
        rejected_tool_reason = rejected_tool_reason or validation_error
        log_event("PARSED TOOL USE", {"parsed_tools": parsed_tools})
        log_event(
            "VALIDATED TOOL USE",
            {"tools": tools, "rejected_tool_reason": rejected_tool_reason},
        )
        if rejected_tool_reason:
            dbg(rejected_tool_reason)
            full = rejected_tool_reason

        def emit(event: str, data: dict) -> bytes:
            log_event("SEND TO CLAUDE CODE SSE EVENT", {"event": event, "data": data})
            return sse(event, data)

        if tools:
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
                            "input_tokens": max(1, len(prompt) // 4),
                            "output_tokens": max(1, len(tools)),
                        },
                    },
                },
            )
            for index, tool in enumerate(tools):
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
                    "usage": {"output_tokens": max(1, len(tools))},
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
                        "input_tokens": max(1, len(prompt) // 4),
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
        if full:
            yield emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": full},
                },
            )
        yield emit("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": max(1, len(full) // 4)},
            },
        )
        yield emit("message_stop", {"type": "message_stop"})

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()
    CONFIG_PATH = args.config
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
