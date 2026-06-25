"""Parse tool calls out of Notion's free-text responses.

Two formats are supported, tried in order:

1. A delimiter-based ``field: value`` / heredoc format (preferred — avoids the
   JSON escaping nightmare for tools like Edit/Write that carry large code blobs).
2. Loose JSON, with a series of repair heuristics for the malformed JSON Notion
   tends to emit (Windows paths, trailing commas, unquoted keys).
"""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

from ..logging_utils import dbg

TOOL_USE_RE = re.compile(
    r"<tool_use\b[^>]*>\s*(.*?)\s*</tool_use>", re.DOTALL | re.IGNORECASE
)
TOOL_USE_OPEN_RE = re.compile(r"<tool_use\b[^>]*>", re.IGNORECASE)
TOOL_USE_CLOSE_RE = re.compile(r"</tool_use\s*>", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HEREDOC_OPEN_RE = re.compile(r"^<<<([A-Za-z_][A-Za-z0-9_]*)\s*$")
DELIMITED_KEY_RE = re.compile(r"^([A-Za-z_-][A-Za-z0-9_-]*)\s*:\s*(.*)$")
DELIMITED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_tool_response(text: str) -> str:
    text = unescape(text).replace("﻿", "")
    # Notion sometimes glues consecutive tags together:
    #     </tool_use<tool_use>
    # The closing tag regex needs the `>`; without it the depth tracker
    # never matches the close and returns zero parsed bodies.
    text = re.sub(
        r"</tool_use\s*(?=<tool_use)",
        "</tool_use>\n",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def contains_tool_use_markup(text: str) -> bool:
    """Detect a genuine attempt at a tool-call block.

    Only a complete, paired ``<tool_use>...</tool_use>`` block counts. Mentions
    of the token in prose or inside code spans (e.g. when Notion explains the
    tool-call format) are stripped first so they don't trigger a false reject.
    The reject path exists for blocks that look like real attempts but fail to
    parse — not for any stray mention of the word.
    """
    stripped = _normalize_tool_response(text)
    stripped = CODE_FENCE_RE.sub("", stripped)
    stripped = INLINE_CODE_RE.sub("", stripped)
    return bool(TOOL_USE_RE.search(stripped))


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
    for idx, candidate in enumerate(attempts):
        try:
            value = json.loads(candidate)
            dbg(f"_parse_loose_json_object: attempt {idx} SUCCESS")
            return value if isinstance(value, dict) else None

        except json.JSONDecodeError as e:
            dbg(f"_parse_loose_json_object: attempt {idx} FAILED")
            dbg(f"error: {e}")
            dbg(f"line={e.lineno} col={e.colno} pos={e.pos}")

            start = max(0, e.pos - 200)
            end = min(len(candidate), e.pos + 200)

            dbg("CONTEXT START")
            dbg(candidate[start:end])
            dbg("CONTEXT END")

    return None


def _parse_delimited_tool_call(raw: str) -> dict | None:
    """Parse the delimiter-based tool format.

    Expected shape (whitespace tolerant):

        ToolName
        field_one: value on one line
        field_two: <<<TAG
        ...multi-line text...
        TAG
        field_three: another inline value

    Returns {"name": ToolName, "input": {...}} or None if it doesn't match.
    """
    lines = raw.splitlines()
    # Skip leading blank lines.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return None
    name = lines[idx].strip()
    if not DELIMITED_NAME_RE.match(name):
        return None
    idx += 1

    input_obj: dict[str, Any] = {}
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        match = DELIMITED_KEY_RE.match(stripped)
        if not match:
            # Anything other than `field:` here means this isn't our format.
            return None
        field, after_colon = match.group(1), match.group(2)
        idx += 1

        heredoc_open = HEREDOC_OPEN_RE.match(after_colon.strip())
        if heredoc_open:
            tag = heredoc_open.group(1)
            collected: list[str] = []
            closed = False
            while idx < len(lines):
                current = lines[idx]
                if current.strip() == tag:
                    closed = True
                    idx += 1
                    break
                collected.append(current)
                idx += 1
            if not closed:
                return None
            input_obj[field] = "\n".join(collected)
            continue

        if after_colon.strip():
            # Inline scalar value on the same line as the key.
            input_obj[field] = after_colon
            continue

        # `field:` with empty value, possibly followed by a heredoc on the next
        # non-blank line. If the next non-blank line is a heredoc opener, use
        # that; otherwise treat the empty value as the empty string.
        peek = idx
        while peek < len(lines) and not lines[peek].strip():
            peek += 1
        if peek < len(lines):
            heredoc_open = HEREDOC_OPEN_RE.match(lines[peek].strip())
            if heredoc_open:
                tag = heredoc_open.group(1)
                idx = peek + 1
                collected = []
                closed = False
                while idx < len(lines):
                    current = lines[idx]
                    if current.strip() == tag:
                        closed = True
                        idx += 1
                        break
                    collected.append(current)
                    idx += 1
                if not closed:
                    return None
                input_obj[field] = "\n".join(collected)
                continue
        input_obj[field] = ""

    # A bare tool name with no fields is a valid call for zero-argument tools
    # (e.g. EnterPlanMode). We're already inside a <tool_use> block, so the name
    # was deliberate — emit it and let validation reject it if the tool requires
    # fields or doesn't exist.
    return {"name": name, "input": input_obj}


def _parse_tool_call(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json|xml|html|yaml|yml|text)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    dbg("TOOL RAW START")
    dbg(raw[:3000])
    dbg("TOOL RAW END")

    # Prefer the delimiter-based format because it avoids the JSON escaping
    # nightmare for tools like Edit/Write that contain large code blobs.
    delimited = _parse_delimited_tool_call(raw)
    if delimited:
        dbg("TOOL PARSED via delimited format")
        return delimited

    tool = _parse_loose_json_object(raw)

    dbg(f"TOOL PARSED via JSON = {tool is not None}")
    if tool and "name" in tool:
        return tool

    call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", raw, re.DOTALL)
    if not call:
        return None
    input_obj = _parse_loose_json_object(call.group(2))
    if input_obj is None:
        return None
    return {"name": call.group(1), "input": input_obj}


def _iter_tool_use_blocks(text: str) -> list[str]:
    """Yield the body of each *top-level* ``<tool_use>`` block.

    A naive non-greedy regex pairs the first ``<tool_use>`` with the first
    ``</tool_use>``, which breaks when a block carries a heredoc value that
    itself contains the delimiter format — for example a tool call that edits
    this very prompt template, whose body teaches the format with a literal
    ``<tool_use>Edit ... </tool_use>`` example. The flat scan would then surface
    that inner example as a second, standalone call and forward it for real.

    Tracking open/close depth folds any nested block into its outer body, so
    only genuine top-level calls are returned.
    """
    bodies: list[str] = []
    depth = 0
    body_start = 0
    pos = 0
    while pos < len(text):
        open_match = TOOL_USE_OPEN_RE.search(text, pos)
        close_match = TOOL_USE_CLOSE_RE.search(text, pos)
        if close_match is None:
            break
        if open_match is not None and open_match.start() < close_match.start():
            if depth == 0:
                body_start = open_match.end()
            depth += 1
            pos = open_match.end()
            continue
        if depth > 0:
            depth -= 1
            if depth == 0:
                bodies.append(text[body_start : close_match.start()])
        pos = close_match.end()
    return bodies


def parse_tool_uses(text: str) -> list[dict]:
    """Extract all tool calls from <tool_use>...</tool_use> blocks."""
    normalized = _normalize_tool_response(text)
    tools: list[dict] = []
    for body in _iter_tool_use_blocks(normalized):
        tool = _parse_tool_call(body.strip())
        if tool:
            tools.append(tool)
    return tools


def parse_tool_use(text: str) -> dict | None:
    tools = parse_tool_uses(text)
    return tools[0] if tools else None
