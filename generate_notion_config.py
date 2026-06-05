"""
Convert a Notion "workflow" curl command (saved as a .txt file) into a
reusable JSON config file with a parameterized body_template.

What it does
------------
1. Parses the curl: URL, method, headers (-H), cookies (-b/--cookie),
   and request body (--data-raw / --data / --data-binary / -d).
2. Cleans the headers (drops noisy tracing headers such as baggage,
   sentry-trace, referer, origin, version-pinned headers, etc.). The cookie
   string (from -b) is folded into headers as `cookie`.
3. Rewrites the JSON body into a body_template where the volatile,
   per-request values become mustache-style placeholders:
       traceId          -> UUID:trace
       threadId         -> NUUID:thread
       transcript ids   -> NUUID:configStep / NUUID:contextStep / NUUID:userStep
       currentDatetime  -> NOW
       createdAt        -> NOW
       user prompt text -> PROMPT
   (each wrapped in double curly braces in the output)
   It also collapses the transcript to one config + one context + one user
   step, drops `updated-config` steps, and flips a few flags for a fresh run
   (createThread=true, generateTitle=true, isPartialTranscript=false).

Usage
-----
    python3 generate_notion_config.py input_curl.txt -o notion_config.json
    python3 generate_notion_config.py input_curl.txt            # prints to stdout

Notes
-----
* The cookie carries your live Notion session tokens. The output config
  contains them verbatim -- treat config.json as a secret and don't commit it.
* This script only reads/writes local files. It never sends the request.
"""

import argparse
import json
import re
import shlex
import sys

# Build the "double curly brace" delimiters at runtime so the literal token
# never has to be typed (keeps editors/templating layers from touching it).
LB = chr(123) * 2  # opening braces
RB = chr(125) * 2  # closing braces


def ph(token):
    """Wrap a token in mustache-style double braces, e.g. ph('NOW') -> NOW."""
    return LB + token + RB


# Volatile values -> placeholder token name.
P_TRACE = ph("UUID:trace")
P_THREAD = ph("NUUID:thread")
P_CONFIG_ID = ph("NUUID:configStep")
P_CONTEXT_ID = ph("NUUID:contextStep")
P_USER_ID = ph("NUUID:userStep")
P_NOW = ph("NOW")
P_PROMPT = ph("PROMPT")


# Headers we don't want to carry into the reusable config.
DROP_HEADERS = {
    "baggage",
    "sentry-trace",
    "referer",
    "origin",
    "notion-client-version",  # version-pinned; let the client default it
    "content-length",
    "host",
    "accept-encoding",
}


def read_curl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def normalize_curl(raw):
    """Join shell line-continuations so shlex can tokenize the whole command."""
    raw = raw.lstrip("\ufeff")  # strip BOM if present
    raw = re.sub(r"\\\r?\n", " ", raw)  # remove backslash-newline continuations
    raw = raw.replace("\r", " ")
    return raw.strip()


def parse_curl(raw):
    """Return dict with url, method, headers(dict), cookie(str|None), data(str|None)."""
    text = normalize_curl(raw)
    tokens = shlex.split(text, posix=True)

    if tokens and tokens[0] == "curl":
        tokens = tokens[1:]

    url = None
    method = None
    headers = {}
    cookie = None
    data = None

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        def take_value(inline_prefix=None):
            """Value for a flag, supporting `--flag=value` and `--flag value`."""
            nonlocal i
            if inline_prefix is not None and tok.startswith(inline_prefix):
                return tok[len(inline_prefix) :]
            i += 1
            return tokens[i] if i < n else ""

        if tok in ("-H", "--header") or tok.startswith("--header="):
            val = (
                take_value("--header=") if tok.startswith("--header=") else take_value()
            )
            if ":" in val:
                k, v = val.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        elif tok in ("-b", "--cookie") or tok.startswith("--cookie="):
            cookie = (
                take_value("--cookie=") if tok.startswith("--cookie=") else take_value()
            )
        elif tok in (
            "--data-raw",
            "--data",
            "--data-binary",
            "--data-ascii",
            "-d",
        ) or any(
            tok.startswith(p + "=")
            for p in ("--data-raw", "--data", "--data-binary", "--data-ascii")
        ):
            matched = False
            for p in ("--data-raw=", "--data-binary=", "--data-ascii=", "--data="):
                if tok.startswith(p):
                    data = tok[len(p) :]
                    matched = True
                    break
            if not matched:
                i += 1
                data = tokens[i] if i < n else ""
        elif tok in ("-X", "--request") or tok.startswith("--request="):
            method = (
                take_value("--request=")
                if tok.startswith("--request=")
                else take_value()
            )
        elif tok in ("--compressed", "--location", "-L", "-s", "--silent", "-i", "-v"):
            pass  # value-less flags we don't care about
        elif not tok.startswith("-") and url is None:
            url = tok
        # Unknown flags: ignore (best effort).
        i += 1

    if method is None:
        method = "POST" if data is not None else "GET"

    return {
        "url": url,
        "method": method.upper(),
        "headers": headers,
        "cookie": cookie,
        "data": data,
    }


def clean_headers(headers, cookie):
    out = {k: v for k, v in headers.items() if k not in DROP_HEADERS}
    if cookie:
        out["cookie"] = cookie
    return out


def notion_id_prefix(body):
    """Derive the shared prefix (first two dash-segments) from a notion id."""
    for step in body.get("transcript", []) or []:
        if isinstance(step, dict) and isinstance(step.get("id"), str):
            parts = step["id"].split("-")
            if len(parts) >= 2:
                return "-".join(parts[:2])
    return ""


def build_body_template(body):
    """Rewrite the parsed request body into a parameterized template."""
    transcript = body.get("transcript", []) or []

    config_step = context_step = user_step = None
    for step in transcript:
        if not isinstance(step, dict):
            continue
        t = step.get("type")
        if t == "config" and config_step is None:
            config_step = step
        elif t == "context" and context_step is None:
            context_step = step
        elif t == "user" and user_step is None:
            user_step = step

    new_transcript = []

    if config_step is not None:
        cs = json.loads(json.dumps(config_step))  # deep copy
        cs["id"] = P_CONFIG_ID
        new_transcript.append(cs)

    if context_step is not None:
        xs = json.loads(json.dumps(context_step))
        xs["id"] = P_CONTEXT_ID
        val = xs.get("value")
        if isinstance(val, dict) and "currentDatetime" in val:
            val["currentDatetime"] = P_NOW
        new_transcript.append(xs)

    if user_step is not None:
        us = json.loads(json.dumps(user_step))
        us["id"] = P_USER_ID
        us["value"] = [[P_PROMPT]]
        if "createdAt" in us:
            us["createdAt"] = P_NOW
        new_transcript.append(us)

    tmpl = json.loads(json.dumps(body))  # deep copy of whole body
    tmpl["transcript"] = new_transcript

    if "traceId" in tmpl:
        tmpl["traceId"] = P_TRACE
    if "threadId" in tmpl:
        tmpl["threadId"] = P_THREAD

    space_id = tmpl.get("spaceId")
    if space_id:
        tmpl["threadParentPointer"] = {
            "table": "space",
            "id": space_id,
            "spaceId": space_id,
        }

    # Flags for a fresh, full run.
    tmpl["createThread"] = True
    tmpl["generateTitle"] = True
    tmpl["isPartialTranscript"] = False

    return tmpl


def build_config(parsed):
    body = json.loads(parsed["data"]) if parsed.get("data") else {}
    return {
        "url": parsed["url"],
        "method": parsed["method"],
        "persist_threads": True,
        "headers": clean_headers(parsed["headers"], parsed["cookie"]),
        "notion_id_prefix": notion_id_prefix(body),
        "strip_system": True,
        "persona": "",
        "body_template": build_body_template(body),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a Notion workflow curl .txt into a JSON config."
    )
    ap.add_argument("input", help="Path to the .txt file containing the curl command.")
    ap.add_argument(
        "-o", "--output", help="Output JSON path. If omitted, prints to stdout."
    )
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default 2).")
    args = ap.parse_args(argv)

    raw = read_curl(args.input)
    parsed = parse_curl(raw)
    if not parsed["url"]:
        sys.exit("Error: could not find a URL in the curl command.")
    if parsed["data"]:
        try:
            json.loads(parsed["data"])
        except json.JSONDecodeError as e:
            sys.exit(f"Error: request body is not valid JSON: {e}")

    config = build_config(parsed)
    text = json.dumps(config, indent=args.indent, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
