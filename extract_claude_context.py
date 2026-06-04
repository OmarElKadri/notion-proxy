#!/usr/bin/env python3
"""
extract_claude_context.py

Given a Claude Code API request payload (JSON string or file),
extracts and prints all context variables:
  {{MCP_SERVERS}}, {{SKILLS}}, {{GLOBAL_CLAUDE_MD}}, {{IMPORTED_CLAUDE_FILES}},
  {{CURRENT_DATE}}, {{WORKSPACE_PATH}}, {{IS_GIT_REPO}}, {{PLATFORM}},
  {{SHELL}}, {{OS_VERSION}}, {{MODEL_NAME}}, {{KNOWLEDGE_CUTOFF}},
  {{MEMORY_PATH}}, {{CURRENT_BRANCH}}, {{MAIN_BRANCH}}, {{GIT_USER}},
  {{GIT_STATUS}}, {{RECENT_COMMITS}}

Usage:
    python extract_claude_context.py payload.json
    python extract_claude_context.py --stdin < payload.json
    python extract_claude_context.py --json '{"model": ...}'
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# ── helpers ──────────────────────────────────────────────────────────────────


def _collect_text(content: Any) -> str:
    """Flatten all text blocks from a messages-style content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _all_texts(payload: dict) -> list[str]:
    """Return every text string visible in the payload (system + messages)."""
    texts: list[str] = []

    # system prompt (string or list of blocks)
    system = payload.get("system")
    if system:
        texts.append(_collect_text(system))

    # messages
    for msg in payload.get("messages", []):
        texts.append(_collect_text(msg.get("content", "")))

    return texts


# ── extraction functions ──────────────────────────────────────────────────────


def extract_mcp_servers(payload: dict) -> list[str]:
    """Names of MCP servers that appear in the tool list or system-reminder."""
    servers: set[str] = set()

    # from tools list
    for tool in payload.get("tools", []):
        name = tool.get("name", "")
        if name.startswith("mcp__"):
            parts = name.split("__")
            if len(parts) >= 2:
                servers.add(parts[1])

    # from system reminders mentioning MCP server instructions
    for text in _all_texts(payload):
        for m in re.finditer(r"##\s+plugin:(\S+)", text):
            servers.add(m.group(1))
        # also pick up "## <name>\nThe official … MCP server"
        for m in re.finditer(r"MCP [Ss]erver.*?\n.*?##\s+(\S+)", text):
            servers.add(m.group(1))

    return sorted(servers)


def extract_skills(payload: dict) -> list[str]:
    """Skill names listed in system-reminder available-skills blocks."""
    skills: list[str] = []
    seen: set[str] = set()

    for text in _all_texts(payload):
        # look for the skills section header (ends at next blank line or end)
        skills_block_match = re.search(
            r"The following skills are available[^\n]*\n(.*?)(?=\n\n|\Z)",
            text,
            re.DOTALL,
        )
        if skills_block_match:
            block = skills_block_match.group(1)
            # match "- skill-name:" (with description) or "- skill-name\n" (bare)
            for m in re.finditer(r"^- ([\w:/-]+)(?::|$)", block, re.MULTILINE):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    skills.append(name)

    return skills


def extract_global_claude_md(payload: dict) -> str:
    """Contents of the global CLAUDE.md (everything between the header and next section)."""
    for text in _all_texts(payload):
        m = re.search(
            r"Contents of [^\n]*CLAUDE\.md[^\n]*\n(.*?)(?=\nContents of |\Z)",
            text,
            re.DOTALL,
        )
        if m:
            return m.group(1).strip()
    return ""


def extract_imported_claude_files(payload: dict) -> dict[str, str]:
    """All @-imported or separately listed CLAUDE/instruction files and their contents."""
    files: dict[str, str] = {}
    for text in _all_texts(payload):
        for m in re.finditer(
            r"Contents of ([^\n]+)\s*\([^)]*\):\n(.*?)(?=\nContents of |\Z)",
            text,
            re.DOTALL,
        ):
            path = m.group(1).strip()
            content = m.group(2).strip()
            files[path] = content
    return files


def extract_current_date(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Today'?s date is\s+([^\.\n]+)", text)
        if m:
            return m.group(1).strip().rstrip(".")
    return ""


def extract_workspace_path(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Primary working directory:\s*(.+)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_is_git_repo(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Is a git repository:\s*(\S+)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_platform(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Platform:\s*(\S+)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_shell(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Shell:\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_os_version(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"OS Version:\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_model_name(payload: dict) -> str:
    # top-level model field first
    if "model" in payload:
        return payload["model"]
    for text in _all_texts(payload):
        m = re.search(r"You are powered by the model named\s+([^\.\n]+)", text)
        if m:
            return m.group(1).strip().rstrip(".")
        m = re.search(r"The exact model ID is\s+([^\.\n]+)", text)
        if m:
            return m.group(1).strip().rstrip(".")
    return ""


def extract_knowledge_cutoff(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Assistant knowledge cutoff is\s+([^\.\n]+)", text)
        if m:
            return m.group(1).strip().rstrip(".")
    return ""


def extract_memory_path(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(
            r"persistent.*?memory system at\s+[`'\"]?([^`'\"\n]+)[`'\"]?", text
        )
        if m:
            return m.group(1).strip().rstrip(".")
    return ""


def extract_current_branch(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Current branch:\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_main_branch(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Main branch[^:]*:\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_git_user(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Git user:\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    return ""


def extract_git_status(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Status:\n(.*?)(?:\n\nRecent commits:|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def extract_recent_commits(payload: dict) -> str:
    for text in _all_texts(payload):
        m = re.search(r"Recent commits:\n(.*?)(?:\n\n|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


# ── main ──────────────────────────────────────────────────────────────────────


@dataclass
class ClaudeContext:
    mcp_servers: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    global_claude_md: str = ""
    imported_claude_files: dict[str, str] = field(default_factory=dict)
    current_date: str = ""
    workspace_path: str = ""
    is_git_repo: str = ""
    platform: str = ""
    shell: str = ""
    os_version: str = ""
    model_name: str = ""
    knowledge_cutoff: str = ""
    memory_path: str = ""
    current_branch: str = ""
    main_branch: str = ""
    git_user: str = ""
    git_status: str = ""
    recent_commits: str = ""


def extract(payload: dict) -> ClaudeContext:
    return ClaudeContext(
        mcp_servers=extract_mcp_servers(payload),
        skills=extract_skills(payload),
        global_claude_md=extract_global_claude_md(payload),
        imported_claude_files=extract_imported_claude_files(payload),
        current_date=extract_current_date(payload),
        workspace_path=extract_workspace_path(payload),
        is_git_repo=extract_is_git_repo(payload),
        platform=extract_platform(payload),
        shell=extract_shell(payload),
        os_version=extract_os_version(payload),
        model_name=extract_model_name(payload),
        knowledge_cutoff=extract_knowledge_cutoff(payload),
        memory_path=extract_memory_path(payload),
        current_branch=extract_current_branch(payload),
        main_branch=extract_main_branch(payload),
        git_user=extract_git_user(payload),
        git_status=extract_git_status(payload),
        recent_commits=extract_recent_commits(payload),
    )


def _section(title: str, value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, list):
        if not value:
            return f"{pad}{{{{ {title} }}}}: (none)\n"
        items = "\n".join(f"{pad}  - {v}" for v in value)
        return f"{pad}{{{{ {title} }}}}:\n{items}\n"
    if isinstance(value, dict):
        if not value:
            return f"{pad}{{{{ {title} }}}}: (none)\n"
        parts = [f"{pad}{{{{ {title} }}}}:"]
        for k, v in value.items():
            parts.append(f"{pad}  [{k}]")
            # show first 200 chars of each file
            snippet = v[:200].replace("\n", "\n" + pad + "    ")
            parts.append(f"{pad}    {snippet}{'…' if len(v) > 200 else ''}")
        return "\n".join(parts) + "\n"
    val = str(value).strip()
    if not val:
        return f"{pad}{{{{ {title} }}}}: (not found)\n"
    if "\n" in val:
        indented = val.replace("\n", "\n" + pad + "  ")
        return f"{pad}{{{{ {title} }}}}:\n{pad}  {indented}\n"
    return f"{pad}{{{{ {title} }}}}: {val}\n"


def print_context(ctx: ClaudeContext) -> None:
    print("=" * 60)
    print("  Claude Code Context Variables")
    print("=" * 60)
    print(_section("MCP_SERVERS", ctx.mcp_servers))
    print(_section("SKILLS", ctx.skills))
    print(_section("GLOBAL_CLAUDE_MD", ctx.global_claude_md))
    print(_section("IMPORTED_CLAUDE_FILES", ctx.imported_claude_files))
    print(_section("CURRENT_DATE", ctx.current_date))
    print(_section("WORKSPACE_PATH", ctx.workspace_path))
    print(_section("IS_GIT_REPO", ctx.is_git_repo))
    print(_section("PLATFORM", ctx.platform))
    print(_section("SHELL", ctx.shell))
    print(_section("OS_VERSION", ctx.os_version))
    print(_section("MODEL_NAME", ctx.model_name))
    print(_section("KNOWLEDGE_CUTOFF", ctx.knowledge_cutoff))
    print(_section("MEMORY_PATH", ctx.memory_path))
    print(_section("CURRENT_BRANCH", ctx.current_branch))
    print(_section("MAIN_BRANCH", ctx.main_branch))
    print(_section("GIT_USER", ctx.git_user))
    print(_section("GIT_STATUS", ctx.git_status))
    print(_section("RECENT_COMMITS", ctx.recent_commits))
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Claude Code context variables from an API payload."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("file", nargs="?", help="Path to JSON payload file")
    source.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    source.add_argument("--json", metavar="JSON", help="Inline JSON string")
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    args = parser.parse_args()

    # load payload
    if args.json:
        raw = args.json
    elif args.stdin or not args.file:
        raw = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8") as fh:
            raw = fh.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON — {e}", file=sys.stderr)
        sys.exit(1)

    ctx = extract(payload)

    if args.output == "json":
        import dataclasses

        print(json.dumps(dataclasses.asdict(ctx), indent=2, ensure_ascii=False))
    else:
        print_context(ctx)


if __name__ == "__main__":
    main()
