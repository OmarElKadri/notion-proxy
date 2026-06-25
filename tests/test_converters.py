"""Tests for Anthropic-to-unified message conversion and transcript building."""

import json

from notion_proxy.converters import (
    UnifiedMessage,
    UnifiedTool,
    build_conversation_transcript,
    build_tools_summary,
    convert_anthropic_messages,
    convert_anthropic_tools,
    extract_system_prompt,
    extract_text_content,
    extract_tool_results_from_anthropic_content,
    extract_tool_uses_from_anthropic_content,
)


def _assistant_msg(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["content"] = tool_calls
    return msg


def _user_msg(content, tool_results=None):
    msg = {"role": "user", "content": content}
    if tool_results is not None:
        msg["content"] = tool_results
    return msg


def test_extract_text_content_string():
    assert extract_text_content("hello") == "hello"


def test_extract_text_content_blocks():
    assert extract_text_content([{"type": "text", "text": "hi"}]) == "hi"
    assert extract_text_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_extract_text_content_ignores_non_text():
    assert extract_text_content([{"type": "tool_use", "name": "Bash"}]) == ""


def test_extract_system_prompt_string():
    assert extract_system_prompt("be helpful") == "be helpful"


def test_extract_system_prompt_blocks():
    assert extract_system_prompt([{"type": "text", "text": "sys1"}, {"type": "text", "text": "sys2"}]) == "sys1\nsys2"


def test_extract_system_prompt_none():
    assert extract_system_prompt(None) == ""


def test_extract_tool_uses_from_content():
    content = [
        {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
        {"type": "text", "text": "done"},
    ]
    tools = extract_tool_uses_from_anthropic_content(content)
    assert len(tools) == 1
    assert tools[0]["id"] == "tu_1"
    assert tools[0]["function"]["name"] == "Bash"
    assert tools[0]["function"]["arguments"] == {"command": "ls"}


def test_extract_tool_uses_from_content_ignores_string_input():
    content = [
        {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": '{"command":"ls"}'},
    ]
    tools = extract_tool_uses_from_anthropic_content(content)
    assert tools[0]["function"]["arguments"] == {"command": "ls"}


def test_extract_tool_results_from_content():
    content = [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"},
        {"type": "text", "text": "ok"},
    ]
    results = extract_tool_results_from_anthropic_content(content)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "tu_1"
    assert results[0]["content"] == "file.txt"


def test_extract_tool_results_with_list_content():
    content = [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "ok"}]},
    ]
    results = extract_tool_results_from_anthropic_content(content)
    assert results[0]["content"] == "ok"


def test_convert_anthropic_messages_basic():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    unified = convert_anthropic_messages(messages)
    assert len(unified) == 2
    assert unified[0] == UnifiedMessage(role="user", content="hello")
    assert unified[1] == UnifiedMessage(role="assistant", content="hi there")


def test_convert_anthropic_messages_with_tool_use():
    messages = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
            ],
        },
    ]
    unified = convert_anthropic_messages(messages)
    assert len(unified) == 2
    assert unified[1].tool_calls[0]["function"]["name"] == "Bash"


def test_convert_anthropic_messages_with_tool_result():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"},
            ],
        },
    ]
    unified = convert_anthropic_messages(messages)
    assert unified[0].tool_results[0]["tool_use_id"] == "tu_1"


def test_convert_anthropic_tools():
    tools = [
        {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}},
    ]
    unified = convert_anthropic_tools(tools)
    assert unified == [UnifiedTool(name="Read", description="Read a file", input_schema={"type": "object"})]


def test_build_tools_summary():
    tools = [
        UnifiedTool(name="Bash", description="Run shell", input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}),
    ]
    summary = build_tools_summary(tools)
    assert "Bash" in summary
    assert "command" in summary
    assert "required" in summary


def test_build_tools_summary_compact_one_line_per_tool():
    """Each tool is a single line with args + required, no multi-paragraph prose."""
    long_desc = "This is a tool. " * 50  # well over the 140 char cap
    tools = [
        UnifiedTool(name="Bash", description=long_desc, input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}),
        UnifiedTool(name="Read", description=None, input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}),
    ]
    summary = build_tools_summary(tools)
    lines = [l for l in summary.splitlines() if l.strip()]
    assert len(lines) == 2  # one line per tool, no extra indented lines
    bash_line = lines[0]
    assert "Bash" in bash_line
    assert "args [command]" in bash_line
    assert "required [command]" in bash_line
    # Description is truncated to a single short line.
    assert len(bash_line) < 250
    assert "…" in bash_line
    read_line = lines[1]
    assert "Read" in read_line
    assert "args [file_path]" in read_line
    assert "required [file_path]" in read_line


def test_build_tools_summary_empty():
    assert build_tools_summary([]) == "(no tools available)"


def test_build_conversation_transcript_basic():
    messages = [
        UnifiedMessage(role="user", content="hello"),
        UnifiedMessage(role="assistant", content="hi"),
    ]
    transcript = build_conversation_transcript(messages)
    assert "User: hello" in transcript
    assert "Assistant: hi" in transcript


def test_build_conversation_transcript_with_tool_use():
    messages = [
        UnifiedMessage(role="user", content="list files"),
        UnifiedMessage(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "tu_1", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}},
            ],
        ),
    ]
    transcript = build_conversation_transcript(messages)
    assert "list files" in transcript
    assert "<tool_use>" in transcript
    assert "Bash" in transcript
    assert "command: ls" in transcript
    assert "</tool_use>" in transcript


def test_build_conversation_transcript_with_tool_result():
    messages = [
        UnifiedMessage(role="assistant", content="", tool_calls=[
            {"id": "tu_1", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}},
        ]),
        UnifiedMessage(role="user", content="", tool_results=[
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"},
        ]),
    ]
    transcript = build_conversation_transcript(messages)
    assert "tool result tu_1" in transcript
    assert "file.txt" in transcript


def test_round_trip_conversation_with_tools():
    """A full turn with user query, assistant tool_use, and user tool_result."""
    payload = {
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "file.txt"},
                ],
            },
        ],
        "tools": [
            {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ],
    }
    messages = convert_anthropic_messages(payload["messages"])
    transcript = build_conversation_transcript(messages)
    assert "User: List files" in transcript
    assert "Assistant:" in transcript
    assert "Bash" in transcript
    assert "command: ls" in transcript
    assert "tool result toolu_01" in transcript
    assert "file.txt" in transcript
