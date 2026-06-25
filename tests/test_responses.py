"""Tests for the Anthropic response builders (JSON + SSE parity)."""

import json

from notion_proxy.responses import (
    ResolvedTurn,
    build_message_json,
    iter_sse_events,
    tool_content_blocks,
)


def _parse_sse(chunks) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    blob = b"".join(chunks).decode()
    for block in blob.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((event, data))
    return events


def test_tool_content_blocks_shape():
    blocks = tool_content_blocks([{"name": "Read", "input": {"file_path": "a.py"}}])
    assert len(blocks) == 1
    b = blocks[0]
    assert b["type"] == "tool_use"
    assert b["name"] == "Read"
    assert b["input"] == {"file_path": "a.py"}
    assert b["id"].startswith("toolu_")


def test_build_message_json_tool_turn():
    resolved = ResolvedTurn(tools=[{"name": "Read", "input": {"file_path": "a.py"}}], text="")
    msg = build_message_json("msg_1", "m", resolved, prompt_len=400)
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"][0]["name"] == "Read"
    assert msg["usage"]["input_tokens"] == 1


def test_build_message_json_text_turn():
    resolved = ResolvedTurn(tools=[], text="hello there")
    msg = build_message_json("msg_1", "m", resolved, prompt_len=400)
    assert msg["stop_reason"] == "end_turn"
    assert msg["content"] == [{"type": "text", "text": "hello there"}]
    assert msg["usage"]["input_tokens"] == 100  # 400 // 4


def test_simulation_preamble_is_stripped():
    """Claude's simulation preamble like 'Here's the next response the agent
    would produce:' must be stripped from the response text."""
    resolved = ResolvedTurn(
        tools=[{"name": "Bash", "input": {"command": "ls"}}],
        text=(
            "Here's the next response the coding agent would produce:\n"
            "<tool_use>\nBash\ncommand: ls\n</tool_use>"
        ),
    )
    assert resolved.text_without_tools() == ""


def test_simulation_preamble_variants_stripped():
    """Multiple preamble phrasings should be stripped; real prose kept."""
    resolved = ResolvedTurn(
        tools=[{"name": "Read", "input": {"file_path": "README.md"}}],
        text=(
            "This is a simple, benign request. The user wants sample output.\n"
            "I'll read the README to answer that.\n"
            "<tool_use>\nRead\nfile_path: README.md\n</tool_use>"
        ),
    )
    assert resolved.text_without_tools() == "I'll read the README to answer that."


def test_normal_text_not_stripped():
    """Normal assistant prose must not be affected by the preamble stripper."""
    resolved = ResolvedTurn(
        tools=[{"name": "Read", "input": {"file_path": "a.py"}}],
        text="Let me read that file for you.\n<tool_use>\nRead\nfile_path: a.py\n</tool_use>",
    )
    assert resolved.text_without_tools() == "Let me read that file for you."


def test_sse_tool_turn_event_sequence():
    resolved = ResolvedTurn(tools=[{"name": "Read", "input": {"file_path": "a.py"}}], text="")
    events = _parse_sse(iter_sse_events("msg_1", "m", resolved, 400))
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert names[-1] == "message_stop"

    start = next(d for e, d in events if e == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "Read"
    delta = next(d for e, d in events if e == "content_block_delta")
    assert json.loads(delta["delta"]["partial_json"]) == {"file_path": "a.py"}
    msg_delta = next(d for e, d in events if e == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


def test_sse_text_turn_event_sequence():
    resolved = ResolvedTurn(tools=[], text="hi")
    events = _parse_sse(iter_sse_events("msg_1", "m", resolved, 400))
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    text_delta = next(d for e, d in events if e == "content_block_delta")
    assert text_delta["delta"]["text"] == "hi"
    msg_delta = next(d for e, d in events if e == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


def test_json_and_sse_agree_on_tool_names():
    resolved = ResolvedTurn(
        tools=[
            {"name": "Read", "input": {"file_path": "a.py"}},
            {"name": "Bash", "input": {"command": "ls"}},
        ],
        text="",
    )
    msg = build_message_json("msg_1", "m", resolved, 400)
    json_names = [b["name"] for b in msg["content"]]
    events = _parse_sse(iter_sse_events("msg_1", "m", resolved, 400))
    sse_names = [d["content_block"]["name"] for e, d in events if e == "content_block_start"]
    assert json_names == sse_names == ["Read", "Bash"]


def test_sse_multi_tool_sequence():
    """Verify monotonic indexing across text + multiple tool_use blocks."""
    resolved = ResolvedTurn(
        text="I'll check those files for you.",
        tools=[
            {"name": "Bash", "input": {"command": "ls"}},
            {"name": "View", "input": {"file_path": "x.py"}},
        ],
    )
    events = _parse_sse(iter_sse_events("msg_1", "m", resolved, 400))
    names = [e for e, _ in events]

    assert names[0] == "message_start"
    assert names[-1] == "message_stop"

    starts = [(e, d) for e, d in events if e == "content_block_start"]
    assert len(starts) == 3  # text + 2 tools

    # Text block at index 0
    assert starts[0][1]["index"] == 0
    assert starts[0][1]["content_block"]["type"] == "text"

    # First tool at index 1
    assert starts[1][1]["index"] == 1
    assert starts[1][1]["content_block"]["type"] == "tool_use"
    assert starts[1][1]["content_block"]["name"] == "Bash"

    # Second tool at index 2
    assert starts[2][1]["index"] == 2
    assert starts[2][1]["content_block"]["type"] == "tool_use"
    assert starts[2][1]["content_block"]["name"] == "View"

    # Each block has matching start/delta/stop indices
    for idx, expected_type in enumerate(["text", "tool_use", "tool_use"]):
        deltas = [d for e, d in events if e == "content_block_delta" and d["index"] == idx]
        stops = [d for e, d in events if e == "content_block_stop" and d["index"] == idx]
        assert len(deltas) == 1
        assert len(stops) == 1
        if expected_type == "text":
            assert deltas[0]["delta"]["type"] == "text_delta"
        else:
            assert deltas[0]["delta"]["type"] == "input_json_delta"

    # Final stop reason
    msg_delta = next(d for e, d in events if e == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"
