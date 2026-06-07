"""Tests for the tool-error nudge that breaks repeat-failure loops."""

from notion_proxy.claude_io import build_error_nudge, recent_tool_errors


def _assistant_call(tool_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _tool_result(tool_id: str, content, is_error: bool) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
                "is_error": is_error,
            }
        ],
    }


def test_recent_tool_errors_pairs_name_and_message():
    payload = {
        "messages": [
            _assistant_call("t1", "Edit"),
            _tool_result("t1", "File does not exist.", True),
        ]
    }
    errors = recent_tool_errors(payload)
    assert errors == [{"name": "Edit", "error": "File does not exist."}]


def test_no_errors_when_last_result_succeeded():
    payload = {
        "messages": [
            _assistant_call("t1", "Read"),
            _tool_result("t1", "ok", False),
        ]
    }
    assert recent_tool_errors(payload) == []


def test_plain_text_user_turn_is_not_a_tool_round_trip():
    # A fresh human query after an old failure must NOT re-trigger the nudge.
    payload = {
        "messages": [
            _assistant_call("t1", "Edit"),
            _tool_result("t1", "File does not exist.", True),
            {"role": "assistant", "content": [{"type": "text", "text": "sorry"}]},
            {"role": "user", "content": "try again please"},
        ]
    }
    assert recent_tool_errors(payload) == []


def test_only_latest_turn_errors_surface():
    # An old error followed by a newer successful tool turn -> nothing to nudge.
    payload = {
        "messages": [
            _assistant_call("t1", "Edit"),
            _tool_result("t1", "boom", True),
            _assistant_call("t2", "Read"),
            _tool_result("t2", "fine", False),
        ]
    }
    assert recent_tool_errors(payload) == []


def test_build_error_nudge_is_empty_without_errors():
    assert build_error_nudge([]) == ""


def test_build_error_nudge_mentions_tool_and_cause():
    nudge = build_error_nudge([{"name": "Edit", "error": "File does not exist."}])
    assert "Edit failed" in nudge
    assert "File does not exist." in nudge
    assert "Do NOT repeat" in nudge
