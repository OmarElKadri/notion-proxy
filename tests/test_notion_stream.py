"""Tests for the Notion NDJSON patch-stream parser."""

from notion_proxy.notion.stream import NotionPatchStream


def _agent_step(*parts: dict) -> dict:
    return {"type": "agent-inference", "value": list(parts)}


def test_patch_start_emits_initial_text():
    parser = NotionPatchStream()
    obj = {"type": "patch-start", "data": {"s": [_agent_step({"type": "text", "content": "Hello"})]}}
    assert list(parser.feed(obj)) == ["Hello"]


def test_append_content_via_x_op():
    parser = NotionPatchStream()
    list(parser.feed({"type": "patch-start", "data": {"s": [_agent_step({"type": "text", "content": ""})]}}))
    patch = {"type": "patch", "v": [{"o": "x", "p": "/s/0/value/0/content", "v": " world"}]}
    assert list(parser.feed(patch)) == [" world"]


def test_append_new_value_part_via_a_op():
    parser = NotionPatchStream()
    list(parser.feed({"type": "patch-start", "data": {"s": [_agent_step()]}}))
    patch = {
        "type": "patch",
        "v": [{"o": "a", "p": "/s/0/value/-", "v": {"type": "text", "content": "X"}}],
    }
    assert list(parser.feed(patch)) == ["X"]


def test_thinking_is_gated_by_include_thinking():
    step = _agent_step({"type": "thinking", "content": "secret"})
    obj = {"type": "patch-start", "data": {"s": [step]}}

    assert list(NotionPatchStream(include_thinking=False).feed(obj)) == []
    assert list(NotionPatchStream(include_thinking=True).feed(obj)) == ["secret"]


def test_patch_sync_resets_without_emitting():
    parser = NotionPatchStream()
    obj = {"type": "patch-sync", "data": {"s": [_agent_step({"type": "text", "content": "ignored"})]}}
    assert list(parser.feed(obj)) == []
    assert parser.steps  # steps were replaced


def test_non_agent_step_emits_nothing():
    parser = NotionPatchStream()
    obj = {"type": "patch-start", "data": {"s": [{"type": "config", "value": []}]}}
    assert list(parser.feed(obj)) == []
