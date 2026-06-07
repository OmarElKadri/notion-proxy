"""Tests for prompt-body rendering and thread-persistence config helpers."""

import re

from notion_proxy.config import (
    persist_notion_threads,
    suppress_notion_thread_persistence,
)
from notion_proxy.prompt import render_body


def test_prompt_and_history_and_now_substitution():
    template = {"text": "{{PROMPT}}", "hist": "{{HISTORY}}", "when": "{{NOW}}"}
    out = render_body(template, "hello", "the history")
    assert out["text"] == "hello"
    assert out["hist"] == "the history"
    assert out["when"].endswith("Z")
    assert "T" in out["when"]  # ISO-8601 timestamp


def test_named_uuid_is_stable_within_one_render():
    template = {"a": "{{UUID:trace}}", "b": "{{UUID:trace}}", "c": "{{UUID:other}}"}
    out = render_body(template, "p", "")
    assert out["a"] == out["b"]
    assert out["a"] != out["c"]


def test_anonymous_uuid_is_unique():
    template = {"a": "{{UUID}}", "b": "{{UUID}}"}
    out = render_body(template, "p", "")
    assert out["a"] != out["b"]


def test_nuuid_uses_thread_bucket_and_prefix():
    template = {"id": "{{NUUID:thread}}"}
    out = render_body(template, "p", "", notion_id_prefix="aaaa-bbbb")
    assert out["id"].startswith("aaaa-bbbb-80")
    assert "-00a9" in out["id"]
    # Stable within one render.
    out2 = render_body({"x": "{{NUUID:thread}}", "y": "{{NUUID:thread}}"}, "p", "")
    assert out2["x"] == out2["y"]


def test_nuuid_default_bucket():
    out = render_body({"id": "{{NUUID:userStep}}"}, "p", "", notion_id_prefix="pre-fix")
    assert "-00aa" in out["id"]


def test_persist_threads_env_override(monkeypatch):
    monkeypatch.setenv("NOTION_PROXY_PERSIST_THREADS", "false")
    assert persist_notion_threads({"persist_threads": True}) is False
    monkeypatch.setenv("NOTION_PROXY_PERSIST_THREADS", "1")
    assert persist_notion_threads({"persist_threads": False}) is True


def test_persist_threads_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("NOTION_PROXY_PERSIST_THREADS", raising=False)
    assert persist_notion_threads({"persist_threads": False}) is False
    assert persist_notion_threads({}) is True  # defaults to True


def test_suppress_thread_persistence_flips_flags():
    body = {
        "createThread": True,
        "generateTitle": True,
        "saveAllThreadOperations": True,
        "setUnreadState": True,
        "unrelated": True,
    }
    suppress_notion_thread_persistence(body)
    assert body["createThread"] is False
    assert body["generateTitle"] is False
    assert body["saveAllThreadOperations"] is False
    assert body["setUnreadState"] is False
    assert body["unrelated"] is True  # untouched
