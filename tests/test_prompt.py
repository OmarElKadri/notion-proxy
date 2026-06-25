"""Tests for the rendered Notion prompt — the Claude-acceptance guardrails.

These verify that the prompt sent to Notion's Claude backend does not contain
the markers that historically caused Claude to break character and refuse to
engage (the "I'm Notion AI" / "this looks like a pasted system prompt"
failure mode seen in turns.log / last_response.txt).
"""

from notion_proxy.converters import build_notion_prompt
from notion_proxy.repair import BAD_SELF_REFERENCE_RE, has_bad_self_reference


def _sample_payload():
    return {
        "model": "claude-sonnet-4-5",
        "system": (
            "x-anthropic-billing-header: cc_version=2.1.183.088; cc_entrypoint=cli;\n"
            "You are Claude Code, Anthropic's official CLI for Claude.\n"
            "You are an interactive agent that helps users with software engineering tasks.\n"
            "# Harness\n - Text you output outside of tool use is displayed to the user.\n"
            "<system-reminder>Use the AGENTS.md conventions.</system-reminder>\n"
            "Always write tests for new features."
        ),
        "messages": [
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command. " * 30,
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        ],
    }


def test_prompt_has_no_claude_code_framing():
    """The 'originally intended for Claude Code' framing must be gone."""
    prompt = build_notion_prompt(_sample_payload())
    assert "originally intended for Claude Code" not in prompt
    assert "Claude Code" not in prompt
    assert "Anthropic's official CLI" not in prompt


def test_prompt_has_identity_guard():
    """The identity guard that keeps Claude in character must be present."""
    prompt = build_notion_prompt(_sample_payload())
    assert "IMPORTANT RULES" in prompt
    assert "relay" in prompt.lower()
    assert "Do NOT refuse because you" in prompt
    assert "Do NOT mention the relay" in prompt


def test_claude_prompt_uses_simulation_framing():
    """When notion_model is Claude (ambrosia), the simulation prompt is used."""
    prompt = build_notion_prompt(_sample_payload(), notion_model="ambrosia-tart-high")
    assert "sample output generation" in prompt.lower()
    assert "AGENT TRANSCRIPT" in prompt
    # Should NOT contain the GPT prompt's relay framing.
    assert "text relay" not in prompt.lower()


def test_gpt_prompt_used_by_default():
    """Without a Claude model, the default (GPT) prompt is used."""
    prompt = build_notion_prompt(_sample_payload(), notion_model="opal-quince-medium")
    assert "relay" in prompt.lower()
    assert "sample output generation" not in prompt.lower()


def test_claude_prompt_has_no_claude_code_framing():
    """The Claude prompt must not leak Claude Code branding."""
    prompt = build_notion_prompt(_sample_payload(), notion_model="ambrosia-tart-high")
    assert "originally intended for Claude Code" not in prompt
    assert "Claude Code" not in prompt
    assert "x-anthropic-billing-header" not in prompt


def test_claude_prompt_drops_system_prompt_entirely():
    """For Claude, the system prompt is dropped completely — it triggers
    Claude's impersonation refusal when it sees its own harness instructions."""
    prompt = build_notion_prompt(_sample_payload(), notion_model="ambrosia-tart-high")
    assert "PROJECT CONTEXT" not in prompt
    assert "Always write tests for new features." not in prompt
    assert "You are Claude Code" not in prompt


def test_prompt_strips_claude_code_system_prompt_branding():
    """The forwarded system prompt must not leak the harness identity."""
    prompt = build_notion_prompt(_sample_payload())
    # Billing header stripped.
    assert "x-anthropic-billing-header" not in prompt
    # Identity declarations stripped.
    assert "You are Claude Code" not in prompt
    assert "Anthropic's official CLI" not in prompt
    # system-reminder blocks stripped.
    assert "<system-reminder>" not in prompt
    # But user/project context is preserved.
    assert "Always write tests for new features." in prompt
    assert "PROJECT CONTEXT" in prompt


def test_prompt_tools_summary_is_compact():
    """Tool descriptions must be truncated, not dumped verbatim."""
    prompt = build_notion_prompt(_sample_payload())
    # The long description ("Run a shell command. " * 30) must be truncated
    # to far fewer occurrences than the full 30.
    assert prompt.count("Run a shell command.") < 10
    assert "…" in prompt
    assert "args [command]" in prompt
    assert "required [command]" in prompt


def test_prompt_does_not_trip_own_self_reference_detector():
    """The rendered prompt itself must not match the bad-self-reference regex.

    If it did, we'd be shipping a prompt that trips our own repair detector
    before Notion even responds.
    """
    prompt = build_notion_prompt(_sample_payload())
    assert not has_bad_self_reference(prompt)


def test_simulated_notion_refusal_still_detected():
    """A simulated 'I'm Notion AI' reply must still trigger the safety net."""
    bad = (
        "I should be straight with you: I'm Notion AI — I don't have a local "
        "codebase mounted, and I can't run those CLI-style tools."
    )
    assert has_bad_self_reference(bad)
    assert BAD_SELF_REFERENCE_RE.search(bad)


def test_prompt_without_system_prompt():
    """A payload with no system prompt still renders cleanly."""
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    }
    prompt = build_notion_prompt(payload)
    assert "IMPORTANT RULES" in prompt
    assert "(no tools available)" in prompt
    assert "User: hello" in prompt
    assert "PROJECT CONTEXT" not in prompt


def test_prompt_with_error_nudge():
    """The error nudge is placed before the system context."""
    prompt = build_notion_prompt(
        _sample_payload(),
        error_nudge="ATTENTION — your previous tool call failed.",
    )
    assert "ATTENTION — your previous tool call failed." in prompt
    # Nudge appears before the project context block.
    assert prompt.index("ATTENTION") < prompt.index("PROJECT CONTEXT")
