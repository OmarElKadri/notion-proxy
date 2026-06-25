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


def test_claude_prompt_uses_request_format_framing():
    """When notion_model is Claude (ambrosia), the request-format prompt is used.

    The prompt frames tool_use blocks as a text-based request format (the user
    runs the action and sends back the result) rather than the old
    "sample output generation" simulation framing, which Claude recognized as
    an impersonation/jailbreak pattern and refused.
    """
    prompt = build_notion_prompt(_sample_payload(), notion_model="ambrosia-tart-high")
    assert "text-based request format" in prompt.lower()
    assert "CONVERSATION" in prompt
    # Should NOT contain the old simulation framing.
    assert "sample output generation" not in prompt.lower()
    assert "AGENT TRANSCRIPT" not in prompt
    # Should NOT contain the GPT prompt's relay framing.
    assert "text relay" not in prompt.lower()


def test_gpt_prompt_used_by_default():
    """Without a Claude model, the default (GPT) prompt is used."""
    prompt = build_notion_prompt(_sample_payload(), notion_model="opal-quince-medium")
    assert "relay" in prompt.lower()
    assert "sample output generation" not in prompt.lower()
    assert "text-based request format" not in prompt.lower()


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


def test_prompt_without_tools_uses_passthrough():
    """A payload with no tools is a non-coding request (e.g. title
    generation) and uses the passthrough prompt — no coding-agent
    framing, no tool_use format instructions, just the conversation."""
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    }
    prompt = build_notion_prompt(payload)
    assert "User: hello" in prompt
    # No coding-agent framing.
    assert "IMPORTANT RULES" not in prompt
    assert "tool_use" not in prompt.lower()
    assert "PROJECT CONTEXT" not in prompt
    assert "(no tools available)" not in prompt


def test_title_generation_request_passes_through():
    """A title-generation request (no tools, title system prompt) should
    pass through with the task instructions preserved and the identity
    declarations stripped, so Claude generates the title instead of
    refusing."""
    payload = {
        "model": "claude-opus-4-8",
        "messages": [
            {
                "role": "user",
                "content": (
                    "<session>\nhow does the ball speed get calculated\n</session>\n\n"
                    "Write the title in the language the user wrote in."
                ),
            }
        ],
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1;"},
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            {
                "type": "text",
                "text": 'Generate a concise title. Return JSON with a "title" field.',
            },
        ],
        "tools": [],
    }
    prompt = build_notion_prompt(payload, notion_model="ambrosia-tart-high")
    # Task instructions preserved.
    assert "Generate a concise title" in prompt
    # Identity declarations and billing header stripped.
    assert "You are Claude Code" not in prompt
    assert "x-anthropic-billing-header" not in prompt
    # No coding-agent framing.
    assert "tool_use" not in prompt.lower()
    assert "text-based request format" not in prompt.lower()


def test_prompt_with_error_nudge():
    """The error nudge is placed before the system context."""
    prompt = build_notion_prompt(
        _sample_payload(),
        error_nudge="ATTENTION — your previous tool call failed.",
    )
    assert "ATTENTION — your previous tool call failed." in prompt
    # Nudge appears before the project context block.
    assert prompt.index("ATTENTION") < prompt.index("PROJECT CONTEXT")


def _history_payload():
    """A payload with conversation history including tool calls and results."""
    return {
        "model": "claude-sonnet-4-5",
        "system": "",
        "messages": [
            {"role": "user", "content": "List the files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll list them."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "README.md\nAssets/",
                    }
                ],
            },
            {"role": "user", "content": "how does the ball speed get calculated"},
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        ],
    }


def test_claude_transcript_hides_prior_tool_use_blocks():
    """For Claude, prior tool calls must NOT be rendered as <tool_use> blocks
    in the transcript. Seeing the "emit <tool_use> → get result" execution
    pattern in history triggers Claude's impersonation refusal."""
    prompt = build_notion_prompt(
        _history_payload(), notion_model="ambrosia-tart-high"
    )
    # The assistant's text is preserved.
    assert "I'll list them." in prompt
    # The tool result is shown as a user message.
    assert "README.md" in prompt
    # But there should be no <tool_use> blocks in the CONVERSATION section
    # (the only <tool_use> blocks should be in the FORMAT examples).
    conversation_section = prompt.split("CONVERSATION:")[1].split(
        "Write your next response"
    )[0]
    assert "<tool_use>" not in conversation_section
    assert "</tool_use>" not in conversation_section


def test_gpt_transcript_shows_prior_tool_use_blocks():
    """For GPT, prior tool calls ARE rendered as <tool_use> blocks — the
    GPT backend doesn't have the impersonation refusal problem."""
    prompt = build_notion_prompt(
        _history_payload(), notion_model="opal-quince-medium"
    )
    conversation_section = prompt.split("CONVERSATION HISTORY:")[1].split(
        "Produce the next assistant response"
    )[0]
    assert "<tool_use>" in conversation_section
    assert "Bash" in conversation_section
