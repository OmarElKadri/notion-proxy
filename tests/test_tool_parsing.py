"""Tests for tool-call parsing and the loose-JSON repair heuristics."""

from notion_proxy.tools.parse import (
    contains_tool_use_markup,
    parse_tool_use,
    parse_tool_uses,
)


def _wrap(inner: str) -> str:
    return f"<tool_use>\n{inner}\n</tool_use>"


def test_delimited_inline_value():
    tool = parse_tool_use(_wrap("Read\nfile_path: /home/a/b.py"))
    assert tool == {"name": "Read", "input": {"file_path": "/home/a/b.py"}}


def test_delimited_heredoc_multiline():
    block = _wrap(
        "Edit\n"
        "file_path: D:\\dev\\main.py\n"
        "old_string: <<<OLD\n"
        "def hello():\n"
        '    print("hi")\n'
        "OLD\n"
        "new_string: <<<NEW\n"
        "def hello():\n"
        '    print("hello")\n'
        "NEW"
    )
    tool = parse_tool_use(block)
    assert tool["name"] == "Edit"
    assert tool["input"]["file_path"] == "D:\\dev\\main.py"
    assert tool["input"]["old_string"] == 'def hello():\n    print("hi")'
    assert tool["input"]["new_string"] == 'def hello():\n    print("hello")'


def test_delimited_empty_then_heredoc():
    block = _wrap("Write\nfile_path: notes.txt\ncontent:\n<<<BODY\nline one\nline two\nBODY")
    tool = parse_tool_use(block)
    assert tool["input"]["content"] == "line one\nline two"


def test_bare_tool_name_with_no_fields_is_parsed():
    # The parser emits a bare-name call with empty input; the validator is the
    # authority on whether that's legal (depends on the tool's required fields).
    assert parse_tool_uses(_wrap("EnterPlanMode")) == [
        {"name": "EnterPlanMode", "input": {}}
    ]


def test_fenced_json_is_stripped():
    block = _wrap('```json\n{"name": "Bash", "input": {"command": "echo hi"}}\n```')
    tool = parse_tool_use(block)
    assert tool == {"name": "Bash", "input": {"command": "echo hi"}}


def test_loose_json_windows_path_repair():
    # Single backslashes in a Windows path are invalid JSON; the repair pass
    # should double them so json.loads recovers the original path.
    block = _wrap(r'{"name": "Read", "input": {"file_path": "D:\dev\a.py"}}')
    tool = parse_tool_use(block)
    assert tool["input"]["file_path"] == r"D:\dev\a.py"


def test_loose_json_unquoted_keys_and_trailing_comma():
    block = _wrap('{name: "Bash", input: {command: "ls",}}')
    tool = parse_tool_use(block)
    assert tool == {"name": "Bash", "input": {"command": "ls"}}


def test_function_call_syntax():
    block = _wrap('Bash({"command": "ls -la"})')
    tool = parse_tool_use(block)
    assert tool == {"name": "Bash", "input": {"command": "ls -la"}}


def test_multiple_tool_use_blocks():
    text = _wrap("Read\nfile_path: a.py") + "\nsome prose\n" + _wrap("Read\nfile_path: b.py")
    tools = parse_tool_uses(text)
    assert [t["input"]["file_path"] for t in tools] == ["a.py", "b.py"]


def test_contains_tool_use_markup():
    assert contains_tool_use_markup("before <tool_use>garbage</tool_use> after")
    assert not contains_tool_use_markup("plain text response")


def test_markup_mention_in_prose_is_not_an_attempt():
    # Notion explaining the format mentions the token but emits no real block.
    # This must NOT count as a malformed attempt, or it clobbers the answer.
    assert not contains_tool_use_markup(
        "When tool use is required, emit one or more <tool_use> blocks."
    )


def test_markup_inside_code_fence_is_not_an_attempt():
    # A fenced example of the format is documentation, not a tool call.
    prose = (
        "Use a prompt like:\n\n"
        "```text\n"
        "output exactly one or more <tool_use> blocks.\n"
        "Do not include explanation outside the blocks.\n"
        "```\n\n"
        "Then your parser enforces it."
    )
    assert not contains_tool_use_markup(prose)
    assert parse_tool_uses(prose) == []


def test_markup_without_parseable_body_yields_no_tools():
    # Markup present but body is not a valid tool call -> empty list. This is the
    # condition that drives the "Rejected malformed tool_use block" path.
    assert parse_tool_uses("<tool_use>\n???\n</tool_use>") == []
