"""Tests for tool validation and schema-based type coercion."""

from notion_proxy.tools.validate import validate_tool_use, validate_tool_uses


def _payload(*tools: dict) -> dict:
    return {"tools": list(tools)}


READ_TOOL = {
    "name": "Read",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}

TYPED_TOOL = {
    "name": "Configure",
    "input_schema": {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "items": {"type": "array"},
            "meta": {"type": "object"},
        },
    },
}


def test_valid_tool_passes():
    checked, error = validate_tool_use(
        _payload(READ_TOOL), {"name": "Read", "input": {"file_path": "a.py"}}
    )
    assert error is None
    assert checked == {"name": "Read", "input": {"file_path": "a.py"}}


def test_type_coercion_from_strings():
    tool = {
        "name": "Configure",
        "input": {
            "count": "3",
            "ratio": "0.5",
            "enabled": "true",
            "items": "[1, 2]",
            "meta": '{"k": "v"}',
        },
    }
    checked, error = validate_tool_use(_payload(TYPED_TOOL), tool)
    assert error is None
    assert checked["input"] == {
        "count": 3,
        "ratio": 0.5,
        "enabled": True,
        "items": [1, 2],
        "meta": {"k": "v"},
    }


def test_unavailable_tool_rejected():
    checked, error = validate_tool_use(
        _payload(READ_TOOL), {"name": "Write", "input": {}}
    )
    assert checked is None
    assert "unavailable tool" in error.lower()


def test_missing_required_field_rejected():
    checked, error = validate_tool_use(
        _payload(READ_TOOL), {"name": "Read", "input": {}}
    )
    assert checked is None
    assert "missing required" in error.lower()


def test_non_dict_input_rejected():
    checked, error = validate_tool_use(
        _payload(READ_TOOL), {"name": "Read", "input": "oops"}
    )
    assert checked is None
    assert "must be an object" in error.lower()


def test_missing_name_rejected():
    checked, error = validate_tool_use(_payload(READ_TOOL), {"input": {}})
    assert checked is None
    assert "missing string tool name" in error.lower()


def test_validate_many_short_circuits_on_first_error():
    tools = [
        {"name": "Read", "input": {"file_path": "a.py"}},
        {"name": "Write", "input": {}},
    ]
    valid, error = validate_tool_uses(_payload(READ_TOOL), tools)
    assert valid == []
    assert error is not None


def test_validate_many_all_valid():
    tools = [
        {"name": "Read", "input": {"file_path": "a.py"}},
        {"name": "Read", "input": {"file_path": "b.py"}},
    ]
    valid, error = validate_tool_uses(_payload(READ_TOOL), tools)
    assert error is None
    assert [t["input"]["file_path"] for t in valid] == ["a.py", "b.py"]
