import pathlib

from ouroboros.llm import LLMClient
from ouroboros.loop import _execute_single_tool


def test_normalize_gigachat_function_call_to_tool_calls():
    msg = {
        "role": "assistant",
        "content": "",
        "function_call": {
            "name": "browse_page",
            "arguments": {"url": "https://example.com", "output": "text"},
        },
    }
    out = LLMClient._normalize_response_message("gigachat", msg)
    assert out.get("tool_calls"), "Legacy function_call must be mapped into tool_calls"
    tc = out["tool_calls"][0]
    assert tc["function"]["name"] == "browse_page"
    assert isinstance(tc["function"]["arguments"], str)
    assert '"url": "https://example.com"' in tc["function"]["arguments"]


def test_execute_single_tool_accepts_dict_arguments(tmp_path):
    class DummyTools:
        CODE_TOOLS = frozenset()

        def execute(self, name, args):
            assert name == "browse_page"
            assert args["url"] == "https://example.com"
            return "ok"

    tc = {
        "id": "call_1",
        "function": {
            "name": "browse_page",
            "arguments": {"url": "https://example.com", "output": "text"},
        },
    }

    result = _execute_single_tool(
        tools=DummyTools(),
        tc=tc,
        drive_logs=pathlib.Path(tmp_path),
        task_id="task_1",
    )
    assert result["result"] == "ok"
    assert result["is_error"] is False


def test_normalize_gigachat_tool_call_from_content_json():
    msg = {
        "role": "assistant",
        "content": '{"name":"browse_page","arguments":{"url":"https://example.com","output":"text"}}',
    }
    out = LLMClient._normalize_response_message("gigachat", msg)
    assert out.get("tool_calls"), "JSON content payload must be mapped into tool_calls"
    tc = out["tool_calls"][0]
    assert tc["function"]["name"] == "browse_page"
    assert '"url": "https://example.com"' in tc["function"]["arguments"]
    assert out["content"] == ""


def test_normalize_messages_for_gigachat_removes_openai_only_fields():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "browse_page", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    out = LLMClient._normalize_messages_for_gigachat(messages)
    assert "tool_calls" not in out[0]
    assert out[0]["function_call"]["name"] == "browse_page"
    assert out[1]["role"] == "function"
    assert "tool_call_id" not in out[1]


def test_normalize_messages_for_gigachat_drops_tool_calls_even_with_function_call():
    messages = [
        {
            "role": "assistant",
            "function_call": {"name": "browse_page", "arguments": "{}"},
            "tool_calls": [{"id": "call_1", "function": {"name": "browse_page", "arguments": "{}"}}],
        },
    ]
    out = LLMClient._normalize_messages_for_gigachat(messages)
    assert out[0]["function_call"]["name"] == "browse_page"
    assert "tool_calls" not in out[0]


def test_pick_gigachat_function_honors_tool_choice_name():
    functions = [
        {"name": "repo_read", "description": "Read repo file", "parameters": {"type": "object", "properties": {}}},
        {"name": "run_shell", "description": "Run shell command", "parameters": {"type": "object", "properties": {}}},
    ]
    picked = LLMClient._pick_gigachat_function(
        functions,
        {"type": "function", "function": {"name": "run_shell"}},
        [{"role": "user", "content": "Open terminal"}],
    )
    assert picked is not None
    assert picked["name"] == "run_shell"
