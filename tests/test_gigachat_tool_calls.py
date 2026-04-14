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
