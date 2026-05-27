import pytest


class _FakeResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.001},
        }


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def _client_with_fake_backend(provider):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test-key", provider=provider)
    fake = _FakeClient()
    client._client = fake
    return client, fake


def test_openai_provider_uses_native_chat_completion_params():
    client, fake = _client_with_fake_backend("openai")
    tools = [{
        "type": "function",
        "function": {
            "name": "sample_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    msg, usage = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gpt-5.2",
        tools=tools,
        reasoning_effort="high",
        max_tokens=123,
    )

    kwargs = fake.chat.completions.kwargs
    assert msg["content"] == "ok"
    assert usage["prompt_tokens"] == 2
    assert kwargs["model"] == "gpt-5.2"
    assert kwargs["max_completion_tokens"] == 123
    assert kwargs["reasoning_effort"] == "high"
    assert "max_tokens" not in kwargs
    assert "extra_body" not in kwargs
    assert "cache_control" not in kwargs["tools"][-1]


def test_openai_provider_omits_reasoning_effort_for_non_reasoning_model():
    client, fake = _client_with_fake_backend("openai")

    client.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4.1",
        reasoning_effort="low",
    )

    kwargs = fake.chat.completions.kwargs
    assert kwargs["model"] == "gpt-4.1"
    assert "reasoning_effort" not in kwargs


def test_openrouter_provider_keeps_openrouter_specific_params():
    client, fake = _client_with_fake_backend("openrouter")
    tools = [{
        "type": "function",
        "function": {
            "name": "sample_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    client.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-sonnet-4.6",
        tools=tools,
        reasoning_effort="low",
        max_tokens=456,
    )

    kwargs = fake.chat.completions.kwargs
    assert kwargs["model"] == "anthropic/claude-sonnet-4.6"
    assert kwargs["max_tokens"] == 456
    assert kwargs["extra_body"]["reasoning"]["effort"] == "low"
    assert kwargs["extra_body"]["provider"]["order"] == ["Anthropic"]
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_provider_aware_defaults(monkeypatch):
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_CODE", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)

    client = LLMClient(api_key="test-key")

    assert client.provider == "openai"
    assert client.default_model() == "gpt-5.2"
    assert client.available_models() == ["gpt-5.2"]


def test_openai_provider_rejects_non_openai_models():
    client, _fake = _client_with_fake_backend("openai")

    assert client.supports_model("gpt-5.2")
    assert client.supports_model("openai/gpt-5.2")
    assert not client.supports_model("anthropic/claude-sonnet-4.6")

    with pytest.raises(ValueError):
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="google/gemini-3-pro-preview",
        )


def test_claude_code_edit_hidden_when_openai_provider(monkeypatch):
    from ouroboros.tools import shell

    monkeypatch.setenv("OUROBOROS_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OUROBOROS_DISABLE_CLAUDE_CODE_EDIT", raising=False)
    names = [entry.name for entry in shell.get_tools()]
    assert "run_shell" in names
    assert "claude_code_edit" not in names


def test_claude_code_edit_can_be_reenabled_explicitly(monkeypatch):
    from ouroboros.tools import shell

    monkeypatch.setenv("OUROBOROS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OUROBOROS_DISABLE_CLAUDE_CODE_EDIT", "0")
    names = [entry.name for entry in shell.get_tools()]
    assert "claude_code_edit" in names
