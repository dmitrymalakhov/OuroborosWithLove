import json
import pathlib

import pytest


def _registry(tmp_path: pathlib.Path, role: str = "admin"):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=pathlib.Path.cwd(), drive_root=tmp_path)
    reg.set_context(ToolContext(
        repo_dir=pathlib.Path.cwd(),
        drive_root=tmp_path,
        current_user_id=1,
        user_role=role,
    ))
    return reg


def _schema_names(schemas):
    return {s["function"]["name"] for s in schemas}


def test_every_registered_tool_has_exactly_one_pack(tmp_path):
    from ouroboros.tools.registry import TOOL_PACKS

    reg = _registry(tmp_path)
    registered = _schema_names(reg.schemas())
    membership = {
        name: sum(name in tools for tools in TOOL_PACKS.values())
        for name in registered
    }

    assert [name for name, count in membership.items() if count == 0] == []
    assert [name for name, count in membership.items() if count > 1] == []


def test_base_pack_stays_small(tmp_path):
    reg = _registry(tmp_path)
    base = _schema_names(reg.schemas(core_only=True))

    assert len(base) <= 12
    assert {"list_tool_packs", "enable_tool_pack", "enable_tools"}.issubset(base)
    assert "credit_pack_check" not in base
    assert "repo_write_commit" not in base


def test_pack_dependencies_add_document_tools_for_credit(tmp_path):
    reg = _registry(tmp_path)
    names = _schema_names(reg.schemas_for_packs(["credit"]))

    assert "credit_pack_check" in names
    assert "credit_deck_challenge" in names
    assert "analyze_document" in names
    assert "send_file" in names
    assert "drive_write" not in names


def test_user_role_pack_filter_hides_admin_packs(tmp_path):
    reg = _registry(tmp_path, role="user")
    packs = {p["name"]: p for p in reg.list_tool_packs()}

    assert "credit" in packs
    assert "hr" in packs
    assert "code_git" not in packs
    assert "admin_control" not in packs
    assert "github" not in packs


def test_self_mod_disabled_hides_self_mod_tools_in_packs(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DISABLE_SELF_MODIFICATION", "1")
    reg = _registry(tmp_path, role="admin")
    names = _schema_names(reg.schemas_for_packs(["code_git", "admin_control", "health_review"]))

    assert "repo_write_commit" not in names
    assert "repo_commit_push" not in names
    assert "claude_code_edit" not in names
    assert "request_restart" not in names
    assert "multi_model_review" not in names
    assert "run_shell" in names


class FakeRouterLLM:
    provider = "openai"

    def __init__(self, mapping):
        self.mapping = mapping

    def supports_model(self, model: str) -> bool:
        return True

    def chat(self, messages, model, tools=None, reasoning_effort="low", max_tokens=512):
        payload = json.loads(messages[-1]["content"])
        request = payload["request"].lower()
        pack, confidence = [], 0.2
        for needle, result in self.mapping.items():
            if needle in request:
                pack, confidence = result
                break
        return {
            "content": json.dumps({
                "packs": pack,
                "confidence": confidence,
                "reason": "test router",
            })
        }, {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.parametrize("text,expected", [
    ("подготовь материалы для кредитного комитета", ["credit"]),
    ("проверь кандидата на роль аналитика", ["hr"]),
    ("прочитай PDF и сделай выводы", ["documents"]),
    ("найди в интернете свежие источники", ["web_browser"]),
    ("измени код и посмотри git diff", ["code_git"]),
    ("привет, как дела", []),
])
def test_fast_router_selects_expected_pack(tmp_path, text, expected):
    from ouroboros.tool_routing import route_tool_packs

    mapping = {
        "кредитного комитета": (["credit"], 0.86),
        "кандидата": (["hr"], 0.82),
        "pdf": (["documents"], 0.9),
        "интернете": (["web_browser"], 0.88),
        "код": (["code_git"], 0.84),
    }
    reg = _registry(tmp_path, role="admin")
    info = route_tool_packs(
        [{"role": "user", "content": text}],
        reg,
        FakeRouterLLM(mapping),
        task_type="task",
        user_role="admin",
        drive_logs=tmp_path / "logs",
        task_id="t1",
        event_queue=None,
        accumulated_usage={},
        user_id=1,
        estimate_cost_fn=lambda *args, **kwargs: 0.0,
        emit_usage_fn=lambda *args, **kwargs: None,
    )

    assert info["selected_packs"] == expected


def test_fast_router_low_confidence_keeps_base_only(tmp_path):
    from ouroboros.tool_routing import route_tool_packs

    reg = _registry(tmp_path, role="admin")
    info = route_tool_packs(
        [{"role": "user", "content": "что-нибудь про комитет"}],
        reg,
        FakeRouterLLM({"комитет": (["credit"], 0.2)}),
        task_type="task",
        user_role="admin",
        drive_logs=tmp_path / "logs",
        task_id="t1",
        event_queue=None,
        accumulated_usage={},
        user_id=1,
        estimate_cost_fn=lambda *args, **kwargs: 0.0,
        emit_usage_fn=lambda *args, **kwargs: None,
    )

    assert info["selected_packs"] == []
