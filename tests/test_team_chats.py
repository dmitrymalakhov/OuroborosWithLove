import pathlib
import tempfile


def test_team_chat_request_notification_and_approval_workspace():
    from supervisor.teams import (
        TEAM_APPROVED,
        TEAM_PENDING,
        append_team_chat_notification,
        get_team_chat,
        request_team_chat,
        set_team_chat_status,
        team_root,
        team_slug_for_chat,
    )

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        chat = {"id": -100123, "type": "supergroup", "title": "Project X"}
        from_user = {"id": 42, "username": "alice"}

        rec, created, should_notify = request_team_chat(drive_root, chat, requested_by=from_user)

        assert created is True
        assert should_notify is True
        assert rec["status"] == TEAM_PENDING
        assert rec["slug"] == team_slug_for_chat(-100123)
        assert not team_root(drive_root, -100123).exists()

        append_team_chat_notification(drive_root, -100123, admin_chat_id=1, message_id=99)
        rec2, created2, should_notify2 = request_team_chat(drive_root, chat, requested_by=from_user)

        assert created2 is False
        assert should_notify2 is False
        assert len(rec2["notifications"]) == 1

        result = set_team_chat_status(drive_root, -100123, TEAM_APPROVED, decided_by=1)

        assert result["old_status"] == TEAM_PENDING
        assert result["status"] == TEAM_APPROVED
        root = team_root(drive_root, -100123)
        assert (root / "logs" / "chat.jsonl").exists()
        assert (root / "inbox" / "messages.jsonl").exists()
        assert get_team_chat(drive_root, -100123)["status"] == TEAM_APPROVED

        repeated = set_team_chat_status(drive_root, -100123, "denied", decided_by=2)
        assert repeated["status"] == TEAM_APPROVED
        assert repeated["old_status"] == TEAM_APPROVED

        forced = set_team_chat_status(
            drive_root,
            -100123,
            "denied",
            decided_by=2,
            allow_terminal_change=True,
        )
        assert forced["status"] == "denied"
        assert forced["old_status"] == TEAM_APPROVED


def test_teamchat_runtime_command_can_override_terminal_decision():
    from supervisor.teamchat import TeamChatRuntime
    from supervisor.teams import TEAM_APPROVED, TEAM_DENIED, get_team_chat, request_team_chat, set_team_chat_status

    class FakeTG:
        def edit_message_text(self, *args, **kwargs):
            return True, "ok"

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_team_chat(
            drive_root,
            {"id": -100123, "type": "supergroup", "title": "Project X"},
            requested_by={"id": 1},
        )
        set_team_chat_status(drive_root, -100123, TEAM_DENIED, decided_by=1)
        sent = []
        runtime = TeamChatRuntime(
            drive_root=drive_root,
            tg=FakeTG(),
            admin_chat_ids_fn=lambda: [1],
            access_user_label_fn=lambda rec: str(rec.get("user_id")),
            is_admin_user_fn=lambda user_id, st: True,
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
            log_chat_fn=lambda *args, **kwargs: None,
            append_jsonl_fn=lambda *args, **kwargs: None,
        )

        assert runtime.handle_command("/teamchat approve -100123", chat_id=1, admin_user_id=1) is True
        assert get_team_chat(drive_root, -100123)["status"] == TEAM_APPROVED
        assert any("Группа разрешена" in args[1] for args, _kwargs in sent)


def test_teamchat_runtime_force_callback_can_override_terminal_decision():
    from supervisor.teamchat import TeamChatRuntime
    from supervisor.teams import TEAM_APPROVED, TEAM_DENIED, get_team_chat, request_team_chat, set_team_chat_status

    class FakeTG:
        def __init__(self):
            self.callbacks = []
            self.edits = []

        def answer_callback_query(self, callback_query_id, text="", show_alert=False):
            self.callbacks.append((callback_query_id, text, show_alert))
            return True, "ok"

        def edit_message_text(self, *args, **kwargs):
            self.edits.append((args, kwargs))
            return True, "ok"

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_team_chat(
            drive_root,
            {"id": -100123, "type": "supergroup", "title": "Project X"},
            requested_by={"id": 1},
        )
        set_team_chat_status(drive_root, -100123, TEAM_DENIED, decided_by=1)
        tg = FakeTG()
        runtime = TeamChatRuntime(
            drive_root=drive_root,
            tg=tg,
            admin_chat_ids_fn=lambda: [1],
            access_user_label_fn=lambda rec: str(rec.get("user_id")),
            is_admin_user_fn=lambda user_id, st: True,
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            log_chat_fn=lambda *args, **kwargs: None,
            append_jsonl_fn=lambda *args, **kwargs: None,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": "teamchat:approve:-100123:force",
            "from": {"id": 1},
            "message": {"chat": {"id": 1}, "message_id": 77},
        })

        assert handled is True
        assert get_team_chat(drive_root, -100123)["status"] == TEAM_APPROVED
        assert tg.callbacks and "Группа разрешена" in tg.callbacks[-1][1]


def test_team_member_seen_is_stored():
    from supervisor.teams import note_team_member_seen, request_team_chat

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        chat = {"id": -100123, "type": "group", "title": "Project X"}
        request_team_chat(drive_root, chat, requested_by={"id": 1})

        note_team_member_seen(
            drive_root,
            -100123,
            {"id": 42, "username": "alice", "first_name": "Alice"},
        )

        rec, _created, _notify = request_team_chat(drive_root, chat, requested_by={"id": 1})
        assert rec["members"]["42"]["username"] == "alice"


def test_team_inbox_tools_require_team_context_and_roundtrip():
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.team import _team_inbox_read, _team_inbox_send

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        ctx = ToolContext(repo_dir=drive_root, drive_root=drive_root)
        assert "only inside" in _team_inbox_read(ctx)

        team_root = drive_root / "teams" / "tg_100123"
        ctx = ToolContext(
            repo_dir=drive_root,
            drive_root=team_root,
            shared_drive_root=drive_root,
            current_chat_id=-100123,
            current_user_id=42,
            team_chat_id=-100123,
            team_slug="tg_100123",
            is_team_workspace=True,
        )

        assert _team_inbox_send(ctx, "coordinate this", topic="plan").startswith("OK:")
        output = _team_inbox_read(ctx, limit=5)
        assert "coordinate this" in output
        assert "#plan" in output


def test_team_context_does_not_include_personal_memory():
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    class Env:
        def __init__(self, repo_dir, drive_root):
            self.repo_dir = repo_dir
            self.drive_root = drive_root
            self.shared_drive_root = drive_root.parent.parent

        def repo_path(self, rel):
            return (self.repo_dir / rel).resolve()

        def drive_path(self, rel):
            return (self.drive_root / rel).resolve()

        def shared_drive_path(self, rel):
            return (self.shared_drive_root / rel).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        repo = root / "repo"
        repo.mkdir()
        (repo / "prompts").mkdir()
        (repo / "prompts" / "SYSTEM.md").write_text("system", encoding="utf-8")
        (repo / "BIBLE.md").write_text("bible", encoding="utf-8")
        (repo / "README.md").write_text("readme", encoding="utf-8")
        (repo / "VERSION").write_text("1.0.0", encoding="utf-8")
        (repo / "pyproject.toml").write_text('version = "1.0.0"\n', encoding="utf-8")

        drive = root / "drive"
        team = drive / "teams" / "tg_100123"
        personal = drive / "users" / "42"
        (team / "memory").mkdir(parents=True)
        (team / "logs").mkdir()
        (drive / "state").mkdir(parents=True)
        (drive / "state" / "state.json").write_text("{}", encoding="utf-8")
        (team / "memory" / "scratchpad.md").write_text("team scratchpad", encoding="utf-8")
        (team / "memory" / "identity.md").write_text("team identity", encoding="utf-8")
        (team / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
        (personal / "memory").mkdir(parents=True)
        (personal / "memory" / "identity.md").write_text("private user identity", encoding="utf-8")

        messages, _info = build_llm_messages(
            Env(repo, team),
            Memory(drive_root=team, repo_dir=repo),
            {
                "id": "task1",
                "type": "task",
                "text": "hello team",
                "is_team_workspace": True,
                "team_slug": "tg_100123",
                "team_chat_id": -100123,
                "chat_type": "supergroup",
            },
        )

        system_text = "\n".join(
            block.get("text", "")
            for block in messages[0]["content"]
            if isinstance(block, dict)
        )
        assert "team identity" in system_text
        assert "team scratchpad" in system_text
        assert "private user identity" not in system_text


def test_telegram_inline_keyboard_methods(monkeypatch):
    from supervisor import telegram
    from supervisor.telegram import TelegramClient

    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeRequests:
        @staticmethod
        def post(url, data=None, timeout=0, files=None):
            calls.append((url, data))
            if url.endswith("/sendMessage"):
                return Response({"ok": True, "result": {"message_id": 123}})
            return Response({"ok": True, "result": True})

    monkeypatch.setattr(telegram, "requests", FakeRequests)
    client = TelegramClient("token")

    ok, err, message_id = client.send_message_with_markup(
        1,
        "Approve?",
        {"inline_keyboard": [[{"text": "ok", "callback_data": "teamchat:approve:-1"}]]},
    )
    assert ok is True
    assert err == "ok"
    assert message_id == 123
    assert "reply_markup" in calls[0][1]

    ok, err = client.answer_callback_query("cb1", "done")
    assert ok is True
    assert err == "ok"

    ok, err = client.edit_message_text(1, 123, "done")
    assert ok is True
    assert err == "ok"


def test_teamchat_trigger_helpers():
    from supervisor.teamchat import is_group_task_trigger, strip_bot_mention

    msg = {"text": "hello"}
    assert is_group_task_trigger(msg, "hello", "", bot_id=10, bot_username="ouro_bot") is False

    assert is_group_task_trigger(msg, "/status", "", bot_id=10, bot_username="ouro_bot") is True
    assert is_group_task_trigger(msg, "hey @ouro_bot help", "", bot_id=10, bot_username="ouro_bot") is True
    assert is_group_task_trigger(
        {"reply_to_message": {"from": {"id": 10, "username": "ouro_bot"}}},
        "continue",
        "",
        bot_id=10,
        bot_username="ouro_bot",
    ) is True
    assert strip_bot_mention("hey @ouro_bot help", "ouro_bot") == "hey help"


def test_teamchat_runtime_callback_rejects_non_admin():
    from supervisor.teamchat import TeamChatRuntime

    class FakeTG:
        def __init__(self):
            self.callbacks = []

        def answer_callback_query(self, callback_query_id, text="", show_alert=False):
            self.callbacks.append((callback_query_id, text, show_alert))
            return True, "ok"

    with tempfile.TemporaryDirectory() as tmp:
        tg = FakeTG()
        runtime = TeamChatRuntime(
            drive_root=pathlib.Path(tmp),
            tg=tg,
            admin_chat_ids_fn=lambda: [1],
            access_user_label_fn=lambda rec: str(rec.get("user_id")),
            is_admin_user_fn=lambda user_id, st: False,
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            log_chat_fn=lambda *args, **kwargs: None,
            append_jsonl_fn=lambda *args, **kwargs: None,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": "teamchat:approve:-100",
            "from": {"id": 2},
        })

        assert handled is True
        assert tg.callbacks == [("cb1", "Только админ Ouroboros может принимать решение.", True)]
