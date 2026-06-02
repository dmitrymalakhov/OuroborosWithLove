import json
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
            self.sent = []

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

        sent = []
        runtime.send_with_budget_fn = lambda *args, **kwargs: sent.append((args, kwargs))

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": "teamchat:approve:-100123:force",
            "from": {"id": 1},
            "message": {"chat": {"id": 1}, "message_id": 77},
        })

        assert handled is True
        assert get_team_chat(drive_root, -100123)["status"] == TEAM_APPROVED
        assert tg.callbacks and "Группа разрешена" in tg.callbacks[-1][1]
        assert sent
        assert sent[0][0][0] == -100123
        assert "/ask" in sent[0][0][1]


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
    from ouroboros.tools.team import _team_chat_history, _team_chat_search, _team_inbox_read, _team_inbox_send
    from ouroboros.utils import append_jsonl

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        ctx = ToolContext(repo_dir=drive_root, drive_root=drive_root)
        assert "only inside" in _team_inbox_read(ctx)
        assert "only inside" in _team_chat_history(ctx)

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

        chat_log = team_root / "logs" / "chat.jsonl"
        chat_log.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(chat_log, {
            "ts": "2026-01-01T00:00:00+00:00",
            "direction": "in",
            "chat_id": -100123,
            "user_id": 42,
            "text": "Discuss Project Phoenix timeline",
        })
        append_jsonl(chat_log, {
            "ts": "2026-01-01T00:01:00+00:00",
            "direction": "out",
            "chat_id": -100123,
            "user_id": 0,
            "text": "I can summarize the timeline.",
        })

        history = _team_chat_history(ctx, limit=2)
        assert "Project Phoenix" in history
        assert "I can summarize" in history

        user_only = _team_chat_history(ctx, limit=2, include_bot=False)
        assert "Project Phoenix" in user_only
        assert "I can summarize" not in user_only

        search = _team_chat_search(ctx, "phoenix", limit=5)
        assert "Project Phoenix" in search
        assert "matches" in search


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


def test_team_poll_tool_queues_create_event_and_requires_team():
    from ouroboros.tools.polls import _team_poll_create
    from ouroboros.tools.registry import ToolContext

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        ctx = ToolContext(repo_dir=drive_root, drive_root=drive_root)
        assert "only inside" in _team_poll_create(ctx, "Pick?", ["A", "B"])

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
            task_id="task1",
        )

        result = _team_poll_create(
            ctx,
            "Pick one?",
            ["Alpha", "Beta"],
            allows_multiple_answers=True,
            is_anonymous=False,
        )

        assert result.startswith("OK:")
        event = ctx.pending_events[0]
        assert event["type"] == "send_poll"
        assert event["chat_id"] == -100123
        assert event["question"] == "Pick one?"
        assert event["options"] == ["Alpha", "Beta"]
        assert event["allows_multiple_answers"] is True
        assert event["is_anonymous"] is False
        assert event["drive_root"] == str(team_root)
        assert event["shared_drive_root"] == str(drive_root)


def test_team_poll_event_answer_results_and_close_roundtrip():
    from types import SimpleNamespace

    from ouroboros.tools.polls import _team_poll_close, _team_poll_results
    from ouroboros.tools.registry import ToolContext
    from ouroboros.utils import append_jsonl
    from supervisor.events import _handle_send_poll, _handle_stop_poll
    from supervisor.polls import record_poll_answer

    class FakeTG:
        def __init__(self):
            self.sent_polls = []
            self.stopped = []

        def send_poll(
            self,
            chat_id,
            question,
            options,
            is_anonymous=False,
            allows_multiple_answers=False,
            open_period_seconds=0,
        ):
            self.sent_polls.append((chat_id, question, options, is_anonymous, allows_multiple_answers, open_period_seconds))
            return True, "ok", {
                "message_id": 77,
                "poll": {
                    "id": "poll123",
                    "question": question,
                    "options": [{"text": options[0], "voter_count": 0}, {"text": options[1], "voter_count": 0}],
                    "total_voter_count": 0,
                    "is_closed": False,
                    "is_anonymous": is_anonymous,
                    "allows_multiple_answers": allows_multiple_answers,
                },
            }

        def stop_poll(self, chat_id, message_id):
            self.stopped.append((chat_id, message_id))
            return True, "ok", {
                "id": "poll123",
                "question": "Pick one?",
                "options": [{"text": "Alpha", "voter_count": 0}, {"text": "Beta", "voter_count": 1}],
                "total_voter_count": 1,
                "is_closed": True,
                "is_anonymous": False,
            }

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        team_root = drive_root / "teams" / "tg_100123"
        (team_root / "logs").mkdir(parents=True)
        sent = []
        tg = FakeTG()
        event_ctx = SimpleNamespace(
            DRIVE_ROOT=drive_root,
            TG=tg,
            append_jsonl=append_jsonl,
            send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
        )

        _handle_send_poll({
            "type": "send_poll",
            "poll_uid": "polluid1",
            "chat_id": -100123,
            "question": "Pick one?",
            "options": ["Alpha", "Beta"],
            "is_anonymous": False,
            "allows_multiple_answers": False,
            "open_period_seconds": 0,
            "drive_root": str(team_root),
            "shared_drive_root": str(drive_root),
            "user_id": 42,
            "task_id": "task1",
        }, event_ctx)

        snapshot = json.loads((team_root / "polls" / "polluid1.json").read_text(encoding="utf-8"))
        assert snapshot["telegram_poll_id"] == "poll123"
        assert snapshot["message_id"] == 77
        assert tg.sent_polls == [(-100123, "Pick one?", ["Alpha", "Beta"], False, False, 0)]

        record_poll_answer(drive_root, {
            "poll_id": "poll123",
            "user": {"id": 42, "username": "alice", "first_name": "Alice"},
            "option_ids": [1],
        })

        tool_ctx = ToolContext(
            repo_dir=drive_root,
            drive_root=team_root,
            shared_drive_root=drive_root,
            current_chat_id=-100123,
            current_user_id=42,
            team_chat_id=-100123,
            team_slug="tg_100123",
            is_team_workspace=True,
        )
        results = _team_poll_results(tool_ctx, poll_ref="poll123", include_voters=True)
        assert "Beta — 1" in results
        assert "@alice" in results
        assert "Alice" in results

        close_result = _team_poll_close(tool_ctx, "polluid1")
        assert close_result.startswith("OK:")
        stop_event = tool_ctx.pending_events[-1]
        assert stop_event["type"] == "stop_poll"
        _handle_stop_poll(stop_event, event_ctx)

        closed = json.loads((team_root / "polls" / "polluid1.json").read_text(encoding="utf-8"))
        assert closed["status"] == "closed"
        assert closed["is_closed"] is True
        assert tg.stopped == [(-100123, 77)]


def test_telegram_inline_keyboard_methods(monkeypatch):
    from supervisor import telegram
    from supervisor.telegram import TelegramClient

    calls = []
    get_calls = []

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
        def get(url, params=None, timeout=0):
            get_calls.append((url, params))
            return Response({"ok": True, "result": []})

        @staticmethod
        def post(url, data=None, timeout=0, files=None):
            calls.append((url, data))
            if url.endswith("/sendMessage"):
                return Response({"ok": True, "result": {"message_id": 123}})
            if url.endswith("/sendPoll"):
                return Response({
                    "ok": True,
                    "result": {
                        "message_id": 456,
                        "poll": {
                            "id": "poll123",
                            "question": data["question"],
                            "options": [
                                {"text": "A", "voter_count": 0},
                                {"text": "B", "voter_count": 0},
                            ],
                            "total_voter_count": 0,
                            "is_closed": False,
                            "is_anonymous": data["is_anonymous"],
                            "allows_multiple_answers": data["allows_multiple_answers"],
                        },
                    },
                })
            if url.endswith("/stopPoll"):
                return Response({
                    "ok": True,
                    "result": {
                        "id": "poll123",
                        "question": "Pick",
                        "options": [{"text": "A", "voter_count": 1}, {"text": "B", "voter_count": 0}],
                        "total_voter_count": 1,
                        "is_closed": True,
                    },
                })
            return Response({"ok": True, "result": True})

    monkeypatch.setattr(telegram, "requests", FakeRequests)
    client = TelegramClient("token")

    assert client.get_updates(10) == []
    allowed = json.loads(get_calls[0][1]["allowed_updates"])
    assert "poll" in allowed
    assert "poll_answer" in allowed

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

    ok, err, message = client.send_poll(1, "Pick", ["A", "B"], is_anonymous=False)
    assert ok is True
    assert err == "ok"
    assert message["poll"]["id"] == "poll123"
    poll_call = next(call for call in calls if call[0].endswith("/sendPoll"))
    assert json.loads(poll_call[1]["options"]) == [{"text": "A"}, {"text": "B"}]
    assert poll_call[1]["is_anonymous"] is False

    ok, err, poll = client.stop_poll(1, 456)
    assert ok is True
    assert err == "ok"
    assert poll["is_closed"] is True


def test_telegram_error_redacts_bot_token():
    from supervisor.telegram import redact_telegram_token

    token = "123456:ABC_SECRET"
    raw = (
        "HTTPError('400 Client Error: Bad Request for url: "
        "https://api.telegram.org/bot123456:ABC_SECRET/editMessageText')"
    )

    redacted = redact_telegram_token(raw, token)

    assert token not in redacted
    assert "bot<telegram-token>" in redacted


def test_teamchat_trigger_helpers():
    from supervisor.teamchat import is_group_task_trigger, prepare_group_task_text, strip_bot_mention

    msg = {"text": "hello"}
    assert is_group_task_trigger(msg, "hello", "", bot_id=10, bot_username="ouro_bot") is False

    assert is_group_task_trigger(msg, "/status", "", bot_id=10, bot_username="ouro_bot") is True
    assert prepare_group_task_text("/ask@ouro_bot hello", "ouro_bot") == "hello"
    assert prepare_group_task_text("/ouro привет", "ouro_bot") == "привет"
    assert prepare_group_task_text("/status", "ouro_bot") == "/status"
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
