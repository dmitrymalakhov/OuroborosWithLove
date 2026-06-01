import json
import pathlib
import tempfile


def _write_jsonl(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FakeAccessTG:
    def __init__(self):
        self.sent_markup = []
        self.edits = []
        self.callbacks = []
        self.next_message_id = 77

    def send_message_with_markup(self, chat_id, text, reply_markup, parse_mode=""):
        self.sent_markup.append((chat_id, text, reply_markup, parse_mode))
        return True, "ok", self.next_message_id

    def edit_message_text(self, chat_id, message_id, text, reply_markup=None, parse_mode=""):
        self.edits.append((chat_id, message_id, text, reply_markup, parse_mode))
        return True, "ok"

    def answer_callback_query(self, callback_query_id, text="", show_alert=False):
        self.callbacks.append((callback_query_id, text, show_alert))
        return True, "ok"


def test_access_user_label_and_id_parsing():
    from supervisor.access_control import access_user_label, parse_access_user_ids

    assert access_user_label({"user_id": 42, "username": "alice"}) == "@alice (id=42)"
    assert access_user_label({"user_id": 42, "first_name": "Alice", "last_name": "Doe"}) == "Alice Doe (id=42)"
    assert parse_access_user_ids(["1,2", "bad", "2", "3"]) == [1, 2, 3]


def test_access_runtime_approve_command_creates_workspace_and_notifies():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import request_user_access, user_root

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(
            drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )
        sent = []
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
        )

        handled = runtime.handle_command("/approve 123", chat_id=1, admin_user_id=1)

        assert handled is True
        assert (user_root(drive_root, 123) / "logs" / "chat.jsonl").exists()
        assert any(args[0] == 456 and "Доступ к боту предоставлен" in args[1] for args, _kwargs in sent)
        assert any(args[0] == 1 and "Доступ: согласовано 1." in args[1] for args, _kwargs in sent)


def test_access_runtime_request_notification_marks_pending():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import list_user_records, request_user_access

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, _created, _notify = request_user_access(
            drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )
        sent = []
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
        )

        assert runtime.send_request_to_admins(rec) == 1
        runtime.mark_request_notified(123)

        updated = list_user_records(drive_root, access_status="pending")[0]
        assert updated["access_admin_notified_at"]
        assert "Запрос доступа" in sent[0][0][1]


def test_access_runtime_request_notification_uses_inline_keyboard():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import list_user_records, request_user_access

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, _created, _notify = request_user_access(
            drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )
        tg = FakeAccessTG()
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            tg=tg,
            log_chat_fn=lambda *args, **kwargs: None,
            append_jsonl_fn=lambda *args, **kwargs: None,
        )

        assert runtime.send_request_to_admins(rec) == 1

        assert tg.sent_markup[0][0] == 1
        keyboard = tg.sent_markup[0][2]["inline_keyboard"]
        callbacks = {button["callback_data"] for row in keyboard for button in row}
        assert "access:approve:123" in callbacks
        assert "access:deny:123" in callbacks
        updated = list_user_records(drive_root, access_status="pending")[0]
        assert updated["access_admin_notified_at"]
        assert updated["access_notifications"][0]["message_id"] == 77


def test_access_runtime_retries_legacy_notified_request_without_inline_message():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import mark_access_request_notified, request_user_access

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(drive_root, user_id=123, chat_id=456, from_user={"username": "alice"})
        mark_access_request_notified(drive_root, 123)
        tg = FakeAccessTG()
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            tg=tg,
        )

        assert runtime.notify_unnotified_requests() == 1
        assert tg.sent_markup


def test_access_runtime_callback_approves_user_and_notifies():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import ACCESS_APPROVED, list_user_records, request_user_access, user_root

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(
            drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )
        tg = FakeAccessTG()
        sent = []
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
            tg=tg,
            is_admin_user_fn=lambda user_id, st: user_id == 1,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": "access:approve:123",
            "from": {"id": 1},
            "message": {"chat": {"id": 1}, "message_id": 77},
        })

        assert handled is True
        approved = list_user_records(drive_root, access_status=ACCESS_APPROVED)
        assert [rec["user_id"] for rec in approved] == [123]
        assert (user_root(drive_root, 123) / "logs" / "chat.jsonl").exists()
        assert any(args[0] == 456 and "Доступ к боту предоставлен" in args[1] for args, _kwargs in sent)
        assert tg.callbacks and "Доступ предоставлен" in tg.callbacks[-1][1]
        assert any("Доступ предоставлен" in edit[2] for edit in tg.edits)


def test_admin_menu_command_sends_user_and_group_sections():
    from supervisor.access_control import AccessRuntime
    from supervisor.teams import request_team_chat
    from supervisor.users import request_user_access

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(drive_root, user_id=123, chat_id=456, from_user={"username": "alice"})
        request_team_chat(
            drive_root,
            {"id": -100123, "type": "supergroup", "title": "Project X"},
            requested_by={"id": 1},
        )
        tg = FakeAccessTG()
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            tg=tg,
        )

        assert runtime.handle_command("/admin", chat_id=1, admin_user_id=1) is True

        assert tg.sent_markup
        assert "Админ-панель" in tg.sent_markup[0][1]
        keyboard = tg.sent_markup[0][2]["inline_keyboard"]
        callbacks = {button["callback_data"] for row in keyboard for button in row}
        assert "admin:users:pending" in callbacks
        assert "admin:groups:pending" in callbacks


def test_admin_user_list_is_numbered_and_keyboard_is_compact():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import ACCESS_APPROVED, request_user_access, set_user_access_status

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(
            drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice", "first_name": "Alice", "last_name": "Doe"},
        )
        set_user_access_status(drive_root, [123], ACCESS_APPROVED, decided_by=1)
        _write_jsonl(drive_root / "logs" / "chat.jsonl", [
            {"direction": "in", "user_id": 123, "text": "/start"},
            {"direction": "out", "user_id": 123, "text": "ok"},
            {"direction": "in", "user_id": 999, "text": "other"},
        ])
        _write_jsonl(drive_root / "users" / "123" / "logs" / "chat.jsonl", [
            {"direction": "in", "user_id": 123, "text": "hello"},
            {"direction": "in", "user_id": 123, "text": ""},
        ])
        _write_jsonl(drive_root / "users" / "123" / "archive" / "chat_20260601_000000.jsonl", [
            {"direction": "in", "user_id": 123, "text": "old request"},
        ])
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
        )

        text = runtime.format_records(ACCESS_APPROVED)
        keyboard = runtime.user_records_keyboard(ACCESS_APPROVED)
        button_texts = [button["text"] for row in keyboard["inline_keyboard"] for button in row]

        assert "1. Alice Doe" in text
        assert "@alice · запросов: 3" in text
        assert "   доступ:" in text
        assert "requested=" not in text
        assert "T" not in text
        assert "⛔️ 1" in button_texts
        assert all(len(label) <= 16 for label in button_texts)


def test_access_runtime_list_callback_refreshes_same_view():
    from supervisor.access_control import AccessRuntime
    from supervisor.users import ACCESS_APPROVED, list_user_records, request_user_access, set_user_access_status

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        request_user_access(drive_root, user_id=123, chat_id=456, from_user={"username": "alice"})
        set_user_access_status(drive_root, [123], ACCESS_APPROVED, decided_by=1)
        tg = FakeAccessTG()
        sent = []
        runtime = AccessRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
            tg=tg,
            is_admin_user_fn=lambda user_id, st: user_id == 1,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": "access:deny:123:users:approved",
            "from": {"id": 1},
            "message": {"chat": {"id": 1}, "message_id": 77},
        })

        assert handled is True
        assert list_user_records(drive_root, access_status=ACCESS_APPROVED) == []
        assert tg.edits
        assert "Пользователи с доступом: пусто." in tg.edits[-1][2]
