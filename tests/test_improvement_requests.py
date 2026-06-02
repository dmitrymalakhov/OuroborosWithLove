import pathlib
import tempfile
import types


class FakeTG:
    def __init__(self):
        self.sent_markup = []
        self.edits = []
        self.callbacks = []
        self.next_message_id = 100

    def send_message_with_markup(self, chat_id, text, reply_markup, parse_mode=""):
        self.sent_markup.append((chat_id, text, reply_markup, parse_mode))
        self.next_message_id += 1
        return True, "ok", self.next_message_id

    def edit_message_text(self, chat_id, message_id, text, reply_markup=None, parse_mode=""):
        self.edits.append((chat_id, message_id, text, reply_markup, parse_mode))
        return True, "ok"

    def answer_callback_query(self, callback_query_id, text="", show_alert=False):
        self.callbacks.append((callback_query_id, text, show_alert))
        return True, "ok"


def _payload():
    return {
        "reason": "Не смог проверить внешний сервис",
        "summary": "Пользователь хотел проверить статус интеграции",
        "missing_requirements": "Не хватило доступа к внешнему сервису",
        "attempted_steps": "Проверил доступные локальные данные",
        "suggested_creator_action": "Добавить безопасный диагностический tool",
        "task_id": "task1",
        "user_id": 123,
        "chat_id": 456,
        "chat_type": "private",
    }


def test_improvement_request_draft_is_not_admin_visible_and_keyboard_vertical():
    from supervisor.unresolved_tasks import (
        STATUS_DRAFT,
        create_draft_report,
        list_reports,
        user_offer_keyboard,
    )

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, created = create_draft_report(drive_root, _payload())

        assert created is True
        assert rec["status"] == STATUS_DRAFT
        assert list_reports(drive_root) == []

        keyboard = user_offer_keyboard(rec["id"])
        rows = keyboard["inline_keyboard"]
        assert len(rows) == 2
        assert rows[0][0]["text"] == "Запросить доработку"
        assert rows[1][0]["text"] == "Не нужно"


def test_user_confirmation_opens_report_and_notifies_admin():
    from supervisor.unresolved_tasks import ImprovementRequestRuntime, STATUS_OPEN, create_draft_report, get_report

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, _created = create_draft_report(drive_root, _payload())
        tg = FakeTG()
        sent = []
        runtime = ImprovementRequestRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
            tg=tg,
            is_admin_user_fn=lambda user_id, st: user_id == 1,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": f"improvement:send:{rec['id']}",
            "from": {"id": 123},
            "message": {"chat": {"id": 456}, "message_id": 77},
        })

        assert handled is True
        updated = get_report(drive_root, rec["id"])
        assert updated["status"] == STATUS_OPEN
        assert len(tg.sent_markup) == 1
        assert tg.sent_markup[0][0] == 1
        assert "Запрос доработки бота" in tg.sent_markup[0][1]
        admin_rows = tg.sent_markup[0][2]["inline_keyboard"]
        assert [row[0]["text"] for row in admin_rows] == ["Взять", "Закрыть", "Отклонить", "Обновить"]
        assert tg.edits and "отправлен" in tg.edits[-1][2]


def test_user_dismissal_does_not_notify_admin():
    from supervisor.unresolved_tasks import ImprovementRequestRuntime, STATUS_DISMISSED, create_draft_report, get_report

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, _created = create_draft_report(drive_root, _payload())
        tg = FakeTG()
        runtime = ImprovementRequestRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            tg=tg,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": f"improvement:dismiss:{rec['id']}",
            "from": {"id": 123},
            "message": {"chat": {"id": 456}, "message_id": 77},
        })

        assert handled is True
        updated = get_report(drive_root, rec["id"])
        assert updated["status"] == STATUS_DISMISSED
        assert tg.sent_markup == []
        assert tg.edits and "не отправляю" in tg.edits[-1][2]


def test_admin_unresolved_callback_rejects_non_admin():
    from supervisor.unresolved_tasks import ImprovementRequestRuntime, create_draft_report, submit_report

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        rec, _created = create_draft_report(drive_root, _payload())
        submit_report(drive_root, rec["id"], submitted_by=123)
        tg = FakeTG()
        runtime = ImprovementRequestRuntime(
            drive_root=drive_root,
            admin_chat_ids_fn=lambda: [1],
            load_state_fn=lambda: {"owner_id": 1},
            send_with_budget_fn=lambda *args, **kwargs: None,
            tg=tg,
            is_admin_user_fn=lambda user_id, st: False,
        )

        handled = runtime.handle_callback({
            "id": "cb1",
            "data": f"unresolved:claim:{rec['id']}",
            "from": {"id": 999},
            "message": {"chat": {"id": 1}, "message_id": 77},
        })

        assert handled is True
        assert tg.callbacks[-1][2] is True
        assert "Только админ" in tg.callbacks[-1][1]


def test_offer_event_creates_draft_and_sends_user_buttons():
    from supervisor.events import dispatch_event
    from supervisor.unresolved_tasks import get_report

    with tempfile.TemporaryDirectory() as tmp:
        drive_root = pathlib.Path(tmp)
        tg = FakeTG()
        ctx = types.SimpleNamespace(
            DRIVE_ROOT=drive_root,
            TG=tg,
            append_jsonl=lambda *args, **kwargs: None,
        )

        dispatch_event({
            "type": "offer_improvement_request",
            **_payload(),
            "drive_root": str(drive_root / "users" / "123"),
            "shared_drive_root": str(drive_root),
        }, ctx)

        assert len(tg.sent_markup) == 1
        assert tg.sent_markup[0][0] == 456
        rows = tg.sent_markup[0][2]["inline_keyboard"]
        assert [row[0]["text"] for row in rows] == ["Запросить доработку", "Не нужно"]
        callback_data = rows[0][0]["callback_data"]
        report_id = callback_data.split(":")[-1]
        assert get_report(drive_root, report_id)["status"] == "draft"
