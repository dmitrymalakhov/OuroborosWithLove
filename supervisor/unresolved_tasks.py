"""Supervisor improvement-request storage and Telegram UI.

The agent can offer a user to request a creator-side improvement when it
believes it did not solve the task. Reports remain drafts until the user
explicitly presses the confirmation button.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from supervisor.state import acquire_file_lock, atomic_write_text, release_file_lock


DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
REPORTS_PATH: pathlib.Path = DRIVE_ROOT / "state" / "unresolved_tasks.json"
REPORTS_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "unresolved_tasks.lock"

STATUS_DRAFT = "draft"
STATUS_OPEN = "open"
STATUS_IN_PROGRESS = "in_progress"
STATUS_RESOLVED = "resolved"
STATUS_DISMISSED = "dismissed"
REPORT_STATUSES = {
    STATUS_DRAFT,
    STATUS_OPEN,
    STATUS_IN_PROGRESS,
    STATUS_RESOLVED,
    STATUS_DISMISSED,
}


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, REPORTS_PATH, REPORTS_LOCK_PATH
    DRIVE_ROOT = drive_root
    REPORTS_PATH = drive_root / "state" / "unresolved_tasks.json"
    REPORTS_LOCK_PATH = drive_root / "locks" / "unresolved_tasks.lock"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_status(value: Any, default: str = STATUS_DRAFT) -> str:
    status = str(value or "").strip().lower()
    return status if status in REPORT_STATUSES else default


def _load_unlocked() -> Dict[str, Any]:
    try:
        if not REPORTS_PATH.exists():
            return {"reports": {}}
        data = json.loads(REPORTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"reports": {}}
        if not isinstance(data.get("reports"), dict):
            data["reports"] = {}
        return data
    except Exception:
        return {"reports": {}}


def _save_unlocked(data: Dict[str, Any]) -> None:
    data.setdefault("reports", {})
    data["updated_at"] = _now_iso()
    atomic_write_text(REPORTS_PATH, json.dumps(data, ensure_ascii=False, indent=2))


def _new_report_id() -> str:
    return uuid.uuid4().hex[:10]


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _text(value: Any, limit: int = 2000) -> str:
    return str(value or "").strip()[:limit]


def _find_existing_report(reports: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_id = str(payload.get("task_id") or "").strip()
    user_id = _as_int_or_none(payload.get("user_id"))
    if not task_id:
        return None
    for rec in reports.values():
        if not isinstance(rec, dict):
            continue
        if _normalize_status(rec.get("status")) not in (STATUS_DRAFT, STATUS_OPEN, STATUS_IN_PROGRESS):
            continue
        if str(rec.get("task_id") or "") != task_id:
            continue
        if user_id is not None and _as_int_or_none(rec.get("user_id")) != user_id:
            continue
        return rec
    return None


def _apply_report_payload(rec: Dict[str, Any], payload: Dict[str, Any]) -> None:
    for key in (
        "reason",
        "summary",
        "missing_requirements",
        "attempted_steps",
        "suggested_creator_action",
        "bot_response_preview",
    ):
        if key in payload:
            rec[key] = _text(payload.get(key), limit=4000 if key == "bot_response_preview" else 2000)

    for key in ("task_id", "chat_type", "team_slug"):
        if key in payload:
            rec[key] = str(payload.get(key) or "")

    for key in ("user_id", "chat_id", "team_chat_id"):
        if key in payload:
            rec[key] = _as_int_or_none(payload.get(key))

    for key in ("drive_root", "shared_drive_root"):
        if key in payload:
            rec[key] = str(payload.get(key) or "")

    if "is_team_workspace" in payload:
        rec["is_team_workspace"] = bool(payload.get("is_team_workspace"))

    rec["updated_at"] = _now_iso()


def create_draft_report(drive_root: pathlib.Path, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Create or update a draft report. Returns (record, created)."""
    init(drive_root)
    lock_fd = acquire_file_lock(REPORTS_LOCK_PATH)
    try:
        data = _load_unlocked()
        reports = data.setdefault("reports", {})
        existing = _find_existing_report(reports, payload)
        created = existing is None
        if existing is None:
            report_id = _new_report_id()
            existing = {
                "id": report_id,
                "status": STATUS_DRAFT,
                "source": "agent_offer",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "offer_notifications": [],
                "admin_notifications": [],
            }
            reports[report_id] = existing
        _apply_report_payload(existing, payload)
        _save_unlocked(data)
        return dict(existing), created
    finally:
        release_file_lock(REPORTS_LOCK_PATH, lock_fd)


def get_report(drive_root: pathlib.Path, report_id: str) -> Optional[Dict[str, Any]]:
    init(drive_root)
    lock_fd = acquire_file_lock(REPORTS_LOCK_PATH)
    try:
        rec = _load_unlocked().setdefault("reports", {}).get(str(report_id or ""))
        return dict(rec) if isinstance(rec, dict) else None
    finally:
        release_file_lock(REPORTS_LOCK_PATH, lock_fd)


def list_reports(
    drive_root: pathlib.Path,
    *,
    status: Optional[str] = None,
    include_drafts: bool = False,
) -> List[Dict[str, Any]]:
    init(drive_root)
    target = _normalize_status(status, default="") if status else None
    lock_fd = acquire_file_lock(REPORTS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rows: List[Dict[str, Any]] = []
        changed = False
        for rec in data.setdefault("reports", {}).values():
            if not isinstance(rec, dict):
                continue
            normalized = _normalize_status(rec.get("status"))
            if normalized != rec.get("status"):
                rec["status"] = normalized
                changed = True
            if not include_drafts and normalized == STATUS_DRAFT:
                continue
            if target and normalized != target:
                continue
            rows.append(dict(rec))
        if changed:
            _save_unlocked(data)
    finally:
        release_file_lock(REPORTS_LOCK_PATH, lock_fd)
    rows.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return rows


def report_counts(drive_root: pathlib.Path) -> Dict[str, int]:
    counts = {status: 0 for status in REPORT_STATUSES}
    for rec in list_reports(drive_root, include_drafts=True):
        status = _normalize_status(rec.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def append_offer_notification(
    drive_root: pathlib.Path,
    report_id: str,
    *,
    chat_id: int,
    message_id: int,
) -> None:
    _append_notification(drive_root, report_id, "offer_notifications", chat_id, message_id)


def append_admin_notification(
    drive_root: pathlib.Path,
    report_id: str,
    *,
    chat_id: int,
    message_id: int,
) -> None:
    _append_notification(drive_root, report_id, "admin_notifications", chat_id, message_id)


def _append_notification(
    drive_root: pathlib.Path,
    report_id: str,
    field: str,
    chat_id: int,
    message_id: int,
) -> None:
    init(drive_root)
    lock_fd = acquire_file_lock(REPORTS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rec = data.setdefault("reports", {}).get(str(report_id or ""))
        if not isinstance(rec, dict):
            return
        rec.setdefault(field, []).append({
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "ts": _now_iso(),
        })
        rec["updated_at"] = _now_iso()
        _save_unlocked(data)
    finally:
        release_file_lock(REPORTS_LOCK_PATH, lock_fd)


def submit_report(drive_root: pathlib.Path, report_id: str, submitted_by: int) -> Tuple[bool, str, Dict[str, Any]]:
    return _set_report_status(
        drive_root,
        report_id,
        STATUS_OPEN,
        actor_user_id=submitted_by,
        field_prefix="submitted",
        ok_message="Запрос на доработку отправлен.",
    )


def dismiss_user_report(drive_root: pathlib.Path, report_id: str, dismissed_by: int) -> Tuple[bool, str, Dict[str, Any]]:
    return _set_report_status(
        drive_root,
        report_id,
        STATUS_DISMISSED,
        actor_user_id=dismissed_by,
        field_prefix="dismissed",
        ok_message="Ок, не отправляю запрос на доработку.",
    )


def set_admin_status(
    drive_root: pathlib.Path,
    report_id: str,
    status: str,
    actor_user_id: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    normalized = _normalize_status(status, default="")
    if normalized not in (STATUS_IN_PROGRESS, STATUS_RESOLVED, STATUS_DISMISSED, STATUS_OPEN):
        return False, "Некорректный статус.", {}
    prefix = {
        STATUS_IN_PROGRESS: "claimed",
        STATUS_RESOLVED: "resolved",
        STATUS_DISMISSED: "dismissed",
        STATUS_OPEN: "reopened",
    }.get(normalized, "updated")
    return _set_report_status(
        drive_root,
        report_id,
        normalized,
        actor_user_id=actor_user_id,
        field_prefix=prefix,
        ok_message=f"Статус обновлён: {normalized}.",
    )


def _set_report_status(
    drive_root: pathlib.Path,
    report_id: str,
    status: str,
    *,
    actor_user_id: int,
    field_prefix: str,
    ok_message: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    init(drive_root)
    lock_fd = acquire_file_lock(REPORTS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rec = data.setdefault("reports", {}).get(str(report_id or ""))
        if not isinstance(rec, dict):
            return False, "Запрос не найден.", {}
        rec["status"] = _normalize_status(status)
        rec[f"{field_prefix}_by"] = int(actor_user_id or 0)
        rec[f"{field_prefix}_at"] = _now_iso()
        rec["updated_at"] = _now_iso()
        _save_unlocked(data)
        return True, ok_message, dict(rec)
    finally:
        release_file_lock(REPORTS_LOCK_PATH, lock_fd)


def user_offer_keyboard(report_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": "Запросить доработку", "callback_data": f"improvement:send:{report_id}"}],
            [{"text": "Не нужно", "callback_data": f"improvement:dismiss:{report_id}"}],
        ]
    }


def admin_report_keyboard(report_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": "Взять", "callback_data": f"unresolved:claim:{report_id}"}],
            [{"text": "Закрыть", "callback_data": f"unresolved:resolve:{report_id}"}],
            [{"text": "Отклонить", "callback_data": f"unresolved:dismiss:{report_id}"}],
            [{"text": "Обновить", "callback_data": f"unresolved:view:{report_id}"}],
        ]
    }


def report_list_keyboard(status: str = STATUS_OPEN, limit: int = 10) -> Dict[str, Any]:
    rows: List[List[Dict[str, str]]] = []
    for rec in list_reports(DRIVE_ROOT, status=status, include_drafts=status == STATUS_DRAFT)[:limit]:
        report_id = str(rec.get("id") or "")
        label = (str(rec.get("summary") or rec.get("reason") or report_id).replace("\n", " ")[:32])
        rows.append([{"text": label or report_id, "callback_data": f"unresolved:view:{report_id}"}])
    rows.append([{"text": "Обновить", "callback_data": f"unresolved:list:{status}"}])
    return {"inline_keyboard": rows}


def format_user_offer_text(rec: Dict[str, Any]) -> str:
    summary = str(rec.get("summary") or rec.get("reason") or "задача решена не полностью").strip()
    missing = str(rec.get("missing_requirements") or "").strip()
    lines = [
        "Похоже, мне не хватило возможностей, чтобы полностью решить задачу.",
        "",
        f"Проблема: {summary}",
    ]
    if missing:
        lines.append(f"Чего не хватило: {missing}")
    lines.extend([
        "",
        "Можно отправить создателю запрос на доработку бота.",
    ])
    return "\n".join(lines)


def format_admin_report_text(rec: Dict[str, Any]) -> str:
    report_id = str(rec.get("id") or "")
    status = _normalize_status(rec.get("status"))
    lines = [
        "Запрос доработки бота",
        f"id: {report_id}",
        f"status: {status}",
        "",
        f"Пользователь хотел: {rec.get('summary') or '-'}",
        f"Что не получилось: {rec.get('reason') or '-'}",
        f"Чего не хватило: {rec.get('missing_requirements') or '-'}",
        f"Что бот попробовал: {rec.get('attempted_steps') or '-'}",
        f"Что доработать: {rec.get('suggested_creator_action') or '-'}",
        "",
        f"task_id: {rec.get('task_id') or '-'}",
        f"user_id: {rec.get('user_id') or '-'}",
        f"chat_id: {rec.get('chat_id') or '-'}",
        f"chat_type: {rec.get('chat_type') or '-'}",
        f"created_at: {str(rec.get('created_at') or '-')[:19]}",
    ]
    return "\n".join(lines)


def format_report_list(drive_root: pathlib.Path, status: str = STATUS_OPEN, limit: int = 10) -> str:
    rows = list_reports(drive_root, status=status, include_drafts=status == STATUS_DRAFT)[:limit]
    title = {
        STATUS_OPEN: "Открытые запросы доработки",
        STATUS_IN_PROGRESS: "Запросы в работе",
        STATUS_RESOLVED: "Закрытые запросы",
        STATUS_DISMISSED: "Отклонённые запросы",
        STATUS_DRAFT: "Черновики запросов",
    }.get(status, "Запросы доработки")
    if not rows:
        return f"{title}: пусто."
    lines = [f"{title}: {len(rows)}", ""]
    for idx, rec in enumerate(rows, start=1):
        summary = str(rec.get("summary") or rec.get("reason") or "").replace("\n", " ")[:80]
        lines.append(f"{idx}. {rec.get('id')} · user {rec.get('user_id') or '-'}")
        lines.append(f"   {summary or '-'}")
    return "\n".join(lines)


@dataclass
class ImprovementRequestRuntime:
    drive_root: pathlib.Path
    admin_chat_ids_fn: Callable[[], List[int]]
    load_state_fn: Callable[[], Dict[str, Any]]
    send_with_budget_fn: Callable[..., Any]
    tg: Any = None
    is_admin_user_fn: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    log_chat_fn: Optional[Callable[..., Any]] = None
    append_jsonl_fn: Optional[Callable[..., Any]] = None

    def _answer_callback(self, callback_query: Dict[str, Any], text: str = "", show_alert: bool = False) -> None:
        callback_id = str(callback_query.get("id") or "")
        if callback_id and self.tg is not None:
            self.tg.answer_callback_query(callback_id, text, show_alert=show_alert)

    def _callback_user_id(self, callback_query: Dict[str, Any]) -> int:
        from_user = callback_query.get("from") or {}
        return int(from_user.get("id") or 0)

    def _callback_chat_id(self, callback_query: Dict[str, Any]) -> int:
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        return int(chat.get("id") or 0)

    def _ensure_admin(self, callback_query: Dict[str, Any]) -> Tuple[bool, int]:
        user_id = self._callback_user_id(callback_query)
        if self.is_admin_user_fn is None:
            return True, user_id
        if self.is_admin_user_fn(user_id, self.load_state_fn()):
            return True, user_id
        self._answer_callback(callback_query, "Только админ может управлять запросами.", show_alert=True)
        return False, user_id

    def _ensure_report_owner(self, callback_query: Dict[str, Any], rec: Dict[str, Any]) -> bool:
        callback_user_id = self._callback_user_id(callback_query)
        callback_chat_id = self._callback_chat_id(callback_query)
        report_chat_id = int(rec.get("chat_id") or 0)
        report_user_id = int(rec.get("user_id") or 0)
        chat_type = str(rec.get("chat_type") or "private").lower()
        if chat_type == "private" and report_user_id and callback_user_id != report_user_id:
            self._answer_callback(callback_query, "Этот запрос может отправить только автор.", show_alert=True)
            return False
        if report_chat_id and callback_chat_id and callback_chat_id != report_chat_id:
            self._answer_callback(callback_query, "Этот запрос относится к другому чату.", show_alert=True)
            return False
        return True

    def _edit_callback_message(self, callback_query: Dict[str, Any], text: str, reply_markup: Optional[Dict[str, Any]] = None) -> None:
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = int(chat.get("id") or 0)
        message_id = int(message.get("message_id") or 0)
        if self.tg is not None and chat_id and message_id:
            ok, _err = self.tg.edit_message_text(chat_id, message_id, text, reply_markup=reply_markup)
            if ok:
                return
        if chat_id:
            self.send_with_budget_fn(chat_id, text, force_budget=True)

    def _send_markup_or_text(self, chat_id: int, text: str, reply_markup: Dict[str, Any]) -> Tuple[bool, int]:
        if self.tg is not None:
            ok, _err, message_id = self.tg.send_message_with_markup(chat_id, text, reply_markup)
            if ok:
                self._log_admin_out(chat_id, text)
                return True, int(message_id or 0)
        self.send_with_budget_fn(chat_id, text, force_budget=True)
        return True, 0

    def _log_admin_out(self, chat_id: int, text: str) -> None:
        if not self.log_chat_fn:
            return
        owner_id = int(self.load_state_fn().get("owner_id") or 0)
        self.log_chat_fn("out", chat_id, owner_id, text, drive_root=self.drive_root)

    def notify_admins(self, rec: Dict[str, Any]) -> int:
        sent = 0
        text = format_admin_report_text(rec)
        keyboard = admin_report_keyboard(str(rec.get("id") or ""))
        for admin_chat_id in self.admin_chat_ids_fn():
            ok, message_id = self._send_markup_or_text(admin_chat_id, text, keyboard)
            if ok:
                sent += 1
                if message_id:
                    append_admin_notification(
                        self.drive_root,
                        str(rec.get("id") or ""),
                        chat_id=admin_chat_id,
                        message_id=message_id,
                    )
        return sent

    def handle_command(self, text: str, chat_id: int, admin_user_id: int) -> bool:
        parts = text.strip().split()
        if not parts:
            return False
        command = parts[0].lower().split("@", 1)[0]
        if command != "/unresolved":
            return False
        status = STATUS_OPEN
        if len(parts) > 1:
            status = {
                "open": STATUS_OPEN,
                "active": STATUS_OPEN,
                "work": STATUS_IN_PROGRESS,
                "in_progress": STATUS_IN_PROGRESS,
                "resolved": STATUS_RESOLVED,
                "closed": STATUS_RESOLVED,
                "dismissed": STATUS_DISMISSED,
            }.get(parts[1].lower(), STATUS_OPEN)
        self._send_markup_or_text(
            chat_id,
            format_report_list(self.drive_root, status=status),
            report_list_keyboard(status=status),
        )
        return True

    def handle_callback(self, callback_query: Dict[str, Any]) -> bool:
        data = str(callback_query.get("data") or "")
        if not (data.startswith("improvement:") or data.startswith("unresolved:")):
            return False

        if data.startswith("improvement:"):
            return self._handle_user_callback(callback_query, data)
        return self._handle_admin_callback(callback_query, data)

    def _handle_user_callback(self, callback_query: Dict[str, Any], data: str) -> bool:
        parts = data.split(":")
        if len(parts) != 3 or parts[1] not in ("send", "dismiss"):
            self._answer_callback(callback_query, "Некорректная кнопка.", show_alert=True)
            return True
        rec = get_report(self.drive_root, parts[2])
        if not rec:
            self._answer_callback(callback_query, "Запрос не найден.", show_alert=True)
            return True
        if not self._ensure_report_owner(callback_query, rec):
            return True
        if parts[1] == "send":
            ok, msg, updated = submit_report(self.drive_root, parts[2], self._callback_user_id(callback_query))
            self._answer_callback(callback_query, msg, show_alert=not ok)
            if ok:
                self.notify_admins(updated)
            self._edit_callback_message(callback_query, msg)
            return True
        ok, msg, _updated = dismiss_user_report(self.drive_root, parts[2], self._callback_user_id(callback_query))
        self._answer_callback(callback_query, msg, show_alert=not ok)
        self._edit_callback_message(callback_query, msg)
        return True

    def _handle_admin_callback(self, callback_query: Dict[str, Any], data: str) -> bool:
        ok_admin, admin_user_id = self._ensure_admin(callback_query)
        if not ok_admin:
            return True
        parts = data.split(":")
        if len(parts) == 3 and parts[1] == "list":
            status = _normalize_status(parts[2], default=STATUS_OPEN)
            self._edit_callback_message(
                callback_query,
                format_report_list(self.drive_root, status=status),
                report_list_keyboard(status=status),
            )
            self._answer_callback(callback_query, "Список обновлён.")
            return True
        if len(parts) != 3 or parts[1] not in ("view", "claim", "resolve", "dismiss"):
            self._answer_callback(callback_query, "Некорректная кнопка.", show_alert=True)
            return True
        report_id = parts[2]
        if parts[1] == "claim":
            ok, msg, rec = set_admin_status(self.drive_root, report_id, STATUS_IN_PROGRESS, admin_user_id)
        elif parts[1] == "resolve":
            ok, msg, rec = set_admin_status(self.drive_root, report_id, STATUS_RESOLVED, admin_user_id)
        elif parts[1] == "dismiss":
            ok, msg, rec = set_admin_status(self.drive_root, report_id, STATUS_DISMISSED, admin_user_id)
        else:
            rec = get_report(self.drive_root, report_id) or {}
            ok, msg = bool(rec), "Обновлено." if rec else "Запрос не найден."
        self._answer_callback(callback_query, msg, show_alert=not ok)
        if ok:
            self._edit_callback_message(callback_query, format_admin_report_text(rec), admin_report_keyboard(report_id))
        return True
