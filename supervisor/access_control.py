"""Supervisor access approval UI and admin command helpers."""

from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from supervisor.teams import (
    TEAM_APPROVED as TEAM_CHAT_APPROVED,
    TEAM_DENIED as TEAM_CHAT_DENIED,
    TEAM_PENDING as TEAM_CHAT_PENDING,
    list_team_chats,
)
from supervisor.unresolved_tasks import STATUS_IN_PROGRESS, STATUS_OPEN, report_counts
from supervisor.users import (
    ACCESS_REQUESTS_LOG,
    ACCESS_APPROVED,
    ACCESS_DENIED,
    ACCESS_PENDING,
    append_access_request_notification,
    ensure_user_workspace,
    list_user_records,
    mark_access_request_notified,
    set_user_access_status,
)

USER_ACTIVITY_VIEW = "last_seen"


def access_user_label(rec: Dict[str, Any]) -> str:
    uid = int(rec.get("user_id") or 0)
    username = str(rec.get("username") or "").strip()
    first = str(rec.get("first_name") or "").strip()
    last = str(rec.get("last_name") or "").strip()
    name = " ".join(part for part in (first, last) if part).strip()
    if username and name:
        return f"{name} (@{username}, id={uid})"
    if username:
        return f"@{username} (id={uid})"
    if name:
        return f"{name} (id={uid})"
    return f"id={uid}"


def parse_access_user_ids(parts: List[str]) -> List[int]:
    ids: List[int] = []
    for part in parts:
        for raw in str(part).replace(",", " ").split():
            try:
                ids.append(int(raw))
            except Exception:
                continue
    seen: set[int] = set()
    unique: List[int] = []
    for uid in ids:
        if uid not in seen:
            unique.append(uid)
            seen.add(uid)
    return unique


def _format_dt(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    try:
        dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%d.%m %H:%M")
    except Exception:
        if "T" in raw:
            date, time = raw.split("T", 1)
            hhmm = time[:5] if len(time) >= 5 else time
            return f"{date} {hhmm}".strip()
        return raw[:16]


def access_request_keyboard(user_id: int) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Разрешить", "callback_data": f"access:approve:{int(user_id)}"},
                {"text": "⛔️ Отказать", "callback_data": f"access:deny:{int(user_id)}"},
            ],
            [{"text": "🛠 Админ-меню", "callback_data": "admin:home"}],
        ]
    }


def _user_name(rec: Dict[str, Any]) -> str:
    first = str(rec.get("first_name") or "").strip()
    last = str(rec.get("last_name") or "").strip()
    name = " ".join(part for part in (first, last) if part).strip()
    username = str(rec.get("username") or "").strip()
    if name:
        return name
    if username:
        return f"@{username}"
    return f"id {int(rec.get('user_id') or 0)}"


def _user_list_identity(rec: Dict[str, Any]) -> str:
    username = str(rec.get("username") or "").strip()
    if username:
        return f"@{username}"
    return f"id {int(rec.get('user_id') or 0)}"


def _user_log_roots(drive_root: pathlib.Path, rec: Dict[str, Any]) -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    raw_root = str(rec.get("drive_root") or "").strip()
    if raw_root:
        roots.append(pathlib.Path(raw_root))
    roots.append(drive_root)

    unique: List[pathlib.Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key and key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _chat_log_paths(root: pathlib.Path) -> List[pathlib.Path]:
    paths = [root / "logs" / "chat.jsonl"]
    archive_dir = root / "archive"
    try:
        paths.extend(sorted(archive_dir.glob("chat_*.jsonl")))
    except Exception:
        pass
    return paths


def _access_request_log_paths(drive_root: pathlib.Path) -> List[pathlib.Path]:
    return [drive_root / "logs" / ACCESS_REQUESTS_LOG]


def _count_user_requests_in_log(path: pathlib.Path, user_ids: set[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    try:
        if not path.exists():
            return counts
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("direction") != "in":
                    continue
                try:
                    row_user_id = int(row.get("user_id") or 0)
                except Exception:
                    continue
                if row_user_id not in user_ids:
                    continue
                if str(row.get("text") or "").strip():
                    counts[row_user_id] = counts.get(row_user_id, 0) + 1
    except Exception:
        return {}
    return counts


def _user_request_counts(drive_root: pathlib.Path, records: List[Dict[str, Any]]) -> Dict[int, int]:
    user_ids: set[int] = set()
    for rec in records:
        try:
            uid = int(rec.get("user_id") or 0)
        except Exception:
            continue
        if uid:
            user_ids.add(uid)
    counts = {uid: 0 for uid in user_ids}
    if not user_ids:
        return counts

    paths: List[pathlib.Path] = []
    seen_paths: set[str] = set()
    for rec in records:
        for root in _user_log_roots(drive_root, rec):
            for path in _chat_log_paths(root):
                key = str(path)
                if key not in seen_paths:
                    paths.append(path)
                    seen_paths.add(key)
    for path in _access_request_log_paths(drive_root):
        key = str(path)
        if key not in seen_paths:
            paths.append(path)
            seen_paths.add(key)

    for path in paths:
        for uid, count in _count_user_requests_in_log(path, user_ids).items():
            counts[uid] = counts.get(uid, 0) + count
    return counts


def _user_detail_line(rec: Dict[str, Any], status: str) -> str:
    username = str(rec.get("username") or "").strip()
    uid = int(rec.get("user_id") or 0)
    when = {
        ACCESS_PENDING: rec.get("access_last_requested_at") or rec.get("access_requested_at"),
        ACCESS_APPROVED: rec.get("access_approved_at") or rec.get("created_at"),
        ACCESS_DENIED: rec.get("access_denied_at") or rec.get("created_at"),
    }.get(status) or rec.get("created_at")
    date_label = {
        ACCESS_PENDING: "запрос",
        ACCESS_APPROVED: "доступ",
        ACCESS_DENIED: "отключён",
    }.get(status, "дата")
    parts = []
    if username:
        parts.append(f"@{username}")
    parts.append(f"id {uid}")
    parts.append(f"{date_label}: {_format_dt(when)}")
    return " · ".join(parts)


def _user_list_detail_lines(rec: Dict[str, Any], status: str, request_count: int) -> List[str]:
    when = {
        ACCESS_PENDING: rec.get("access_last_requested_at") or rec.get("access_requested_at"),
        ACCESS_APPROVED: rec.get("access_approved_at") or rec.get("created_at"),
        ACCESS_DENIED: rec.get("access_denied_at") or rec.get("created_at"),
    }.get(status) or rec.get("created_at")
    date_label = {
        ACCESS_PENDING: "запрос",
        ACCESS_APPROVED: "доступ",
        ACCESS_DENIED: "отключён",
    }.get(status, "дата")
    return [
        f"   {_user_list_identity(rec)} · запросов: {request_count}",
        f"   {date_label}: {_format_dt(when)}",
    ]


def _user_access_status_label(rec: Dict[str, Any]) -> str:
    status = str(rec.get("access_status") or "").strip().lower()
    return {
        ACCESS_PENDING: "заявка",
        ACCESS_APPROVED: "доступ",
        ACCESS_DENIED: "отключён",
    }.get(status, status or "-")


def _user_activity_sort_key(rec: Dict[str, Any]) -> Tuple[str, int]:
    return (
        str(rec.get("last_seen_at") or rec.get("created_at") or ""),
        int(rec.get("user_id") or 0),
    )


def _sort_users_by_activity(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=_user_activity_sort_key, reverse=True)


def _user_activity_detail_lines(rec: Dict[str, Any], request_count: int) -> List[str]:
    when = rec.get("last_seen_at") or rec.get("created_at")
    return [
        f"   {_user_list_identity(rec)} · статус: {_user_access_status_label(rec)} · запросов: {request_count}",
        f"   последний раз: {_format_dt(when)}",
    ]


def _team_name(rec: Dict[str, Any]) -> str:
    return str(rec.get("title") or "").strip() or f"chat {int(rec.get('chat_id') or 0)}"


def _team_detail_line(rec: Dict[str, Any], status: str) -> str:
    chat_id = int(rec.get("chat_id") or 0)
    when = {
        TEAM_CHAT_PENDING: rec.get("requested_at") or rec.get("created_at"),
        TEAM_CHAT_APPROVED: rec.get("approved_at") or rec.get("created_at"),
        TEAM_CHAT_DENIED: rec.get("denied_at") or rec.get("created_at"),
    }.get(status) or rec.get("created_at")
    date_label = {
        TEAM_CHAT_PENDING: "запрос",
        TEAM_CHAT_APPROVED: "доступ",
        TEAM_CHAT_DENIED: "отключена",
    }.get(status, "дата")
    return f"chat {chat_id} · {date_label}: {_format_dt(when)}"


def access_decision_text(rec: Dict[str, Any]) -> str:
    status = str(rec.get("access_status") or ACCESS_PENDING)
    decided_by = rec.get("access_decided_by")
    if status == ACCESS_APPROVED:
        verdict = "✅ Доступ предоставлен"
    elif status == ACCESS_DENIED:
        verdict = "⛔️ Доступ отклонён"
    else:
        verdict = "⏳ Ожидает решения"
    suffix = f"\nРешил admin user_id={decided_by}" if decided_by else ""
    return f"{verdict}: {access_user_label(rec)}{suffix}"


def collect_admin_chat_ids(drive_root: pathlib.Path, load_state_fn: Callable[[], Dict[str, Any]]) -> List[int]:
    state = load_state_fn()
    chat_ids: List[int] = []
    owner_chat_id = state.get("owner_chat_id")
    if owner_chat_id:
        chat_ids.append(int(owner_chat_id))
    for rec in list_user_records(drive_root, role="admin"):
        chat_id = rec.get("chat_id")
        if chat_id:
            chat_ids.append(int(chat_id))

    seen: set[int] = set()
    unique: List[int] = []
    for chat_id in chat_ids:
        if chat_id not in seen:
            unique.append(chat_id)
            seen.add(chat_id)
    return unique


@dataclass
class AccessRuntime:
    drive_root: pathlib.Path
    admin_chat_ids_fn: Callable[[], List[int]]
    load_state_fn: Callable[[], Dict[str, Any]]
    send_with_budget_fn: Callable[..., Any]
    tg: Any = None
    is_admin_user_fn: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    log_chat_fn: Optional[Callable[..., Any]] = None
    append_jsonl_fn: Optional[Callable[..., Any]] = None

    def _log_admin_out(self, chat_id: int, text: str) -> None:
        if not self.log_chat_fn:
            return
        owner_id = int(self.load_state_fn().get("owner_id") or 0)
        self.log_chat_fn("out", chat_id, owner_id, text, drive_root=self.drive_root)

    def _log_error(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.append_jsonl_fn:
            return
        try:
            import datetime

            row = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": event_type,
            }
            row.update(payload)
            self.append_jsonl_fn(self.drive_root / "logs" / "supervisor.jsonl", row)
        except Exception:
            pass

    def _send_markup_or_text(
        self,
        chat_id: int,
        text: str,
        reply_markup: Dict[str, Any],
        *,
        force_budget: bool = False,
    ) -> Tuple[bool, int]:
        if self.tg is not None:
            ok, err, message_id = self.tg.send_message_with_markup(chat_id, text, reply_markup)
            if ok:
                self._log_admin_out(chat_id, text)
                return True, int(message_id or 0)
            self._log_error("access_admin_markup_send_failed", {
                "chat_id": int(chat_id),
                "error": err,
            })

        self.send_with_budget_fn(chat_id, text, force_budget=force_budget)
        return True, 0

    def _edit_or_send_callback_view(
        self,
        callback_query: Dict[str, Any],
        text: str,
        reply_markup: Dict[str, Any],
    ) -> None:
        message = callback_query.get("message") or {}
        message_chat = message.get("chat") or {}
        message_chat_id = int(message_chat.get("id") or 0)
        message_id = int(message.get("message_id") or 0)
        if self.tg is not None and message_chat_id and message_id:
            ok, err = self.tg.edit_message_text(message_chat_id, message_id, text, reply_markup=reply_markup)
            if ok:
                return
            self._log_error("admin_menu_edit_failed", {
                "chat_id": message_chat_id,
                "message_id": message_id,
                "error": err,
            })
        if message_chat_id:
            self._send_markup_or_text(message_chat_id, text, reply_markup, force_budget=True)

    def _answer_callback(self, callback_query: Dict[str, Any], text: str = "", show_alert: bool = False) -> None:
        callback_id = str(callback_query.get("id") or "")
        if callback_id and self.tg is not None:
            self.tg.answer_callback_query(callback_id, text, show_alert=show_alert)

    def _callback_admin_user_id(self, callback_query: Dict[str, Any]) -> int:
        from_user = callback_query.get("from") or {}
        return int(from_user.get("id") or 0)

    def _ensure_callback_admin(self, callback_query: Dict[str, Any]) -> Tuple[bool, int]:
        user_id = self._callback_admin_user_id(callback_query)
        if self.is_admin_user_fn is None:
            return True, user_id
        if self.is_admin_user_fn(user_id, self.load_state_fn()):
            return True, user_id
        self._answer_callback(
            callback_query,
            "Только админ Ouroboros может управлять доступами.",
            show_alert=True,
        )
        return False, user_id

    def send_request_to_admins(self, rec: Dict[str, Any]) -> int:
        pending_count = len(list_user_records(self.drive_root, access_status=ACCESS_PENDING))
        uid = int(rec.get("user_id") or 0)
        body = (
            "🔐 Запрос доступа\n\n"
            f"{_user_name(rec)}\n"
            f"{_user_detail_line(rec, ACCESS_PENDING)}\n\n"
            f"В очереди: {pending_count}"
        )
        sent = 0
        owner_id = int(self.load_state_fn().get("owner_id") or 0) or None
        for admin_chat_id in self.admin_chat_ids_fn():
            if self.tg is not None:
                ok, err, message_id = self.tg.send_message_with_markup(
                    admin_chat_id,
                    body,
                    access_request_keyboard(uid),
                )
                if ok:
                    append_access_request_notification(
                        self.drive_root,
                        uid,
                        admin_chat_id=admin_chat_id,
                        message_id=message_id,
                    )
                    self._log_admin_out(admin_chat_id, body)
                    sent += 1
                else:
                    self._log_error("user_access_admin_notify_failed", {
                        "user_id": uid,
                        "admin_chat_id": int(admin_chat_id),
                        "error": err,
                    })
                continue

            self.send_with_budget_fn(
                admin_chat_id,
                body,
                log_drive_root=self.drive_root,
                log_user_id=owner_id,
            )
            sent += 1
        return sent

    def notify_unnotified_requests(self) -> int:
        sent = 0
        for rec in list_user_records(self.drive_root, access_status=ACCESS_PENDING):
            has_inline_notification = bool(rec.get("access_notifications"))
            if rec.get("access_admin_notified_at") and (self.tg is None or has_inline_notification):
                continue
            count = self.send_request_to_admins(rec)
            if count:
                mark_access_request_notified(self.drive_root, int(rec["user_id"]))
                sent += count
        return sent

    def mark_request_notified(self, user_id: int) -> None:
        mark_access_request_notified(self.drive_root, user_id)

    def format_records(self, status: str = ACCESS_PENDING) -> str:
        records = list_user_records(self.drive_root, access_status=status)
        title = {
            ACCESS_PENDING: "👤 Заявки пользователей",
            ACCESS_APPROVED: "✅ Пользователи с доступом",
            ACCESS_DENIED: "⛔️ Отключённые пользователи",
        }.get(status, "Пользователи")
        if not records:
            return f"{title}: пусто."

        visible = records[:20]
        request_counts = _user_request_counts(self.drive_root, visible)
        lines = [f"{title}: {len(records)}", ""]
        for idx, rec in enumerate(visible, start=1):
            uid = int(rec.get("user_id") or 0)
            lines.append(f"{idx}. {_user_name(rec)}")
            lines.extend(_user_list_detail_lines(rec, status, request_counts.get(uid, 0)))
        if len(records) > len(visible):
            lines.append(f"\nПоказаны первые {len(visible)} из {len(records)}.")
        if status == ACCESS_PENDING:
            lines.extend([
                "",
                "Быстрые команды:",
                "/approve all",
            ])
        return "\n".join(lines)

    def format_recent_users(self) -> str:
        records = _sort_users_by_activity(list_user_records(self.drive_root))
        title = "🕘 Пользователи по активности"
        if not records:
            return f"{title}: пусто."

        visible = records[:20]
        request_counts = _user_request_counts(self.drive_root, visible)
        lines = [f"{title}: {len(records)}", ""]
        for idx, rec in enumerate(visible, start=1):
            uid = int(rec.get("user_id") or 0)
            lines.append(f"{idx}. {_user_name(rec)}")
            lines.extend(_user_activity_detail_lines(rec, request_counts.get(uid, 0)))
        if len(records) > len(visible):
            lines.append(f"\nПоказаны первые {len(visible)} из {len(records)}.")
        return "\n".join(lines)

    def format_admin_home(self) -> str:
        user_pending = len(list_user_records(self.drive_root, access_status=ACCESS_PENDING))
        user_approved = len(list_user_records(self.drive_root, access_status=ACCESS_APPROVED))
        user_denied = len(list_user_records(self.drive_root, access_status=ACCESS_DENIED))
        group_pending = len(list_team_chats(self.drive_root, status=TEAM_CHAT_PENDING))
        group_approved = len(list_team_chats(self.drive_root, status=TEAM_CHAT_APPROVED))
        group_denied = len(list_team_chats(self.drive_root, status=TEAM_CHAT_DENIED))
        improvement_counts = report_counts(self.drive_root)
        improvements_open = improvement_counts.get(STATUS_OPEN, 0)
        improvements_in_progress = improvement_counts.get(STATUS_IN_PROGRESS, 0)
        return (
            "🛠 Админ-панель\n\n"
            "Пользователи\n"
            f"• заявки: {user_pending}\n"
            f"• с доступом: {user_approved}\n"
            f"• отключены: {user_denied}\n\n"
            "Группы\n"
            f"• заявки: {group_pending}\n"
            f"• с доступом: {group_approved}\n"
            f"• отключены: {group_denied}\n\n"
            "Запросы доработки\n"
            f"• открытые: {improvements_open}\n"
            f"• в работе: {improvements_in_progress}"
        )

    def admin_home_keyboard(self) -> Dict[str, Any]:
        user_pending = len(list_user_records(self.drive_root, access_status=ACCESS_PENDING))
        user_approved = len(list_user_records(self.drive_root, access_status=ACCESS_APPROVED))
        user_denied = len(list_user_records(self.drive_root, access_status=ACCESS_DENIED))
        group_pending = len(list_team_chats(self.drive_root, status=TEAM_CHAT_PENDING))
        group_approved = len(list_team_chats(self.drive_root, status=TEAM_CHAT_APPROVED))
        group_denied = len(list_team_chats(self.drive_root, status=TEAM_CHAT_DENIED))
        improvement_counts = report_counts(self.drive_root)
        improvements_open = improvement_counts.get(STATUS_OPEN, 0)
        return {
            "inline_keyboard": [
                [{"text": f"👤 Заявки: {user_pending}", "callback_data": "admin:users:pending"}],
                [
                    {"text": f"✅ Доступ: {user_approved}", "callback_data": "admin:users:approved"},
                    {"text": f"⛔️ Отключены: {user_denied}", "callback_data": "admin:users:denied"},
                ],
                [{"text": "🕘 По активности", "callback_data": f"admin:users:{USER_ACTIVITY_VIEW}"}],
                [{"text": f"👥 Заявки групп: {group_pending}", "callback_data": "admin:groups:pending"}],
                [
                    {"text": f"✅ Группы: {group_approved}", "callback_data": "admin:groups:approved"},
                    {"text": f"⛔️ Отключены: {group_denied}", "callback_data": "admin:groups:denied"},
                ],
                [{"text": f"🛠 Доработки: {improvements_open}", "callback_data": "unresolved:list:open"}],
                [{"text": "🔄 Обновить", "callback_data": "admin:home"}],
            ]
        }

    @staticmethod
    def _numbered_rows(buttons: List[Dict[str, str]], columns: int = 4) -> List[List[Dict[str, str]]]:
        rows: List[List[Dict[str, str]]] = []
        for idx in range(0, len(buttons), columns):
            rows.append(buttons[idx:idx + columns])
        return rows

    def _user_decision_rows(
        self,
        records: List[Dict[str, Any]],
        view_name: str,
    ) -> List[List[Dict[str, str]]]:
        rows: List[List[Dict[str, str]]] = []
        buttons: List[Dict[str, str]] = []
        for idx, rec in enumerate(records, start=1):
            uid = int(rec.get("user_id") or 0)
            status = str(rec.get("access_status") or ACCESS_APPROVED).strip().lower()
            if status == ACCESS_PENDING:
                rows.append([
                    {"text": f"✅ {idx}", "callback_data": f"access:approve:{uid}:users:{view_name}"},
                    {"text": f"⛔️ {idx}", "callback_data": f"access:deny:{uid}:users:{view_name}"},
                ])
            elif status == ACCESS_APPROVED:
                buttons.append({"text": f"⛔️ {idx}", "callback_data": f"access:deny:{uid}:users:{view_name}"})
            else:
                buttons.append({"text": f"✅ {idx}", "callback_data": f"access:approve:{uid}:users:{view_name}"})
        if buttons:
            rows.extend(self._numbered_rows(buttons))
        return rows

    def _user_navigation_rows(self, refresh_callback: str) -> List[List[Dict[str, str]]]:
        return [
            [
                {"text": "Заявки", "callback_data": "admin:users:pending"},
                {"text": "Доступ", "callback_data": "admin:users:approved"},
                {"text": "Отключены", "callback_data": "admin:users:denied"},
            ],
            [{"text": "🕘 Активность", "callback_data": f"admin:users:{USER_ACTIVITY_VIEW}"}],
            [
                {"text": "⬅️ Меню", "callback_data": "admin:home"},
                {"text": "🔄 Обновить", "callback_data": refresh_callback},
            ],
        ]

    def user_records_keyboard(self, status: str) -> Dict[str, Any]:
        records = list_user_records(self.drive_root, access_status=status)[:20]
        rows = self._user_decision_rows(records, status)
        rows.extend(self._user_navigation_rows(f"admin:users:{status}"))
        return {"inline_keyboard": rows}

    def recent_user_records_keyboard(self) -> Dict[str, Any]:
        records = _sort_users_by_activity(list_user_records(self.drive_root))[:20]
        rows = self._user_decision_rows(records, USER_ACTIVITY_VIEW)
        rows.extend(self._user_navigation_rows(f"admin:users:{USER_ACTIVITY_VIEW}"))
        return {"inline_keyboard": rows}

    def format_team_records(self, status: str) -> str:
        records = list_team_chats(self.drive_root, status=status)
        title = {
            TEAM_CHAT_PENDING: "👥 Заявки групп",
            TEAM_CHAT_APPROVED: "✅ Группы с доступом",
            TEAM_CHAT_DENIED: "⛔️ Отключённые группы",
        }.get(status, "Группы")
        if not records:
            return f"{title}: пусто."
        visible = records[:20]
        lines = [f"{title}: {len(records)}", ""]
        for idx, rec in enumerate(visible, start=1):
            lines.append(f"{idx}. {_team_name(rec)}")
            lines.append(f"   {_team_detail_line(rec, status)}")
        if len(records) > len(visible):
            lines.append(f"\nПоказаны первые {len(visible)} из {len(records)}.")
        return "\n".join(lines)

    def team_records_keyboard(self, status: str) -> Dict[str, Any]:
        records = list_team_chats(self.drive_root, status=status)[:20]
        rows: List[List[Dict[str, str]]] = []
        buttons: List[Dict[str, str]] = []
        for idx, rec in enumerate(records, start=1):
            chat_id = int(rec.get("chat_id") or 0)
            if status == TEAM_CHAT_PENDING:
                rows.append([
                    {"text": f"✅ {idx}", "callback_data": f"teamchat:approve:{chat_id}"},
                    {"text": f"⛔️ {idx}", "callback_data": f"teamchat:deny:{chat_id}"},
                ])
            elif status == TEAM_CHAT_APPROVED:
                buttons.append({"text": f"⛔️ {idx}", "callback_data": f"teamchat:deny:{chat_id}:force"})
            else:
                buttons.append({"text": f"✅ {idx}", "callback_data": f"teamchat:approve:{chat_id}:force"})
        if buttons:
            rows.extend(self._numbered_rows(buttons))
        rows.extend([
            [
                {"text": "Заявки", "callback_data": "admin:groups:pending"},
                {"text": "Доступ", "callback_data": "admin:groups:approved"},
                {"text": "Отключены", "callback_data": "admin:groups:denied"},
            ],
            [
                {"text": "⬅️ Меню", "callback_data": "admin:home"},
                {"text": "🔄 Обновить", "callback_data": f"admin:groups:{status}"},
            ],
        ])
        return {"inline_keyboard": rows}

    def send_admin_home(self, chat_id: int) -> None:
        self._send_markup_or_text(
            chat_id,
            self.format_admin_home(),
            self.admin_home_keyboard(),
            force_budget=True,
        )

    def send_user_admin_view(self, chat_id: int, status: str) -> None:
        self._send_markup_or_text(
            chat_id,
            self.format_records(status),
            self.user_records_keyboard(status),
            force_budget=True,
        )

    def send_recent_user_admin_view(self, chat_id: int) -> None:
        self._send_markup_or_text(
            chat_id,
            self.format_recent_users(),
            self.recent_user_records_keyboard(),
            force_budget=True,
        )

    def send_group_admin_view(self, chat_id: int, status: str) -> None:
        self._send_markup_or_text(
            chat_id,
            self.format_team_records(status),
            self.team_records_keyboard(status),
            force_budget=True,
        )

    def edit_access_notifications(self, rec: Dict[str, Any]) -> None:
        if self.tg is None:
            return
        text = access_decision_text(rec)
        for item in rec.get("access_notifications") or []:
            try:
                admin_chat_id = int(item.get("admin_chat_id") or 0)
                message_id = int(item.get("message_id") or 0)
                if admin_chat_id and message_id:
                    self.tg.edit_message_text(admin_chat_id, message_id, text)
            except Exception:
                continue

    def decide_user(
        self,
        user_id: int,
        target_status: str,
        admin_user_id: int,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        results = set_user_access_status(
            self.drive_root,
            [user_id],
            target_status,
            decided_by=admin_user_id,
        )
        self.notify_users_decision(results, target_status)
        result = results[0] if results else {"status": "missing", "record": {}}
        if result.get("status") == "missing":
            return False, f"Пользователь {user_id} не найден.", {}
        rec = result.get("record") or {}
        self.edit_access_notifications(rec)
        old_status = result.get("old_status")
        actual_status = result.get("status")
        if actual_status != target_status:
            msg = f"Нельзя изменить: {access_decision_text(rec)}"
        elif old_status == target_status:
            msg = f"Уже {target_status}: {access_user_label(rec)}"
        elif target_status == ACCESS_APPROVED:
            msg = f"Доступ предоставлен: {access_user_label(rec)}"
        else:
            msg = f"Доступ отклонён: {access_user_label(rec)}"
        return True, msg, rec

    def notify_users_decision(self, results: List[Dict[str, Any]], status: str) -> None:
        for result in results:
            if result.get("status") == "missing":
                continue
            if result.get("old_status") == result.get("status"):
                continue
            rec = result.get("record") or {}
            chat_id = int(rec.get("chat_id") or 0)
            uid = int(rec.get("user_id") or 0)
            if not chat_id or not uid:
                continue

            if status == ACCESS_APPROVED:
                user_root, _, _ = ensure_user_workspace(
                    self.drive_root,
                    uid,
                    chat_id,
                    rec,
                    role="user",
                    use_global_root=False,
                    access_status=ACCESS_APPROVED,
                )
                self.send_with_budget_fn(
                    chat_id,
                    "✅ Доступ к боту предоставлен. Можно писать задачу.",
                    log_drive_root=user_root,
                    log_user_id=uid,
                )
            elif status == ACCESS_DENIED:
                self.send_with_budget_fn(
                    chat_id,
                    "⛔️ Доступ к боту отклонён администратором.",
                    log_drive_root=self.drive_root,
                    log_user_id=uid,
                )

    def handle_command(self, text: str, chat_id: int, admin_user_id: int) -> bool:
        parts = text.strip().split()
        if not parts:
            return False

        command = parts[0].lower()
        user_status_aliases = {
            "pending": ACCESS_PENDING,
            "requests": ACCESS_PENDING,
            "approved": ACCESS_APPROVED,
            "users": ACCESS_APPROVED,
            "denied": ACCESS_DENIED,
            "rejected": ACCESS_DENIED,
        }
        user_activity_aliases = {
            USER_ACTIVITY_VIEW,
            "activity",
            "active",
            "recent",
            "last",
            "last_seen",
        }
        group_status_aliases = {
            "pending": TEAM_CHAT_PENDING,
            "requests": TEAM_CHAT_PENDING,
            "approved": TEAM_CHAT_APPROVED,
            "denied": TEAM_CHAT_DENIED,
            "rejected": TEAM_CHAT_DENIED,
        }

        if command == "/admin":
            if len(parts) == 1:
                self.send_admin_home(chat_id)
                return True
            section = parts[1].lower()
            status_name = parts[2].lower() if len(parts) > 2 else "pending"
            if section in ("users", "user", "access"):
                if status_name in user_activity_aliases:
                    self.send_recent_user_admin_view(chat_id)
                    return True
                self.send_user_admin_view(chat_id, user_status_aliases.get(status_name, ACCESS_PENDING))
                return True
            if section in ("groups", "group", "teamchat", "teams"):
                self.send_group_admin_view(chat_id, group_status_aliases.get(status_name, TEAM_CHAT_PENDING))
                return True
            self.send_admin_home(chat_id)
            return True

        aliases = {
            "/approve": "approve",
            "/allow": "approve",
            "/deny": "deny",
            "/reject": "deny",
        }
        action = aliases.get(command)
        args = parts[1:]

        if command in ("/access", "/requests"):
            if len(parts) == 1:
                self.send_with_budget_fn(chat_id, self.format_records(ACCESS_PENDING), force_budget=True)
                return True
            subcommand = parts[1].lower()
            if subcommand in ("list", "pending"):
                self.send_with_budget_fn(chat_id, self.format_records(ACCESS_PENDING), force_budget=True)
                return True
            if subcommand in ("approved", "users"):
                self.send_with_budget_fn(chat_id, self.format_records(ACCESS_APPROVED), force_budget=True)
                return True
            if subcommand in ("denied", "rejected"):
                self.send_with_budget_fn(chat_id, self.format_records(ACCESS_DENIED), force_budget=True)
                return True
            if subcommand in user_activity_aliases:
                self.send_with_budget_fn(chat_id, self.format_recent_users(), force_budget=True)
                return True
            if subcommand in ("approve", "allow"):
                action = "approve"
                args = parts[2:]
            elif subcommand in ("deny", "reject"):
                action = "deny"
                args = parts[2:]
            else:
                self.send_with_budget_fn(
                    chat_id,
                    "Команды доступа: /access, /approve all, /approve <user_id>, /deny <user_id>",
                    force_budget=True,
                )
                return True

        if action not in ("approve", "deny"):
            return False

        target_status = ACCESS_APPROVED if action == "approve" else ACCESS_DENIED
        if args and args[0].lower() == "all":
            user_ids = [int(rec["user_id"]) for rec in list_user_records(self.drive_root, access_status=ACCESS_PENDING)]
        else:
            user_ids = parse_access_user_ids(args)

        if not user_ids:
            if target_status == ACCESS_APPROVED:
                self.send_with_budget_fn(chat_id, "Нет pending-запросов для согласования.", force_budget=True)
            else:
                self.send_with_budget_fn(chat_id, "Укажи user_id для отказа: /deny <user_id>", force_budget=True)
            return True

        results = set_user_access_status(
            self.drive_root,
            user_ids,
            target_status,
            decided_by=admin_user_id,
        )
        self.notify_users_decision(results, target_status)
        for result in results:
            rec = result.get("record") or {}
            if rec:
                self.edit_access_notifications(rec)

        updated = [r for r in results if r.get("status") == target_status and r.get("old_status") != target_status]
        unchanged = [r for r in results if r.get("status") == target_status and r.get("old_status") == target_status]
        missing = [r for r in results if r.get("status") == "missing"]
        verb = "согласовано" if target_status == ACCESS_APPROVED else "отклонено"
        lines = [f"Доступ: {verb} {len(updated)}."]
        if unchanged:
            lines.append(f"Без изменений: {len(unchanged)}.")
        if missing:
            lines.append("Не найдены: " + ", ".join(str(r.get("user_id")) for r in missing))
        pending_left = len(list_user_records(self.drive_root, access_status=ACCESS_PENDING))
        lines.append(f"Pending осталось: {pending_left}.")
        self.send_with_budget_fn(chat_id, "\n".join(lines), force_budget=True)
        return True

    def handle_callback(self, callback_query: Dict[str, Any]) -> bool:
        data = str(callback_query.get("data") or "")
        if not (data.startswith("access:") or data.startswith("admin:")):
            return False

        ok_admin, admin_user_id = self._ensure_callback_admin(callback_query)
        if not ok_admin:
            return True

        if data == "admin:home":
            self._edit_or_send_callback_view(
                callback_query,
                self.format_admin_home(),
                self.admin_home_keyboard(),
            )
            self._answer_callback(callback_query, "Админ-меню обновлено.")
            return True

        if data.startswith("admin:"):
            parts = data.split(":")
            if len(parts) != 3:
                self._answer_callback(callback_query, "Некорректный раздел меню.", show_alert=True)
                return True
            section, status_name = parts[1], parts[2]
            if section == "users":
                if status_name == USER_ACTIVITY_VIEW:
                    self._edit_or_send_callback_view(
                        callback_query,
                        self.format_recent_users(),
                        self.recent_user_records_keyboard(),
                    )
                    self._answer_callback(callback_query, "Активность пользователей обновлена.")
                    return True
                status = {
                    "pending": ACCESS_PENDING,
                    "approved": ACCESS_APPROVED,
                    "denied": ACCESS_DENIED,
                }.get(status_name)
                if not status:
                    self._answer_callback(callback_query, "Некорректный статус пользователей.", show_alert=True)
                    return True
                self._edit_or_send_callback_view(
                    callback_query,
                    self.format_records(status),
                    self.user_records_keyboard(status),
                )
                self._answer_callback(callback_query, "Пользователи обновлены.")
                return True
            if section == "groups":
                status = {
                    "pending": TEAM_CHAT_PENDING,
                    "approved": TEAM_CHAT_APPROVED,
                    "denied": TEAM_CHAT_DENIED,
                }.get(status_name)
                if not status:
                    self._answer_callback(callback_query, "Некорректный статус групп.", show_alert=True)
                    return True
                self._edit_or_send_callback_view(
                    callback_query,
                    self.format_team_records(status),
                    self.team_records_keyboard(status),
                )
                self._answer_callback(callback_query, "Группы обновлены.")
                return True
            self._answer_callback(callback_query, "Некорректный раздел меню.", show_alert=True)
            return True

        parts = data.split(":")
        if len(parts) not in (3, 5) or parts[1] not in ("approve", "deny"):
            self._answer_callback(callback_query, "Некорректная кнопка доступа.", show_alert=True)
            return True
        user_ids = parse_access_user_ids([parts[2]])
        if not user_ids:
            self._answer_callback(callback_query, "Некорректный user_id.", show_alert=True)
            return True

        target_status = ACCESS_APPROVED if parts[1] == "approve" else ACCESS_DENIED
        ok, msg, rec = self.decide_user(user_ids[0], target_status, admin_user_id)
        self._answer_callback(callback_query, msg, show_alert=not ok)

        message = callback_query.get("message") or {}
        message_chat = message.get("chat") or {}
        message_chat_id = int(message_chat.get("id") or 0)
        message_id = int(message.get("message_id") or 0)
        if ok and len(parts) == 5 and parts[3] == "users":
            if parts[4] == USER_ACTIVITY_VIEW:
                self._edit_or_send_callback_view(
                    callback_query,
                    self.format_recent_users(),
                    self.recent_user_records_keyboard(),
                )
                return True
            status = {
                "pending": ACCESS_PENDING,
                "approved": ACCESS_APPROVED,
                "denied": ACCESS_DENIED,
            }.get(parts[4])
            if status:
                self._edit_or_send_callback_view(
                    callback_query,
                    self.format_records(status),
                    self.user_records_keyboard(status),
                )
                return True
        if ok and self.tg is not None and message_chat_id and message_id:
            self.tg.edit_message_text(message_chat_id, message_id, access_decision_text(rec))
        return True
