"""Team chat approval UI and routing helpers.

This module keeps Telegram group approval/button behavior out of the launcher.
The persistent registry and workspace creation live in supervisor.teams.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from supervisor.teams import (
    TEAM_APPROVED,
    TEAM_DENIED,
    TEAM_PENDING,
    append_team_chat_notification,
    ensure_team_workspace,
    is_group_chat_type,
    list_team_chats,
    request_team_chat,
    set_team_chat_status,
    team_chat_status,
)

log = logging.getLogger(__name__)
GROUP_TRIGGER_COMMANDS = {"ask", "ai", "ouro", "ouroboros"}


def team_chat_label(rec: Dict[str, Any]) -> str:
    title = str(rec.get("title") or "").strip() or "(без названия)"
    chat_id = int(rec.get("chat_id") or 0)
    status = str(rec.get("status") or TEAM_PENDING)
    return f"{title} (chat_id={chat_id}, status={status})"


def team_chat_keyboard(chat_id: int) -> Dict[str, Any]:
    return {
        "inline_keyboard": [[
            {"text": "✅ Разрешить", "callback_data": f"teamchat:approve:{int(chat_id)}"},
            {"text": "⛔️ Запретить", "callback_data": f"teamchat:deny:{int(chat_id)}"},
        ]]
    }


def team_chat_request_text(
    rec: Dict[str, Any],
    access_user_label_fn: Callable[[Dict[str, Any]], str],
) -> str:
    requested_by = rec.get("requested_by") or rec.get("last_requested_by") or {}
    requester = (
        access_user_label_fn(requested_by)
        if isinstance(requested_by, dict) and requested_by.get("user_id")
        else "неизвестно"
    )
    return (
        "👥 Бота добавили в Telegram-группу\n"
        f"Группа: {team_chat_label(rec)}\n"
        f"Добавил/запросил: {requester}\n\n"
        "Разрешить боту общаться с участниками и выполнять командные задачи в этой группе?"
    )


def team_chat_decision_text(rec: Dict[str, Any]) -> str:
    status = str(rec.get("status") or TEAM_PENDING)
    decided_by = rec.get("decided_by")
    if status == TEAM_APPROVED:
        verdict = "✅ Разрешено"
    elif status == TEAM_DENIED:
        verdict = "⛔️ Запрещено"
    else:
        verdict = "⏳ Ожидает решения"
    suffix = f"\nРешил admin user_id={decided_by}" if decided_by else ""
    return f"{verdict}: {team_chat_label(rec)}{suffix}"


def parse_team_chat_id(raw: str) -> Optional[int]:
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def message_mentions_bot(msg: Dict[str, Any], text: str, bot_username: str = "") -> bool:
    username = str(bot_username or "").strip().lstrip("@").lower()
    if username and f"@{username}" in str(text or "").lower():
        return True
    for ent in msg.get("entities") or msg.get("caption_entities") or []:
        if not isinstance(ent, dict):
            continue
        if ent.get("type") == "mention":
            offset = int(ent.get("offset") or 0)
            length = int(ent.get("length") or 0)
            mention = str(text or "")[offset:offset + length].lower()
            if username and mention == f"@{username}":
                return True
    return False


def message_is_reply_to_bot(msg: Dict[str, Any], bot_id: int = 0, bot_username: str = "") -> bool:
    username = str(bot_username or "").strip().lstrip("@").lower()
    reply = msg.get("reply_to_message") or {}
    reply_from = reply.get("from") or {}
    if bot_id and int(reply_from.get("id") or 0) == int(bot_id):
        return True
    if username and str(reply_from.get("username") or "").lower() == username:
        return True
    return False


def is_group_task_trigger(
    msg: Dict[str, Any],
    text: str,
    caption: str,
    *,
    bot_id: int = 0,
    bot_username: str = "",
) -> bool:
    trigger_text = text or caption
    stripped = str(trigger_text or "").strip()
    if stripped.startswith("/"):
        return True
    return (
        message_mentions_bot(msg, trigger_text, bot_username=bot_username)
        or message_is_reply_to_bot(msg, bot_id=bot_id, bot_username=bot_username)
    )


def strip_bot_mention(text: str, bot_username: str = "") -> str:
    username = str(bot_username or "").strip().lstrip("@")
    if not username:
        return text
    stripped = re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", stripped).strip()


def strip_group_command(text: str, bot_username: str = "") -> str:
    raw = str(text or "").strip()
    if not raw.startswith("/"):
        return text
    first, sep, rest = raw.partition(" ")
    match = re.match(r"^/([A-Za-z0-9_]+)(?:@([A-Za-z0-9_]+))?$", first)
    if not match:
        return text
    command = match.group(1).lower()
    target = (match.group(2) or "").lower()
    username = str(bot_username or "").strip().lstrip("@").lower()
    targets_this_bot = bool(username and target == username)
    if command in GROUP_TRIGGER_COMMANDS or targets_this_bot:
        return rest.strip() if sep else ""
    return text


def prepare_group_task_text(text: str, bot_username: str = "") -> str:
    without_mention = strip_bot_mention(text, bot_username=bot_username)
    return strip_group_command(without_mention, bot_username=bot_username).strip()


def group_approved_text(bot_username: str = "") -> str:
    command = "/ask"
    username = str(bot_username or "").strip().lstrip("@")
    if username:
        command = f"/ask@{username}"
    return (
        "✅ Ouroboros подключён к группе.\n\n"
        "Если Telegram privacy mode включён, я вижу только команды и ответы на мои сообщения. "
        f"Пиши так: `{command} текст задачи`, или отвечай reply на это сообщение.\n\n"
        "Чтобы я видел обычные сообщения группы, отключи privacy mode у бота в BotFather."
    )


@dataclass
class TeamChatRuntime:
    drive_root: pathlib.Path
    tg: Any
    admin_chat_ids_fn: Callable[[], List[int]]
    access_user_label_fn: Callable[[Dict[str, Any]], str]
    is_admin_user_fn: Callable[[int, Dict[str, Any]], bool]
    load_state_fn: Callable[[], Dict[str, Any]]
    send_with_budget_fn: Callable[..., Any]
    log_chat_fn: Callable[..., Any]
    append_jsonl_fn: Callable[..., Any]
    bot_id: int = 0
    bot_username: str = ""

    def send_request_to_admins(self, rec: Dict[str, Any]) -> int:
        chat_id = int(rec.get("chat_id") or 0)
        if not chat_id:
            return 0
        sent = 0
        owner_id = int(self.load_state_fn().get("owner_id") or 0) or None
        body = team_chat_request_text(rec, self.access_user_label_fn)
        for admin_chat_id in self.admin_chat_ids_fn():
            ok, err, message_id = self.tg.send_message_with_markup(
                admin_chat_id,
                body,
                team_chat_keyboard(chat_id),
            )
            if ok:
                append_team_chat_notification(
                    self.drive_root,
                    chat_id,
                    admin_chat_id=admin_chat_id,
                    message_id=message_id,
                )
                self.log_chat_fn(
                    "out",
                    admin_chat_id,
                    owner_id or 0,
                    body,
                    drive_root=self.drive_root,
                )
                sent += 1
            else:
                self.append_jsonl_fn(self.drive_root / "logs" / "supervisor.jsonl", {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "team_chat_admin_notify_failed",
                    "chat_id": chat_id,
                    "admin_chat_id": admin_chat_id,
                    "error": err,
                })
        return sent

    def edit_notifications(self, rec: Dict[str, Any]) -> None:
        text = team_chat_decision_text(rec)
        for item in rec.get("notifications") or []:
            try:
                admin_chat_id = int(item.get("admin_chat_id") or 0)
                message_id = int(item.get("message_id") or 0)
                if admin_chat_id and message_id:
                    self.tg.edit_message_text(admin_chat_id, message_id, text)
            except Exception:
                log.debug("Failed to edit team chat notification", exc_info=True)

    def decide(
        self,
        chat_id: int,
        target_status: str,
        admin_user_id: int,
        *,
        force_terminal: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        result = set_team_chat_status(
            self.drive_root,
            chat_id,
            target_status,
            decided_by=admin_user_id,
            allow_terminal_change=force_terminal,
        )
        if result.get("status") == "missing":
            return False, f"Группа {chat_id} не найдена.", {}
        rec = result.get("record") or {}
        old_status = result.get("old_status")
        actual_status = result.get("status")
        if actual_status != target_status:
            msg = f"Уже {actual_status}: {team_chat_label(rec)}"
        elif old_status == target_status:
            msg = f"Уже {target_status}: {team_chat_label(rec)}"
        elif target_status == TEAM_APPROVED:
            msg = f"Группа разрешена: {team_chat_label(rec)}"
            self.notify_group_approved(rec)
        else:
            msg = f"Группа запрещена: {team_chat_label(rec)}"
        self.edit_notifications(rec)
        return True, msg, rec

    def notify_group_approved(self, rec: Dict[str, Any]) -> None:
        chat_id = int(rec.get("chat_id") or 0)
        if not chat_id:
            return
        owner_id = int(self.load_state_fn().get("owner_id") or 0) or None
        team_root = ensure_team_workspace(self.drive_root, chat_id)
        self.send_with_budget_fn(
            chat_id,
            group_approved_text(self.bot_username),
            log_drive_root=team_root,
            log_user_id=owner_id,
        )

    def format_records(self, status: str = TEAM_PENDING) -> str:
        records = list_team_chats(self.drive_root, status=status)
        title = {
            TEAM_PENDING: "Группы ожидают approval",
            TEAM_APPROVED: "Разрешённые группы",
            TEAM_DENIED: "Запрещённые группы",
        }.get(status, "Группы")
        if not records:
            return f"{title}: пусто."
        lines = [f"{title}: {len(records)}"]
        for rec in records[:30]:
            lines.append(f"- {team_chat_label(rec)}")
        if len(records) > 30:
            lines.append(f"... ещё {len(records) - 30}")
        if status == TEAM_PENDING:
            lines.extend([
                "",
                "Команды:",
                "/teamchat approve <chat_id>",
                "/teamchat deny <chat_id>",
            ])
        return "\n".join(lines)

    def handle_command(self, text: str, chat_id: int, admin_user_id: int) -> bool:
        parts = text.strip().split()
        if not parts or parts[0].lower() != "/teamchat":
            return False

        subcommand = parts[1].lower() if len(parts) > 1 else "pending"
        if subcommand in ("pending", "approved", "denied"):
            status = {
                "pending": TEAM_PENDING,
                "approved": TEAM_APPROVED,
                "denied": TEAM_DENIED,
            }[subcommand]
            self.send_with_budget_fn(chat_id, self.format_records(status), force_budget=True)
            return True

        if subcommand not in ("approve", "allow", "deny", "reject"):
            self.send_with_budget_fn(
                chat_id,
                "Команды групп: /teamchat pending, /teamchat approved, /teamchat denied, "
                "/teamchat approve <chat_id>, /teamchat deny <chat_id>",
                force_budget=True,
            )
            return True

        if len(parts) < 3:
            self.send_with_budget_fn(chat_id, "Укажи chat_id: /teamchat approve <chat_id>", force_budget=True)
            return True

        target_chat_id = parse_team_chat_id(parts[2])
        if target_chat_id is None:
            self.send_with_budget_fn(chat_id, f"Некорректный chat_id: {parts[2]}", force_budget=True)
            return True

        target_status = TEAM_APPROVED if subcommand in ("approve", "allow") else TEAM_DENIED
        _ok, msg, _rec = self.decide(target_chat_id, target_status, admin_user_id, force_terminal=True)
        self.send_with_budget_fn(chat_id, msg, force_budget=True)
        return True

    def handle_callback(self, callback_query: Dict[str, Any]) -> bool:
        data = str(callback_query.get("data") or "")
        if not data.startswith("teamchat:"):
            return False

        callback_id = str(callback_query.get("id") or "")
        from_user = callback_query.get("from") or {}
        user_id = int(from_user.get("id") or 0)
        state = self.load_state_fn()
        if not self.is_admin_user_fn(user_id, state):
            if callback_id:
                self.tg.answer_callback_query(
                    callback_id,
                    "Только админ Ouroboros может принимать решение.",
                    show_alert=True,
                )
            return True

        parts = data.split(":")
        if len(parts) not in (3, 4) or parts[1] not in ("approve", "deny"):
            if callback_id:
                self.tg.answer_callback_query(callback_id, "Некорректная кнопка.", show_alert=True)
            return True
        force_terminal = len(parts) == 4 and parts[3] == "force"

        target_chat_id = parse_team_chat_id(parts[2])
        if target_chat_id is None:
            if callback_id:
                self.tg.answer_callback_query(callback_id, "Некорректный chat_id.", show_alert=True)
            return True

        target_status = TEAM_APPROVED if parts[1] == "approve" else TEAM_DENIED
        ok, msg, rec = self.decide(target_chat_id, target_status, user_id, force_terminal=force_terminal)
        if callback_id:
            self.tg.answer_callback_query(callback_id, msg, show_alert=not ok)

        message = callback_query.get("message") or {}
        message_chat = message.get("chat") or {}
        message_chat_id = int(message_chat.get("id") or 0)
        message_id = int(message.get("message_id") or 0)
        if ok and message_chat_id and message_id:
            self.tg.edit_message_text(message_chat_id, message_id, team_chat_decision_text(rec))
        return True

    def handle_added_update(self, update: Dict[str, Any]) -> bool:
        member_update = update.get("my_chat_member")
        if not isinstance(member_update, dict):
            return False

        chat = member_update.get("chat") or {}
        if not is_group_chat_type(chat.get("type")):
            return True

        new_status = str((member_update.get("new_chat_member") or {}).get("status") or "").lower()
        if new_status not in ("member", "administrator"):
            return True

        from_user = member_update.get("from") or {}
        rec, _created, should_notify = request_team_chat(self.drive_root, chat, requested_by=from_user)
        if team_chat_status(rec) == TEAM_PENDING and should_notify:
            if self.send_request_to_admins(rec) > 0:
                self.append_jsonl_fn(self.drive_root / "logs" / "events.jsonl", {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "team_chat_access_request",
                    "chat_id": int(chat.get("id") or 0),
                    "source": "my_chat_member",
                })
        return True

    def is_group_task_trigger(self, msg: Dict[str, Any], text: str, caption: str) -> bool:
        return is_group_task_trigger(
            msg,
            text,
            caption,
            bot_id=self.bot_id,
            bot_username=self.bot_username,
        )

    def strip_bot_mention(self, text: str) -> str:
        return strip_bot_mention(text, bot_username=self.bot_username)

    def prepare_group_task_text(self, text: str) -> str:
        return prepare_group_task_text(text, bot_username=self.bot_username)
