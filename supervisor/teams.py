"""Supervisor team chat registry and shared workspace helpers.

Telegram groups/supergroups are team workspaces only after an Ouroboros admin
approves them. Pending and denied chats stay silent and do not run LLM work.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    acquire_file_lock,
    atomic_write_text,
    release_file_lock,
)

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
TEAM_CHATS_PATH: pathlib.Path = DRIVE_ROOT / "state" / "team_chats.json"
TEAM_CHATS_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "team_chats.lock"

TEAM_PENDING = "pending"
TEAM_APPROVED = "approved"
TEAM_DENIED = "denied"
TEAM_STATUSES = {TEAM_PENDING, TEAM_APPROVED, TEAM_DENIED}
GROUP_CHAT_TYPES = {"group", "supergroup"}


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, TEAM_CHATS_PATH, TEAM_CHATS_LOCK_PATH
    DRIVE_ROOT = drive_root
    TEAM_CHATS_PATH = drive_root / "state" / "team_chats.json"
    TEAM_CHATS_LOCK_PATH = drive_root / "locks" / "team_chats.lock"


def is_group_chat_type(chat_type: Any) -> bool:
    return str(chat_type or "").lower() in GROUP_CHAT_TYPES


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _chat_key(chat_id: int) -> str:
    return str(int(chat_id))


def team_slug_for_chat(chat_id: int) -> str:
    return f"tg_{abs(int(chat_id))}"


def team_root(drive_root: pathlib.Path, chat_id: int) -> pathlib.Path:
    return (drive_root / "teams" / team_slug_for_chat(chat_id)).resolve()


def _normalize_status(value: Any, default: str = TEAM_PENDING) -> str:
    status = str(value or "").strip().lower()
    return status if status in TEAM_STATUSES else default


def _load_unlocked() -> Dict[str, Any]:
    try:
        if not TEAM_CHATS_PATH.exists():
            return {"team_chats": {}}
        data = json.loads(TEAM_CHATS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"team_chats": {}}
        if not isinstance(data.get("team_chats"), dict):
            data["team_chats"] = {}
        return data
    except Exception:
        log.warning("Failed to load team chat registry", exc_info=True)
        return {"team_chats": {}}


def _save_unlocked(data: Dict[str, Any]) -> None:
    data.setdefault("team_chats", {})
    data["updated_at"] = _now_iso()
    atomic_write_text(TEAM_CHATS_PATH, json.dumps(data, ensure_ascii=False, indent=2))


def _user_snapshot(from_user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = from_user or {}
    snap: Dict[str, Any] = {}
    if src.get("id") is not None:
        try:
            snap["user_id"] = int(src.get("id") or 0)
        except Exception:
            pass
    for key in ("username", "first_name", "last_name"):
        val = str(src.get(key) or "").strip()
        if val:
            snap[key] = val
    return snap


def _chat_title(chat: Dict[str, Any]) -> str:
    return str(chat.get("title") or chat.get("username") or "").strip()


def _update_chat_metadata(
    rec: Dict[str, Any],
    *,
    chat: Dict[str, Any],
    requested_by: Optional[Dict[str, Any]] = None,
) -> None:
    chat_id = int(chat.get("id") or rec.get("chat_id") or 0)
    rec["chat_id"] = chat_id
    rec["chat_type"] = str(chat.get("type") or rec.get("chat_type") or "").lower()
    title = _chat_title(chat)
    if title:
        rec["title"] = title
    rec["slug"] = team_slug_for_chat(chat_id)
    rec["drive_root"] = str(team_root(DRIVE_ROOT, chat_id))
    rec["last_seen_at"] = _now_iso()
    if requested_by:
        snap = _user_snapshot(requested_by)
        if snap:
            rec.setdefault("requested_by", snap)
            rec["last_requested_by"] = snap


def _ensure_team_workspace_files(root: pathlib.Path) -> None:
    for sub in ("memory", "logs", "task_results", "uploads", "inbox", "polls", "locks", "archive", "index"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for rel in ("logs/chat.jsonl", "inbox/messages.jsonl"):
        p = root / rel
        if not p.exists():
            p.write_text("", encoding="utf-8")
    try:
        from ouroboros.memory import Memory
        Memory(drive_root=root).ensure_files()
    except Exception:
        log.warning("Failed to initialize team memory files at %s", root, exc_info=True)


def ensure_team_workspace(drive_root: pathlib.Path, chat_id: int) -> pathlib.Path:
    root = team_root(drive_root, chat_id)
    _ensure_team_workspace_files(root)
    return root


def request_team_chat(
    drive_root: pathlib.Path,
    chat: Dict[str, Any],
    requested_by: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Create or refresh a pending team-chat request.

    Returns (record, created, should_notify_admins). Approved workspaces are
    initialized, pending/denied chats are only recorded.
    """
    init(drive_root)
    chat_id = int(chat.get("id") or 0)
    key = _chat_key(chat_id)
    created = False
    should_notify = False

    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        data = _load_unlocked()
        chats = data.setdefault("team_chats", {})
        rec = chats.get(key)
        now = _now_iso()
        if not isinstance(rec, dict):
            rec = {
                "chat_id": chat_id,
                "status": TEAM_PENDING,
                "created_at": now,
                "requested_at": now,
                "notifications": [],
                "members": {},
            }
            chats[key] = rec
            created = True
            should_notify = True
        else:
            rec["status"] = _normalize_status(rec.get("status"))
            if rec["status"] == TEAM_PENDING and not rec.get("notifications"):
                should_notify = True

        _update_chat_metadata(rec, chat=chat, requested_by=requested_by)
        _save_unlocked(data)
        out = dict(rec)
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)

    if out.get("status") == TEAM_APPROVED:
        ensure_team_workspace(drive_root, chat_id)
    return out, created, should_notify


def get_team_chat(drive_root: pathlib.Path, chat_id: int) -> Optional[Dict[str, Any]]:
    init(drive_root)
    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        rec = _load_unlocked().setdefault("team_chats", {}).get(_chat_key(chat_id))
        return dict(rec) if isinstance(rec, dict) else None
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)


def list_team_chats(
    drive_root: pathlib.Path,
    *,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    init(drive_root)
    target = _normalize_status(status, default="") if status else None
    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rows: List[Dict[str, Any]] = []
        changed = False
        for rec in data.setdefault("team_chats", {}).values():
            if not isinstance(rec, dict):
                continue
            normalized = _normalize_status(rec.get("status"))
            if normalized != rec.get("status"):
                rec["status"] = normalized
                changed = True
            if target and normalized != target:
                continue
            rows.append(dict(rec))
        if changed:
            _save_unlocked(data)
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)

    rows.sort(key=lambda r: str(r.get("created_at") or r.get("last_seen_at") or ""), reverse=True)
    return rows


def append_team_chat_notification(
    drive_root: pathlib.Path,
    chat_id: int,
    *,
    admin_chat_id: int,
    message_id: int,
) -> None:
    init(drive_root)
    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rec = data.setdefault("team_chats", {}).get(_chat_key(chat_id))
        if not isinstance(rec, dict):
            return
        notifications = rec.setdefault("notifications", [])
        notifications.append({
            "admin_chat_id": int(admin_chat_id),
            "message_id": int(message_id),
            "ts": _now_iso(),
        })
        rec["admin_notified_at"] = _now_iso()
        _save_unlocked(data)
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)


def set_team_chat_status(
    drive_root: pathlib.Path,
    chat_id: int,
    status: str,
    *,
    decided_by: Optional[int] = None,
    allow_terminal_change: bool = False,
) -> Dict[str, Any]:
    init(drive_root)
    target = _normalize_status(status)
    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rec = data.setdefault("team_chats", {}).get(_chat_key(chat_id))
        if not isinstance(rec, dict):
            return {"status": "missing", "chat_id": int(chat_id)}
        old_status = _normalize_status(rec.get("status"))
        if old_status in (TEAM_APPROVED, TEAM_DENIED):
            if old_status == target or not allow_terminal_change:
                return {"status": old_status, "old_status": old_status, "record": dict(rec), "chat_id": int(chat_id)}
        rec["status"] = target
        rec["decided_at"] = _now_iso()
        if decided_by is not None:
            rec["decided_by"] = int(decided_by)
        if target == TEAM_APPROVED:
            rec["approved_at"] = rec["decided_at"]
        elif target == TEAM_DENIED:
            rec["denied_at"] = rec["decided_at"]
        _save_unlocked(data)
        out = dict(rec)
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)

    if target == TEAM_APPROVED:
        ensure_team_workspace(drive_root, chat_id)
    return {"status": target, "old_status": old_status, "record": out, "chat_id": int(chat_id)}


def note_team_member_seen(
    drive_root: pathlib.Path,
    chat_id: int,
    from_user: Optional[Dict[str, Any]],
) -> None:
    snap = _user_snapshot(from_user)
    uid = snap.get("user_id")
    if not uid:
        return
    init(drive_root)
    lock_fd = acquire_file_lock(TEAM_CHATS_LOCK_PATH)
    try:
        data = _load_unlocked()
        rec = data.setdefault("team_chats", {}).get(_chat_key(chat_id))
        if not isinstance(rec, dict):
            return
        members = rec.setdefault("members", {})
        row = members.setdefault(str(uid), {})
        row.update(snap)
        row["last_seen_at"] = _now_iso()
        _save_unlocked(data)
    finally:
        release_file_lock(TEAM_CHATS_LOCK_PATH, lock_fd)


def team_chat_status(rec: Optional[Dict[str, Any]]) -> str:
    if not isinstance(rec, dict):
        return ""
    return _normalize_status(rec.get("status"))
