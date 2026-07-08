"""Supervisor user registry and per-user workspace helpers.

Multi-user mode keeps the codebase, Telegram bot, and budget global, while
normal users get isolated memory/log roots under DRIVE_ROOT/users/<user_id>/.
The admin/owner keeps using the legacy DRIVE_ROOT so existing identity and
evolution state remain intact.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from supervisor.state import (
    acquire_file_lock,
    append_jsonl,
    atomic_write_text,
    release_file_lock,
)

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
USERS_PATH: pathlib.Path = DRIVE_ROOT / "state" / "users.json"
USERS_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "users.lock"
ACCESS_APPROVED = "approved"
ACCESS_PENDING = "pending"
ACCESS_DENIED = "denied"
ACCESS_STATUSES = {ACCESS_APPROVED, ACCESS_PENDING, ACCESS_DENIED}
ACCESS_REQUESTS_LOG = "access_requests.jsonl"


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, USERS_PATH, USERS_LOCK_PATH
    DRIVE_ROOT = drive_root
    USERS_PATH = drive_root / "state" / "users.json"
    USERS_LOCK_PATH = drive_root / "locks" / "users.lock"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _user_key(user_id: int) -> str:
    return str(int(user_id))


def user_root(drive_root: pathlib.Path, user_id: int) -> pathlib.Path:
    return (drive_root / "users" / _user_key(user_id)).resolve()


def _load_users_unlocked() -> Dict[str, Any]:
    try:
        if not USERS_PATH.exists():
            return {"users": {}}
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"users": {}}
        users = data.get("users")
        if not isinstance(users, dict):
            data["users"] = {}
        return data
    except Exception:
        log.warning("Failed to load users registry", exc_info=True)
        return {"users": {}}


def _save_users_unlocked(data: Dict[str, Any]) -> None:
    data.setdefault("users", {})
    data["updated_at"] = _now_iso()
    atomic_write_text(USERS_PATH, json.dumps(data, ensure_ascii=False, indent=2))


def _normalize_access_status(value: Any, default: str = ACCESS_APPROVED) -> str:
    status = str(value or "").strip().lower()
    if status in ACCESS_STATUSES:
        return status
    return default


def user_access_status(rec: Dict[str, Any]) -> str:
    """Return a normalized access status for a user registry record."""
    if str(rec.get("role") or "").lower() == "admin":
        return ACCESS_APPROVED
    return _normalize_access_status(rec.get("access_status"), default=ACCESS_APPROVED)


def _apply_access_status(
    rec: Dict[str, Any],
    status: str,
    decided_by: Optional[int] = None,
) -> None:
    status = _normalize_access_status(status, default=ACCESS_PENDING)
    now = _now_iso()
    rec["access_status"] = status
    if status == ACCESS_APPROVED:
        rec["access_approved_at"] = now
        rec.pop("access_denied_at", None)
    elif status == ACCESS_DENIED:
        rec["access_denied_at"] = now
    elif status == ACCESS_PENDING:
        rec.setdefault("access_requested_at", now)
        rec["access_last_requested_at"] = now
    if decided_by is not None:
        rec["access_decided_by"] = int(decided_by)


def _migrate_legacy_access(rec: Dict[str, Any]) -> str:
    """Existing user records predate access approval and keep their access."""
    if str(rec.get("role") or "").lower() == "admin":
        rec["access_status"] = ACCESS_APPROVED
        rec.setdefault("access_approved_at", rec.get("created_at") or _now_iso())
        return ACCESS_APPROVED

    if "access_status" not in rec:
        rec["access_status"] = ACCESS_APPROVED
        rec.setdefault("access_approved_at", rec.get("created_at") or _now_iso())
        rec.setdefault("access_approved_reason", "legacy_existing_user")
        return ACCESS_APPROVED

    status = _normalize_access_status(rec.get("access_status"), default=ACCESS_APPROVED)
    rec["access_status"] = status
    return status


def _update_user_metadata(
    rec: Dict[str, Any],
    *,
    user_id: int,
    chat_id: int,
    from_user: Optional[Dict[str, Any]],
    role: Optional[str] = None,
    drive_root: Optional[pathlib.Path] = None,
) -> None:
    from_user = from_user or {}
    rec["user_id"] = int(user_id)
    rec["chat_id"] = int(chat_id)
    if role:
        rec["role"] = role
    if drive_root is not None:
        rec["drive_root"] = str(drive_root)
    rec["last_seen_at"] = _now_iso()
    if from_user.get("username"):
        rec["username"] = str(from_user.get("username"))
    if from_user.get("first_name"):
        rec["first_name"] = str(from_user.get("first_name"))
    if from_user.get("last_name"):
        rec["last_name"] = str(from_user.get("last_name"))


def _ensure_workspace_files(root: pathlib.Path) -> None:
    for sub in ("memory", "logs", "task_results", "locks", "archive", "index"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    chat_path = root / "logs" / "chat.jsonl"
    if not chat_path.exists():
        chat_path.write_text("", encoding="utf-8")

    # Reuse the Memory defaults so user workspaces stay compatible with the
    # rest of the agent without copying the admin's private identity.
    try:
        from ouroboros.memory import Memory
        Memory(drive_root=root).ensure_files()
    except Exception:
        log.warning("Failed to initialize user memory files at %s", root, exc_info=True)


def ensure_user_workspace(
    drive_root: pathlib.Path,
    user_id: int,
    chat_id: int,
    from_user: Optional[Dict[str, Any]] = None,
    role: str = "user",
    use_global_root: bool = False,
    access_status: Optional[str] = None,
) -> Tuple[pathlib.Path, bool, Dict[str, Any]]:
    """Ensure a user has a registry entry and workspace.

    Returns (effective_drive_root, created, user_record). Admin callers can pass
    use_global_root=True to preserve the legacy global memory location.
    """
    init(drive_root)
    from_user = from_user or {}
    key = _user_key(user_id)
    root = drive_root if use_global_root else user_root(drive_root, user_id)
    created = False

    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        users = data.setdefault("users", {})
        rec = users.get(key)
        if not isinstance(rec, dict):
            rec = {
                "user_id": int(user_id),
                "chat_id": int(chat_id),
                "role": role,
                "drive_root": str(root),
                "created_at": _now_iso(),
            }
            rec["access_status"] = _normalize_access_status(
                access_status,
                default=ACCESS_APPROVED,
            )
            if rec["access_status"] == ACCESS_APPROVED:
                rec["access_approved_at"] = rec["created_at"]
            elif rec["access_status"] == ACCESS_PENDING:
                rec["access_requested_at"] = rec["created_at"]
            users[key] = rec
            created = True
        else:
            _migrate_legacy_access(rec)

        _update_user_metadata(
            rec,
            user_id=user_id,
            chat_id=chat_id,
            from_user=from_user,
            role=role,
            drive_root=root,
        )
        if role == "admin":
            _apply_access_status(rec, ACCESS_APPROVED)
        elif access_status is not None:
            _apply_access_status(rec, access_status)
        _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)

    _ensure_workspace_files(root)
    return root, created, dict(rec)


def request_user_access(
    drive_root: pathlib.Path,
    user_id: int,
    chat_id: int,
    from_user: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Create or refresh a user access request.

    Returns (user_record, created, should_notify_admins). Legacy records without
    access_status are treated as approved so existing users keep access.
    """
    init(drive_root)
    key = _user_key(user_id)
    root = user_root(drive_root, user_id)
    created = False
    should_notify_admins = False

    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        users = data.setdefault("users", {})
        rec = users.get(key)
        if not isinstance(rec, dict):
            now = _now_iso()
            rec = {
                "user_id": int(user_id),
                "chat_id": int(chat_id),
                "role": "user",
                "drive_root": str(root),
                "created_at": now,
                "access_status": ACCESS_PENDING,
                "access_requested_at": now,
                "access_last_requested_at": now,
            }
            users[key] = rec
            created = True
            should_notify_admins = True
        else:
            status = _migrate_legacy_access(rec)
            if status == ACCESS_PENDING:
                rec["access_last_requested_at"] = _now_iso()
                should_notify_admins = not bool(rec.get("access_admin_notified_at"))
            elif status == ACCESS_DENIED:
                rec["access_last_requested_at"] = _now_iso()

        _update_user_metadata(
            rec,
            user_id=user_id,
            chat_id=chat_id,
            from_user=from_user,
            role=str(rec.get("role") or "user").lower(),
            drive_root=pathlib.Path(str(rec.get("drive_root") or root)),
        )
        _save_users_unlocked(data)
        return dict(rec), created, should_notify_admins
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)


def log_access_request_message(
    drive_root: pathlib.Path,
    user_id: int,
    chat_id: int,
    text: str,
    *,
    from_user: Optional[Dict[str, Any]] = None,
    access_status: str = ACCESS_PENDING,
) -> None:
    """Audit an access-request message without adding it to chat history."""
    init(drive_root)
    from_user = from_user or {}
    row: Dict[str, Any] = {
        "ts": _now_iso(),
        "type": "user_access_request_message",
        "direction": "in",
        "chat_id": int(chat_id),
        "user_id": int(user_id),
        "access_status": _normalize_access_status(access_status, default=ACCESS_PENDING),
        "text": str(text or ""),
    }
    for key in ("username", "first_name", "last_name"):
        value = str(from_user.get(key) or "").strip()
        if value:
            row[key] = value
    append_jsonl(DRIVE_ROOT / "logs" / ACCESS_REQUESTS_LOG, row)


def mark_access_request_notified(drive_root: pathlib.Path, user_id: int) -> None:
    init(drive_root)
    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        rec = data.setdefault("users", {}).get(_user_key(user_id))
        if isinstance(rec, dict):
            rec["access_admin_notified_at"] = _now_iso()
            _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)


def append_access_request_notification(
    drive_root: pathlib.Path,
    user_id: int,
    *,
    admin_chat_id: int,
    message_id: int,
) -> None:
    init(drive_root)
    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        rec = data.setdefault("users", {}).get(_user_key(user_id))
        if not isinstance(rec, dict):
            return
        notifications = rec.setdefault("access_notifications", [])
        notifications.append({
            "admin_chat_id": int(admin_chat_id),
            "message_id": int(message_id),
            "ts": _now_iso(),
        })
        rec["access_admin_notified_at"] = _now_iso()
        _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)


def list_user_records(
    drive_root: pathlib.Path,
    *,
    access_status: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    init(drive_root)
    target_status = _normalize_access_status(access_status, default="") if access_status else None
    target_role = str(role or "").strip().lower() or None

    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        records: List[Dict[str, Any]] = []
        changed = False
        for rec in data.setdefault("users", {}).values():
            if not isinstance(rec, dict):
                continue
            before = rec.get("access_status")
            status = _migrate_legacy_access(rec)
            if before != rec.get("access_status"):
                changed = True
            if target_status and status != target_status:
                continue
            if target_role and str(rec.get("role") or "").lower() != target_role:
                continue
            records.append(dict(rec))
        if changed:
            _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)

    def _sort_key(item: Dict[str, Any]) -> Tuple[str, int]:
        return (
            str(item.get("access_requested_at") or item.get("created_at") or ""),
            int(item.get("user_id") or 0),
        )

    return sorted(records, key=_sort_key)


def set_user_access_status(
    drive_root: pathlib.Path,
    user_ids: Iterable[int],
    status: str,
    *,
    decided_by: Optional[int] = None,
) -> List[Dict[str, Any]]:
    init(drive_root)
    target_status = _normalize_access_status(status, default=ACCESS_PENDING)
    results: List[Dict[str, Any]] = []

    lock_fd = acquire_file_lock(USERS_LOCK_PATH)
    try:
        data = _load_users_unlocked()
        users = data.setdefault("users", {})
        for user_id in user_ids:
            key = _user_key(int(user_id))
            rec = users.get(key)
            if not isinstance(rec, dict):
                results.append({
                    "user_id": int(user_id),
                    "status": "missing",
                    "old_status": "",
                    "record": {},
                })
                continue

            old_status = _migrate_legacy_access(rec)
            if str(rec.get("role") or "").lower() == "admin":
                _apply_access_status(rec, ACCESS_APPROVED, decided_by=decided_by)
                new_status = ACCESS_APPROVED
            else:
                _apply_access_status(rec, target_status, decided_by=decided_by)
                new_status = target_status
            results.append({
                "user_id": int(user_id),
                "status": new_status,
                "old_status": old_status,
                "record": dict(rec),
            })
        _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)

    return results
