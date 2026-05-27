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
from typing import Any, Dict, Optional, Tuple

from supervisor.state import (
    acquire_file_lock,
    atomic_write_text,
    release_file_lock,
)

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
USERS_PATH: pathlib.Path = DRIVE_ROOT / "state" / "users.json"
USERS_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "users.lock"


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
            users[key] = rec
            created = True

        rec["chat_id"] = int(chat_id)
        rec["role"] = role
        rec["drive_root"] = str(root)
        rec["last_seen_at"] = _now_iso()
        if from_user.get("username"):
            rec["username"] = str(from_user.get("username"))
        if from_user.get("first_name"):
            rec["first_name"] = str(from_user.get("first_name"))
        if from_user.get("last_name"):
            rec["last_name"] = str(from_user.get("last_name"))
        _save_users_unlocked(data)
    finally:
        release_file_lock(USERS_LOCK_PATH, lock_fd)

    _ensure_workspace_files(root)
    return root, created, dict(rec)

