"""Telegram team poll state and result collection helpers."""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from supervisor.state import acquire_file_lock, atomic_write_text, release_file_lock
from ouroboros.utils import append_jsonl

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
TEAM_POLLS_PATH: pathlib.Path = DRIVE_ROOT / "state" / "team_polls.json"
TEAM_POLLS_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "team_polls.lock"

POLL_ACTIVE = "active"
POLL_CLOSED = "closed"
POLL_FAILED = "failed"


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, TEAM_POLLS_PATH, TEAM_POLLS_LOCK_PATH
    DRIVE_ROOT = pathlib.Path(drive_root)
    TEAM_POLLS_PATH = DRIVE_ROOT / "state" / "team_polls.json"
    TEAM_POLLS_LOCK_PATH = DRIVE_ROOT / "locks" / "team_polls.lock"


def new_poll_uid() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_unlocked() -> Dict[str, Any]:
    try:
        if not TEAM_POLLS_PATH.exists():
            return {"polls": {}, "telegram_poll_ids": {}}
        data = json.loads(TEAM_POLLS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"polls": {}, "telegram_poll_ids": {}}
        if not isinstance(data.get("polls"), dict):
            data["polls"] = {}
        if not isinstance(data.get("telegram_poll_ids"), dict):
            data["telegram_poll_ids"] = {}
        return data
    except Exception:
        log.warning("Failed to load team poll registry", exc_info=True)
        return {"polls": {}, "telegram_poll_ids": {}}


def _save_unlocked(data: Dict[str, Any]) -> None:
    data.setdefault("polls", {})
    data.setdefault("telegram_poll_ids", {})
    data["updated_at"] = _now_iso()
    atomic_write_text(TEAM_POLLS_PATH, json.dumps(data, ensure_ascii=False, indent=2))


def _team_poll_dir(team_drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(team_drive_root) / "polls"


def _snapshot_path(rec: Dict[str, Any]) -> Optional[pathlib.Path]:
    drive_root = str(rec.get("drive_root") or "").strip()
    poll_uid = str(rec.get("poll_uid") or "").strip()
    if not drive_root or not poll_uid:
        return None
    return _team_poll_dir(pathlib.Path(drive_root)) / f"{poll_uid}.json"


def _write_snapshot(rec: Dict[str, Any]) -> None:
    path = _snapshot_path(rec)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(rec, ensure_ascii=False, indent=2))


def _append_poll_log(rec: Dict[str, Any], event: Dict[str, Any]) -> None:
    drive_root = str(rec.get("drive_root") or "").strip()
    if not drive_root:
        return
    append_jsonl(pathlib.Path(drive_root) / "logs" / "polls.jsonl", event)


def _chat_log_poll(rec: Dict[str, Any], text: str) -> None:
    drive_root = str(rec.get("drive_root") or "").strip()
    if not drive_root:
        return
    append_jsonl(pathlib.Path(drive_root) / "logs" / "chat.jsonl", {
        "ts": _now_iso(),
        "direction": "out",
        "chat_id": rec.get("chat_id"),
        "user_id": 0,
        "text": text,
    })


def _poll_options_from_poll(poll: Dict[str, Any], fallback: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_options = poll.get("options") if isinstance(poll, dict) else None
    if isinstance(raw_options, list) and raw_options:
        for item in raw_options:
            if not isinstance(item, dict):
                continue
            rows.append({
                "text": str(item.get("text") or ""),
                "persistent_id": str(item.get("persistent_id") or ""),
                "voter_count": int(item.get("voter_count") or 0),
            })
    if rows:
        return rows
    return [{"text": str(opt), "persistent_id": "", "voter_count": 0} for opt in fallback]


def _apply_poll_object(rec: Dict[str, Any], poll: Dict[str, Any]) -> None:
    if not isinstance(poll, dict):
        return
    rec["telegram_poll_id"] = str(poll.get("id") or rec.get("telegram_poll_id") or "")
    rec["question"] = str(poll.get("question") or rec.get("question") or "")
    rec["options"] = _poll_options_from_poll(poll, [str(o.get("text") or "") for o in rec.get("options") or []])
    rec["total_voter_count"] = int(poll.get("total_voter_count") or rec.get("total_voter_count") or 0)
    rec["is_closed"] = bool(poll.get("is_closed"))
    rec["is_anonymous"] = bool(poll.get("is_anonymous", rec.get("is_anonymous", True)))
    rec["allows_multiple_answers"] = bool(poll.get("allows_multiple_answers", rec.get("allows_multiple_answers", False)))
    rec["allows_revoting"] = bool(poll.get("allows_revoting", rec.get("allows_revoting", True)))
    if poll.get("open_period") is not None:
        rec["open_period"] = int(poll.get("open_period") or 0)
    if poll.get("close_date") is not None:
        rec["close_date"] = int(poll.get("close_date") or 0)
    if rec["is_closed"]:
        rec["status"] = POLL_CLOSED
        rec.setdefault("closed_at", _now_iso())
    rec["last_poll_update_at"] = _now_iso()


def _answer_voter_key(answer: Dict[str, Any]) -> str:
    user = answer.get("user") if isinstance(answer, dict) else None
    if isinstance(user, dict) and user.get("id") is not None:
        return f"user:{int(user.get('id') or 0)}"
    voter_chat = answer.get("voter_chat") if isinstance(answer, dict) else None
    if isinstance(voter_chat, dict) and voter_chat.get("id") is not None:
        return f"chat:{int(voter_chat.get('id') or 0)}"
    return f"unknown:{uuid.uuid4().hex[:8]}"


def _snapshot_user(user: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if user.get("id") is not None:
        out["id"] = int(user.get("id") or 0)
    for key in ("username", "first_name", "last_name"):
        val = str(user.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def _recompute_counts_from_answers(rec: Dict[str, Any]) -> None:
    answers = rec.get("answers")
    options = rec.get("options")
    if not isinstance(answers, dict) or not answers or not isinstance(options, list):
        return
    counts = [0 for _ in options]
    total = 0
    for row in answers.values():
        if not isinstance(row, dict):
            continue
        option_ids = row.get("option_ids")
        if not isinstance(option_ids, list) or not option_ids:
            continue
        total += 1
        for raw_idx in option_ids:
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if 0 <= idx < len(counts):
                counts[idx] += 1
    for idx, count in enumerate(counts):
        if isinstance(options[idx], dict):
            options[idx]["voter_count"] = count
    rec["total_voter_count"] = total


def record_poll_sent(
    shared_drive_root: pathlib.Path,
    team_drive_root: pathlib.Path,
    *,
    poll_uid: str,
    chat_id: int,
    message_id: int,
    poll: Dict[str, Any],
    question: str,
    options: List[str],
    created_by: Optional[int] = None,
    task_id: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    init(shared_drive_root)
    now = _now_iso()
    poll_uid = str(poll_uid or "").strip() or new_poll_uid()
    telegram_poll_id = str(poll.get("id") or "")
    rec: Dict[str, Any] = {
        "poll_uid": poll_uid,
        "telegram_poll_id": telegram_poll_id,
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "drive_root": str(pathlib.Path(team_drive_root).resolve()),
        "question": str(question or ""),
        "options": [{"text": str(opt), "persistent_id": "", "voter_count": 0} for opt in options],
        "status": POLL_ACTIVE,
        "is_closed": False,
        "created_at": now,
        "created_by": int(created_by) if created_by is not None else None,
        "task_id": str(task_id or ""),
        "answers": {},
    }
    if settings:
        rec["settings"] = dict(settings)
        for key in ("is_anonymous", "allows_multiple_answers"):
            if key in settings:
                rec[key] = bool(settings[key])
    _apply_poll_object(rec, poll)

    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    try:
        data = _load_unlocked()
        data.setdefault("polls", {})[poll_uid] = rec
        if telegram_poll_id:
            data.setdefault("telegram_poll_ids", {})[telegram_poll_id] = poll_uid
        _save_unlocked(data)
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)

    _write_snapshot(rec)
    _append_poll_log(rec, {
        "ts": now,
        "type": "team_poll_sent",
        "poll_uid": poll_uid,
        "telegram_poll_id": telegram_poll_id,
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "question": rec.get("question"),
    })
    _chat_log_poll(rec, f"[poll] {rec.get('question')}")
    return rec


def record_poll_failed(
    shared_drive_root: pathlib.Path,
    team_drive_root: pathlib.Path,
    *,
    poll_uid: str,
    chat_id: int,
    question: str,
    options: List[str],
    error: str,
    created_by: Optional[int] = None,
    task_id: str = "",
) -> Dict[str, Any]:
    init(shared_drive_root)
    now = _now_iso()
    poll_uid = str(poll_uid or "").strip() or new_poll_uid()
    rec: Dict[str, Any] = {
        "poll_uid": poll_uid,
        "telegram_poll_id": "",
        "chat_id": int(chat_id),
        "message_id": 0,
        "drive_root": str(pathlib.Path(team_drive_root).resolve()),
        "question": str(question or ""),
        "options": [{"text": str(opt), "persistent_id": "", "voter_count": 0} for opt in options],
        "status": POLL_FAILED,
        "is_closed": True,
        "created_at": now,
        "created_by": int(created_by) if created_by is not None else None,
        "task_id": str(task_id or ""),
        "error": str(error or ""),
        "answers": {},
    }

    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    try:
        data = _load_unlocked()
        data.setdefault("polls", {})[poll_uid] = rec
        _save_unlocked(data)
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)

    _write_snapshot(rec)
    _append_poll_log(rec, {
        "ts": now,
        "type": "team_poll_failed",
        "poll_uid": poll_uid,
        "chat_id": int(chat_id),
        "question": question,
        "error": str(error or ""),
    })
    return rec


def find_poll_record(
    shared_drive_root: pathlib.Path,
    *,
    poll_ref: str = "",
    telegram_poll_id: str = "",
    chat_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    init(shared_drive_root)
    ref = str(poll_ref or telegram_poll_id or "").strip()
    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    try:
        data = _load_unlocked()
        polls = data.setdefault("polls", {})
        uid = data.setdefault("telegram_poll_ids", {}).get(ref) if ref else None
        if uid and isinstance(polls.get(uid), dict):
            return dict(polls[uid])
        if ref and isinstance(polls.get(ref), dict):
            return dict(polls[ref])
        for rec in polls.values():
            if not isinstance(rec, dict):
                continue
            if chat_id is not None and int(rec.get("chat_id") or 0) != int(chat_id):
                continue
            if not ref:
                continue
            if ref in (
                str(rec.get("poll_uid") or ""),
                str(rec.get("telegram_poll_id") or ""),
                str(rec.get("message_id") or ""),
            ):
                return dict(rec)
        return None
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)


def record_poll_answer(shared_drive_root: pathlib.Path, poll_answer: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    init(shared_drive_root)
    telegram_poll_id = str(poll_answer.get("poll_id") or "").strip()
    if not telegram_poll_id:
        return None

    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    rec: Optional[Dict[str, Any]] = None
    try:
        data = _load_unlocked()
        uid = data.setdefault("telegram_poll_ids", {}).get(telegram_poll_id)
        if not uid or not isinstance(data.setdefault("polls", {}).get(uid), dict):
            return None
        rec = data["polls"][uid]
        answers = rec.setdefault("answers", {})
        voter_key = _answer_voter_key(poll_answer)
        option_ids = poll_answer.get("option_ids")
        if not isinstance(option_ids, list):
            option_ids = []
        row: Dict[str, Any] = {
            "ts": _now_iso(),
            "poll_id": telegram_poll_id,
            "option_ids": [int(x) for x in option_ids if isinstance(x, int) or str(x).isdigit()],
        }
        option_persistent_ids = poll_answer.get("option_persistent_ids")
        if isinstance(option_persistent_ids, list):
            row["option_persistent_ids"] = [str(x) for x in option_persistent_ids]
        user = poll_answer.get("user")
        if isinstance(user, dict):
            row["user"] = _snapshot_user(user)
        voter_chat = poll_answer.get("voter_chat")
        if isinstance(voter_chat, dict):
            row["voter_chat"] = _snapshot_user(voter_chat)
        answers[voter_key] = row
        rec["last_answer_at"] = row["ts"]
        _recompute_counts_from_answers(rec)
        data["polls"][uid] = rec
        _save_unlocked(data)
        rec = dict(rec)
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)

    if rec is None:
        return None
    _write_snapshot(rec)
    _append_poll_log(rec, {
        "ts": _now_iso(),
        "type": "team_poll_answer",
        "poll_uid": rec.get("poll_uid"),
        "telegram_poll_id": telegram_poll_id,
        "answer": poll_answer,
    })
    return rec


def record_poll_update(
    shared_drive_root: pathlib.Path,
    poll: Dict[str, Any],
    *,
    status: str = "",
) -> Optional[Dict[str, Any]]:
    init(shared_drive_root)
    telegram_poll_id = str(poll.get("id") or "").strip()
    if not telegram_poll_id:
        return None

    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    rec: Optional[Dict[str, Any]] = None
    try:
        data = _load_unlocked()
        uid = data.setdefault("telegram_poll_ids", {}).get(telegram_poll_id)
        if not uid or not isinstance(data.setdefault("polls", {}).get(uid), dict):
            return None
        rec = data["polls"][uid]
        _apply_poll_object(rec, poll)
        if status:
            rec["status"] = str(status)
            if status == POLL_CLOSED:
                rec["is_closed"] = True
                rec["closed_at"] = _now_iso()
        data["polls"][uid] = rec
        _save_unlocked(data)
        rec = dict(rec)
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)

    if rec is None:
        return None
    _write_snapshot(rec)
    _append_poll_log(rec, {
        "ts": _now_iso(),
        "type": "team_poll_update",
        "poll_uid": rec.get("poll_uid"),
        "telegram_poll_id": telegram_poll_id,
        "status": rec.get("status"),
    })
    return rec


def list_team_poll_records(
    shared_drive_root: pathlib.Path,
    *,
    chat_id: Optional[int] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    init(shared_drive_root)
    lock_fd = acquire_file_lock(TEAM_POLLS_LOCK_PATH)
    try:
        rows = []
        for rec in _load_unlocked().setdefault("polls", {}).values():
            if not isinstance(rec, dict):
                continue
            if chat_id is not None and int(rec.get("chat_id") or 0) != int(chat_id):
                continue
            rows.append(dict(rec))
    finally:
        release_file_lock(TEAM_POLLS_LOCK_PATH, lock_fd)
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return rows[:max(1, int(limit or 20))]


def handle_poll_update(shared_drive_root: pathlib.Path, update: Dict[str, Any]) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Handle Telegram poll/poll_answer updates without routing them to chat tasks."""
    poll_answer = update.get("poll_answer") if isinstance(update, dict) else None
    if isinstance(poll_answer, dict):
        return True, record_poll_answer(shared_drive_root, poll_answer)
    poll_update = update.get("poll") if isinstance(update, dict) else None
    if isinstance(poll_update, dict):
        return True, record_poll_update(shared_drive_root, poll_update)
    return False, None
