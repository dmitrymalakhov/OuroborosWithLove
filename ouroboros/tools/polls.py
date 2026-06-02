"""Telegram team poll tools."""

from __future__ import annotations

import json
import pathlib
import uuid
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


MAX_QUESTION_CHARS = 300
MAX_OPTION_CHARS = 100
MAX_OPTIONS = 12


def _require_team(ctx: ToolContext) -> str:
    if not ctx.is_team_workspace or not ctx.team_slug or not ctx.current_chat_id:
        return "⚠️ Poll tools are available only inside an approved Telegram group workspace."
    return ""


def _scope(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "user_id": ctx.current_user_id,
        "user_role": ctx.user_role,
        "drive_root": str(ctx.drive_root),
        "shared_drive_root": str(ctx.shared_drive_root or ctx.drive_root),
        "chat_type": ctx.chat_type,
        "team_chat_id": ctx.team_chat_id,
        "team_slug": ctx.team_slug,
        "is_team_workspace": ctx.is_team_workspace,
        "task_id": ctx.task_id or "",
    }


def _coerce_options(options: Any) -> List[str]:
    raw: List[Any]
    if isinstance(options, list):
        raw = options
    elif isinstance(options, str):
        stripped = options.strip()
        parsed = None
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            raw = parsed
        else:
            sep = "\n" if "\n" in stripped else ","
            raw = [part for part in stripped.split(sep)]
    else:
        raw = []
    out: List[str] = []
    for item in raw:
        text = str(item.get("text") if isinstance(item, dict) else item).strip()
        if not text:
            continue
        if text not in out:
            out.append(text[:MAX_OPTION_CHARS])
    return out


def _validate_poll(question: str, options: List[str]) -> str:
    if not question.strip():
        return "⚠️ Poll question is empty."
    if len(question) > MAX_QUESTION_CHARS:
        return f"⚠️ Poll question is too long ({len(question)} chars, max {MAX_QUESTION_CHARS})."
    if len(options) < 2:
        return "⚠️ Poll needs at least 2 answer options."
    if len(options) > MAX_OPTIONS:
        return f"⚠️ Poll supports at most {MAX_OPTIONS} options."
    for opt in options:
        if len(opt) > MAX_OPTION_CHARS:
            return f"⚠️ Poll option is too long ({len(opt)} chars, max {MAX_OPTION_CHARS}): {opt[:40]}"
    return ""


def _safe_open_period(value: int) -> int:
    try:
        seconds = int(value or 0)
    except Exception:
        return 0
    if seconds <= 0:
        return 0
    return max(5, min(seconds, 2_628_000))


def _team_poll_create(
    ctx: ToolContext,
    question: str,
    options: Any,
    allows_multiple_answers: bool = False,
    is_anonymous: bool = False,
    open_period_seconds: int = 0,
) -> str:
    err = _require_team(ctx)
    if err:
        return err
    q = str(question or "").strip()
    opts = _coerce_options(options)
    err = _validate_poll(q, opts)
    if err:
        return err

    poll_uid = uuid.uuid4().hex[:12]
    ctx.pending_events.append({
        "type": "send_poll",
        "poll_uid": poll_uid,
        "chat_id": int(ctx.current_chat_id),
        "question": q,
        "options": opts,
        "allows_multiple_answers": bool(allows_multiple_answers),
        "is_anonymous": bool(is_anonymous),
        "open_period_seconds": _safe_open_period(open_period_seconds),
        **_scope(ctx),
    })
    note = "anonymous counts only" if is_anonymous else "non-anonymous answers will be collected"
    return (
        "OK: Telegram poll queued\n"
        f"- poll_uid: {poll_uid}\n"
        f"- question: {q}\n"
        f"- options: {len(opts)}\n"
        f"- collection: {note}"
    )


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _poll_records_from_workspace(ctx: ToolContext, limit: int = 50) -> List[Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    poll_dir = ctx.drive_root / "polls"
    if poll_dir.exists():
        for path in sorted(poll_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            rec = _load_json(path)
            uid = str(rec.get("poll_uid") or path.stem)
            if rec:
                records[uid] = rec

    state_path = (ctx.shared_drive_root or ctx.drive_root) / "state" / "team_polls.json"
    state = _load_json(state_path)
    for rec in (state.get("polls") or {}).values():
        if not isinstance(rec, dict):
            continue
        if ctx.team_chat_id is not None and int(rec.get("chat_id") or 0) != int(ctx.team_chat_id):
            continue
        uid = str(rec.get("poll_uid") or "")
        if uid:
            records[uid] = rec

    rows = list(records.values())
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return rows[:max(1, int(limit or 50))]


def _poll_matches(rec: Dict[str, Any], poll_ref: str) -> bool:
    ref = str(poll_ref or "").strip().lower()
    if not ref:
        return True
    values = (
        str(rec.get("poll_uid") or ""),
        str(rec.get("telegram_poll_id") or ""),
        str(rec.get("message_id") or ""),
        str(rec.get("question") or ""),
    )
    return any(ref in val.lower() for val in values)


def _option_counts(rec: Dict[str, Any]) -> List[int]:
    options = rec.get("options") if isinstance(rec.get("options"), list) else []
    counts = [int(opt.get("voter_count") or 0) if isinstance(opt, dict) else 0 for opt in options]
    answers = rec.get("answers")
    if not isinstance(answers, dict) or not answers:
        return counts
    counts = [0 for _ in options]
    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        option_ids = answer.get("option_ids")
        if not isinstance(option_ids, list):
            continue
        for raw_idx in option_ids:
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if 0 <= idx < len(counts):
                counts[idx] += 1
    return counts


def _voter_label(answer: Dict[str, Any]) -> str:
    user = answer.get("user")
    if isinstance(user, dict):
        username = str(user.get("username") or "").strip()
        first = str(user.get("first_name") or "").strip()
        last = str(user.get("last_name") or "").strip()
        name = " ".join(part for part in (first, last) if part)
        label = f"id={user.get('id')}"
        if username:
            label += f" @{username}"
        if name:
            label += f" {name}"
        return label
    voter_chat = answer.get("voter_chat")
    if isinstance(voter_chat, dict):
        title = str(voter_chat.get("title") or voter_chat.get("username") or "").strip()
        return f"chat={voter_chat.get('id')}" + (f" {title}" if title else "")
    return "unknown voter"


def _format_poll(rec: Dict[str, Any], include_voters: bool) -> str:
    uid = str(rec.get("poll_uid") or "?")
    status = str(rec.get("status") or ("closed" if rec.get("is_closed") else "active"))
    telegram_poll_id = str(rec.get("telegram_poll_id") or "")
    header = f"Poll {uid} ({status})"
    if telegram_poll_id:
        header += f" telegram_poll_id={telegram_poll_id}"
    lines = [
        header,
        f"Question: {rec.get('question') or ''}",
        f"Message: chat_id={rec.get('chat_id')} message_id={rec.get('message_id')}",
        f"Total voters: {int(rec.get('total_voter_count') or 0)}",
        "Options:",
    ]
    options = rec.get("options") if isinstance(rec.get("options"), list) else []
    counts = _option_counts(rec)
    for idx, opt in enumerate(options):
        text = str(opt.get("text") if isinstance(opt, dict) else opt)
        count = counts[idx] if idx < len(counts) else 0
        lines.append(f"{idx + 1}. {text} — {count}")

    answers = rec.get("answers")
    if include_voters and isinstance(answers, dict) and answers:
        lines.append("Voters:")
        for answer in sorted(answers.values(), key=lambda row: str(row.get("ts") or "")):
            if not isinstance(answer, dict):
                continue
            selected = []
            for raw_idx in answer.get("option_ids") or []:
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if 0 <= idx < len(options):
                    opt = options[idx]
                    selected.append(str(opt.get("text") if isinstance(opt, dict) else opt))
            selected_text = ", ".join(selected) if selected else "(retracted)"
            lines.append(f"- {_voter_label(answer)}: {selected_text}")
    elif include_voters and rec.get("is_anonymous"):
        lines.append("Voters: unavailable for anonymous polls; only counts are collected.")
    return "\n".join(lines)


def _team_poll_results(
    ctx: ToolContext,
    poll_ref: str = "",
    include_voters: bool = True,
    limit: int = 5,
) -> str:
    err = _require_team(ctx)
    if err:
        return err
    try:
        limit = max(1, min(int(limit or 5), 20))
    except Exception:
        limit = 5
    rows = [rec for rec in _poll_records_from_workspace(ctx, limit=50) if _poll_matches(rec, poll_ref)]
    rows = rows[:limit]
    if not rows:
        return "No team polls found." if not poll_ref else f"No team polls matched: {poll_ref}"
    return "\n\n".join(_format_poll(rec, bool(include_voters)) for rec in rows)


def _team_poll_close(ctx: ToolContext, poll_ref: str) -> str:
    err = _require_team(ctx)
    if err:
        return err
    ref = str(poll_ref or "").strip()
    if not ref:
        return "⚠️ poll_ref is required. Use team_poll_results first if you need the poll_uid."
    ctx.pending_events.append({
        "type": "stop_poll",
        "poll_ref": ref,
        "chat_id": int(ctx.current_chat_id),
        **_scope(ctx),
    })
    return f"OK: poll close queued for {ref}."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("team_poll_create", {
            "name": "team_poll_create",
            "description": (
                "Create a native Telegram poll in the current approved group and collect answers. "
                "Use non-anonymous polls when user-level answer collection is needed."
            ),
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "Poll question, max 300 characters."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Answer options, 2-12 items, max 100 characters each.",
                },
                "allows_multiple_answers": {"type": "boolean", "default": False},
                "is_anonymous": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, only aggregate counts can be collected.",
                },
                "open_period_seconds": {
                    "type": "integer",
                    "default": 0,
                    "description": "Optional auto-close period in seconds. 0 means no explicit auto-close.",
                },
            }, "required": ["question", "options"]},
        }, _team_poll_create),
        ToolEntry("team_poll_results", {
            "name": "team_poll_results",
            "description": "Read collected Telegram poll results for the current approved group workspace.",
            "parameters": {"type": "object", "properties": {
                "poll_ref": {
                    "type": "string",
                    "description": "Optional poll_uid, telegram poll_id, message_id, or question fragment.",
                },
                "include_voters": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include non-anonymous voter identities and selected options.",
                },
                "limit": {"type": "integer", "default": 5, "description": "Maximum polls to return."},
            }, "required": []},
        }, _team_poll_results),
        ToolEntry("team_poll_close", {
            "name": "team_poll_close",
            "description": "Close a Telegram poll previously created by this bot in the current approved group.",
            "parameters": {"type": "object", "properties": {
                "poll_ref": {
                    "type": "string",
                    "description": "poll_uid, telegram poll_id, or message_id from team_poll_results.",
                },
            }, "required": ["poll_ref"]},
        }, _team_poll_close),
    ]
