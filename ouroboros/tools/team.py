"""Team workspace tools: shared inbox and membership context."""

from __future__ import annotations

import json
import pathlib
import uuid
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import append_jsonl, short, utc_now_iso


def _require_team(ctx: ToolContext) -> str:
    if not ctx.is_team_workspace or not ctx.team_slug:
        return "⚠️ This tool is available only inside an approved team workspace."
    return ""


def _inbox_path(ctx: ToolContext) -> pathlib.Path:
    return ctx.drive_root / "inbox" / "messages.jsonl"


def _chat_log_path(ctx: ToolContext) -> pathlib.Path:
    return ctx.drive_root / "logs" / "chat.jsonl"


def _clean_limit(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(parsed, maximum))


def _team_registry_record(ctx: ToolContext) -> Dict[str, Any]:
    path = (ctx.shared_drive_root or ctx.drive_root) / "state" / "team_chats.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("team_chats") or {}
        if ctx.team_chat_id is not None:
            rec = rows.get(str(int(ctx.team_chat_id)))
            if isinstance(rec, dict):
                return rec
        for rec in rows.values():
            if isinstance(rec, dict) and rec.get("slug") == ctx.team_slug:
                return rec
    except Exception:
        pass
    return {}


def _read_jsonl_tail(path: pathlib.Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-max(1, limit):]:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _format_chat_entry(entry: Dict[str, Any], max_text_chars: int = 900) -> str:
    direction_raw = str(entry.get("direction") or "").lower()
    direction = "→" if direction_raw in ("out", "outgoing") else "←"
    ts = str(entry.get("ts") or "")[:16]
    user_id = entry.get("user_id")
    who = "bot" if direction == "→" else f"user={user_id or '?'}"
    text = short(str(entry.get("text") or ""), max_text_chars)
    return f"{direction} [{ts}] {who}: {text}"


def _team_inbox_send(ctx: ToolContext, message: str, topic: str = "") -> str:
    err = _require_team(ctx)
    if err:
        return err
    text = str(message or "").strip()
    if not text:
        return "⚠️ Empty message."
    path = _inbox_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": utc_now_iso(),
        "team_slug": ctx.team_slug,
        "team_chat_id": ctx.team_chat_id,
        "chat_id": ctx.current_chat_id,
        "user_id": ctx.current_user_id,
        "task_id": ctx.task_id,
        "topic": str(topic or "").strip(),
        "message": text,
    }
    append_jsonl(path, entry)
    return f"OK: team inbox message written ({entry['id']})."


def _team_inbox_read(ctx: ToolContext, limit: int = 20, since_id: str = "") -> str:
    err = _require_team(ctx)
    if err:
        return err
    limit = max(1, min(int(limit or 20), 100))
    entries = _read_jsonl_tail(_inbox_path(ctx), limit=limit + 50)
    if since_id:
        seen = False
        filtered = []
        for entry in entries:
            if seen:
                filtered.append(entry)
            elif str(entry.get("id") or "") == str(since_id):
                seen = True
        entries = filtered
    entries = entries[-limit:]
    if not entries:
        return "Team inbox is empty."
    lines = [f"Team inbox ({len(entries)} messages):"]
    for entry in entries:
        ts = str(entry.get("ts") or "")[:16]
        topic = str(entry.get("topic") or "").strip()
        prefix = f"[{entry.get('id')}] {ts}"
        if topic:
            prefix += f" #{topic}"
        sender = entry.get("user_id")
        if sender:
            prefix += f" user={sender}"
        lines.append(f"- {prefix}: {short(str(entry.get('message') or ''), 500)}")
    return "\n".join(lines)


def _team_chat_history(ctx: ToolContext, limit: int = 50, offset: int = 0, include_bot: bool = True) -> str:
    err = _require_team(ctx)
    if err:
        return err
    limit = _clean_limit(limit, default=50, minimum=1, maximum=200)
    offset = _clean_limit(offset, default=0, minimum=0, maximum=10_000)
    entries = _read_jsonl_tail(_chat_log_path(ctx), limit=limit + offset + 50)
    if not include_bot:
        entries = [entry for entry in entries if str(entry.get("direction") or "").lower() not in ("out", "outgoing")]
    if offset:
        entries = entries[:-offset] if offset < len(entries) else []
    entries = entries[-limit:]
    if not entries:
        return "Team chat history is empty."
    lines = [f"Team chat history ({len(entries)} messages):"]
    lines.extend(_format_chat_entry(entry) for entry in entries)
    return "\n".join(lines)


def _team_chat_search(
    ctx: ToolContext,
    query: str,
    limit: int = 20,
    max_scan: int = 1000,
    include_bot: bool = True,
    case_sensitive: bool = False,
) -> str:
    err = _require_team(ctx)
    if err:
        return err
    needle = str(query or "").strip()
    if not needle:
        return "⚠️ Empty search query."
    limit = _clean_limit(limit, default=20, minimum=1, maximum=100)
    max_scan = _clean_limit(max_scan, default=1000, minimum=50, maximum=5000)
    entries = _read_jsonl_tail(_chat_log_path(ctx), limit=max_scan)
    if not include_bot:
        entries = [entry for entry in entries if str(entry.get("direction") or "").lower() not in ("out", "outgoing")]

    if case_sensitive:
        matches = [entry for entry in entries if needle in str(entry.get("text") or "")]
    else:
        needle_lower = needle.lower()
        matches = [entry for entry in entries if needle_lower in str(entry.get("text") or "").lower()]
    matches = matches[-limit:]
    if not matches:
        return f"No team chat messages matched: {needle}"
    lines = [f"Team chat search for {needle!r}: {len(matches)} matches"]
    lines.extend(_format_chat_entry(entry) for entry in matches)
    return "\n".join(lines)


def _team_members(ctx: ToolContext) -> str:
    err = _require_team(ctx)
    if err:
        return err
    rec = _team_registry_record(ctx)
    members = rec.get("members") if isinstance(rec, dict) else {}
    if not isinstance(members, dict) or not members:
        return "No team members have been observed yet."
    lines = [f"Team members for {ctx.team_slug}: {len(members)}"]
    for row in members.values():
        if not isinstance(row, dict):
            continue
        uid = row.get("user_id")
        username = str(row.get("username") or "").strip()
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        name = " ".join(part for part in (first, last) if part).strip()
        label = f"id={uid}"
        if username:
            label += f" @{username}"
        if name:
            label += f" {name}"
        lines.append(f"- {label}")
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("team_inbox_send", {
            "name": "team_inbox_send",
            "description": "Write a coordination message to the current approved team workspace inbox.",
            "parameters": {"type": "object", "properties": {
                "message": {"type": "string", "description": "Message to share with other team tasks/agents."},
                "topic": {"type": "string", "description": "Optional short topic label."},
            }, "required": ["message"]},
        }, _team_inbox_send),
        ToolEntry("team_inbox_read", {
            "name": "team_inbox_read",
            "description": "Read recent coordination messages from the current approved team workspace inbox.",
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 20},
                "since_id": {"type": "string", "description": "Optional message id; return messages after it."},
            }, "required": []},
        }, _team_inbox_read),
        ToolEntry("team_chat_history", {
            "name": "team_chat_history",
            "description": "Read recent Telegram group messages from the current approved team workspace.",
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Number of recent messages to return."},
                "offset": {"type": "integer", "default": 0, "description": "Skip this many newest messages."},
                "include_bot": {"type": "boolean", "default": True, "description": "Include bot replies."},
            }, "required": []},
        }, _team_chat_history),
        ToolEntry("team_chat_search", {
            "name": "team_chat_search",
            "description": "Search Telegram group chat history in the current approved team workspace.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Text to search for in group chat messages."},
                "limit": {"type": "integer", "default": 20, "description": "Maximum matches to return."},
                "max_scan": {"type": "integer", "default": 1000, "description": "Maximum recent log entries to scan."},
                "include_bot": {"type": "boolean", "default": True, "description": "Search bot replies too."},
                "case_sensitive": {"type": "boolean", "default": False},
            }, "required": ["query"]},
        }, _team_chat_search),
        ToolEntry("team_members", {
            "name": "team_members",
            "description": "List Telegram users observed in the current approved team workspace.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _team_members),
    ]
