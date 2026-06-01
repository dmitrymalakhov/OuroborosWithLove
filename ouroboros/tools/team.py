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
        ToolEntry("team_members", {
            "name": "team_members",
            "description": "List Telegram users observed in the current approved team workspace.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _team_members),
    ]
