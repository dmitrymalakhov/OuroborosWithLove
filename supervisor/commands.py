"""Supervisor slash-command runtime.

Keeps owner/admin control commands out of the Telegram launcher loop.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


ADMIN_ONLY_COMMANDS = frozenset({
    "/admin",
    "/access",
    "/requests",
    "/approve",
    "/allow",
    "/deny",
    "/reject",
    "/teamchat",
    "/unresolved",
})


def is_admin_only_command(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped.startswith("/"):
        return False
    command = stripped.split(maxsplit=1)[0].lower().split("@", 1)[0]
    return command in ADMIN_ONLY_COMMANDS


@dataclass
class SupervisorCommandRuntime:
    access_runtime: Any
    teamchat_runtime: Any
    improvement_runtime: Any
    load_state_fn: Callable[[], Dict[str, Any]]
    save_state_fn: Callable[[Dict[str, Any]], None]
    send_with_budget_fn: Callable[..., Any]
    kill_workers_fn: Callable[[], Any]
    safe_restart_fn: Callable[..., Tuple[bool, str]]
    status_text_fn: Callable[..., str]
    workers: Dict[int, Any]
    pending: List[Dict[str, Any]]
    running: Dict[str, Dict[str, Any]]
    soft_timeout_sec: int
    hard_timeout_sec: int
    queue_review_task_fn: Callable[..., Any]
    sort_pending_fn: Callable[[], Any]
    persist_queue_snapshot_fn: Callable[..., Any]
    consciousness: Any
    launcher_file: str
    execv_fn: Callable[[str, List[str]], Any] = os.execv
    executable: str = sys.executable

    def handle(self, text: str, chat_id: int, user_id: int, tg_offset: int = 0) -> Optional[Any]:
        """Handle supervisor slash-commands.

        Returns:
            True  — terminal command fully handled.
            str   — supervisor note to prepend before falling through to the LLM.
            ""    — not recognized.
        """
        lowered = text.strip().lower()

        if self.teamchat_runtime.handle_command(text, chat_id, user_id):
            return True

        if self.access_runtime.handle_command(text, chat_id, user_id):
            return True

        if self.improvement_runtime.handle_command(text, chat_id, user_id):
            return True

        if lowered.startswith("/panic"):
            self.send_with_budget_fn(chat_id, "🛑 PANIC: stopping everything now.")
            self.kill_workers_fn()
            state = self.load_state_fn()
            state["tg_offset"] = tg_offset
            self.save_state_fn(state)
            raise SystemExit("PANIC")

        if lowered.startswith("/restart"):
            state = self.load_state_fn()
            state["session_id"] = uuid.uuid4().hex
            state["tg_offset"] = tg_offset
            self.save_state_fn(state)
            self.send_with_budget_fn(chat_id, "♻️ Restarting (soft).")
            ok, msg = self.safe_restart_fn(reason="owner_restart", unsynced_policy="rescue_and_reset")
            if not ok:
                self.send_with_budget_fn(chat_id, f"⚠️ Restart cancelled: {msg}")
                return True
            self.kill_workers_fn()
            self.execv_fn(self.executable, [self.executable, self.launcher_file])

        # Dual-path commands: supervisor handles + LLM sees a note.
        if lowered.startswith("/status"):
            status = self.status_text_fn(
                self.workers,
                self.pending,
                self.running,
                self.soft_timeout_sec,
                self.hard_timeout_sec,
            )
            self.send_with_budget_fn(chat_id, status, force_budget=True)
            return "[Supervisor handled /status - status text already sent to chat]\n"

        if lowered.startswith("/review"):
            self.queue_review_task_fn(reason="owner:/review", force=True)
            return "[Supervisor handled /review - review task queued]\n"

        if lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")
            state = self.load_state_fn()
            state["evolution_mode_enabled"] = bool(turn_on)
            self.save_state_fn(state)
            if not turn_on:
                self.pending[:] = [task for task in self.pending if str(task.get("type")) != "evolution"]
                self.sort_pending_fn()
                self.persist_queue_snapshot_fn(reason="evolve_off")
            state_str = "ON" if turn_on else "OFF"
            self.send_with_budget_fn(chat_id, f"🧬 Evolution: {state_str}")
            return f"[Supervisor handled /evolve - evolution toggled {state_str}]\n"

        if lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1"):
                result = self.consciousness.start()
                self.send_with_budget_fn(chat_id, f"🧠 {result}")
            elif action in ("stop", "off", "0"):
                result = self.consciousness.stop()
                self.send_with_budget_fn(chat_id, f"🧠 {result}")
            else:
                bg_status = "running" if self.consciousness.is_running else "stopped"
                self.send_with_budget_fn(chat_id, f"🧠 Background consciousness: {bg_status}")
            return f"[Supervisor handled /bg {action}]\n"

        return ""
