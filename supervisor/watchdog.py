"""Supervisor watchdog helpers."""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


def start_chat_watchdog(
    *,
    load_state_fn: Callable[[], Dict[str, Any]],
    get_chat_agent_fn: Callable[..., Any],
    send_with_budget_fn: Callable[..., Any],
    reset_chat_agent_fn: Callable[[], Any],
    drive_root: pathlib.Path,
    soft_timeout_sec: int,
    hard_timeout_sec: int,
    interval_sec: int = 30,
) -> threading.Thread:
    """Start daemon watchdog for the direct-mode chat agent."""

    def _loop() -> None:
        soft_warned = False
        while True:
            time.sleep(interval_sec)
            try:
                state = load_state_fn()
                owner_uid: Optional[int] = int(state.get("owner_id") or 0) or None
                agent = get_chat_agent_fn(user_id=owner_uid, drive_root=drive_root, user_role="admin")
                if not agent._busy:
                    soft_warned = False
                    continue

                now = time.time()
                idle_sec = now - agent._last_progress_ts
                total_sec = now - agent._task_started_ts

                if idle_sec >= hard_timeout_sec:
                    if state.get("owner_chat_id"):
                        send_with_budget_fn(
                            int(state["owner_chat_id"]),
                            f"⚠️ Task stuck ({int(total_sec)}s without progress). Restarting agent.",
                        )
                    reset_chat_agent_fn()
                    soft_warned = False
                    continue

                if idle_sec >= soft_timeout_sec and not soft_warned:
                    soft_warned = True
                    if state.get("owner_chat_id"):
                        send_with_budget_fn(
                            int(state["owner_chat_id"]),
                            f"⏱️ Task running for {int(total_sec)}s, "
                            f"last progress {int(idle_sec)}s ago. Continuing.",
                        )
            except Exception:
                log.debug("Failed to check/notify chat watchdog", exc_info=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
