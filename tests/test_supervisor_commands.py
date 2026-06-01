class NoopApprovalRuntime:
    def __init__(self, handled=False):
        self.handled = handled
        self.calls = []

    def handle_command(self, text, chat_id, user_id):
        self.calls.append((text, chat_id, user_id))
        return self.handled


class FakeConsciousness:
    def __init__(self):
        self.is_running = False

    def start(self):
        self.is_running = True
        return "started"

    def stop(self):
        self.is_running = False
        return "stopped"


def make_runtime(**overrides):
    from supervisor.commands import SupervisorCommandRuntime

    state = overrides.pop("state", {})
    sent = overrides.pop("sent", [])
    pending = overrides.pop("pending", [])
    snapshots = overrides.pop("snapshots", [])
    reviews = overrides.pop("reviews", [])

    defaults = dict(
        access_runtime=NoopApprovalRuntime(),
        teamchat_runtime=NoopApprovalRuntime(),
        load_state_fn=lambda: state,
        save_state_fn=lambda value: state.update(value),
        send_with_budget_fn=lambda *args, **kwargs: sent.append((args, kwargs)),
        kill_workers_fn=lambda: None,
        safe_restart_fn=lambda **kwargs: (False, "blocked"),
        status_text_fn=lambda *args: "status text",
        workers={},
        pending=pending,
        running={},
        soft_timeout_sec=60,
        hard_timeout_sec=120,
        queue_review_task_fn=lambda **kwargs: reviews.append(kwargs),
        sort_pending_fn=lambda: None,
        persist_queue_snapshot_fn=lambda **kwargs: snapshots.append(kwargs),
        consciousness=FakeConsciousness(),
        launcher_file="launcher.py",
        execv_fn=lambda *args: None,
        executable="python",
    )
    defaults.update(overrides)
    return SupervisorCommandRuntime(**defaults), sent, pending, snapshots, reviews, state


def test_supervisor_status_is_dual_path():
    runtime, sent, _pending, _snapshots, _reviews, _state = make_runtime()

    result = runtime.handle("/status", chat_id=1, user_id=2, tg_offset=99)

    assert "Supervisor handled /status" in result
    assert sent == [((1, "status text"), {"force_budget": True})]


def test_supervisor_evolve_off_removes_pending_evolution_tasks():
    runtime, sent, pending, snapshots, _reviews, state = make_runtime(
        state={"evolution_mode_enabled": True},
        pending=[{"type": "evolution"}, {"type": "chat"}],
    )

    result = runtime.handle("/evolve off", chat_id=1, user_id=2)

    assert "evolution toggled OFF" in result
    assert state["evolution_mode_enabled"] is False
    assert pending == [{"type": "chat"}]
    assert snapshots == [{"reason": "evolve_off"}]
    assert sent[-1][0] == (1, "🧬 Evolution: OFF")


def test_supervisor_approval_commands_take_precedence():
    teamchat = NoopApprovalRuntime(handled=True)
    access = NoopApprovalRuntime(handled=False)
    runtime, _sent, _pending, _snapshots, _reviews, _state = make_runtime(
        teamchat_runtime=teamchat,
        access_runtime=access,
    )

    assert runtime.handle("/teamchat pending", chat_id=1, user_id=2) is True
    assert len(teamchat.calls) == 1
    assert access.calls == []


def test_admin_only_command_detection():
    from supervisor.commands import is_admin_only_command

    assert is_admin_only_command("/admin")
    assert is_admin_only_command("/admin@ouro_bot")
    assert is_admin_only_command("/approve 123")
    assert is_admin_only_command("/teamchat pending")
    assert not is_admin_only_command("/status")
    assert not is_admin_only_command("hello /admin")
