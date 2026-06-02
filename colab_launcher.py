# ============================
# Ouroboros — Runtime launcher (entry point, executed from repository)
# ============================
# Thin orchestrator: secrets, bootstrap, main loop.
# Heavy logic lives in supervisor/ package.

import logging
import os, sys, time, pathlib, subprocess, datetime, threading, queue as _queue_mod
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)

# ----------------------------
# 0) Install launcher deps
# ----------------------------
def install_launcher_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests"],
        check=True,
    )

install_launcher_deps()

def ensure_claude_code_cli() -> bool:
    """Best-effort install of Claude Code CLI for Anthropic-powered code edits."""
    local_bin = str(pathlib.Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    subprocess.run(["bash", "-lc", "curl -fsSL https://claude.ai/install.sh | bash"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    subprocess.run(["bash", "-lc", "command -v npm >/dev/null 2>&1 && npm install -g @anthropic-ai/claude-code"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    return has_cli

# ----------------------------
# 0.1) provide apply_patch shim
# ----------------------------
from ouroboros.apply_patch import install as install_apply_patch
from ouroboros.llm import (
    default_code_model,
    default_fallback_models,
    default_light_model,
    default_main_model,
    normalize_llm_provider,
)
install_apply_patch()

# ----------------------------
# 1) Secrets + runtime config
# ----------------------------
from google.colab import userdata  # type: ignore
from google.colab import drive  # type: ignore

_LEGACY_CFG_WARNED: Set[str] = set()

def _userdata_get(name: str) -> Optional[str]:
    try:
        return userdata.get(name)
    except Exception:
        return None

def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = _userdata_get(name)
    if v is None or str(v).strip() == "":
        v = os.environ.get(name, default)
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v

def get_cfg(name: str, default: Optional[str] = None, allow_legacy_secret: bool = False) -> Optional[str]:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return v
    if allow_legacy_secret:
        legacy = _userdata_get(name)
        if legacy is not None and str(legacy).strip() != "":
            if name not in _LEGACY_CFG_WARNED:
                print(f"[cfg] DEPRECATED: move {name} from Colab Secrets to config cell/env.")
                _LEGACY_CFG_WARNED.add(name)
            return legacy
    return default


def _parse_int_cfg(raw: Optional[str], default: int, minimum: int = 0) -> int:
    try:
        val = int(str(raw))
    except Exception:
        val = default
    return max(minimum, val)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = get_cfg(name, default="1" if default else "0", allow_legacy_secret=True)
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


DISABLE_SELF_MODIFICATION = _env_flag("OUROBOROS_DISABLE_SELF_MODIFICATION", default=False)
LLM_PROVIDER = normalize_llm_provider(get_cfg("OUROBOROS_LLM_PROVIDER", default="openrouter", allow_legacy_secret=True))
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", required=(LLM_PROVIDER == "openrouter"))
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", required=True)
TOTAL_BUDGET_DEFAULT = get_secret("TOTAL_BUDGET", required=True)
GITHUB_TOKEN = get_secret("GITHUB_TOKEN", default="", required=not DISABLE_SELF_MODIFICATION)

# Robust TOTAL_BUDGET parsing — handles \r\n, spaces, and other junk from Colab Secrets
# Example: user enters "8 800" → Colab stores as "8\r\n800" → we need 8800
try:
    import re
    _raw_budget = str(TOTAL_BUDGET_DEFAULT or "")
    _clean_budget = re.sub(r'[^0-9.\-]', '', _raw_budget)  # keep only digits, dot, minus
    TOTAL_BUDGET_LIMIT = float(_clean_budget) if _clean_budget else 0.0
    if _raw_budget.strip() != _clean_budget:
        log.warning(f"TOTAL_BUDGET cleaned: {_raw_budget!r} → {TOTAL_BUDGET_LIMIT}")
except Exception as e:
    log.warning(f"Failed to parse TOTAL_BUDGET ({TOTAL_BUDGET_DEFAULT!r}): {e}")
    TOTAL_BUDGET_LIMIT = 0.0

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", default="", required=(LLM_PROVIDER == "openai"))
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", default="")
GITHUB_USER = get_cfg("GITHUB_USER", default=None, allow_legacy_secret=True)
GITHUB_REPO = get_cfg("GITHUB_REPO", default=None, allow_legacy_secret=True)
assert GITHUB_USER and str(GITHUB_USER).strip(), "GITHUB_USER not set. Add it to your config cell (see README)."
assert GITHUB_REPO and str(GITHUB_REPO).strip(), "GITHUB_REPO not set. Add it to your config cell (see README)."
MAX_WORKERS = int(get_cfg("OUROBOROS_MAX_WORKERS", default="5", allow_legacy_secret=True) or "5")
MODEL_MAIN = get_cfg("OUROBOROS_MODEL", default=default_main_model(LLM_PROVIDER), allow_legacy_secret=True)
MODEL_CODE = get_cfg("OUROBOROS_MODEL_CODE", default=default_code_model(LLM_PROVIDER), allow_legacy_secret=True)
MODEL_LIGHT = get_cfg("OUROBOROS_MODEL_LIGHT", default=default_light_model(LLM_PROVIDER), allow_legacy_secret=True)
MODEL_FALLBACK_LIST = get_cfg(
    "OUROBOROS_MODEL_FALLBACK_LIST",
    default=default_fallback_models(LLM_PROVIDER),
    allow_legacy_secret=True,
)
DISABLE_CLAUDE_CODE_EDIT = get_cfg(
    "OUROBOROS_DISABLE_CLAUDE_CODE_EDIT",
    default="1" if LLM_PROVIDER == "openai" else "0",
    allow_legacy_secret=True,
)
if LLM_PROVIDER == "openai":
    for _name, _model in (
        ("OUROBOROS_MODEL", MODEL_MAIN),
        ("OUROBOROS_MODEL_CODE", MODEL_CODE),
        ("OUROBOROS_MODEL_LIGHT", MODEL_LIGHT),
    ):
        _model_str = str(_model or "").strip()
        assert "/" not in _model_str or _model_str.startswith("openai/"), (
            f"{_name}={_model_str!r} is not compatible with OUROBOROS_LLM_PROVIDER=openai. "
            "Use a native OpenAI model id such as 'gpt-5.2', 'gpt-5.4-mini', or an openai/... id."
        )
ADMIN_USER_IDS_RAW = get_cfg("OUROBOROS_ADMIN_USER_IDS", default="", allow_legacy_secret=True) or ""


def _parse_admin_user_ids(raw: str) -> Set[int]:
    ids: Set[int] = set()
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except Exception:
            log.warning("Ignoring invalid OUROBOROS_ADMIN_USER_IDS entry: %r", part)
    return ids


ADMIN_USER_IDS = _parse_admin_user_ids(ADMIN_USER_IDS_RAW)
REQUIRE_USER_APPROVAL = _env_flag("OUROBOROS_REQUIRE_USER_APPROVAL", default=True)
APPROVED_USER_IDS = _parse_admin_user_ids(
    ",".join([
        get_cfg("OUROBOROS_APPROVED_USER_IDS", default="", allow_legacy_secret=True) or "",
        os.environ.get("ALLOWED_TELEGRAM_IDS", ""),
    ])
)


def _is_admin_user(user_id: int, st: Dict[str, Any]) -> bool:
    if ADMIN_USER_IDS:
        return int(user_id) in ADMIN_USER_IDS
    return int(user_id) == int(st.get("owner_id") or 0)

BUDGET_REPORT_EVERY_MESSAGES = 10
SOFT_TIMEOUT_SEC = max(60, int(get_cfg("OUROBOROS_SOFT_TIMEOUT_SEC", default="600", allow_legacy_secret=True) or "600"))
HARD_TIMEOUT_SEC = max(120, int(get_cfg("OUROBOROS_HARD_TIMEOUT_SEC", default="1800", allow_legacy_secret=True) or "1800"))
DIAG_HEARTBEAT_SEC = _parse_int_cfg(
    get_cfg("OUROBOROS_DIAG_HEARTBEAT_SEC", default="30", allow_legacy_secret=True),
    default=30,
    minimum=0,
)
DIAG_SLOW_CYCLE_SEC = _parse_int_cfg(
    get_cfg("OUROBOROS_DIAG_SLOW_CYCLE_SEC", default="20", allow_legacy_secret=True),
    default=20,
    minimum=0,
)
AUDIO_TRANSCRIPTION_ENABLED = _env_flag("OUROBOROS_TRANSCRIBE_AUDIO", default=True)
AUDIO_TRANSCRIPTION_MODEL = get_cfg("OUROBOROS_TRANSCRIPTION_MODEL", default="whisper-1", allow_legacy_secret=True) or "whisper-1"
AUDIO_TRANSCRIPTION_LANGUAGE = get_cfg("OUROBOROS_TRANSCRIPTION_LANGUAGE", default="", allow_legacy_secret=True) or ""
AUDIO_TRANSCRIPTION_PROMPT = get_cfg("OUROBOROS_TRANSCRIPTION_PROMPT", default="", allow_legacy_secret=True) or ""
AUDIO_TRANSCRIPTION_MAX_BYTES = _parse_int_cfg(
    get_cfg("OUROBOROS_TRANSCRIPTION_MAX_BYTES", default="25000000", allow_legacy_secret=True),
    default=25_000_000,
    minimum=1,
)

os.environ["OUROBOROS_LLM_PROVIDER"] = LLM_PROVIDER
os.environ["OPENROUTER_API_KEY"] = str(OPENROUTER_API_KEY or "")
os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY or "")
os.environ["ANTHROPIC_API_KEY"] = str(ANTHROPIC_API_KEY or "")
os.environ["GITHUB_USER"] = str(GITHUB_USER)
os.environ["GITHUB_REPO"] = str(GITHUB_REPO)
os.environ["OUROBOROS_MODEL"] = str(MODEL_MAIN or default_main_model(LLM_PROVIDER))
os.environ["OUROBOROS_MODEL_CODE"] = str(MODEL_CODE or default_code_model(LLM_PROVIDER))
if MODEL_LIGHT:
    os.environ["OUROBOROS_MODEL_LIGHT"] = str(MODEL_LIGHT)
if MODEL_FALLBACK_LIST:
    os.environ["OUROBOROS_MODEL_FALLBACK_LIST"] = str(MODEL_FALLBACK_LIST)
os.environ["OUROBOROS_DISABLE_CLAUDE_CODE_EDIT"] = str(DISABLE_CLAUDE_CODE_EDIT)
os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(DIAG_HEARTBEAT_SEC)
os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(DIAG_SLOW_CYCLE_SEC)
os.environ["TELEGRAM_BOT_TOKEN"] = str(TELEGRAM_BOT_TOKEN)
os.environ["OUROBOROS_TRANSCRIPTION_MODEL"] = str(AUDIO_TRANSCRIPTION_MODEL)
if AUDIO_TRANSCRIPTION_LANGUAGE:
    os.environ["OUROBOROS_TRANSCRIPTION_LANGUAGE"] = str(AUDIO_TRANSCRIPTION_LANGUAGE)

if str(ANTHROPIC_API_KEY or "").strip() and str(DISABLE_CLAUDE_CODE_EDIT).strip().lower() not in ("1", "true", "yes", "on"):
    ensure_claude_code_cli()

# ----------------------------
# 2) Mount Drive
# ----------------------------
if not pathlib.Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/Ouroboros").resolve()
REPO_DIR = pathlib.Path("/content/ouroboros_repo").resolve()

for sub in ["state", "logs", "memory", "index", "locks", "archive", "users", "teams"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
REPO_DIR.mkdir(parents=True, exist_ok=True)

# Clear stale owner mailbox files from previous session
try:
    from ouroboros.owner_inject import get_pending_path
    # Clean legacy global file
    _stale_inject = get_pending_path(DRIVE_ROOT)
    if _stale_inject.exists():
        _stale_inject.unlink(missing_ok=True)
    # Clean per-task mailbox dir
    _mailbox_dir = DRIVE_ROOT / "memory" / "owner_mailbox"
    if _mailbox_dir.exists():
        for _f in _mailbox_dir.iterdir():
            _f.unlink(missing_ok=True)
except Exception:
    pass

CHAT_LOG_PATH = DRIVE_ROOT / "logs" / "chat.jsonl"
if not CHAT_LOG_PATH.exists():
    CHAT_LOG_PATH.write_text("", encoding="utf-8")

# ----------------------------
# 3) Git constants
# ----------------------------
BRANCH_DEV = get_cfg(
    "OUROBOROS_BRANCH_DEV",
    default="main" if DISABLE_SELF_MODIFICATION else "ouroboros",
    allow_legacy_secret=True,
) or "main"
BRANCH_STABLE = get_cfg(
    "OUROBOROS_BRANCH_STABLE",
    default="main" if DISABLE_SELF_MODIFICATION else "ouroboros-stable",
    allow_legacy_secret=True,
) or BRANCH_DEV
if DISABLE_SELF_MODIFICATION and not str(GITHUB_TOKEN or "").strip():
    REMOTE_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
else:
    REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

# ----------------------------
# 4) Initialize supervisor modules
# ----------------------------
from supervisor.state import (
    init as state_init, load_state, save_state, append_jsonl,
    update_budget_from_usage, status_text, rotate_chat_log_if_needed,
    init_state,
)
state_init(DRIVE_ROOT, TOTAL_BUDGET_LIMIT)
init_state()
_st_admin = load_state()
if ADMIN_USER_IDS:
    _st_admin["admin_user_ids"] = sorted(ADMIN_USER_IDS)
else:
    _st_admin.pop("admin_user_ids", None)
save_state(_st_admin)

from supervisor.users import (
    ACCESS_APPROVED, ACCESS_PENDING,
    init as users_init, ensure_user_workspace,
    request_user_access, user_access_status,
)
users_init(DRIVE_ROOT)

from supervisor.access_control import AccessRuntime, access_user_label, collect_admin_chat_ids
from supervisor.commands import SupervisorCommandRuntime, is_admin_only_command

from supervisor.teams import (
    TEAM_APPROVED, TEAM_DENIED, TEAM_PENDING,
    ensure_team_workspace,
    is_group_chat_type,
    note_team_member_seen, request_team_chat,
    team_chat_status,
    team_slug_for_chat,
    init as teams_init,
)
teams_init(DRIVE_ROOT)

from supervisor.polls import init as polls_init, handle_poll_update
polls_init(DRIVE_ROOT)

from supervisor.teamchat import TeamChatRuntime
from supervisor.unresolved_tasks import ImprovementRequestRuntime

from supervisor.telegram import (
    TelegramClient,
    init as telegram_init,
    log_chat,
    save_incoming_audio,
    save_incoming_document,
    send_with_budget,
)
from supervisor.transcription import (
    audio_upload_name,
    format_audio_task_text,
    is_audio_attachment,
    transcribe_audio_file,
)
TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))
try:
    _BOT_INFO = TG.get_me()
except Exception:
    log.warning("Failed to fetch Telegram bot identity; group mention detection will use fallback", exc_info=True)
    _BOT_INFO = {}
BOT_ID = int(_BOT_INFO.get("id") or 0)
BOT_USERNAME = str(_BOT_INFO.get("username") or "").strip().lstrip("@")
telegram_init(
    drive_root=DRIVE_ROOT,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    budget_report_every=BUDGET_REPORT_EVERY_MESSAGES,
    tg_client=TG,
)

from supervisor.git_ops import (
    init as git_ops_init, ensure_repo_present, checkout_and_reset,
    sync_runtime_dependencies, import_test, safe_restart,
)
git_ops_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, remote_url=REMOTE_URL,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.queue import (
    enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
    persist_queue_snapshot, restore_pending_from_snapshot,
    cancel_task_by_id, queue_review_task, sort_pending,
)

from supervisor.workers import (
    init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
    spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
    handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
)
from supervisor.watchdog import start_chat_watchdog
workers_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, max_workers=MAX_WORKERS,
    soft_timeout=SOFT_TIMEOUT_SEC, hard_timeout=HARD_TIMEOUT_SEC,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.events import dispatch_event

# ----------------------------
# 5) Bootstrap repo
# ----------------------------
ensure_repo_present()
if DISABLE_SELF_MODIFICATION:
    subprocess.run(["git", "checkout", BRANCH_DEV], cwd=str(REPO_DIR), check=False)
    st_bootstrap = load_state()
    st_bootstrap["current_branch"] = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_DIR),
        capture_output=True, text=True, check=False,
    ).stdout.strip() or BRANCH_DEV
    st_bootstrap["current_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR),
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    save_state(st_bootstrap)
    ok, msg = True, "OK: self-modification disabled"
else:
    ok, msg = safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
    assert ok, f"Bootstrap failed: {msg}"

# ----------------------------
# 6) Start workers
# ----------------------------
kill_workers()
spawn_workers(MAX_WORKERS)
restored_pending = restore_pending_from_snapshot()
persist_queue_snapshot(reason="startup")
if restored_pending > 0:
    st_boot = load_state()
    if st_boot.get("owner_chat_id"):
        send_with_budget(int(st_boot["owner_chat_id"]),
                         f"♻️ Restored pending queue from snapshot: {restored_pending} tasks.")

append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "type": "launcher_start",
    "branch": load_state().get("current_branch"),
    "sha": load_state().get("current_sha"),
    "max_workers": MAX_WORKERS,
    "model_default": MODEL_MAIN, "model_code": MODEL_CODE, "model_light": MODEL_LIGHT,
    "soft_timeout_sec": SOFT_TIMEOUT_SEC, "hard_timeout_sec": HARD_TIMEOUT_SEC,
    "worker_start_method": str(os.environ.get("OUROBOROS_WORKER_START_METHOD") or ""),
    "disable_self_modification": DISABLE_SELF_MODIFICATION,
    "require_user_approval": REQUIRE_USER_APPROVAL,
    "approved_user_ids_count": len(APPROVED_USER_IDS),
    "diag_heartbeat_sec": DIAG_HEARTBEAT_SEC,
    "diag_slow_cycle_sec": DIAG_SLOW_CYCLE_SEC,
    "audio_transcription_enabled": AUDIO_TRANSCRIPTION_ENABLED,
    "audio_transcription_model": AUDIO_TRANSCRIPTION_MODEL,
    "audio_transcription_language": AUDIO_TRANSCRIPTION_LANGUAGE,
    "telegram_bot_username": BOT_USERNAME,
    "telegram_can_read_all_group_messages": _BOT_INFO.get("can_read_all_group_messages"),
})

# ----------------------------
# 6.1) Auto-resume after restart
# ----------------------------
auto_resume_after_restart()

# ----------------------------
# 6.3) Background consciousness
# ----------------------------
from ouroboros.consciousness import BackgroundConsciousness

def _get_owner_chat_id() -> Optional[int]:
    try:
        st = load_state()
        cid = st.get("owner_chat_id")
        return int(cid) if cid else None
    except Exception:
        return None

_consciousness = BackgroundConsciousness(
    drive_root=DRIVE_ROOT,
    repo_dir=REPO_DIR,
    event_queue=get_event_q(),
    owner_chat_id_fn=_get_owner_chat_id,
)

def reset_chat_agent():
    """Reset the direct-mode chat agent (called by watchdog on hangs)."""
    import supervisor.workers as _w
    with _w._chat_agents_lock:
        _w._chat_agents.clear()


_watchdog_thread = start_chat_watchdog(
    load_state_fn=load_state,
    get_chat_agent_fn=_get_chat_agent,
    send_with_budget_fn=send_with_budget,
    reset_chat_agent_fn=reset_chat_agent,
    drive_root=DRIVE_ROOT,
    soft_timeout_sec=SOFT_TIMEOUT_SEC,
    hard_timeout_sec=HARD_TIMEOUT_SEC,
)

# ----------------------------
# 7) Main loop
# ----------------------------
import types
_event_ctx = types.SimpleNamespace(
    DRIVE_ROOT=DRIVE_ROOT,
    REPO_DIR=REPO_DIR,
    BRANCH_DEV=BRANCH_DEV,
    BRANCH_STABLE=BRANCH_STABLE,
    TG=TG,
    WORKERS=WORKERS,
    PENDING=PENDING,
    RUNNING=RUNNING,
    MAX_WORKERS=MAX_WORKERS,
    send_with_budget=send_with_budget,
    load_state=load_state,
    save_state=save_state,
    update_budget_from_usage=update_budget_from_usage,
    append_jsonl=append_jsonl,
    enqueue_task=enqueue_task,
    cancel_task_by_id=cancel_task_by_id,
    queue_review_task=queue_review_task,
    persist_queue_snapshot=persist_queue_snapshot,
    safe_restart=safe_restart,
    kill_workers=kill_workers,
    spawn_workers=spawn_workers,
    sort_pending=sort_pending,
    consciousness=_consciousness,
)


def _safe_qsize(q: Any) -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


def _admin_chat_ids():
    return collect_admin_chat_ids(DRIVE_ROOT, load_state)


_access = AccessRuntime(
    drive_root=DRIVE_ROOT,
    admin_chat_ids_fn=_admin_chat_ids,
    load_state_fn=load_state,
    send_with_budget_fn=send_with_budget,
    tg=TG,
    is_admin_user_fn=_is_admin_user,
    log_chat_fn=log_chat,
    append_jsonl_fn=append_jsonl,
)

_teamchat = TeamChatRuntime(
    drive_root=DRIVE_ROOT,
    tg=TG,
    admin_chat_ids_fn=_admin_chat_ids,
    access_user_label_fn=access_user_label,
    is_admin_user_fn=_is_admin_user,
    load_state_fn=load_state,
    send_with_budget_fn=send_with_budget,
    log_chat_fn=log_chat,
    append_jsonl_fn=append_jsonl,
    bot_id=BOT_ID,
    bot_username=BOT_USERNAME,
)

_improvements = ImprovementRequestRuntime(
    drive_root=DRIVE_ROOT,
    admin_chat_ids_fn=_admin_chat_ids,
    load_state_fn=load_state,
    send_with_budget_fn=send_with_budget,
    tg=TG,
    is_admin_user_fn=_is_admin_user,
    log_chat_fn=log_chat,
    append_jsonl_fn=append_jsonl,
)

_supervisor_commands = SupervisorCommandRuntime(
    access_runtime=_access,
    teamchat_runtime=_teamchat,
    improvement_runtime=_improvements,
    load_state_fn=load_state,
    save_state_fn=save_state,
    send_with_budget_fn=send_with_budget,
    kill_workers_fn=kill_workers,
    safe_restart_fn=safe_restart,
    status_text_fn=status_text,
    workers=WORKERS,
    pending=PENDING,
    running=RUNNING,
    soft_timeout_sec=SOFT_TIMEOUT_SEC,
    hard_timeout_sec=HARD_TIMEOUT_SEC,
    queue_review_task_fn=queue_review_task,
    sort_pending_fn=sort_pending,
    persist_queue_snapshot_fn=persist_queue_snapshot,
    consciousness=_consciousness,
    launcher_file=__file__,
)

offset = int(load_state().get("tg_offset") or 0)
_last_diag_heartbeat_ts = 0.0
_last_message_ts: float = time.time()  # Start in active mode after restart
_ACTIVE_MODE_SEC: int = 300  # 5 min of activity = active polling mode

# Auto-start background consciousness (creator's policy: always on by default)
try:
    _consciousness.start()
    log.info("🧠 Background consciousness auto-started (default: always on)")
except Exception as e:
    log.warning("consciousness auto-start failed: %s", e)

while True:
    loop_started_ts = time.time()
    rotate_chat_log_if_needed(DRIVE_ROOT)
    ensure_workers_healthy()

    # Drain worker events
    event_q = get_event_q()
    while True:
        try:
            evt = event_q.get_nowait()
        except _queue_mod.Empty:
            break
        dispatch_event(evt, _event_ctx)

    enforce_task_timeouts()
    enqueue_evolution_task_if_needed()
    assign_tasks()
    persist_queue_snapshot(reason="main_loop")

    _now = time.time()
    # Poll Telegram — adaptive: fast when active, long-poll when idle
    _active = (_now - _last_message_ts) < _ACTIVE_MODE_SEC
    _poll_timeout = 0 if _active else 10
    try:
        updates = TG.get_updates(offset=offset, timeout=_poll_timeout)
    except Exception as e:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "telegram_poll_error", "offset": offset, "error": repr(e),
            },
        )
        time.sleep(1.5)
        continue

    for upd in updates:
        offset = int(upd["update_id"]) + 1
        poll_handled, poll_rec = handle_poll_update(DRIVE_ROOT, upd)
        if poll_handled:
            if poll_rec:
                _last_message_ts = time.time()
            continue

        callback_query = upd.get("callback_query")
        if isinstance(callback_query, dict):
            if _improvements.handle_callback(callback_query):
                continue
            if _access.handle_callback(callback_query):
                continue
            if _teamchat.handle_callback(callback_query):
                continue

        if _teamchat.handle_added_update(upd):
            continue

        msg = upd.get("message") or upd.get("edited_message") or {}
        if not msg:
            continue

        chat = msg.get("chat") or {}
        chat_id = int(chat["id"])
        chat_type = str(chat.get("type") or "private").lower()
        from_user = msg.get("from") or {}
        user_id = int(from_user.get("id") or 0)
        text = str(msg.get("text") or "")
        caption = str(msg.get("caption") or "")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Extract attachment metadata first; actual downloads happen only after
        # the workspace has been resolved and approved.
        image_file_id = ""
        pending_audio = None
        pending_audio_type = ""
        pending_document = None
        if msg.get("photo"):
            best_photo = msg["photo"][-1]
            image_file_id = str(best_photo.get("file_id") or "")
        elif msg.get("voice"):
            pending_audio = msg["voice"]
            pending_audio_type = "voice"
        elif msg.get("audio"):
            pending_audio = msg["audio"]
            pending_audio_type = "audio"
        elif msg.get("document"):
            doc = msg["document"]
            mime_type = str(doc.get("mime_type") or "")
            if mime_type.startswith("image/"):
                image_file_id = str(doc.get("file_id") or "")
            elif is_audio_attachment(doc):
                pending_audio = doc
                pending_audio_type = "document_audio"
            else:
                pending_document = doc

        st = load_state()
        is_team_chat = False
        team_slug = ""
        team_chat_id = None
        is_group_chat = is_group_chat_type(chat_type)

        if is_group_chat:
            rec, _created, should_notify = request_team_chat(DRIVE_ROOT, chat, requested_by=from_user)
            status = team_chat_status(rec)
            if status == TEAM_PENDING:
                if should_notify:
                    if _teamchat.send_request_to_admins(rec) > 0:
                        append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
                            "ts": now_iso,
                            "type": "team_chat_access_request",
                            "chat_id": chat_id,
                            "source": "message_fallback",
                        })
                _last_message_ts = time.time()
                continue
            if status == TEAM_DENIED:
                _last_message_ts = time.time()
                continue
            if status != TEAM_APPROVED:
                _last_message_ts = time.time()
                continue

            user_drive_root = ensure_team_workspace(DRIVE_ROOT, chat_id)
            note_team_member_seen(DRIVE_ROOT, chat_id, from_user)
            is_team_chat = True
            team_chat_id = chat_id
            team_slug = team_slug_for_chat(chat_id)
            is_admin = _is_admin_user(user_id, st)
            user_role = "admin" if is_admin else "user"
        else:
            user_drive_root = None
            is_admin = False
            user_role = "user"

        owner_missing = st.get("owner_id") is None
        user_allowed_as_admin = (not ADMIN_USER_IDS) or (user_id in ADMIN_USER_IDS)
        if not is_group_chat and owner_missing and user_allowed_as_admin:
            st["owner_id"] = user_id
            st["owner_chat_id"] = chat_id
            st["last_owner_message_at"] = now_iso
            if ADMIN_USER_IDS:
                st["admin_user_ids"] = sorted(ADMIN_USER_IDS)
            save_state(st)
            ensure_user_workspace(
                DRIVE_ROOT, user_id, chat_id, from_user,
                role="admin", use_global_root=True,
            )
            log_chat("in", chat_id, user_id, text, drive_root=DRIVE_ROOT)
            send_with_budget(chat_id, "✅ Owner registered. Ouroboros online.")
            _access.notify_unnotified_requests()
            continue

        if not is_group_chat:
            is_admin = _is_admin_user(user_id, st)
            user_role = "admin" if is_admin else "user"
        preapproved_user = user_id in APPROVED_USER_IDS
        if not is_group_chat and REQUIRE_USER_APPROVAL and not is_admin and not preapproved_user:
            access_rec, access_created, should_notify_admins = request_user_access(
                DRIVE_ROOT, user_id, chat_id, from_user,
            )
            access_status = user_access_status(access_rec)
            if access_status != ACCESS_APPROVED:
                request_text = text or caption or "[non-text access request]"
                log_chat("in", chat_id, user_id, request_text, drive_root=DRIVE_ROOT)
                append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
                    "ts": now_iso,
                    "type": "user_access_request",
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "access_status": access_status,
                    "created": access_created,
                })
                if access_status == ACCESS_PENDING and should_notify_admins:
                    if _access.send_request_to_admins(access_rec) > 0:
                        _access.mark_request_notified(user_id)

                if access_status == ACCESS_PENDING:
                    reply = (
                        "🔐 Запрос доступа отправлен администратору. "
                        "Я напишу здесь, когда доступ будет предоставлен."
                    )
                else:
                    reply = "⛔️ Доступ к боту отклонён администратором."
                st["last_user_access_request_at"] = now_iso
                _last_message_ts = time.time()
                save_state(st)
                send_with_budget(
                    chat_id,
                    reply,
                    log_drive_root=DRIVE_ROOT,
                    log_user_id=user_id,
                )
                continue

        if not is_group_chat:
            user_drive_root, user_created, _user_record = ensure_user_workspace(
                DRIVE_ROOT, user_id, chat_id, from_user,
                role=user_role, use_global_root=is_admin,
                access_status=ACCESS_APPROVED,
            )
        else:
            user_created = False
        if is_admin:
            _access.notify_unnotified_requests()

        image_data = None
        if image_file_id:
            b64, mime = TG.download_file_base64(image_file_id)
            if b64:
                image_data = (b64, mime, caption)

        saved_audio = None
        audio_transcription = ""
        audio_download_failed = False
        audio_transcription_failed = False
        audio_transcription_error = ""
        if pending_audio is not None:
            file_id = str(pending_audio.get("file_id") or "")
            if file_id:
                file_bytes, detected_mime, telegram_path = TG.download_file_bytes(
                    file_id,
                    max_bytes=AUDIO_TRANSCRIPTION_MAX_BYTES,
                )
                if file_bytes is not None:
                    duration_sec = int(pending_audio.get("duration") or 0)
                    saved_audio = save_incoming_audio(
                        user_drive_root,
                        file_bytes=file_bytes,
                        original_name=audio_upload_name(
                            pending_audio,
                            attachment_type=pending_audio_type,
                            message_id=int(msg.get("message_id") or 0),
                            telegram_path=telegram_path,
                        ),
                        mime_type=str(pending_audio.get("mime_type") or detected_mime or ""),
                        telegram_file_id=file_id,
                        telegram_file_unique_id=str(pending_audio.get("file_unique_id") or ""),
                        caption=caption,
                        message_id=int(msg.get("message_id") or 0),
                        attachment_type=pending_audio_type,
                        duration_sec=duration_sec,
                    )
                    append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                        "ts": now_iso,
                        "type": "telegram_audio_saved",
                        "path": saved_audio.get("path"),
                        "mime_type": saved_audio.get("mime_type"),
                        "size_bytes": saved_audio.get("size_bytes"),
                        "duration_sec": saved_audio.get("duration_sec"),
                        "attachment_type": saved_audio.get("attachment_type"),
                    })
                    if AUDIO_TRANSCRIPTION_ENABLED:
                        try:
                            TG.send_chat_action(chat_id, "typing")
                            audio_path = user_drive_root / str(saved_audio["path"])
                            audio_transcription = transcribe_audio_file(
                                audio_path,
                                model=AUDIO_TRANSCRIPTION_MODEL,
                                language=AUDIO_TRANSCRIPTION_LANGUAGE,
                                prompt=AUDIO_TRANSCRIPTION_PROMPT,
                            )
                            transcript_path = audio_path.with_name(audio_path.name + ".transcript.txt")
                            transcript_path.write_text(audio_transcription, encoding="utf-8")
                            saved_audio["transcript_path"] = str(transcript_path.relative_to(user_drive_root))
                            append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "type": "telegram_audio_transcribed",
                                "path": saved_audio.get("path"),
                                "transcript_path": saved_audio.get("transcript_path"),
                                "model": AUDIO_TRANSCRIPTION_MODEL,
                                "transcript_chars": len(audio_transcription),
                                "duration_sec": saved_audio.get("duration_sec"),
                            })
                        except Exception as e:
                            audio_transcription_failed = True
                            audio_transcription_error = f"{type(e).__name__}: {e}"[:1000]
                            append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "type": "telegram_audio_transcription_failed",
                                "path": saved_audio.get("path"),
                                "model": AUDIO_TRANSCRIPTION_MODEL,
                                "error": audio_transcription_error,
                            })
                    else:
                        audio_transcription_failed = True
                        audio_transcription_error = "audio transcription disabled"
                        append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "type": "telegram_audio_transcription_disabled",
                            "path": saved_audio.get("path"),
                        })
                else:
                    append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                        "ts": now_iso,
                        "type": "telegram_audio_download_failed",
                        "attachment_type": pending_audio_type,
                        "mime_type": str(pending_audio.get("mime_type") or ""),
                    })
                    audio_download_failed = True
            else:
                audio_download_failed = True

        saved_document = None
        document_download_failed = False
        if pending_document is not None:
            file_id = str(pending_document.get("file_id") or "")
            if file_id:
                file_bytes, detected_mime, _telegram_path = TG.download_file_bytes(file_id)
                if file_bytes:
                    saved_document = save_incoming_document(
                        user_drive_root,
                        file_bytes=file_bytes,
                        original_name=str(pending_document.get("file_name") or "telegram_document"),
                        mime_type=str(pending_document.get("mime_type") or detected_mime or ""),
                        telegram_file_id=file_id,
                        telegram_file_unique_id=str(pending_document.get("file_unique_id") or ""),
                        caption=caption,
                        message_id=int(msg.get("message_id") or 0),
                    )
                    append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                        "ts": now_iso,
                        "type": "telegram_document_saved",
                        "path": saved_document.get("path"),
                        "mime_type": saved_document.get("mime_type"),
                        "size_bytes": saved_document.get("size_bytes"),
                    })
                else:
                    append_jsonl(user_drive_root / "logs" / "events.jsonl", {
                        "ts": now_iso,
                        "type": "telegram_document_download_failed",
                        "file_name": str(pending_document.get("file_name") or ""),
                        "mime_type": str(pending_document.get("mime_type") or ""),
                    })
                    document_download_failed = True
            else:
                document_download_failed = True

        log_text = text
        if saved_audio and not audio_transcription_failed:
            log_text = (text or caption or "") + (
                f"\n[attached audio transcribed: {saved_audio['path']}]\n{audio_transcription}"
            )
        elif saved_audio and audio_transcription_failed:
            log_text = (text or caption or "") + (
                f"\n[attached audio saved: {saved_audio['path']}]\n"
                f"[audio transcription failed: {audio_transcription_error}]"
            )
        elif audio_download_failed:
            log_text = (text or caption or "") + "\n[attached audio download failed]"
        elif saved_document:
            log_text = (text or caption or "") + f"\n[attached document saved: {saved_document['path']}]"
        elif document_download_failed:
            log_text = (text or caption or "") + "\n[attached document download failed]"
        log_chat("in", chat_id, user_id, log_text, drive_root=user_drive_root)
        if is_team_chat:
            st["last_team_message_at"] = now_iso
        elif is_admin:
            st["last_owner_message_at"] = now_iso
        else:
            st["last_user_message_at"] = now_iso
        _last_message_ts = time.time()
        save_state(st)

        team_trigger_text = text
        if is_team_chat and not team_trigger_text and audio_transcription:
            team_trigger_text = audio_transcription
        if is_team_chat and not _teamchat.is_group_task_trigger(msg, team_trigger_text, caption):
            continue
        if is_team_chat and team_trigger_text:
            text = _teamchat.prepare_group_task_text(team_trigger_text)

        if not is_team_chat and user_created and not is_admin and text.strip().lower().startswith("/start"):
            send_with_budget(
                chat_id,
                "✅ User workspace initialized. You can use this bot as your personal agent.",
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue

        # --- Supervisor commands ---
        if is_admin_only_command(text) and not is_admin:
            send_with_budget(
                chat_id,
                "⚠️ Admin menu and access management are admin-only.",
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue

        if text.strip().lower().startswith("/") and is_admin:
            try:
                result = _supervisor_commands.handle(text, chat_id, user_id, tg_offset=offset)
                if result is True:
                    continue  # terminal command, fully handled
                elif result:  # non-empty string = dual-path note
                    text = result + text  # prepend note, fall through to LLM
            except SystemExit:
                raise
            except Exception:
                log.warning("Supervisor command handler error", exc_info=True)

        if not is_team_chat and text.strip().lower().startswith("/start") and not is_admin:
            send_with_budget(
                chat_id,
                "✅ Your personal workspace is ready. Send a message to start working with the agent.",
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue

        if document_download_failed:
            send_with_budget(
                chat_id,
                "⚠️ I could not download the attached document from Telegram. Please send the file again.",
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue
        if audio_download_failed:
            send_with_budget(
                chat_id,
                "⚠️ Не смог скачать голосовое/аудио из Telegram. Пришли его ещё раз.",
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue
        if saved_audio and audio_transcription_failed:
            detail = f"\nПричина: {audio_transcription_error}" if is_admin and audio_transcription_error else ""
            send_with_budget(
                chat_id,
                "⚠️ Аудиофайл сохранил, но не смог распознать через OpenAI Whisper." + detail,
                log_drive_root=user_drive_root,
                log_user_id=user_id,
            )
            continue

        # All other messages (and dual-path commands) → direct chat with Ouroboros
        if not text and not image_data and not saved_document and not saved_audio:
            continue  # empty message, skip

        # Feed observation to consciousness
        if is_admin:
            observation = text or audio_transcription
            _consciousness.inject_observation(f"Owner message: {observation[:100]}")

        agent = _get_chat_agent(user_id=user_id, drive_root=user_drive_root, user_role=user_role)
        audio_note = ""
        if saved_audio:
            audio_note = format_audio_task_text(text, caption, saved_audio, audio_transcription)

        if agent._busy or getattr(agent, "_dispatching", False):
            # BUSY PATH: inject into active conversation (single consumer)
            if image_data:
                if text:
                    agent.inject_message(text)
                send_with_budget(
                    chat_id,
                    "📎 Photo received, but a task is in progress. Send again when I'm free.",
                    log_drive_root=user_drive_root,
                    log_user_id=user_id,
                )
            elif saved_audio:
                send_with_budget(
                    chat_id,
                    f"🎙️ Аудио распознано: {saved_audio['filename']}. Добавил текст к текущей задаче.",
                    is_progress=True,
                    log_drive_root=user_drive_root,
                    log_user_id=user_id,
                )
                agent.inject_message(audio_note)
            elif saved_document:
                send_with_budget(
                    chat_id,
                    f"📎 Файл получил: {saved_document['filename']}. Добавил его к текущей задаче, разберу после текущего шага.",
                    is_progress=True,
                    log_drive_root=user_drive_root,
                    log_user_id=user_id,
                )
                doc_note = (
                    f"{text or caption or 'Document attached.'}\n\n"
                    "[Telegram document saved]\n"
                    f"- path: {saved_document['path']}\n"
                    f"- filename: {saved_document['filename']}\n"
                    f"- mime_type: {saved_document['mime_type']}\n"
                    f"- size_bytes: {saved_document['size_bytes']}\n"
                    "Use analyze_document(path='<path>', source='drive') if this file is relevant."
                )
                agent.inject_message(doc_note)
            elif text:
                agent.inject_message(text)

        else:
            # FREE PATH: mark dispatching before starting the thread so any
            # immediately following Telegram update for this user is injected
            # into the same conversation instead of launching a parallel task.
            agent._dispatching = True
            final_text = text
            if saved_audio:
                send_with_budget(
                    chat_id,
                    f"🎙️ Аудио распознано: {saved_audio['filename']}. Сейчас отвечу по тексту.",
                    is_progress=True,
                    log_drive_root=user_drive_root,
                    log_user_id=user_id,
                )
                final_text = audio_note
            elif saved_document:
                send_with_budget(
                    chat_id,
                    f"📎 Файл получил: {saved_document['filename']}. Сохранил в workspace, сейчас открою и разберу содержимое.",
                    is_progress=True,
                    log_drive_root=user_drive_root,
                    log_user_id=user_id,
                )
                doc_note = (
                    "\n\n[Telegram document saved]\n"
                    f"- path: {saved_document['path']}\n"
                    f"- filename: {saved_document['filename']}\n"
                    f"- mime_type: {saved_document['mime_type']}\n"
                    f"- size_bytes: {saved_document['size_bytes']}\n"
                    "Use analyze_document(path='<path>', source='drive') if the user asks about this file."
                )
                final_text = (text or caption or "Please analyze the attached document.") + doc_note
            if is_admin:
                _consciousness.pause()

            def _run_task_and_resume(
                cid, txt, img,
                _user_id=user_id,
                _user_role=user_role,
                _user_drive_root=user_drive_root,
                _chat_type=chat_type,
                _team_chat_id=team_chat_id,
                _team_slug=team_slug,
                _is_team_workspace=is_team_chat,
                _is_admin=is_admin,
                _agent=agent,
            ):
                try:
                    handle_chat_direct(
                        cid, txt, img,
                        user_id=_user_id,
                        user_role=_user_role,
                        drive_root=_user_drive_root,
                        chat_type=_chat_type,
                        team_chat_id=_team_chat_id,
                        team_slug=_team_slug,
                        is_team_workspace=_is_team_workspace,
                    )
                finally:
                    _agent._dispatching = False
                    if _is_admin:
                        _consciousness.resume()

            _t = threading.Thread(
                target=_run_task_and_resume,
                args=(chat_id, final_text, image_data),
                daemon=True,
            )
            try:
                _t.start()
            except Exception as _te:
                agent._dispatching = False
                log.error("Failed to start chat thread: %s", _te)
                if is_admin:
                    _consciousness.resume()  # ensure resume if thread fails to start

    st = load_state()
    st["tg_offset"] = offset
    save_state(st)

    now_epoch = time.time()
    loop_duration_sec = now_epoch - loop_started_ts

    if DIAG_SLOW_CYCLE_SEC > 0 and loop_duration_sec >= float(DIAG_SLOW_CYCLE_SEC):
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "main_loop_slow_cycle",
                "duration_sec": round(loop_duration_sec, 3),
                "pending_count": len(PENDING),
                "running_count": len(RUNNING),
            },
        )

    if DIAG_HEARTBEAT_SEC > 0 and (now_epoch - _last_diag_heartbeat_ts) >= float(DIAG_HEARTBEAT_SEC):
        workers_total = len(WORKERS)
        workers_alive = sum(1 for w in WORKERS.values() if w.proc.is_alive())
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "main_loop_heartbeat",
                "offset": offset,
                "workers_total": workers_total,
                "workers_alive": workers_alive,
                "pending_count": len(PENDING),
                "running_count": len(RUNNING),
                "event_q_size": _safe_qsize(event_q),
                "running_task_ids": list(RUNNING.keys())[:5],
                "spent_usd": st.get("spent_usd"),
            },
        )
        _last_diag_heartbeat_ts = now_epoch

    # Short sleep in active mode (fast response), longer when idle (save CPU)
    _loop_sleep = 0.1 if (_now - _last_message_ts) < _ACTIVE_MODE_SEC else 0.5
    time.sleep(_loop_sleep)
