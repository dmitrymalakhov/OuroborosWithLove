"""
Ouroboros — Tool registry (SSOT).

Plugin architecture: each module in tools/ exports get_tools().
ToolRegistry collects all tools, provides schemas() and execute().
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ouroboros.utils import safe_relpath


@dataclass
class BrowserState:
    """Per-task browser lifecycle state (Playwright). Isolated from generic ToolContext."""

    pw_instance: Any = None
    browser: Any = None
    page: Any = None
    last_screenshot_b64: Optional[str] = None


@dataclass
class ToolContext:
    """Tool execution context — passed from the agent before each task."""

    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    shared_drive_root: Optional[pathlib.Path] = None
    branch_dev: str = "ouroboros"
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    current_chat_id: Optional[int] = None
    chat_type: str = "private"
    current_user_id: Optional[int] = None
    user_role: str = "admin"
    team_chat_id: Optional[int] = None
    team_slug: str = ""
    is_team_workspace: bool = False
    current_task_type: Optional[str] = None
    last_push_succeeded: bool = False
    emit_progress_fn: Callable[[str], None] = field(default=lambda _: None)

    # LLM-driven model/effort switch (set by switch_model tool, read by loop.py)
    active_model_override: Optional[str] = None
    active_effort_override: Optional[str] = None

    # Per-task browser state
    browser_state: BrowserState = field(default_factory=BrowserState)

    # Budget tracking (set by loop.py for real-time usage events)
    event_queue: Optional[Any] = None
    task_id: Optional[str] = None

    # Task depth for fork bomb protection
    task_depth: int = 0

    # True when running inside handle_chat_direct (not a queued worker task)
    is_direct_chat: bool = False

    def __post_init__(self) -> None:
        if self.shared_drive_root is None:
            self.shared_drive_root = self.drive_root
        self.chat_type = str(self.chat_type or "private").lower()
        self.user_role = str(self.user_role or "user").lower()
        self.team_slug = str(self.team_slug or "")
        self.is_team_workspace = bool(self.is_team_workspace)

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()

    def shared_drive_path(self, rel: str) -> pathlib.Path:
        root = self.shared_drive_root or self.drive_root
        return (root / safe_relpath(rel)).resolve()

    def drive_logs(self) -> pathlib.Path:
        return (self.drive_root / "logs").resolve()

    def event_scope(self) -> Dict[str, Any]:
        """Return the multi-user routing scope for events emitted by tools."""
        return {
            "chat_id": self.current_chat_id,
            "chat_type": self.chat_type,
            "user_id": self.current_user_id,
            "user_role": self.user_role,
            "drive_root": str(self.drive_root),
            "shared_drive_root": str(self.shared_drive_root or self.drive_root),
            "team_chat_id": self.team_chat_id,
            "team_slug": self.team_slug,
            "is_team_workspace": self.is_team_workspace,
        }


@dataclass
class ToolEntry:
    """Single tool descriptor: name, schema, handler, metadata."""

    name: str
    schema: Dict[str, Any]
    handler: Callable  # fn(ctx: ToolContext, **args) -> str
    is_code_tool: bool = False
    timeout_sec: int = 120
    pack: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "normal"


TOOL_PACKS: Dict[str, Set[str]] = {
    "base": {
        "list_tool_packs", "enable_tool_pack", "list_available_tools", "enable_tools",
        "chat_history", "update_scratchpad", "compact_context", "offer_improvement_request",
    },
    "files": {
        "drive_read", "drive_list", "drive_write",
    },
    "documents": {
        "analyze_document", "index_document", "search_document", "extract_archive", "download_url_to_drive", "send_file",
    },
    "document_editing": {
        "inspect_pdf_for_edit", "edit_pdf", "inspect_word_for_edit", "edit_word",
    },
    "web_browser": {
        "web_search", "browse_page", "browser_action",
    },
    "vision": {
        "analyze_screenshot", "vlm_query", "edit_image", "send_photo",
    },
    "code_git": {
        "repo_read", "repo_list", "repo_write_commit", "repo_commit_push",
        "git_status", "git_diff", "run_shell", "claude_code_edit",
        "codebase_digest",
    },
    "memory": {
        "update_identity", "send_owner_message", "knowledge_read", "knowledge_write",
        "knowledge_list", "summarize_dialogue",
    },
    "orchestration": {
        "schedule_task", "get_task_result", "wait_for_task", "cancel_task",
        "forward_to_worker",
    },
    "team": {
        "team_inbox_send", "team_inbox_read", "team_chat_history", "team_chat_search",
        "team_members", "team_poll_create", "team_poll_results", "team_poll_close",
    },
    "presentation": {
        "create_presentation", "inspect_presentation_for_edit",
        "edit_presentation", "convert_pptx_to_pdf",
    },
    "spreadsheets": {
        "inspect_excel_template", "fill_excel_template",
        "inspect_excel_charts", "create_excel_line_chart",
    },
    "credit": {
        "credit_pack_check", "credit_metrics_check", "credit_deck_challenge",
        "credit_speaker_qna", "credit_committee_readiness", "credit_memo_draft",
        "credit_deck_outline",
    },
    "hr": {
        "hr_vacancy_audit", "hr_role_profile", "hr_candidate_screen",
        "hr_interview_kit", "hr_onboarding_checklist",
    },
    "health_review": {
        "request_review", "multi_model_review", "codebase_health",
        "generate_evolution_stats",
    },
    "admin_control": {
        "request_restart", "promote_to_stable", "toggle_evolution",
        "toggle_consciousness", "switch_model",
    },
    "github": {
        "list_github_issues", "get_github_issue", "comment_on_issue",
        "close_github_issue", "create_github_issue",
    },
}


TOOL_PACK_DESCRIPTIONS: Dict[str, str] = {
    "base": "Always-on minimal discovery, memory, and context tools.",
    "files": "Read, list, and write workspace files.",
    "documents": "Analyze PDFs, archives, office documents, and downloadable files.",
    "document_editing": "Inspect and edit PDF/DOCX copies with redactions, overlays, text edits, comments, and form/table updates.",
    "web_browser": "Search the web, browse pages, and automate browser actions.",
    "vision": "Analyze screenshots/images, edit images, and send images back to Telegram.",
    "code_git": "Inspect and modify the repository, run shell/code tools, and use git.",
    "memory": "Persistent memory, knowledge topics, identity, and dialogue summaries.",
    "orchestration": "Schedule, poll, cancel, and route background tasks.",
    "team": "Approved Telegram team workspace history, inbox, members, and polls.",
    "presentation": "Create PowerPoint decks, edit existing PPTX copies, and export PPTX presentations to PDF.",
    "spreadsheets": "Inspect/fill Excel .xlsx workbooks and create or verify native charts.",
    "credit": "Corporate credit committee preparation, challenge, memo, metrics, and Q&A workflows.",
    "hr": "Hiring playbook workflows: vacancy audit, role profile, screening, interviews, onboarding.",
    "health_review": "Codebase health checks, evolution stats, and multi-model review.",
    "admin_control": "Runtime restart, promotion, model switching, and evolution/consciousness controls.",
    "github": "GitHub issue tracking workflows.",
}


TOOL_PACK_DEPENDENCIES: Dict[str, Set[str]] = {
    "document_editing": {"documents"},
    "spreadsheets": {"documents"},
    "credit": {"documents"},
    "hr": {"documents"},
}


TOOL_PACK_ALIASES: Dict[str, str] = {
    "browser": "web_browser",
    "web": "web_browser",
    "git": "code_git",
    "code": "code_git",
    "github_issues": "github",
    "credit_committee": "credit",
    "hiring": "hr",
    "pdf_editing": "document_editing",
    "pdf_edit": "document_editing",
    "pdf_edits": "document_editing",
    "pdf_editor": "document_editing",
    "word_editing": "document_editing",
    "word": "document_editing",
    "docx": "document_editing",
    "word_edit": "document_editing",
    "word_editor": "document_editing",
    "image": "vision",
    "images": "vision",
    "image_edit": "vision",
    "image_editing": "vision",
    "photo": "vision",
    "photos": "vision",
    "spreadsheet": "spreadsheets",
    "excel": "spreadsheets",
    "presentation_editing": "presentation",
    "presentation_editor": "presentation",
    "presentations": "presentation",
    "slides": "presentation",
    "deck": "presentation",
    "pptx": "presentation",
}


BASE_TOOL_PACK = "base"
BASE_TOOL_NAMES = TOOL_PACKS[BASE_TOOL_PACK]


def normalize_tool_pack(name: str) -> str:
    """Return canonical tool pack name, accepting a small alias set."""
    raw = str(name or "").strip().lower().replace("-", "_")
    return TOOL_PACK_ALIASES.get(raw, raw)


def tool_pack_for_name(tool_name: str) -> str:
    """Return the configured pack for a tool name, or empty string if unmapped."""
    for pack, names in TOOL_PACKS.items():
        if tool_name in names:
            return pack
    return ""


SELF_MOD_DISABLED_TOOL_NAMES = {
    "repo_write_commit", "repo_commit_push", "claude_code_edit",
    "request_restart", "promote_to_stable", "toggle_evolution",
    "multi_model_review", "generate_evolution_stats",
}


SELF_MODIFICATION_ADMIN_ONLY_NOTICE = (
    "Самомодификация и доступ к репозиторию доступны только админу Ouroboros."
)

SELF_MODIFICATION_ADMIN_ONLY_PACKS = {"code_git", "admin_control", "health_review"}

REPOSITORY_SELF_MOD_TOOL_NAMES = (
    TOOL_PACKS["code_git"]
    | {
        "codebase_health", "request_review", "multi_model_review",
        "generate_evolution_stats", "request_restart", "promote_to_stable",
        "toggle_evolution", "toggle_consciousness",
    }
)


ADMIN_ONLY_TOOL_NAMES = {
    # Codebase, shell, git, and deployment surface.
    "repo_read", "repo_list", "repo_write_commit", "repo_commit_push",
    "codebase_digest", "codebase_health",
    "run_shell", "claude_code_edit",
    "git_status", "git_diff",
    "request_restart", "promote_to_stable", "request_review",
    "toggle_evolution", "toggle_consciousness",
    "multi_model_review", "generate_evolution_stats",
    # External project management surface.
    "list_github_issues", "get_github_issue", "comment_on_issue",
    "close_github_issue", "create_github_issue",
    # Users share one budget; only admin can alter model/effort explicitly.
    "switch_model",
}


class ToolRegistry:
    """Ouroboros tool registry (SSOT).

    To add a tool: create a module in ouroboros/tools/,
    export get_tools() -> List[ToolEntry].
    """

    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self._entries: Dict[str, ToolEntry] = {}
        self._ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
        self._load_modules()

    def _load_modules(self) -> None:
        """Auto-discover tool modules in ouroboros/tools/ that export get_tools()."""
        import importlib
        import pkgutil
        import ouroboros.tools as tools_pkg
        for _importer, modname, _ispkg in pkgutil.iter_modules(tools_pkg.__path__):
            if modname.startswith("_") or modname == "registry":
                continue
            try:
                mod = importlib.import_module(f"ouroboros.tools.{modname}")
                if hasattr(mod, "get_tools"):
                    for entry in mod.get_tools():
                        self._assign_pack(entry)
                        self._entries[entry.name] = entry
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load tool module %s", modname, exc_info=True)

    def _assign_pack(self, entry: ToolEntry) -> None:
        """Attach configured pack metadata when the tool module did not set it."""
        if not entry.pack:
            entry.pack = tool_pack_for_name(entry.name)

    def set_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    def register(self, entry: ToolEntry) -> None:
        """Register a new tool (for extension by Ouroboros)."""
        self._assign_pack(entry)
        self._entries[entry.name] = entry

    # --- Contract ---

    def _self_modification_disabled(self) -> bool:
        return os.environ.get("OUROBOROS_DISABLE_SELF_MODIFICATION", "").strip().lower() in ("1", "true", "yes", "on")

    def _is_admin_context(self) -> bool:
        return str(getattr(self._ctx, "user_role", "admin") or "user").lower() == "admin"

    def _is_tool_allowed(self, name: str) -> bool:
        if self._self_modification_disabled():
            if name in SELF_MOD_DISABLED_TOOL_NAMES:
                return False
        if self._is_admin_context():
            return True
        return name not in ADMIN_ONLY_TOOL_NAMES

    def unavailable_tool_message(self, name: str) -> str:
        """Return a user-facing reason when a known tool exists but is blocked."""
        entry = self._entries.get(name)
        if entry is None or self._is_tool_allowed(name):
            return ""
        if not self._is_admin_context() and name in ADMIN_ONLY_TOOL_NAMES:
            if name in REPOSITORY_SELF_MOD_TOOL_NAMES or entry.pack in SELF_MODIFICATION_ADMIN_ONLY_PACKS:
                return f"⚠️ Tool `{name}` is admin-only. {SELF_MODIFICATION_ADMIN_ONLY_NOTICE}"
            return f"⚠️ Tool `{name}` is admin-only in multi-user mode."
        if self._self_modification_disabled() and name in SELF_MOD_DISABLED_TOOL_NAMES:
            return f"⚠️ Tool `{name}` is disabled because self-modification is disabled by configuration."
        return f"⚠️ Tool `{name}` is unavailable in this runtime."

    def unavailable_tool_pack_message(self, pack: str) -> str:
        """Return a user-facing reason when a known pack exists but is blocked."""
        canonical = normalize_tool_pack(pack)
        if canonical not in TOOL_PACKS:
            return ""
        direct_entries = [e for e in self._entries.values() if e.pack == canonical]
        if not direct_entries or self.get_tools_by_pack(canonical, include_dependencies=False):
            return ""
        direct_tool_names = {e.name for e in direct_entries}
        if not self._is_admin_context() and direct_tool_names & ADMIN_ONLY_TOOL_NAMES:
            if canonical in SELF_MODIFICATION_ADMIN_ONLY_PACKS or direct_tool_names & REPOSITORY_SELF_MOD_TOOL_NAMES:
                return f"⚠️ Pack `{canonical}` is admin-only. {SELF_MODIFICATION_ADMIN_ONLY_NOTICE}"
            return f"⚠️ Pack `{canonical}` is admin-only in multi-user mode."
        if self._self_modification_disabled() and direct_tool_names & SELF_MOD_DISABLED_TOOL_NAMES:
            return f"⚠️ Pack `{canonical}` is disabled because self-modification is disabled by configuration."
        return ""

    def blocked_tool_packs_notice(self) -> str:
        """Summarize hidden packs without marking them as available."""
        blocked = []
        for pack in TOOL_PACKS:
            if self.unavailable_tool_pack_message(pack):
                blocked.append(pack)
        if not blocked:
            return ""
        self_mod_packs = [p for p in blocked if p in SELF_MODIFICATION_ADMIN_ONLY_PACKS]
        other_packs = [p for p in blocked if p not in SELF_MODIFICATION_ADMIN_ONLY_PACKS]
        parts = []
        if self_mod_packs:
            names = ", ".join(f"`{p}`" for p in self_mod_packs)
            parts.append(f"⚠️ Admin-only packs hidden for this user: {names}. {SELF_MODIFICATION_ADMIN_ONLY_NOTICE}")
        if other_packs:
            names = ", ".join(f"`{p}`" for p in other_packs)
            parts.append(f"⚠️ Admin-only packs hidden for this user: {names}.")
        return "\n".join(parts)

    def available_tools(self) -> List[str]:
        return [e.name for e in self._entries.values() if self._is_tool_allowed(e.name)]

    def _expand_packs(self, packs: Iterable[str], include_base: bool = True) -> Set[str]:
        expanded: Set[str] = set()
        if include_base:
            expanded.add(BASE_TOOL_PACK)
        stack = [normalize_tool_pack(p) for p in packs if str(p or "").strip()]
        while stack:
            pack = normalize_tool_pack(stack.pop())
            if pack not in TOOL_PACKS or pack in expanded:
                continue
            expanded.add(pack)
            stack.extend(TOOL_PACK_DEPENDENCIES.get(pack, set()))
        return expanded

    def list_tool_packs(self) -> List[Dict[str, Any]]:
        """Return allowed tool packs with descriptions and visible tool counts."""
        result: List[Dict[str, Any]] = []
        for pack in TOOL_PACKS:
            direct_tools = self.get_tools_by_pack(pack, include_dependencies=False)
            if not direct_tools and pack != BASE_TOOL_PACK:
                continue
            result.append({
                "name": pack,
                "description": TOOL_PACK_DESCRIPTIONS.get(pack, ""),
                "tool_count": len(direct_tools),
                "tools": direct_tools,
                "dependencies": sorted(TOOL_PACK_DEPENDENCIES.get(pack, set())),
            })
        return result

    def get_tools_by_pack(self, pack: str, include_dependencies: bool = False) -> List[str]:
        """Return allowed tools in a pack, preserving registry load order."""
        pack = normalize_tool_pack(pack)
        packs = self._expand_packs([pack], include_base=False) if include_dependencies else {pack}
        if not packs or any(p not in TOOL_PACKS for p in packs):
            return []
        return [
            e.name for e in self._entries.values()
            if e.pack in packs and self._is_tool_allowed(e.name)
        ]

    def schemas_for_packs(self, packs: Iterable[str], include_base: bool = True) -> List[Dict[str, Any]]:
        """Return schemas for base plus selected packs and their dependencies."""
        selected = self._expand_packs(packs, include_base=include_base)
        return [
            {"type": "function", "function": e.schema}
            for e in self._entries.values()
            if e.pack in selected and self._is_tool_allowed(e.name)
        ]

    def schemas(self, core_only: bool = False) -> List[Dict[str, Any]]:
        if not core_only:
            return [
                {"type": "function", "function": e.schema}
                for e in self._entries.values()
                if self._is_tool_allowed(e.name)
            ]
        # Backward-compatible name: "core" now means the minimal base pack.
        return self.schemas_for_packs([BASE_TOOL_PACK], include_base=False)

    def list_non_core_tools(self) -> List[Dict[str, str]]:
        """Return name+description+pack of all tools outside the base pack."""
        result = []
        for e in self._entries.values():
            if not self._is_tool_allowed(e.name):
                continue
            if e.pack != BASE_TOOL_PACK:
                desc = e.schema.get("description", "No description")
                result.append({"name": e.name, "description": desc, "pack": e.pack})
        return result

    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the full schema for a specific tool."""
        entry = self._entries.get(name)
        if entry and self._is_tool_allowed(name):
            return {"type": "function", "function": entry.schema}
        return None

    def get_timeout(self, name: str) -> int:
        """Return timeout_sec for the named tool (default 120)."""
        entry = self._entries.get(name)
        return entry.timeout_sec if entry is not None else 120

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        entry = self._entries.get(name)
        if entry is None:
            return f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(self.available_tools()))}"
        if not self._is_tool_allowed(name):
            return self.unavailable_tool_message(name)
        try:
            return entry.handler(self._ctx, **args)
        except TypeError as e:
            return f"⚠️ TOOL_ARG_ERROR ({name}): {e}"
        except Exception as e:
            return f"⚠️ TOOL_ERROR ({name}): {e}"

    def override_handler(self, name: str, handler) -> None:
        """Override the handler for a registered tool (used for closure injection)."""
        entry = self._entries.get(name)
        if entry:
            self._entries[name] = ToolEntry(
                name=entry.name,
                schema=entry.schema,
                handler=handler,
                is_code_tool=entry.is_code_tool,
                timeout_sec=entry.timeout_sec,
                pack=entry.pack,
                tags=entry.tags,
                risk_level=entry.risk_level,
            )

    @property
    def CODE_TOOLS(self) -> frozenset:
        return frozenset(e.name for e in self._entries.values() if e.is_code_tool)
