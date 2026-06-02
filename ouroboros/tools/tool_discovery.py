"""Tool discovery meta-tools — lets the agent see and enable tool packs."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ouroboros.tools.registry import ToolContext, ToolEntry

if TYPE_CHECKING:
    from ouroboros.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Module-level registry reference — set by set_registry() after ToolRegistry is created.
# loop.py also overrides these handlers with closures that have access to per-loop state
# (e.g. the _enabled_extra_tools set); the module-level ref serves as a fallback for
# any context where the tool is called without going through run_llm_loop.
_registry: Optional["ToolRegistry"] = None


def set_registry(reg: "ToolRegistry") -> None:
    global _registry
    _registry = reg


def _list_available_tools(ctx: ToolContext, **kwargs) -> str:
    if _registry is None:
        return "Tool discovery not available in this context."
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for tool in _registry.list_non_core_tools():
        grouped.setdefault(str(tool.get("pack") or "other"), []).append(tool)
    if not grouped:
        return "No additional tools are available."
    lines = [
        "Additional tools are grouped by pack. Prefer `list_tool_packs` and "
        "`enable_tool_pack`; use `enable_tools` only for exact one-off tools.\n"
    ]
    for pack, tools in sorted(grouped.items()):
        sample = ", ".join(t["name"] for t in tools[:5])
        suffix = "" if len(tools) <= 5 else f", +{len(tools) - 5} more"
        lines.append(f"- **{pack}** ({len(tools)}): {sample}{suffix}")
    return "\n".join(lines)


def _list_tool_packs(ctx: ToolContext, **kwargs) -> str:
    if _registry is None:
        return "Tool pack discovery not available in this context."
    packs = _registry.list_tool_packs()
    if not packs:
        return "No tool packs are available."
    lines = ["**Available tool packs** (use `enable_tool_pack` to activate one):\n"]
    for pack in packs:
        deps = pack.get("dependencies") or []
        dep_text = f" deps={','.join(deps)}" if deps else ""
        lines.append(
            f"- **{pack['name']}** ({pack['tool_count']} tools{dep_text}): "
            f"{pack['description']}"
        )
    return "\n".join(lines)


def _enable_tool_pack(ctx: ToolContext, pack: str = "", **kwargs) -> str:
    if _registry is None:
        return "Tool pack enablement not available in this context."
    names = [n.strip() for n in str(pack or "").split(",") if n.strip()]
    if not names:
        return "No tool pack specified."
    parts = []
    known = {p["name"] for p in _registry.list_tool_packs()}
    for name in names:
        tools = _registry.get_tools_by_pack(name, include_dependencies=True)
        canonical = next((p for p in known if p == name.strip().lower().replace("-", "_")), name)
        if tools:
            parts.append(f"✅ Pack `{canonical}` is registered and callable: {', '.join(tools)}")
        else:
            parts.append(f"❌ Pack not found or unavailable: {name}")
    return "\n".join(parts)


def _enable_tools(ctx: ToolContext, tools: str = "", **kwargs) -> str:
    if _registry is None:
        return "Tool enablement not available in this context."
    names = [n.strip() for n in tools.split(",") if n.strip()]
    if not names:
        return "No tools specified."
    found = []
    not_found = []
    for name in names:
        schema = _registry.get_schema_by_name(name)
        if schema:
            found.append(f"{name}: {schema['function'].get('description', '')[:100]}")
        else:
            not_found.append(name)
    parts = []
    if found:
        parts.append("✅ Tools are registered and callable:\n" + "\n".join(f"  - {s}" for s in found))
    if not_found:
        parts.append(f"❌ Not found: {', '.join(not_found)}")
    return "\n".join(parts)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_tool_packs",
            schema={
                "name": "list_tool_packs",
                "description": (
                    "List thematic tool packs such as documents, web_browser, code_git, "
                    "credit, hr, team, memory, and admin_control. Use this before enabling "
                    "tools when the active pack does not fit the task."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            handler=_list_tool_packs,
        ),
        ToolEntry(
            name="enable_tool_pack",
            schema={
                "name": "enable_tool_pack",
                "description": (
                    "Enable one or more thematic tool packs for the remainder of this task. "
                    "Examples: enable_tool_pack(pack='credit'), enable_tool_pack(pack='documents,files')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pack": {
                            "type": "string",
                            "description": "Tool pack name, or comma-separated pack names.",
                        }
                    },
                    "required": ["pack"],
                },
            },
            handler=_enable_tool_pack,
        ),
        ToolEntry(
            name="list_available_tools",
            schema={
                "name": "list_available_tools",
                "description": (
                    "List additional tools grouped by pack. Prefer list_tool_packs and "
                    "enable_tool_pack; use this only when you need exact tool names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            handler=_list_available_tools,
        ),
        ToolEntry(
            name="enable_tools",
            schema={
                "name": "enable_tools",
                "description": (
                    "Enable specific additional tools by name (comma-separated). "
                    "Their schemas will be added to your active tool set for the "
                    "remainder of this task. Prefer enable_tool_pack for normal use. "
                    "Example: enable_tools(tools='multi_model_review,generate_evolution_stats')"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tools": {
                            "type": "string",
                            "description": "Comma-separated tool names to enable",
                        }
                    },
                    "required": ["tools"],
                },
            },
            handler=_enable_tools,
        ),
    ]
