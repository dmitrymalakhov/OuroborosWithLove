"""Tool pack routing and dynamic schema enablement for the LLM loop."""

from __future__ import annotations

import json
import os
import pathlib
import queue
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.llm import LLMClient, add_usage, default_light_model
from ouroboros.tools.registry import BASE_TOOL_PACK, ToolRegistry, normalize_tool_pack
from ouroboros.utils import append_jsonl, utc_now_iso


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _message_content_to_text(msg.get("content")).strip()
    return ""


def _parse_tool_router_response(content: str) -> Dict[str, Any]:
    raw = str(content or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    data = json.loads(raw)
    packs_raw = data.get("packs") or []
    if isinstance(packs_raw, str):
        packs_raw = [packs_raw]
    packs = [normalize_tool_pack(str(p)) for p in packs_raw if str(p or "").strip()]
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "packs": packs,
        "confidence": confidence,
        "reason": str(data.get("reason") or "").strip(),
    }


def route_tool_packs(
    messages: List[Dict[str, Any]],
    tools_registry: ToolRegistry,
    llm: LLMClient,
    task_type: str,
    user_role: str,
    drive_logs: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    accumulated_usage: Dict[str, Any],
    user_id: Optional[int],
    estimate_cost_fn: Callable[..., float],
    emit_usage_fn: Callable[..., None],
) -> Dict[str, Any]:
    """Use the lightweight model to choose the initial non-base tool packs."""
    info: Dict[str, Any] = {
        "selected_packs": [],
        "confidence": 0.0,
        "reason": "",
        "source": "base",
        "model": "",
    }
    if os.environ.get("OUROBOROS_TOOL_ROUTER_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
        info["source"] = "disabled"
        return info

    request_text = _last_user_text(messages)
    available_packs = [p for p in tools_registry.list_tool_packs() if p["name"] != BASE_TOOL_PACK]
    allowed_pack_names = {p["name"] for p in available_packs}
    if not request_text or not available_packs:
        return info

    try:
        threshold = float(os.environ.get("OUROBOROS_TOOL_ROUTER_CONFIDENCE", "0.55"))
    except (TypeError, ValueError):
        threshold = 0.55
    try:
        max_packs = max(1, int(os.environ.get("OUROBOROS_TOOL_ROUTER_MAX_PACKS", "3")))
    except (TypeError, ValueError):
        max_packs = 3

    router_model = os.environ.get("OUROBOROS_MODEL_LIGHT", "").strip() or default_light_model(llm.provider)
    if not llm.supports_model(router_model):
        router_model = default_light_model(llm.provider)
    info["model"] = router_model

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-pack router for an LLM agent. Choose 0 to 3 thematic "
                "tool packs needed for the user's request. Do not choose tools. "
                "Return strict JSON only: {\"packs\":[\"pack\"],\"confidence\":0.0,\"reason\":\"short\"}. "
                "If the request is ordinary conversation or ambiguous, return packs=[] and low confidence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "task_type": task_type,
                "user_role": user_role,
                "request": request_text[:6000],
                "available_packs": [
                    {
                        "name": p["name"],
                        "description": p["description"],
                        "dependencies": p.get("dependencies") or [],
                    }
                    for p in available_packs
                ],
            }, ensure_ascii=False),
        },
    ]

    try:
        msg, usage = llm.chat(
            messages=router_messages,
            model=router_model,
            tools=None,
            reasoning_effort="low",
            max_tokens=512,
        )
        add_usage(accumulated_usage, usage)
        cost = float(usage.get("cost") or 0)
        if not cost:
            cost = estimate_cost_fn(
                router_model,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                int(usage.get("cached_tokens") or 0),
                int(usage.get("cache_write_tokens") or 0),
            )
        emit_usage_fn(
            event_queue, task_id, router_model, usage, cost,
            category="tool_router", user_id=user_id, user_role=user_role
        )

        parsed = _parse_tool_router_response(msg.get("content") or "")
        selected = []
        for pack in parsed["packs"]:
            if pack in allowed_pack_names and pack not in selected:
                selected.append(pack)
            if len(selected) >= max_packs:
                break
        info.update({
            "selected_packs": selected if parsed["confidence"] >= threshold else [],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "source": "light_model",
        })
        if parsed["confidence"] < threshold:
            info["reason"] = parsed["reason"] or "router confidence below threshold"
        return info
    except Exception as e:
        info.update({"source": "error", "error": repr(e)})
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "tool_router_error",
            "task_id": task_id,
            "model": router_model,
            "error": repr(e),
        })
        return info


def setup_dynamic_tools(
    tools_registry: ToolRegistry,
    tool_schemas: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    active_packs: Optional[Iterable[str]] = None,
    drive_logs: Optional[pathlib.Path] = None,
    task_id: str = "",
):
    """Wire pack discovery handlers onto the current mutable schema list."""
    active_tool_names = {
        s.get("function", {}).get("name")
        for s in tool_schemas
        if s.get("function", {}).get("name")
    }
    active_pack_names = {BASE_TOOL_PACK}
    for pack in active_packs or []:
        normalized = normalize_tool_pack(str(pack))
        if normalized:
            active_pack_names.add(normalized)
    enabled_extra: set = set()

    def _append_schema(schema: Dict[str, Any]) -> bool:
        name = schema.get("function", {}).get("name")
        if not name or name in active_tool_names:
            return False
        tool_schemas.append(schema)
        active_tool_names.add(name)
        enabled_extra.add(name)
        return True

    def _handle_list_packs(ctx=None, **kwargs):
        packs = tools_registry.list_tool_packs()
        notice = tools_registry.blocked_tool_packs_notice()
        if not packs:
            return "\n\n".join(p for p in ["No tool packs are available.", notice] if p)
        lines = ["**Available tool packs**:\n"]
        for pack in packs:
            name = pack["name"]
            marker = "active" if name in active_pack_names else "inactive"
            deps = pack.get("dependencies") or []
            dep_text = f", deps={','.join(deps)}" if deps else ""
            lines.append(
                f"- **{name}** ({marker}, {pack['tool_count']} tools{dep_text}): "
                f"{pack['description']}"
            )
        if notice:
            lines.extend(["", notice])
        return "\n".join(lines)

    def _handle_enable_pack(ctx=None, pack: str = "", **kwargs):
        names = [normalize_tool_pack(n) for n in str(pack or "").split(",") if n.strip()]
        if not names:
            return "No tool pack specified."
        known = {p["name"] for p in tools_registry.list_tool_packs()}
        enabled, not_found = [], []
        denied = []
        for name in names:
            if name not in known:
                denial = tools_registry.unavailable_tool_pack_message(name)
                if denial:
                    denied.append(denial)
                else:
                    not_found.append(name)
                continue
            added = []
            for schema in tools_registry.schemas_for_packs([name], include_base=False):
                if _append_schema(schema):
                    added.append(schema["function"]["name"])
            active_pack_names.add(name)
            enabled.append(f"{name} ({len(added)} tools)" if added else f"{name} (already active)")
            if drive_logs is not None:
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "tool_pack_enabled",
                    "task_id": task_id,
                    "pack": name,
                    "added_tools": added,
                    "active_tool_count": len(active_tool_names),
                })
        parts = []
        if enabled:
            parts.append(f"✅ Enabled packs: {', '.join(enabled)}")
        if denied:
            parts.extend(denied)
        if not_found:
            parts.append(f"❌ Packs not found or unavailable: {', '.join(not_found)}")
        return "\n".join(parts) if parts else "No tool pack specified."

    def _handle_list_tools(ctx=None, **kwargs):
        notice = tools_registry.blocked_tool_packs_notice()
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for tool in tools_registry.list_non_core_tools():
            if tool["name"] in active_tool_names:
                continue
            grouped.setdefault(str(tool.get("pack") or "other"), []).append(tool)
        if not grouped:
            return "\n\n".join(p for p in ["All allowed tools are already active.", notice] if p)
        lines = [
            "Additional inactive tools are grouped by pack. Prefer `enable_tool_pack`; "
            "use `enable_tools` only for exact one-off tools.\n"
        ]
        for pack, tools in sorted(grouped.items()):
            sample = ", ".join(t["name"] for t in tools[:5])
            suffix = "" if len(tools) <= 5 else f", +{len(tools) - 5} more"
            lines.append(f"- **{pack}** ({len(tools)}): {sample}{suffix}")
        if notice:
            lines.extend(["", notice])
        return "\n".join(lines)

    def _handle_enable_tools(ctx=None, tools: str = "", **kwargs):
        names = [n.strip() for n in tools.split(",") if n.strip()]
        enabled, not_found, denied = [], [], []
        for name in names:
            schema = tools_registry.get_schema_by_name(name)
            if schema and _append_schema(schema):
                enabled.append(name)
                if drive_logs is not None:
                    append_jsonl(drive_logs / "events.jsonl", {
                        "ts": utc_now_iso(),
                        "type": "tool_enabled",
                        "task_id": task_id,
                        "tool": name,
                        "active_tool_count": len(active_tool_names),
                    })
            elif schema and name in active_tool_names:
                enabled.append(f"{name} (already active)")
            else:
                denial = tools_registry.unavailable_tool_message(name)
                if denial:
                    denied.append(denial)
                else:
                    not_found.append(name)
        parts = []
        if enabled:
            parts.append(f"✅ Enabled: {', '.join(enabled)}")
        if denied:
            parts.extend(denied)
        if not_found:
            parts.append(f"❌ Not found: {', '.join(not_found)}")
        return "\n".join(parts) if parts else "No tools specified."

    tools_registry.override_handler("list_tool_packs", _handle_list_packs)
    tools_registry.override_handler("enable_tool_pack", _handle_enable_pack)
    tools_registry.override_handler("list_available_tools", _handle_list_tools)
    tools_registry.override_handler("enable_tools", _handle_enable_tools)

    packs = tools_registry.list_tool_packs()
    inactive_packs = [p["name"] for p in packs if p["name"] not in active_pack_names]
    if inactive_packs:
        messages.append({
            "role": "system",
            "content": (
                f"Tool packs active: {', '.join(sorted(active_pack_names))}. "
                f"Active tool schemas: {len(tool_schemas)}. "
                f"Additional packs available: {', '.join(inactive_packs)}. "
                f"Use `list_tool_packs` to inspect packs and `enable_tool_pack` "
                f"to activate another pack when the current tools do not fit the task. "
                f"Use `enable_tools` only for exact one-off tools."
            ),
        })

    return tool_schemas, enabled_extra


def setup_initial_tool_schemas(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm: LLMClient,
    drive_logs: pathlib.Path,
    task_id: str,
    task_type: str,
    user_role: str,
    event_queue: Optional[queue.Queue],
    accumulated_usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    user_id: Optional[int],
    estimate_cost_fn: Callable[..., float],
    emit_usage_fn: Callable[..., None],
) -> List[Dict[str, Any]]:
    """Select initial packs, log router output, and wire dynamic pack discovery."""
    router_info = route_tool_packs(
        messages, tools, llm, task_type, user_role, drive_logs, task_id,
        event_queue, accumulated_usage, user_id, estimate_cost_fn, emit_usage_fn
    )
    selected_packs = list(router_info.get("selected_packs") or [])
    tool_schemas = tools.schemas_for_packs(selected_packs)
    router_info["active_tool_count"] = len(tool_schemas)
    router_info["active_packs"] = sorted({BASE_TOOL_PACK, *selected_packs})
    llm_trace["tool_router"] = router_info
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "tool_router_result",
        "task_id": task_id,
        **router_info,
    })
    tool_schemas, _enabled_extra_tools = setup_dynamic_tools(
        tools, tool_schemas, messages,
        active_packs=router_info["active_packs"],
        drive_logs=drive_logs,
        task_id=task_id,
    )
    return tool_schemas


_route_tool_packs = route_tool_packs
