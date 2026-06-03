"""PDF editing tools.

These tools complement analyze_document: documents.py extracts PDF text for
understanding, while this module writes user-approved corrections into a PDF
copy using redaction and overlay operations.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
from typing import Any, Dict, List, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import safe_relpath

PDF_MIME_TYPE = "application/pdf"
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_INSPECT_PAGES = 25
MAX_SEARCH_MATCHES = 120
MIN_EDIT_CONFIDENCE = 0.75


def _require_fitz():
    try:
        import fitz

        return fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required for PDF editing tools. Install dependencies from requirements.txt.") from exc


def _scope(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "user_id": ctx.current_user_id,
        "user_role": ctx.user_role,
        "drive_root": str(ctx.drive_root),
        "shared_drive_root": str(ctx.shared_drive_root or ctx.drive_root),
    }


def _resolve_workspace_file(ctx: ToolContext, path: str, source: str = "drive") -> Tuple[pathlib.Path, pathlib.Path]:
    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty string")

    source = str(source or "drive").strip().lower()
    if source not in {"drive", "repo"}:
        raise ValueError("source must be 'drive' or 'repo'")
    if source == "repo" and str(ctx.user_role or "user").lower() != "admin":
        raise PermissionError("source='repo' is admin-only in multi-user mode")

    root = (ctx.repo_dir if source == "repo" else ctx.drive_root).resolve()
    resolved = (root / safe_relpath(path)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal is not allowed")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return resolved, root


def _resolve_pdf_path(ctx: ToolContext, path: str, source: str = "drive") -> Tuple[pathlib.Path, pathlib.Path]:
    resolved, root = _resolve_workspace_file(ctx, path, source)
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf files are supported by PDF editing tools.")
    size = resolved.stat().st_size
    if size > MAX_PDF_BYTES:
        raise ValueError(f"PDF is too large: {size} bytes")
    return resolved, root


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9А-Яа-я._ -]+", "-", str(value or "document"))
    value = re.sub(r"\s+", "-", value.strip(" .-_"))
    return value[:80] or "document"


def _safe_output_path(root: pathlib.Path, output_path: str, source_name: str, overwrite: bool) -> Tuple[pathlib.Path, str]:
    if output_path:
        rel = safe_relpath(output_path)
        if pathlib.PurePosixPath(rel).suffix.lower() != ".pdf":
            raise ValueError("output_path must end with .pdf")
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        stem = _safe_stem(pathlib.Path(source_name).stem)
        rel = str(pathlib.PurePosixPath("pdf_edits") / f"{stem}-edited-{stamp}.pdf")

    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Path traversal is not allowed")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        stem = path.stem
        suffix = path.suffix
        parent_rel = pathlib.PurePosixPath(rel).parent
        counter = 2
        while path.exists():
            rel = str(parent_rel / f"{stem}-{counter}{suffix}")
            path = (root / rel).resolve()
            counter += 1
    return path, rel


def _parse_page_ranges(page_ranges: str, total_pages: int, max_pages: int = MAX_INSPECT_PAGES) -> List[int]:
    raw = str(page_ranges or "").strip()
    if not raw:
        return list(range(1, min(total_pages, max_pages) + 1))

    normalized = raw.replace("—", "-").replace("–", "-")
    tokens = [token for token in re.split(r"[,;\s]+", normalized) if token]
    pages: List[int] = []
    seen = set()
    for token in tokens:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(f"Invalid page_ranges token: {token!r}. Use format like '1-3,7'.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            raise ValueError(f"Invalid page range: {token!r}")
        for page in range(start, end + 1):
            if page > total_pages:
                continue
            if page not in seen:
                pages.append(page)
                seen.add(page)
            if len(pages) >= max_pages:
                return pages
    if not pages:
        raise ValueError("page_ranges did not select any pages")
    return pages


def _clip(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+\n", "\n", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...(truncated)"


def _rect_line(rect: Any) -> str:
    return f"[{rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f}]"


def _rect_payload(rect: Any) -> List[float]:
    return [round(float(rect.x0), 2), round(float(rect.y0), 2), round(float(rect.x1), 2), round(float(rect.y1), 2)]


def _parse_color(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        raw = value.strip().lower()
        names = {
            "black": (0.0, 0.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "red": (1.0, 0.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "green": (0.0, 0.5, 0.0),
            "yellow": (1.0, 1.0, 0.0),
        }
        if raw in names:
            return names[raw]
        if re.fullmatch(r"#?[0-9a-f]{6}", raw):
            raw = raw.lstrip("#")
            return tuple(int(raw[idx:idx + 2], 16) / 255 for idx in (0, 2, 4))  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        nums = [float(v) for v in value]
        if any(v > 1 for v in nums):
            nums = [max(0.0, min(255.0, v)) / 255 for v in nums]
        return tuple(max(0.0, min(1.0, v)) for v in nums)  # type: ignore[return-value]
    raise ValueError(f"Invalid color: {value!r}")


def _parse_rect(fitz: Any, value: Any) -> Any:
    if isinstance(value, dict):
        coords = [value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1")]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        coords = list(value)
    else:
        raise ValueError("rect must be [x0, y0, x1, y1] or {x0,y0,x1,y1}")
    try:
        x0, y0, x1, y1 = [float(v) for v in coords]
    except Exception as exc:
        raise ValueError("rect coordinates must be numbers") from exc
    if x1 <= x0 or y1 <= y0:
        raise ValueError("rect must have positive width and height")
    return fitz.Rect(x0, y0, x1, y1)


def _coerce_operations(operations: Any) -> List[Dict[str, Any]]:
    if isinstance(operations, str):
        operations = json.loads(operations)
    if not isinstance(operations, list):
        raise ValueError("operations must be a list of objects")
    result = []
    for item in operations:
        if not isinstance(item, dict):
            raise ValueError("each operation must be an object")
        result.append(item)
    return result


def _operation_confirmed(op: Dict[str, Any]) -> Tuple[bool, str]:
    if bool(op.get("confirmed")):
        return True, ""
    confidence = op.get("confidence")
    if confidence is None:
        return False, "missing confirmation; pass confirmed=true after user approval"
    try:
        confidence_value = float(confidence)
    except Exception:
        return False, "invalid confidence; pass confirmed=true after user approval"
    if confidence_value < MIN_EDIT_CONFIDENCE:
        return False, f"low confidence {confidence_value:.2f}; ask user or pass confirmed=true"
    return True, ""


def _selected_rects(page: Any, search_text: str, occurrence: Any = 1, all_matches: bool = False) -> List[Any]:
    if not search_text:
        raise ValueError("search_text is required")
    rects = list(page.search_for(str(search_text)))
    if not rects:
        return []
    if all_matches:
        return rects
    try:
        idx = max(1, int(occurrence or 1)) - 1
    except Exception:
        idx = 0
    if idx >= len(rects):
        return []
    return [rects[idx]]


def _insert_replacement_text(page: Any, rect: Any, text: str, font_size: float, color: Tuple[float, float, float]) -> None:
    # Slightly widen short redaction boxes so replacement text has room without changing the original page layout.
    extra_width = max(18.0, font_size * max(1, len(text)) * 0.45)
    box = rect.__class__(rect.x0 - 0.5, rect.y0 - 0.5, rect.x1 + extra_width, rect.y1 + 2.0)
    remaining = page.insert_textbox(box, text, fontsize=font_size, fontname="helv", color=color)
    if remaining < 0:
        page.insert_text((rect.x0, rect.y0 + font_size), text, fontsize=font_size, fontname="helv", color=color)


def _inspect_pdf_for_edit(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    search_text: str = "",
    page_ranges: str = "",
    max_matches: int = MAX_SEARCH_MATCHES,
) -> str:
    fitz = _require_fitz()
    pdf_path, _root = _resolve_pdf_path(ctx, path, source)
    max_matches = max(1, min(int(max_matches or MAX_SEARCH_MATCHES), MAX_SEARCH_MATCHES))

    ctx.emit_progress_fn(f"Открыл PDF `{pdf_path.name}`. Анализирую страницы, текстовые совпадения и поля формы.")
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass:
            raise ValueError("Encrypted PDFs are not supported by PDF editing tools yet.")
        pages = _parse_page_ranges(page_ranges, len(doc), max_pages=MAX_INSPECT_PAGES)
        lines = [
            "# PDF Edit Inspection",
            f"- path: {path}",
            f"- file: {pdf_path.name}",
            f"- pages: {len(doc)}",
            f"- selected_pages: {','.join(str(p) for p in pages)}",
            f"- encrypted: {'yes' if doc.needs_pass else 'no'}",
            "",
            "## Page Previews",
        ]
        for page_no in pages:
            page = doc[page_no - 1]
            rect = page.rect
            text = _clip(page.get_text("text"), 700)
            lines.append(f"- page {page_no}: size=[{rect.width:.1f} x {rect.height:.1f}] text={text or '[no extractable text]'}")

        lines.extend(["", "## Search Matches"])
        match_count = 0
        if search_text:
            for page_no in pages:
                page = doc[page_no - 1]
                for occurrence, rect in enumerate(page.search_for(str(search_text)), start=1):
                    match_count += 1
                    lines.append(
                        f"- page {page_no}, occurrence {occurrence}: text={search_text!r}, rect={_rect_line(rect)}"
                    )
                    if match_count >= max_matches:
                        break
                if match_count >= max_matches:
                    break
        if not search_text:
            lines.append("- search_text not provided")
        elif match_count == 0:
            lines.append("- none")

        lines.extend(["", "## Form Fields"])
        fields = []
        for page_no in pages:
            page = doc[page_no - 1]
            widgets = list(page.widgets() or [])
            for widget in widgets:
                fields.append(
                    f"- page {page_no}: name={widget.field_name!r}, type={widget.field_type_string}, "
                    f"value={widget.field_value!r}, rect={_rect_line(widget.rect)}"
                )
        lines.extend(fields or ["- none"])
        lines.append("")
        lines.append("Next: show the user an edit plan with page/operation/source/confidence before calling edit_pdf.")
        return "\n".join(lines)
    finally:
        doc.close()


def _apply_replace_text(fitz: Any, page: Any, op: Dict[str, Any]) -> List[str]:
    search_text = str(op.get("search_text") or "")
    replacement = str(op.get("replacement_text") if op.get("replacement_text") is not None else op.get("text") or "")
    rects = _selected_rects(page, search_text, op.get("occurrence", 1), bool(op.get("all_matches")))
    if not rects:
        return [f"replace_text: no match for {search_text!r}"]
    fill = _parse_color(op.get("fill_color"), (1.0, 1.0, 1.0))
    color = _parse_color(op.get("text_color") or op.get("color"), (0.0, 0.0, 0.0))
    font_size = float(op.get("font_size") or 10)
    for rect in rects:
        page.add_redact_annot(rect, fill=fill)
    page.apply_redactions()
    for rect in rects:
        _insert_replacement_text(page, rect, replacement, font_size, color)
    return [f"replace_text: {search_text!r} -> {replacement!r} at {_rect_payload(rect)}" for rect in rects]


def _apply_redact_text(page: Any, op: Dict[str, Any]) -> List[str]:
    search_text = str(op.get("search_text") or "")
    rects = _selected_rects(page, search_text, op.get("occurrence", 1), bool(op.get("all_matches")))
    if not rects:
        return [f"redact_text: no match for {search_text!r}"]
    fill = _parse_color(op.get("fill_color"), (1.0, 1.0, 1.0))
    for rect in rects:
        page.add_redact_annot(rect, fill=fill)
    page.apply_redactions()
    return [f"redact_text: {search_text!r} at {_rect_payload(rect)}" for rect in rects]


def _apply_redact_area(fitz: Any, page: Any, op: Dict[str, Any]) -> List[str]:
    rect = _parse_rect(fitz, op.get("rect"))
    fill = _parse_color(op.get("fill_color"), (1.0, 1.0, 1.0))
    page.add_redact_annot(rect, fill=fill)
    page.apply_redactions()
    return [f"redact_area: {_rect_payload(rect)}"]


def _apply_add_text(page: Any, op: Dict[str, Any]) -> List[str]:
    text = str(op.get("text") or "")
    if not text:
        return ["add_text: missing text"]
    x = float(op.get("x"))
    y = float(op.get("y"))
    font_size = float(op.get("font_size") or 10)
    color = _parse_color(op.get("text_color") or op.get("color"), (0.0, 0.0, 0.0))
    page.insert_text((x, y), text, fontsize=font_size, fontname="helv", color=color)
    return [f"add_text: {text!r} at [{x:.1f}, {y:.1f}]"]


def _apply_add_comment(fitz: Any, page: Any, op: Dict[str, Any]) -> List[str]:
    text = str(op.get("text") or "")
    if not text:
        return ["add_comment: missing text"]
    x = float(op.get("x"))
    y = float(op.get("y"))
    annot = page.add_text_annot(fitz.Point(x, y), text)
    annot.update()
    return [f"add_comment: {text!r} at [{x:.1f}, {y:.1f}]"]


def _apply_set_form_field(doc: Any, op: Dict[str, Any]) -> List[str]:
    field_name = str(op.get("field_name") or "")
    value = str(op.get("value") or "")
    if not field_name:
        return ["set_form_field: missing field_name"]
    applied = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for widget in list(page.widgets() or []):
            if widget.field_name == field_name:
                widget.field_value = value
                widget.update()
                applied.append(f"set_form_field: {field_name!r}={value!r} on page {page_idx + 1}")
    return applied or [f"set_form_field: field not found {field_name!r}"]


def _edit_pdf(
    ctx: ToolContext,
    path: str,
    operations: Any,
    output_path: str = "",
    overwrite: bool = False,
    send_to_chat: bool = True,
) -> str:
    fitz = _require_fitz()
    pdf_path, root = _resolve_pdf_path(ctx, path, "drive")
    output, rel_output = _safe_output_path(root, output_path, pdf_path.name, bool(overwrite))
    operation_items = _coerce_operations(operations)

    ctx.emit_progress_fn(f"Редактирую копию PDF `{pdf_path.name}`: {len(operation_items)} операций.")
    doc = fitz.open(str(pdf_path))
    applied: List[str] = []
    rejected: List[str] = []
    try:
        if doc.needs_pass:
            raise ValueError("Encrypted PDFs are not supported by PDF editing tools yet.")

        for idx, op in enumerate(operation_items, start=1):
            op_type = str(op.get("type") or op.get("operation") or "").strip().lower()
            label = str(op.get("field") or op.get("label") or op_type or f"operation_{idx}")
            ok, reason = _operation_confirmed(op)
            if not ok:
                rejected.append(f"{label}: {reason}")
                continue

            try:
                if op_type == "set_form_field":
                    results = _apply_set_form_field(doc, op)
                else:
                    page_no = int(op.get("page") or 0)
                    if page_no < 1 or page_no > len(doc):
                        rejected.append(f"{label}: invalid page {page_no}")
                        continue
                    page = doc[page_no - 1]
                    if op_type == "replace_text":
                        results = _apply_replace_text(fitz, page, op)
                    elif op_type == "redact_text":
                        results = _apply_redact_text(page, op)
                    elif op_type == "redact_area":
                        results = _apply_redact_area(fitz, page, op)
                    elif op_type == "add_text":
                        results = _apply_add_text(page, op)
                    elif op_type == "add_comment":
                        results = _apply_add_comment(fitz, page, op)
                    else:
                        rejected.append(f"{label}: unsupported operation type {op_type!r}")
                        continue
            except Exception as exc:
                rejected.append(f"{label}: {type(exc).__name__}: {exc}")
                continue

            real_results = [result for result in results if "no match" not in result and "not found" not in result and "missing " not in result]
            if real_results:
                applied.extend(f"{label}: {result}" for result in real_results)
            else:
                rejected.extend(f"{label}: {result}" for result in results)

        if not applied:
            lines = [
                "⚠️ No PDF edits were applied",
                f"- source_path: {path}",
                f"- rejected: {len(rejected)}",
                "- output_path: not_created",
                "- telegram_delivery: skipped",
            ]
            if rejected:
                lines.append("## Rejected")
                lines.extend(f"- {item}" for item in rejected[:80])
            return "\n".join(lines)

        doc.save(str(output), garbage=4, deflate=True)
    finally:
        doc.close()

    queued = False
    if send_to_chat and ctx.current_chat_id:
        ctx.pending_events.append(
            {
                "type": "send_document",
                "chat_id": ctx.current_chat_id,
                "path": rel_output,
                "caption": "Отредактированный PDF",
                "filename": output.name,
                "mime_type": PDF_MIME_TYPE,
                **_scope(ctx),
            }
        )
        queued = True

    lines = [
        "OK: PDF edited",
        f"- source_path: {path}",
        f"- output_path: {rel_output}",
        f"- applied: {len(applied)}",
        f"- rejected: {len(rejected)}",
        f"- telegram_delivery: {'queued' if queued else 'skipped'}",
    ]
    if applied:
        lines.append("## Applied")
        lines.extend(f"- {item}" for item in applied[:80])
    if rejected:
        lines.append("## Rejected")
        lines.extend(f"- {item}" for item in rejected[:80])
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "inspect_pdf_for_edit",
            {
                "name": "inspect_pdf_for_edit",
                "description": (
                    "Inspect a PDF before editing it. Use this when the user wants corrections in a PDF, "
                    "not merely a summary. It reports page previews, search matches with coordinates, and "
                    "fillable form fields. After inspection, show the user an edit plan before calling edit_pdf."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the PDF in Drive or repo."},
                        "source": {
                            "type": "string",
                            "enum": ["drive", "repo"],
                            "default": "drive",
                            "description": "Where to inspect the PDF. repo is admin-only.",
                        },
                        "search_text": {
                            "type": "string",
                            "description": "Optional exact text to locate before planning a redaction/replacement.",
                        },
                        "page_ranges": {
                            "type": "string",
                            "description": "Optional 1-based page ranges to inspect, e.g. '1-3,7'.",
                        },
                        "max_matches": {
                            "type": "integer",
                            "default": MAX_SEARCH_MATCHES,
                            "description": "Maximum search matches to report.",
                        },
                    },
                    "required": ["path"],
                },
            },
            _inspect_pdf_for_edit,
            timeout_sec=60,
        ),
        ToolEntry(
            "edit_pdf",
            {
                "name": "edit_pdf",
                "description": (
                    "Apply confirmed edits to a copy of a PDF and optionally send it to Telegram. "
                    "Supports replace_text (redaction + overlay), redact_text, redact_area, add_text, "
                    "add_comment, and set_form_field. Use only after a user-approved edit plan. Original "
                    "PDF files are never modified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Drive path to the source PDF."},
                        "operations": {
                            "type": "array",
                            "description": (
                                "Confirmed edit operations. Each item needs type and confirmed=true or "
                                "confidence>=0.75. Page numbers are 1-based. Rects use PDF points."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "replace_text",
                                            "redact_text",
                                            "redact_area",
                                            "add_text",
                                            "add_comment",
                                            "set_form_field",
                                        ],
                                    },
                                    "page": {"type": "integer"},
                                    "search_text": {"type": "string"},
                                    "replacement_text": {"type": "string"},
                                    "text": {"type": "string"},
                                    "rect": {},
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "font_size": {"type": "number"},
                                    "text_color": {},
                                    "fill_color": {},
                                    "occurrence": {"type": "integer"},
                                    "all_matches": {"type": "boolean"},
                                    "field_name": {"type": "string"},
                                    "value": {},
                                    "confidence": {"type": "number"},
                                    "confirmed": {"type": "boolean"},
                                },
                            },
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output .pdf path in Drive. Defaults to pdf_edits/<name>-edited-<timestamp>.pdf.",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                            "description": "Overwrite output_path if it exists; otherwise create a deduplicated filename.",
                        },
                        "send_to_chat": {
                            "type": "boolean",
                            "default": True,
                            "description": "Queue the edited PDF for Telegram delivery when there is an active chat.",
                        },
                    },
                    "required": ["path", "operations"],
                },
            },
            _edit_pdf,
            timeout_sec=60,
        ),
    ]
