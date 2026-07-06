"""PowerPoint/PPTX editing tools.

These tools complement analyze_document and create_presentation: they inspect
and edit a copy of an existing PPTX package while preserving unmodified slides,
images, charts, tables, masters, layouts, and theme files.
"""

from __future__ import annotations

import copy
import datetime
import json
import pathlib
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Tuple

from ouroboros.tools.presentations import PPTX_MIME_TYPE
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import safe_relpath

MAX_PPTX_BYTES = 75 * 1024 * 1024
MAX_SEARCH_MATCHES = 160
MAX_PREVIEW_SLIDES = 80
MAX_TARGET_LINES = 160
MIN_EDIT_CONFIDENCE = 0.75

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SLIDE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
NOTES_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)


def _tag(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


A_P = _tag("a", "p")
A_PPR = _tag("a", "pPr")
A_R = _tag("a", "r")
A_RPR = _tag("a", "rPr")
A_T = _tag("a", "t")
A_END_PARA_RPR = _tag("a", "endParaRPr")
A_TBL = _tag("a", "tbl")
A_TR = _tag("a", "tr")
A_TC = _tag("a", "tc")
A_TXBODY = _tag("a", "txBody")
A_LATIN = _tag("a", "latin")
A_SOLID_FILL = _tag("a", "solidFill")
A_SRGB_CLR = _tag("a", "srgbClr")
P_SP = _tag("p", "sp")
P_PIC = _tag("p", "pic")
P_CXNSP = _tag("p", "cxnSp")
P_GRPSP = _tag("p", "grpSp")
P_GRAPHIC_FRAME = _tag("p", "graphicFrame")
P_CN_VPR = _tag("p", "cNvPr")
P_TXBODY = _tag("p", "txBody")
C_CHART = _tag("c", "chart")
RELATIONSHIP = f"{{{PKG_REL_NS}}}Relationship"
SLIDE_RE = re.compile(r"ppt/slides/slide(\d+)\.xml")


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


def _resolve_pptx_path(ctx: ToolContext, path: str, source: str = "drive") -> Tuple[pathlib.Path, pathlib.Path]:
    resolved, root = _resolve_workspace_file(ctx, path, source)
    if resolved.suffix.lower() != ".pptx":
        raise ValueError("Only .pptx files are supported by PowerPoint editing tools.")
    size = resolved.stat().st_size
    if size > MAX_PPTX_BYTES:
        raise ValueError(f"PPTX is too large: {size} bytes")
    return resolved, root


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9А-Яа-я._ -]+", "-", str(value or "presentation"))
    value = re.sub(r"\s+", "-", value.strip(" .-_"))
    return value[:80] or "presentation"


def _safe_output_path(root: pathlib.Path, output_path: str, source_name: str, overwrite: bool) -> Tuple[pathlib.Path, str]:
    if output_path:
        rel = safe_relpath(output_path)
        if pathlib.PurePosixPath(rel).suffix.lower() != ".pptx":
            raise ValueError("output_path must end with .pptx")
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        stem = _safe_stem(pathlib.Path(source_name).stem)
        rel = str(pathlib.PurePosixPath("presentation_edits") / f"{stem}-edited-{stamp}.pptx")

    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Path traversal is not allowed")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        stem, suffix = path.stem, path.suffix
        parent_rel = pathlib.PurePosixPath(rel).parent
        counter = 2
        while path.exists():
            rel = str(parent_rel / f"{stem}-{counter}{suffix}")
            path = (root / rel).resolve()
            counter += 1
    return path, rel


def _slide_number(name: str) -> int:
    match = SLIDE_RE.fullmatch(name)
    return int(match.group(1)) if match else 0


def _slide_parts(names: Iterable[str]) -> List[str]:
    return sorted((name for name in names if SLIDE_RE.fullmatch(name)), key=_slide_number)


def _rels_part_for(part: str) -> str:
    parent = posixpath.dirname(part)
    return posixpath.join(parent, "_rels", posixpath.basename(part) + ".rels")


def _resolve_rel_target(source_part: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def _relationships(zf: zipfile.ZipFile, names: set[str], source_part: str, rel_type: str = "") -> Dict[str, str]:
    rels_part = _rels_part_for(source_part)
    if rels_part not in names:
        return {}
    root = ET.fromstring(zf.read(rels_part))
    result: Dict[str, str] = {}
    for rel in root.findall(RELATIONSHIP):
        if rel_type and rel.get("Type") != rel_type:
            continue
        rel_id = str(rel.get("Id") or "")
        target = str(rel.get("Target") or "")
        if rel_id and target:
            result[rel_id] = _resolve_rel_target(source_part, target)
    return result


def _ordered_slide_parts(zf: zipfile.ZipFile, names: set[str]) -> List[str]:
    fallback = _slide_parts(names)
    if "ppt/presentation.xml" not in names:
        return fallback
    slide_rels = _relationships(zf, names, "ppt/presentation.xml", SLIDE_REL_TYPE)
    if not slide_rels:
        return fallback
    root = _read_xml_part(zf, "ppt/presentation.xml")
    ordered: List[str] = []
    for slide_id in root.findall(f".//{_tag('p', 'sldId')}"):
        rel_id = slide_id.get(f"{{{NS['r']}}}id")
        target = slide_rels.get(str(rel_id or ""))
        if target in names and target not in ordered:
            ordered.append(target)
    if not ordered:
        return fallback
    for part in fallback:
        if part not in ordered:
            ordered.append(part)
    return ordered


def _notes_part(zf: zipfile.ZipFile, names: set[str], slide_name: str) -> str:
    notes = _relationships(zf, names, slide_name, NOTES_REL_TYPE)
    for target in notes.values():
        if target in names:
            return target
    name = f"ppt/notesSlides/notesSlide{_slide_number(slide_name)}.xml"
    return name if name in names else ""


def _notes_map(zf: zipfile.ZipFile, names: set[str], slides: List[str]) -> Dict[str, str]:
    return {slide: note for slide in slides if (note := _notes_part(zf, names, slide))}


def _read_xml_part(zf: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(zf.read(name))


def _xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _node_text(node: ET.Element) -> str:
    return "".join(child.text or "" for child in node.iter(A_T)).strip()


def _paragraphs(root: ET.Element) -> List[str]:
    return [_node_text(para) for para in root.iter(A_P) if _node_text(para)]


def _clip(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...(truncated)"


def _shape_identity(shape: ET.Element) -> Tuple[str, str]:
    cnv = shape.find(f".//{P_CN_VPR}")
    if cnv is None:
        return "", ""
    return str(cnv.get("id") or ""), str(cnv.get("name") or "")


def _shape_lines(root: ET.Element, slide_no: int) -> List[str]:
    lines: List[str] = []
    elements = [node for node in root.iter() if node.tag in {P_SP, P_GRAPHIC_FRAME}]
    table_index = 0
    for node in elements:
        shape_id, name = _shape_identity(node)
        text = _node_text(node)
        if node.find(f".//{A_TBL}") is not None:
            table_index += 1
            kind = f"table={table_index}"
        else:
            kind = "shape"
        if text or kind.startswith("table"):
            lines.append(
                f"- slide {slide_no}: {kind} id={shape_id or '?'} name={name or '?'} text={_clip(text, 140)!r}"
            )
        if len(lines) >= MAX_TARGET_LINES:
            break
    return lines


def _find_matches(part_label: str, paragraphs: List[str], search_text: str, max_matches: int) -> List[str]:
    if not search_text:
        return []
    matches: List[str] = []
    for idx, text in enumerate(paragraphs, start=1):
        start = 0
        while True:
            pos = text.find(search_text, start)
            if pos < 0:
                break
            snippet = _clip(text[max(0, pos - 80): pos + len(search_text) + 80], 220)
            matches.append(f"- {part_label} paragraph {idx}: snippet={snippet!r}")
            if len(matches) >= max_matches:
                return matches
            start = pos + max(1, len(search_text))
    return matches


def _inspect_presentation_for_edit(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    search_text: str = "",
    max_matches: int = MAX_SEARCH_MATCHES,
) -> str:
    pptx_path, _root = _resolve_pptx_path(ctx, path, source)
    max_matches = max(1, min(int(max_matches or MAX_SEARCH_MATCHES), MAX_SEARCH_MATCHES))

    ctx.emit_progress_fn(f"Открыл презентацию `{pptx_path.name}` для правки. Анализирую слайды и объекты.")
    with zipfile.ZipFile(pptx_path) as zf:
        names = set(zf.namelist())
        slides = _ordered_slide_parts(zf, names)
        preview_lines: List[str] = []
        match_lines: List[str] = []
        target_lines: List[str] = []
        totals = {"pictures": 0, "tables": 0, "charts": 0, "connectors": 0, "groups": 0}

        for idx, slide_name in enumerate(slides[:MAX_PREVIEW_SLIDES], start=1):
            root = _read_xml_part(zf, slide_name)
            texts = _paragraphs(root)
            title = texts[0] if texts else "[no extractable text]"
            tables = len(list(root.iter(A_TBL)))
            counts = {
                "pictures": len(list(root.iter(P_PIC))),
                "tables": tables,
                "charts": len(list(root.iter(C_CHART))),
                "connectors": len(list(root.iter(P_CXNSP))),
                "groups": len(list(root.iter(P_GRPSP))),
            }
            for key, value in counts.items():
                totals[key] += value
            preview_lines.append(
                f"- slide {idx}: title={_clip(title)!r}; text_blocks={len(texts)}; "
                f"pictures={counts['pictures']}; tables={tables}; charts={counts['charts']}"
            )
            match_lines.extend(_find_matches(f"slide {idx}", texts, search_text, max_matches - len(match_lines)))
            if len(target_lines) < MAX_TARGET_LINES:
                target_lines.extend(_shape_lines(root, idx))

            note_name = _notes_part(zf, names, slide_name)
            if note_name:
                note_texts = _paragraphs(_read_xml_part(zf, note_name))
                match_lines.extend(_find_matches(f"notes {idx}", note_texts, search_text, max_matches - len(match_lines)))
            if len(match_lines) >= max_matches:
                match_lines = match_lines[:max_matches]

    if len(slides) > MAX_PREVIEW_SLIDES:
        preview_lines.append(f"- preview limited to first {MAX_PREVIEW_SLIDES} of {len(slides)} slides")

    return "\n".join([
        "# Presentation Edit Inspection",
        f"- path: {path}",
        f"- file: {pptx_path.name}",
        f"- slides: {len(slides)}",
        f"- pictures: {totals['pictures']}",
        f"- tables: {totals['tables']}",
        f"- charts: {totals['charts']}",
        f"- connectors: {totals['connectors']}",
        f"- groups: {totals['groups']}",
        "",
        "## Slide Preview",
        *(preview_lines or ["- none"]),
        "",
        "## Search Matches",
        *(match_lines or ["- search_text not provided" if not search_text else "- none"]),
        "",
        "## Editable Targets",
        *(target_lines[:MAX_TARGET_LINES] or ["- no text-bearing shapes or tables found"]),
        "",
        "Next: show the user an edit plan with operation/slide/target/confidence before calling edit_presentation.",
    ])


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


def _count_matches(text: str, search_text: str) -> int:
    if not search_text:
        return 0
    count = start = 0
    while True:
        idx = text.find(search_text, start)
        if idx < 0:
            return count
        count += 1
        start = idx + max(1, len(search_text))


def _replace_nth(text: str, search_text: str, replacement: str, occurrence: int) -> str:
    count = start = 0
    while True:
        idx = text.find(search_text, start)
        if idx < 0:
            return text
        count += 1
        if count == occurrence:
            return text[:idx] + replacement + text[idx + len(search_text):]
        start = idx + max(1, len(search_text))


def _set_paragraph_text(para: ET.Element, text: str) -> None:
    ppr = para.find(A_PPR)
    run = para.find(A_R)
    rpr = run.find(A_RPR) if run is not None else None
    end_rpr = para.find(A_END_PARA_RPR)
    for child in list(para):
        para.remove(child)
    if ppr is not None:
        para.append(copy.deepcopy(ppr))
    new_run = ET.SubElement(para, A_R)
    if rpr is not None:
        new_run.append(copy.deepcopy(rpr))
    text_node = ET.SubElement(new_run, A_T)
    text_node.text = str(text or "")
    if end_rpr is not None:
        para.append(copy.deepcopy(end_rpr))


def _replace_all_in_para(para: ET.Element, search_text: str, replacement: str, allow_reflow: bool) -> Tuple[int, str]:
    para_text = _node_text(para)
    para_count = _count_matches(para_text, search_text)
    if not para_count:
        return 0, "not_found"
    text_nodes = list(para.iter(A_T))
    run_count = sum(_count_matches(node.text or "", search_text) for node in text_nodes)
    if run_count == para_count:
        for node in text_nodes:
            if node.text:
                node.text = node.text.replace(search_text, replacement)
        return para_count, "run"
    if allow_reflow:
        _set_paragraph_text(para, para_text.replace(search_text, replacement))
        return para_count, "paragraph_reflow"
    return 0, "requires_allow_reflow"


def _replace_one_in_para(
    para: ET.Element,
    search_text: str,
    replacement: str,
    occurrence_in_para: int,
    allow_reflow: bool,
) -> Tuple[int, str]:
    seen = 0
    for node in para.iter(A_T):
        node_count = _count_matches(node.text or "", search_text)
        if seen + node_count >= occurrence_in_para:
            node_occurrence = occurrence_in_para - seen
            node.text = _replace_nth(node.text or "", search_text, replacement, node_occurrence)
            return 1, "run"
        seen += node_count
    if allow_reflow:
        para_text = _node_text(para)
        _set_paragraph_text(para, _replace_nth(para_text, search_text, replacement, occurrence_in_para))
        return 1, "paragraph_reflow"
    return 0, "requires_allow_reflow"


def _parse_slide_selection(value: Any, total: int) -> List[int]:
    if value in (None, "", []):
        return list(range(1, total + 1))
    if isinstance(value, int):
        raw_items = [str(value)]
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [item for item in re.split(r"[,;\s]+", str(value).replace("—", "-").replace("–", "-")) if item]

    slides: List[int] = []
    seen = set()
    for item in raw_items:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item.strip())
        if not match:
            raise ValueError(f"Invalid slide selector: {item!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > total:
            raise ValueError(f"Slide selector out of range: {item!r}")
        for slide in range(start, end + 1):
            if slide not in seen:
                slides.append(slide)
                seen.add(slide)
    return slides


def _operation_slides(op: Dict[str, Any], total: int, required: bool = False) -> List[int]:
    selector = op.get("slides", op.get("slide", None))
    if required and selector in (None, "", []):
        raise ValueError("slide or slides is required for this operation")
    return _parse_slide_selection(selector, total)


def _replace_in_part(
    root: ET.Element,
    search_text: str,
    replacement: str,
    all_matches: bool,
    state: Dict[str, int],
    allow_reflow: bool,
    label: str,
) -> Tuple[List[str], List[str], bool]:
    applied: List[str] = []
    rejected: List[str] = []
    changed = False
    for idx, para in enumerate(root.iter(A_P), start=1):
        para_count = _count_matches(_node_text(para), search_text)
        if not para_count:
            continue
        if all_matches:
            count, mode = _replace_all_in_para(para, search_text, replacement, allow_reflow)
            if count:
                applied.append(f"replace_text: {count} match(es) in {label} paragraph {idx} via {mode}")
                changed = True
            else:
                rejected.append(f"replace_text: match spans multiple runs in {label} paragraph {idx}; pass allow_reflow=true")
            continue
        if state["remaining"] > para_count:
            state["remaining"] -= para_count
            continue
        count, mode = _replace_one_in_para(para, search_text, replacement, state["remaining"], allow_reflow)
        if count:
            applied.append(f"replace_text: {search_text!r} -> {replacement!r} in {label} paragraph {idx} via {mode}")
            changed = True
            state["done"] = 1
        else:
            rejected.append(f"replace_text: match spans multiple runs in {label} paragraph {idx}; pass allow_reflow=true")
        break
    return applied, rejected, changed


def _make_paragraph(text: str, template: ET.Element | None) -> ET.Element:
    para = ET.Element(A_P)
    if template is not None and template.find(A_PPR) is not None:
        para.append(copy.deepcopy(template.find(A_PPR)))
    run = ET.SubElement(para, A_R)
    old_run = template.find(A_R) if template is not None else None
    old_rpr = old_run.find(A_RPR) if old_run is not None else None
    if old_rpr is not None:
        run.append(copy.deepcopy(old_rpr))
    text_node = ET.SubElement(run, A_T)
    text_node.text = str(text or "")
    return para


def _set_text_body(body: ET.Element, text_or_paragraphs: Any) -> int:
    if isinstance(text_or_paragraphs, list):
        texts = [str(item) for item in text_or_paragraphs if str(item) != ""]
    else:
        texts = str(text_or_paragraphs if text_or_paragraphs is not None else "").splitlines() or [""]
    existing = [child for child in list(body) if child.tag == A_P]
    template = existing[0] if existing else None
    for para in existing:
        body.remove(para)
    for text in texts:
        body.append(_make_paragraph(text, template))
    return len(texts)


def _shape_matches(shape: ET.Element, shape_id: Any = None, shape_name: Any = None) -> bool:
    sid, name = _shape_identity(shape)
    if shape_id not in (None, "") and sid == str(shape_id):
        return True
    if shape_name not in (None, "") and name == str(shape_name):
        return True
    return shape_id in (None, "") and shape_name in (None, "")


def _apply_set_shape_text(root: ET.Element, op: Dict[str, Any], slide_no: int) -> Tuple[List[str], List[str], bool]:
    shape_id = op.get("shape_id")
    shape_name = op.get("shape_name") or op.get("name")
    if shape_id in (None, "") and shape_name in (None, ""):
        return [], ["set_shape_text: shape_id or shape_name is required"], False
    for shape in root.iter(P_SP):
        if not _shape_matches(shape, shape_id, shape_name):
            continue
        body = shape.find(P_TXBODY)
        if body is None:
            return [], [f"set_shape_text: shape {shape_id or shape_name!r} has no text body"], False
        count = _set_text_body(body, op.get("paragraphs", op.get("text", "")))
        return [f"set_shape_text: slide {slide_no} shape {shape_id or shape_name!r} paragraphs={count}"], [], True
    return [], [f"set_shape_text: shape {shape_id or shape_name!r} not found on slide {slide_no}"], False


def _apply_set_table_cell(root: ET.Element, op: Dict[str, Any], slide_no: int) -> Tuple[List[str], List[str], bool]:
    try:
        table_index = max(1, int(op.get("table_index") or 1))
        row_index = max(1, int(op.get("row") or op.get("row_index") or 1))
        col_index = max(1, int(op.get("col") or op.get("col_index") or 1))
    except Exception:
        return [], ["set_table_cell: table_index, row, and col must be integers"], False
    tables = list(root.iter(A_TBL))
    if table_index > len(tables):
        return [], [f"set_table_cell: table {table_index} not found on slide {slide_no}"], False
    rows = tables[table_index - 1].findall(A_TR)
    if row_index > len(rows):
        return [], [f"set_table_cell: row {row_index} not found in table {table_index} on slide {slide_no}"], False
    cells = rows[row_index - 1].findall(A_TC)
    if col_index > len(cells):
        return [], [f"set_table_cell: col {col_index} not found in table {table_index} row {row_index}"], False
    body = cells[col_index - 1].find(A_TXBODY)
    if body is None:
        return [], [f"set_table_cell: cell {row_index},{col_index} has no text body"], False
    _set_text_body(body, op.get("text", op.get("value", "")))
    return [f"set_table_cell: slide {slide_no} table {table_index} row {row_index} col {col_index}"], [], True


def _normalize_color(value: Any) -> str:
    raw = str(value or "").strip().lower()
    names = {"black": "000000", "white": "ffffff", "red": "ff0000", "blue": "0000ff", "green": "008000"}
    raw = names.get(raw, raw).lstrip("#")
    if not re.fullmatch(r"[0-9a-f]{6}", raw):
        raise ValueError(f"Invalid color: {value!r}")
    return raw.upper()


def _ensure_rpr(run: ET.Element) -> ET.Element:
    rpr = run.find(A_RPR)
    if rpr is None:
        rpr = ET.Element(A_RPR)
        run.insert(0, rpr)
    return rpr


def _style_rpr(rpr: ET.Element, op: Dict[str, Any]) -> None:
    if "bold" in op:
        rpr.set("b", "1" if bool(op.get("bold")) else "0")
    if op.get("font_size_pt") not in (None, ""):
        size = max(1, min(400, float(op.get("font_size_pt"))))
        rpr.set("sz", str(int(round(size * 100))))
    if op.get("font_face"):
        latin = rpr.find(A_LATIN)
        if latin is None:
            latin = ET.SubElement(rpr, A_LATIN)
        latin.set("typeface", str(op.get("font_face")))
    if op.get("color"):
        for child in list(rpr):
            if child.tag == A_SOLID_FILL:
                rpr.remove(child)
        fill = ET.Element(A_SOLID_FILL)
        ET.SubElement(fill, A_SRGB_CLR, {"val": _normalize_color(op.get("color"))})
        rpr.insert(0, fill)


def _apply_text_style(root: ET.Element, op: Dict[str, Any], slide_no: int) -> Tuple[List[str], List[str], bool]:
    shape_id = op.get("shape_id")
    shape_name = op.get("shape_name") or op.get("name")
    containers: List[ET.Element] = []
    if shape_id in (None, "") and shape_name in (None, ""):
        containers = [root]
    else:
        for shape in root.iter():
            if shape.tag in {P_SP, P_GRAPHIC_FRAME} and _shape_matches(shape, shape_id, shape_name):
                containers.append(shape)
    if not containers:
        return [], [f"apply_text_style: target {shape_id or shape_name!r} not found on slide {slide_no}"], False
    run_count = 0
    for container in containers:
        for run in container.iter(A_R):
            _style_rpr(_ensure_rpr(run), op)
            run_count += 1
    if not run_count:
        return [], [f"apply_text_style: no text runs found on slide {slide_no}"], False
    return [f"apply_text_style: slide {slide_no} text_runs={run_count}"], [], True


def _copy_pptx_with_updates(source: pathlib.Path, output: pathlib.Path, updates: Dict[str, bytes]) -> None:
    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(output, "w") as dst:
        for info in src.infolist():
            data = updates.get(info.filename, src.read(info.filename))
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.comment = info.comment
            dst.writestr(new_info, data)


def _edit_presentation(
    ctx: ToolContext,
    path: str,
    operations: Any,
    output_path: str = "",
    overwrite: bool = False,
    send_to_chat: bool = True,
) -> str:
    pptx_path, root_dir = _resolve_pptx_path(ctx, path, "drive")
    output, rel_output = _safe_output_path(root_dir, output_path, pptx_path.name, bool(overwrite))
    if output == pptx_path:
        raise ValueError("output_path must not overwrite the source PPTX")
    operation_items = _coerce_operations(operations)

    ctx.emit_progress_fn(f"Редактирую копию презентации `{pptx_path.name}`: {len(operation_items)} операций.")
    applied: List[str] = []
    rejected: List[str] = []
    changed_parts: set[str] = set()
    loaded_roots: Dict[str, ET.Element] = {}

    with zipfile.ZipFile(pptx_path, "r") as zf:
        names = set(zf.namelist())
        slides = _ordered_slide_parts(zf, names)
        if not slides:
            raise ValueError("PPTX contains no slide XML parts")
        notes_by_slide = _notes_map(zf, names, slides)

        def root_for(part: str) -> ET.Element:
            if part not in loaded_roots:
                loaded_roots[part] = _read_xml_part(zf, part)
            return loaded_roots[part]

        for idx, op in enumerate(operation_items, start=1):
            op_type = str(op.get("type") or op.get("operation") or "").strip().lower()
            label = str(op.get("field") or op.get("label") or op_type or f"operation_{idx}")
            ok, reason = _operation_confirmed(op)
            if not ok:
                rejected.append(f"{label}: {reason}")
                continue
            try:
                op_applied, op_rejected, op_changed = _run_operation(op_type, op, slides, notes_by_slide, root_for)
            except Exception as exc:
                rejected.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            applied.extend(f"{label}: {item}" for item in op_applied)
            rejected.extend(f"{label}: {item}" for item in op_rejected)
            changed_parts.update(op_changed)

    if not applied or not changed_parts:
        return "\n".join([
            "⚠️ No presentation edits were applied",
            f"- source_path: {path}",
            f"- rejected: {len(rejected)}",
            "- output_path: not_created",
            "- telegram_delivery: skipped",
            *(["## Rejected", *[f"- {item}" for item in rejected[:80]]] if rejected else []),
        ])

    updates = {part: _xml_bytes(loaded_roots[part]) for part in changed_parts}
    _copy_pptx_with_updates(pptx_path, output, updates)

    queued = False
    if send_to_chat and ctx.current_chat_id:
        ctx.pending_events.append({
            "type": "send_document",
            "chat_id": ctx.current_chat_id,
            "path": rel_output,
            "caption": "Отредактированная презентация",
            "filename": output.name,
            "mime_type": PPTX_MIME_TYPE,
            **_scope(ctx),
        })
        queued = True

    lines = [
        "OK: PowerPoint presentation edited",
        f"- source_path: {path}",
        f"- output_path: {rel_output}",
        f"- applied: {len(applied)}",
        f"- rejected: {len(rejected)}",
        f"- changed_parts: {len(changed_parts)}",
        f"- telegram_delivery: {'queued' if queued else 'skipped'}",
    ]
    if applied:
        lines.append("## Applied")
        lines.extend(f"- {item}" for item in applied[:80])
    if rejected:
        lines.append("## Rejected")
        lines.extend(f"- {item}" for item in rejected[:80])
    return "\n".join(lines)


def _run_operation(
    op_type: str,
    op: Dict[str, Any],
    slides: List[str],
    notes_by_slide: Dict[str, str],
    root_for,
) -> Tuple[List[str], List[str], set[str]]:
    changed: set[str] = set()
    applied: List[str] = []
    rejected: List[str] = []
    if op_type == "replace_text":
        return _run_replace_text(op, slides, notes_by_slide, root_for)
    if op_type not in {"set_shape_text", "set_table_cell", "apply_text_style"}:
        return [], [f"unsupported operation type {op_type!r}"], changed
    for slide_no in _operation_slides(op, len(slides), required=op_type != "apply_text_style"):
        part = slides[slide_no - 1]
        root = root_for(part)
        if op_type == "set_shape_text":
            a, r, did_change = _apply_set_shape_text(root, op, slide_no)
        elif op_type == "set_table_cell":
            a, r, did_change = _apply_set_table_cell(root, op, slide_no)
        else:
            a, r, did_change = _apply_text_style(root, op, slide_no)
        applied.extend(a)
        rejected.extend(r)
        if did_change:
            changed.add(part)
    return applied, rejected, changed


def _run_replace_text(
    op: Dict[str, Any],
    slides: List[str],
    notes_by_slide: Dict[str, str],
    root_for,
) -> Tuple[List[str], List[str], set[str]]:
    search_text = str(op.get("search_text") or "")
    replacement = str(op.get("replacement_text") if op.get("replacement_text") is not None else op.get("text") or "")
    if not search_text:
        return [], ["replace_text: missing search_text"], set()
    all_matches = bool(op.get("all_matches"))
    allow_reflow = bool(op.get("allow_reflow"))
    state = {"remaining": max(1, int(op.get("occurrence") or 1)), "done": 0}
    changed: set[str] = set()
    applied: List[str] = []
    rejected: List[str] = []
    for slide_no in _operation_slides(op, len(slides)):
        parts = [(slides[slide_no - 1], f"slide {slide_no}")]
        note = notes_by_slide.get(slides[slide_no - 1], "")
        if bool(op.get("include_notes")) and note:
            parts.append((note, f"notes {slide_no}"))
        for part, label in parts:
            if state["done"] and not all_matches:
                break
            a, r, did_change = _replace_in_part(
                root_for(part),
                search_text,
                replacement,
                all_matches,
                state,
                allow_reflow,
                label,
            )
            applied.extend(a)
            rejected.extend(r)
            if did_change:
                changed.add(part)
        if state["done"] and not all_matches:
            break
    if not applied:
        rejected.append(f"replace_text: no match for {search_text!r}")
    return applied, rejected, changed


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "inspect_presentation_for_edit",
            {
                "name": "inspect_presentation_for_edit",
                "description": (
                    "Inspect an existing .pptx before editing it. Use this when the user wants the original "
                    "presentation corrected or restyled, not reconstructed. It reports slide text, exact matches, "
                    "editable shape IDs/names, tables, charts, pictures, and other preserved object counts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the PPTX in Drive or repo."},
                        "source": {
                            "type": "string",
                            "enum": ["drive", "repo"],
                            "default": "drive",
                            "description": "Where to inspect the PPTX. repo is admin-only.",
                        },
                        "search_text": {"type": "string", "description": "Optional exact text to locate."},
                        "max_matches": {
                            "type": "integer",
                            "default": MAX_SEARCH_MATCHES,
                            "description": "Maximum search matches to report.",
                        },
                    },
                    "required": ["path"],
                },
            },
            _inspect_presentation_for_edit,
            timeout_sec=60,
        ),
        ToolEntry(
            "edit_presentation",
            {
                "name": "edit_presentation",
                "description": (
                    "Apply confirmed edits to a copy of an existing .pptx and optionally send it to Telegram. "
                    "Supports replace_text, set_shape_text, set_table_cell, and apply_text_style. The tool edits "
                    "only selected slide/notes XML parts and preserves unmodified PPTX package objects such as "
                    "tables, diagrams, pictures, charts, masters, layouts, media, relationships, and theme files. "
                    "Original PPTX files are never modified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Drive path to the source PPTX."},
                        "operations": {
                            "type": "array",
                            "description": "Confirmed edit operations. Each item needs confirmed=true or confidence>=0.75.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["replace_text", "set_shape_text", "set_table_cell", "apply_text_style"],
                                    },
                                    "slide": {"type": "integer"},
                                    "slides": {
                                        "description": (
                                            "Slide number, list, or range string like '1-3,7'. "
                                            "For apply_text_style only, omit this to style all slides."
                                        )
                                    },
                                    "shape_id": {"type": "integer"},
                                    "shape_name": {"type": "string"},
                                    "search_text": {"type": "string"},
                                    "replacement_text": {"type": "string"},
                                    "text": {"type": "string"},
                                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                                    "occurrence": {"type": "integer"},
                                    "all_matches": {"type": "boolean"},
                                    "include_notes": {"type": "boolean"},
                                    "allow_reflow": {"type": "boolean"},
                                    "table_index": {"type": "integer"},
                                    "row": {"type": "integer"},
                                    "col": {"type": "integer"},
                                    "value": {},
                                    "font_face": {"type": "string"},
                                    "font_size_pt": {"type": "number"},
                                    "color": {"type": "string"},
                                    "bold": {"type": "boolean"},
                                    "confidence": {"type": "number"},
                                    "confirmed": {"type": "boolean"},
                                },
                            },
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output .pptx path in Drive. Defaults to presentation_edits/<name>-edited-<timestamp>.pptx.",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                            "description": "Overwrite output_path if it exists. Source PPTX is still never overwritten.",
                        },
                        "send_to_chat": {
                            "type": "boolean",
                            "default": True,
                            "description": "Queue the edited PPTX for Telegram delivery when there is an active chat.",
                        },
                    },
                    "required": ["path", "operations"],
                },
            },
            _edit_presentation,
            timeout_sec=60,
        ),
    ]
