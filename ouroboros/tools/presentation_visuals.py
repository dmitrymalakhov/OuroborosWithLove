"""Reusable visual primitives for generated PPTX slides."""

from __future__ import annotations

import re
from typing import Any, Dict, List
from xml.sax.saxutils import escape


EMU_PER_POINT = 12_700
MIN_TEXT_SIZE = 1_150

THEMES: Dict[str, Dict[str, str]] = {
    "professional": {
        "bg": "FFFFFF",
        "text": "172033",
        "muted": "64748B",
        "accent": "0F766E",
        "accent2": "EAB308",
        "panel": "F8FAFC",
        "line": "CBD5E1",
        "inverse": "FFFFFF",
    },
    "dark": {
        "bg": "111827",
        "text": "F8FAFC",
        "muted": "CBD5E1",
        "accent": "38BDF8",
        "accent2": "F59E0B",
        "panel": "1F2937",
        "line": "475569",
        "inverse": "0F172A",
    },
    "clean": {
        "bg": "F7F7F2",
        "text": "1F2933",
        "muted": "56616F",
        "accent": "B45309",
        "accent2": "2563EB",
        "panel": "FFFFFF",
        "line": "D7D7CB",
        "inverse": "FFFFFF",
    },
    "vivid": {
        "bg": "FFFDF8",
        "text": "172033",
        "muted": "5B677A",
        "accent": "C2410C",
        "accent2": "0E7490",
        "panel": "FFFFFF",
        "line": "F3C7A8",
        "inverse": "FFFFFF",
    },
}


def clean_text(value: Any, limit: int = 1600) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def xml_text(text: Any) -> str:
    return escape(str(text or ""), {"\"": "&quot;"})


def paragraph(text: str, size: int, color: str, bold: bool = False, bullet: bool = False, align: str = "l") -> str:
    bold_attr = ' b="1"' if bold else ""
    ppr = f'<a:pPr algn="{align}"/>'
    if bullet:
        ppr = '<a:pPr marL="342900" indent="-171450"><a:buChar char="&#8226;"/></a:pPr>'
    return (
        "<a:p>"
        f"{ppr}"
        "<a:r>"
        f'<a:rPr lang="en-US" sz="{size}"{bold_attr}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        "</a:rPr>"
        f"<a:t>{xml_text(text)}</a:t>"
        "</a:r>"
        f'<a:endParaRPr lang="en-US" sz="{size}"/>'
        "</a:p>"
    )


def _estimate_wrapped_lines(text: str, size: int, cx: int, bullet: bool = False) -> int:
    size_pt = max(8.0, size / 100)
    width_pt = max(20.0, cx / EMU_PER_POINT)
    chars_per_line = max(8, int(width_pt / (size_pt * 0.54)))
    if bullet:
        chars_per_line = max(8, chars_per_line - 3)
    lines = 0
    for part in str(text or "").splitlines() or [""]:
        length = max(1, len(part.strip()))
        lines += max(1, (length + chars_per_line - 1) // chars_per_line)
    return lines


def _estimated_text_height_pt(items: List[Dict[str, Any]], cx: int) -> float:
    height = 0.0
    for item in items:
        size = int(item.get("size", 2200))
        size_pt = size / 100
        lines = _estimate_wrapped_lines(item.get("text", ""), size, cx, bool(item.get("bullet", False)))
        height += lines * size_pt * 1.16
        height += max(1.5, size_pt * 0.12)
    return height


def _fit_textbox_items(
    paragraphs: List[Dict[str, Any]],
    cx: int,
    cy: int,
    margin: int,
    min_size: int = MIN_TEXT_SIZE,
) -> List[Dict[str, Any]]:
    items = [dict(item) for item in paragraphs if clean_text(item.get("text"))]
    if not items:
        return []
    usable_cx = max(1, cx - 2 * margin)
    usable_cy = max(1, cy - 2 * margin)
    available_pt = max(10.0, usable_cy / EMU_PER_POINT)

    for _ in range(16):
        if _estimated_text_height_pt(items, usable_cx) <= available_pt:
            return items
        changed = False
        for item in items:
            size = int(item.get("size", 2200))
            if size > min_size:
                item["size"] = max(min_size, int(size * 0.92))
                changed = True
        if not changed:
            break

    fitted: List[Dict[str, Any]] = []
    omitted = 0
    for item in items:
        candidate = fitted + [item]
        if _estimated_text_height_pt(candidate, usable_cx) <= available_pt:
            fitted.append(item)
        else:
            omitted += 1

    if omitted and fitted:
        marker = {
            "text": "...",
            "size": min_size,
            "color": fitted[-1].get("color", "000000"),
            "bullet": bool(fitted[-1].get("bullet", False)),
        }
        while fitted and _estimated_text_height_pt(fitted + [marker], usable_cx) > available_pt:
            fitted.pop()
        if fitted:
            fitted.append(marker)
    return fitted or items[:1]


def textbox(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    paragraphs: List[Dict[str, Any]],
    fill: str = "",
    line: str = "",
    margin: int = 80_000,
    min_size: int = MIN_TEXT_SIZE,
    geometry: str = "rect",
) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    para_xml = "".join(
        paragraph(
            item.get("text", ""),
            int(item.get("size", 2200)),
            item.get("color", "000000"),
            bool(item.get("bold", False)),
            bool(item.get("bullet", False)),
            item.get("align", "l"),
        )
        for item in _fit_textbox_items(paragraphs, cx, cy, margin, min_size)
    ) or "<a:p/>"
    return (
        "<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{xml_text(name)}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        "<p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="{geometry}"><a:avLst/></a:prstGeom>'
        f"{fill_xml}{line_xml}</p:spPr>"
        "<p:txBody>"
        f'<a:bodyPr wrap="square" lIns="{margin}" tIns="{margin}" rIns="{margin}" bIns="{margin}">'
        '<a:normAutofit fontScale="65000" lnSpcReduction="20000"/></a:bodyPr>'
        f"<a:lstStyle/>{para_xml}</p:txBody></p:sp>"
    )


def rect(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, fill: str, line: str = "") -> str:
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    return (
        "<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{xml_text(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        "<p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}'
        "</p:spPr></p:sp>"
    )


def image_frame(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, theme: Dict[str, str]) -> str:
    pad = 100_000
    fx = max(0, x - pad)
    fy = max(0, y - pad)
    return textbox(
        shape_id,
        name,
        fx,
        fy,
        cx + pad * 2,
        cy + pad * 2,
        [],
        fill=theme["panel"],
        line=theme["line"],
        margin=0,
        geometry="roundRect",
    )


def title_size(text: str) -> int:
    length = len(text or "")
    if length > 80:
        return 3000
    if length > 48:
        return 3600
    return 4400


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, acx, acy = a
    bx, by, bcx, bcy = b
    return ax < bx + bcx and ax + acx > bx and ay < by + bcy and ay + acy > by
