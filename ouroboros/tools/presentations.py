"""Presentation generation tools.

The LLM designs slide content; this module turns the structured outline into a
PowerPoint-compatible .pptx package in the user's Drive workspace.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import zipfile
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import safe_relpath


PPTX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
MAX_SLIDES = 60
MAX_BULLETS = 12
MAX_TEXT_CHARS = 1600
SLIDE_W = 12_192_000
SLIDE_H = 6_858_000


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
}


def _scope(ctx: ToolContext) -> dict:
    return {
        "user_id": ctx.current_user_id,
        "user_role": ctx.user_role,
        "drive_root": str(ctx.drive_root),
        "shared_drive_root": str(ctx.shared_drive_root or ctx.drive_root),
    }


def _clean_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def _text_list(value: Any, limit: int = 220) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [line.strip(" -\t") for line in value.splitlines()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    items = [_clean_text(item, limit) for item in raw_items]
    return [item for item in items if item][:MAX_BULLETS]


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    value = value.strip("-._")
    return value[:60] or "presentation"


def _resolve_output_path(ctx: ToolContext, output_path: str, title: str, overwrite: bool) -> tuple[pathlib.Path, str]:
    if output_path:
        rel = safe_relpath(output_path)
        if not rel.lower().endswith(".pptx"):
            rel += ".pptx"
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        rel = f"presentations/{_slugify(title)}-{stamp}.pptx"

    target = ctx.drive_path(rel)
    root = ctx.drive_root.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal is not allowed.")

    if target.exists() and not overwrite:
        stem = target.stem
        suffix = target.suffix
        parent = target.parent
        for idx in range(2, 1000):
            candidate = parent / f"{stem}-{idx}{suffix}"
            if not candidate.exists():
                target = candidate
                rel = str(target.relative_to(root))
                break

    target.parent.mkdir(parents=True, exist_ok=True)
    return target, str(target.relative_to(root))


def _coerce_slides(slides: Any) -> List[Dict[str, Any]]:
    if isinstance(slides, str):
        slides = json.loads(slides)
    if not isinstance(slides, list):
        raise ValueError("slides must be a list of slide objects")
    result: List[Dict[str, Any]] = []
    for raw in slides[:MAX_SLIDES]:
        if not isinstance(raw, dict):
            raw = {"title": str(raw)}
        layout = _clean_text(raw.get("layout") or "auto", 40).lower()
        if layout not in {"auto", "title", "section", "bullets", "two_column", "quote", "closing"}:
            layout = "auto"
        slide = {
            "layout": layout,
            "title": _clean_text(raw.get("title"), 180),
            "subtitle": _clean_text(raw.get("subtitle"), 260),
            "body": _clean_text(raw.get("body"), 900),
            "bullets": _text_list(raw.get("bullets")),
            "left_title": _clean_text(raw.get("left_title"), 120),
            "left_bullets": _text_list(raw.get("left_bullets")),
            "right_title": _clean_text(raw.get("right_title"), 120),
            "right_bullets": _text_list(raw.get("right_bullets")),
            "quote": _clean_text(raw.get("quote"), 500),
            "attribution": _clean_text(raw.get("attribution"), 140),
            "speaker_notes": _clean_text(raw.get("speaker_notes"), 1200),
        }
        if slide["layout"] == "auto":
            if slide["left_bullets"] or slide["right_bullets"]:
                slide["layout"] = "two_column"
            elif slide["quote"]:
                slide["layout"] = "quote"
            elif "thank" in slide["title"].lower() or "next step" in slide["title"].lower():
                slide["layout"] = "closing"
            else:
                slide["layout"] = "bullets"
        result.append(slide)
    return result


def _presentation_slides(
    title: str,
    subtitle: str,
    slides: List[Dict[str, Any]],
    include_title_slide: bool,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if include_title_slide:
        result.append({
            "layout": "title",
            "title": title,
            "subtitle": subtitle,
            "body": "",
            "bullets": [],
            "left_title": "",
            "left_bullets": [],
            "right_title": "",
            "right_bullets": [],
            "quote": "",
            "attribution": "",
            "speaker_notes": "",
        })
    result.extend(slides)
    if not result:
        result.append({
            "layout": "title",
            "title": title or "Presentation",
            "subtitle": subtitle,
            "body": "",
            "bullets": [],
            "left_title": "",
            "left_bullets": [],
            "right_title": "",
            "right_bullets": [],
            "quote": "",
            "attribution": "",
            "speaker_notes": "",
        })
    return result[:MAX_SLIDES]


def _x(text: Any) -> str:
    return escape(str(text or ""), {"\"": "&quot;"})


def _paragraph(text: str, size: int, color: str, bold: bool = False, bullet: bool = False, align: str = "l") -> str:
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
        f"<a:t>{_x(text)}</a:t>"
        "</a:r>"
        f'<a:endParaRPr lang="en-US" sz="{size}"/>'
        "</a:p>"
    )


def _textbox(
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
) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    para_xml = "".join(
        _paragraph(
            item.get("text", ""),
            int(item.get("size", 2200)),
            item.get("color", "000000"),
            bool(item.get("bold", False)),
            bool(item.get("bullet", False)),
            item.get("align", "l"),
        )
        for item in paragraphs
        if _clean_text(item.get("text"))
    )
    if not para_xml:
        para_xml = "<a:p/>"
    return (
        "<p:sp>"
        "<p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{_x(name)}"/>'
        '<p:cNvSpPr txBox="1"/>'
        "<p:nvPr/>"
        "</p:nvSpPr>"
        "<p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f"{fill_xml}{line_xml}"
        "</p:spPr>"
        "<p:txBody>"
        f'<a:bodyPr wrap="square" lIns="{margin}" tIns="{margin}" rIns="{margin}" bIns="{margin}"><a:spAutoFit/></a:bodyPr>'
        "<a:lstStyle/>"
        f"{para_xml}"
        "</p:txBody>"
        "</p:sp>"
    )


def _rect(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, fill: str) -> str:
    return (
        "<p:sp>"
        "<p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{_x(name)}"/>'
        "<p:cNvSpPr/>"
        "<p:nvPr/>"
        "</p:nvSpPr>"
        "<p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        "<a:ln><a:noFill/></a:ln>"
        "</p:spPr>"
        "</p:sp>"
    )


def _title_size(text: str) -> int:
    length = len(text or "")
    if length > 80:
        return 3000
    if length > 48:
        return 3600
    return 4400


def _render_slide_shapes(slide: Dict[str, Any], theme: Dict[str, str], slide_no: int, total: int) -> str:
    shapes: List[str] = []
    sid = 2
    layout = slide["layout"]
    title = slide["title"] or ("Slide " + str(slide_no))

    shapes.append(_rect(sid, "Top accent", 0, 0, SLIDE_W, 90_000, theme["accent"]))
    sid += 1

    if layout == "title":
        shapes.append(_rect(sid, "Accent block", 0, 0, 430_000, SLIDE_H, theme["accent"]))
        sid += 1
        shapes.append(_textbox(
            sid, "Title", 900_000, 1_900_000, 10_500_000, 1_250_000,
            [{"text": title, "size": _title_size(title), "color": theme["text"], "bold": True}],
            margin=0,
        ))
        sid += 1
        if slide["subtitle"]:
            shapes.append(_textbox(
                sid, "Subtitle", 900_000, 3_160_000, 9_800_000, 760_000,
                [{"text": slide["subtitle"], "size": 2400, "color": theme["muted"]}],
                margin=0,
            ))
            sid += 1
        shapes.append(_rect(sid, "Divider", 900_000, 4_080_000, 2_200_000, 65_000, theme["accent2"]))
        sid += 1
        return "".join(shapes)

    if layout == "section":
        shapes.append(_rect(sid, "Section background", 0, 90_000, SLIDE_W, SLIDE_H - 90_000, theme["accent"]))
        sid += 1
        shapes.append(_textbox(
            sid, "Section title", 920_000, 2_250_000, 10_300_000, 1_150_000,
            [{"text": title, "size": _title_size(title), "color": theme["inverse"], "bold": True, "align": "ctr"}],
            margin=0,
        ))
        sid += 1
        if slide["subtitle"] or slide["body"]:
            shapes.append(_textbox(
                sid, "Section subtitle", 1_450_000, 3_500_000, 9_250_000, 820_000,
                [{"text": slide["subtitle"] or slide["body"], "size": 2200, "color": theme["inverse"], "align": "ctr"}],
                margin=0,
            ))
        return "".join(shapes)

    shapes.append(_textbox(
        sid, "Slide title", 650_000, 420_000, 10_950_000, 720_000,
        [{"text": title, "size": 3100, "color": theme["text"], "bold": True}],
        margin=0,
    ))
    sid += 1

    if layout == "two_column":
        shapes.append(_textbox(
            sid, "Left column", 700_000, 1_520_000, 5_150_000, 4_650_000,
            ([{"text": slide["left_title"] or "Option A", "size": 2200, "color": theme["accent"], "bold": True}]
             + [{"text": item, "size": 1850, "color": theme["text"], "bullet": True} for item in slide["left_bullets"]]),
            fill=theme["panel"],
            line=theme["line"],
        ))
        sid += 1
        shapes.append(_textbox(
            sid, "Right column", 6_320_000, 1_520_000, 5_150_000, 4_650_000,
            ([{"text": slide["right_title"] or "Option B", "size": 2200, "color": theme["accent"], "bold": True}]
             + [{"text": item, "size": 1850, "color": theme["text"], "bullet": True} for item in slide["right_bullets"]]),
            fill=theme["panel"],
            line=theme["line"],
        ))
        sid += 1
    elif layout == "quote":
        quote = slide["quote"] or slide["body"] or "Key idea"
        shapes.append(_rect(sid, "Quote mark", 820_000, 1_560_000, 100_000, 3_950_000, theme["accent2"]))
        sid += 1
        shapes.append(_textbox(
            sid, "Quote", 1_080_000, 1_580_000, 9_980_000, 2_700_000,
            [{"text": quote, "size": 3000 if len(quote) < 180 else 2400, "color": theme["text"], "bold": True}],
            margin=0,
        ))
        sid += 1
        if slide["attribution"]:
            shapes.append(_textbox(
                sid, "Attribution", 1_080_000, 4_460_000, 8_800_000, 420_000,
                [{"text": slide["attribution"], "size": 1800, "color": theme["muted"]}],
                margin=0,
            ))
            sid += 1
    else:
        body_items: List[Dict[str, Any]] = []
        if slide["body"]:
            body_items.append({"text": slide["body"], "size": 2050, "color": theme["muted"]})
        for bullet in slide["bullets"]:
            body_items.append({"text": bullet, "size": 2000, "color": theme["text"], "bullet": True})
        if not body_items:
            body_items.append({"text": slide["subtitle"] or "Add details here.", "size": 2200, "color": theme["muted"]})
        shapes.append(_textbox(
            sid, "Content", 850_000, 1_500_000, 10_550_000, 4_700_000,
            body_items,
            fill=theme["panel"] if layout == "closing" else "",
            line=theme["line"] if layout == "closing" else "",
        ))
        sid += 1

    footer = f"{slide_no}/{total}"
    shapes.append(_textbox(
        sid, "Footer", 10_800_000, 6_330_000, 760_000, 260_000,
        [{"text": footer, "size": 1150, "color": theme["muted"], "align": "r"}],
        margin=0,
    ))
    return "".join(shapes)


def _slide_xml(slide: Dict[str, Any], theme: Dict[str, str], slide_no: int, total: int) -> str:
    shapes = _render_slide_shapes(slide, theme, slide_no, total)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld>"
        "<p:bg><p:bgPr>"
        f'<a:solidFill><a:srgbClr val="{theme["bg"]}"/></a:solidFill>'
        "<a:effectLst/>"
        "</p:bgPr></p:bg>"
        "<p:spTree>"
        "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{shapes}"
        "</p:spTree>"
        "</p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        "</p:sld>"
    )


def _slide_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
        'Target="../slideLayouts/slideLayout1.xml"/>'
        "</Relationships>"
    )


def _content_types_xml(slide_count: int) -> str:
    overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        f"{overrides}"
        "</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def _presentation_xml(slide_count: int) -> str:
    slide_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "<p:defaultTextStyle/>"
        "</p:presentation>"
    )


def _presentation_rels_xml(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, slide_count + 1):
        rels.append(
            f'<Relationship Id="rId{i + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{i}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def _slide_master_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'
        '</p:sldMaster>'
    )


def _slide_master_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        '</Relationships>'
    )


def _slide_layout_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )


def _slide_layout_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        '</Relationships>'
    )


def _theme_xml(theme: Dict[str, str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Ouroboros Theme">'
        '<a:themeElements><a:clrScheme name="Ouroboros">'
        '<a:dk1><a:srgbClr val="111827"/></a:dk1>'
        '<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F2937"/></a:dk2>'
        '<a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>'
        f'<a:accent1><a:srgbClr val="{theme["accent"]}"/></a:accent1>'
        f'<a:accent2><a:srgbClr val="{theme["accent2"]}"/></a:accent2>'
        '<a:accent3><a:srgbClr val="2563EB"/></a:accent3>'
        '<a:accent4><a:srgbClr val="7C3AED"/></a:accent4>'
        '<a:accent5><a:srgbClr val="DB2777"/></a:accent5>'
        '<a:accent6><a:srgbClr val="059669"/></a:accent6>'
        '<a:hlink><a:srgbClr val="2563EB"/></a:hlink>'
        '<a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>'
        '</a:clrScheme>'
        '<a:fontScheme name="Ouroboros"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface="Arial"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="Ouroboros"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="6350" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements></a:theme>'
    )


def _core_props_xml(title: str) -> str:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{_x(title)}</dc:title>"
        "<dc:creator>Ouroboros</dc:creator>"
        "<cp:lastModifiedBy>Ouroboros</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def _app_props_xml(slide_count: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Ouroboros</Application>"
        "<PresentationFormat>On-screen Show (16:9)</PresentationFormat>"
        f"<Slides>{slide_count}</Slides>"
        "<Notes>0</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips>"
        "<ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate>"
        "<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>"
        "</Properties>"
    )


def _write_pptx(path: pathlib.Path, title: str, slides: List[Dict[str, Any]], theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["professional"])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(slides)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("docProps/core.xml", _core_props_xml(title))
        zf.writestr("docProps/app.xml", _app_props_xml(len(slides)))
        zf.writestr("ppt/presentation.xml", _presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", _presentation_rels_xml(len(slides)))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", _slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _slide_master_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", _slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _slide_layout_rels_xml())
        zf.writestr("ppt/theme/theme1.xml", _theme_xml(theme))
        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", _slide_xml(slide, theme, idx, len(slides)))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", _slide_rels_xml())


def _write_notes(path: pathlib.Path, rel_path: str, title: str, slides: List[Dict[str, Any]]) -> str:
    notes = []
    for idx, slide in enumerate(slides, start=1):
        note = slide.get("speaker_notes") or ""
        if note:
            notes.append(f"## Slide {idx}: {slide.get('title') or title}\n\n{note}")
    if not notes:
        return ""
    notes_rel = re.sub(r"\.pptx$", ".notes.md", rel_path, flags=re.IGNORECASE)
    notes_path = path.with_suffix(".notes.md")
    notes_path.write_text(f"# Speaker Notes: {title}\n\n" + "\n\n".join(notes) + "\n", encoding="utf-8")
    return notes_rel


def _create_presentation(
    ctx: ToolContext,
    title: str,
    slides: Any,
    subtitle: str = "",
    theme: str = "professional",
    output_path: str = "",
    include_title_slide: bool = True,
    overwrite: bool = False,
    send_to_chat: bool = True,
) -> str:
    title = _clean_text(title, 180) or "Presentation"
    subtitle = _clean_text(subtitle, 260)
    theme = _clean_text(theme, 40).lower() or "professional"
    if theme not in THEMES:
        theme = "professional"

    _slides = _presentation_slides(title, subtitle, _coerce_slides(slides), bool(include_title_slide))
    output, rel_output = _resolve_output_path(ctx, output_path, title, bool(overwrite))

    ctx.emit_progress_fn(f"Creating presentation `{output.name}` with {len(_slides)} slides.")
    _write_pptx(output, title, _slides, theme)
    notes_rel = _write_notes(output, rel_output, title, _slides)

    size = output.stat().st_size
    queued = False
    if send_to_chat and ctx.current_chat_id:
        ctx.pending_events.append({
            "type": "send_document",
            "chat_id": ctx.current_chat_id,
            "path": rel_output,
            "caption": title,
            "filename": output.name,
            "mime_type": PPTX_MIME_TYPE,
            **_scope(ctx),
        })
        queued = True

    result = [
        "OK: presentation created",
        f"- path: {rel_output}",
        f"- slides: {len(_slides)}",
        f"- size_bytes: {size}",
        f"- theme: {theme}",
    ]
    if notes_rel:
        result.append(f"- speaker_notes_path: {notes_rel}")
    if queued:
        result.append("- telegram_delivery: queued")
    elif send_to_chat:
        result.append("- telegram_delivery: skipped (no active chat)")
    return "\n".join(result)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("create_presentation", {
            "name": "create_presentation",
            "description": (
                "Create a PowerPoint .pptx file from a structured slide outline. "
                "Use when a user asks for a deck, slides, or a presentation. "
                "First decide the slide content, then pass concise structured slides here; "
                "the tool saves the PPTX in the user's Drive workspace and can queue it for Telegram delivery."
            ),
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Presentation title."},
                "subtitle": {"type": "string", "description": "Optional subtitle for the cover slide."},
                "slides": {
                    "type": "array",
                    "description": "Content slides after the optional cover slide.",
                    "items": {"type": "object", "properties": {
                        "layout": {
                            "type": "string",
                            "enum": ["auto", "section", "bullets", "two_column", "quote", "closing"],
                            "default": "auto",
                        },
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                        "body": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "left_title": {"type": "string"},
                        "left_bullets": {"type": "array", "items": {"type": "string"}},
                        "right_title": {"type": "string"},
                        "right_bullets": {"type": "array", "items": {"type": "string"}},
                        "quote": {"type": "string"},
                        "attribution": {"type": "string"},
                        "speaker_notes": {"type": "string"},
                    }},
                },
                "theme": {
                    "type": "string",
                    "enum": ["professional", "dark", "clean"],
                    "default": "professional",
                    "description": "Visual theme.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path in user Drive. Defaults to presentations/<title>-timestamp.pptx.",
                },
                "include_title_slide": {
                    "type": "boolean",
                    "default": True,
                    "description": "Add a generated cover slide before the supplied content slides.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite output_path if it already exists; otherwise create a deduplicated filename.",
                },
                "send_to_chat": {
                    "type": "boolean",
                    "default": True,
                    "description": "Queue the PPTX for Telegram delivery when the task has an active chat.",
                },
            }, "required": ["title", "slides"]},
        }, _create_presentation, timeout_sec=60),
    ]
