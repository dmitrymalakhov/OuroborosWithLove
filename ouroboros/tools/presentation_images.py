"""Image helpers for generated PPTX presentations."""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from ouroboros.tools.registry import ToolContext
from ouroboros.utils import safe_relpath


SLIDE_W = 12_192_000
SLIDE_H = 6_858_000
MAX_IMAGES_PER_SLIDE = 12
MAX_IMAGE_BYTES = 15 * 1024 * 1024

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def image_content_type_defaults() -> str:
    return "".join(
        f'<Default Extension="{suffix.lstrip(".")}" ContentType="{content_type}"/>'
        for suffix, content_type in IMAGE_TYPES.items()
    )


def _clean_text(value: Any, limit: int = 1600) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def _x(text: Any) -> str:
    return escape(str(text or ""), {"\"": "&quot;"})


def coerce_images(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for raw in value[:MAX_IMAGES_PER_SLIDE]:
        if isinstance(raw, str):
            raw = {"path": raw}
        if not isinstance(raw, dict):
            continue
        path = _clean_text(raw.get("path"), 260)
        if not path:
            continue
        fit = _clean_text(raw.get("fit") or "contain", 20).lower()
        if fit not in {"contain", "stretch"}:
            fit = "contain"
        item = {
            "path": path,
            "alt_text": _clean_text(raw.get("alt_text") or raw.get("caption"), 180),
            "fit": fit,
        }
        for key in ("x", "y", "w", "h"):
            if key in raw:
                item[key] = raw.get(key)
        result.append(item)
    return result


def _image_size(data: bytes, suffix: str) -> tuple[int, int]:
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"):
        idx = 2
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            idx += 2
            if marker in {0xD8, 0xD9}:
                continue
            size = int.from_bytes(data[idx:idx + 2], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                return int.from_bytes(data[idx + 5:idx + 7], "big"), int.from_bytes(data[idx + 3:idx + 5], "big")
            idx += max(2, size)
    return 0, 0


def prepare_slide_images(ctx: ToolContext, slides: List[Dict[str, Any]]) -> None:
    media_idx = 1
    root = ctx.drive_root.resolve()
    for slide in slides:
        prepared = []
        for rel_idx, image in enumerate(slide.get("images") or [], start=2):
            rel = safe_relpath(str(image.get("path") or ""))
            source = ctx.drive_path(rel)
            try:
                source.relative_to(root)
            except ValueError:
                raise ValueError("Image path traversal is not allowed.")
            suffix = source.suffix.lower()
            if suffix not in IMAGE_TYPES:
                raise ValueError(f"Unsupported image type for PPTX: {rel}. Use PNG, JPEG, GIF, or WEBP.")
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"Image not found: {rel}")
            data = source.read_bytes()
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image is too large for PPTX: {rel} ({len(data)} bytes)")
            width, height = _image_size(data, suffix)
            prepared.append({
                **image,
                "path": rel,
                "name": source.name,
                "data": data,
                "media_name": f"image{media_idx}{suffix}",
                "rel_id": f"rId{rel_idx}",
                "width_px": width,
                "height_px": height,
            })
            media_idx += 1
        slide["_prepared_images"] = prepared


def _fraction_emu(value: Any, total: int, default: int) -> int:
    try:
        number = float(value)
    except Exception:
        return default
    if 0 <= number <= 1:
        return int(total * number)
    return int(number)


def _auto_image_box(idx: int, total: int) -> tuple[int, int, int, int]:
    if total == 1:
        return 6_900_000, 1_520_000, 4_520_000, 3_480_000
    if total == 2:
        return 6_900_000, 1_420_000 + idx * 2_120_000, 4_520_000, 1_820_000
    if 3 <= total <= 4:
        cols = 2
        gap = 140_000
        area_x, area_y = 6_780_000, 1_470_000
        cell_cx, cell_cy = 2_180_000, 1_680_000
        col = idx % cols
        row = idx // cols
        return area_x + col * (cell_cx + gap), area_y + row * (cell_cy + gap), cell_cx, cell_cy

    cols = min(max(total, 1), 4)
    rows = (max(total, 1) + cols - 1) // cols
    area_x, area_y = 900_000, 5_220_000
    area_cx, area_cy = 10_400_000, 950_000
    gap = 130_000
    cell_cx = (area_cx - gap * (cols - 1)) // cols
    cell_cy = (area_cy - gap * (rows - 1)) // rows
    col = idx % cols
    row = idx // cols
    return area_x + col * (cell_cx + gap), area_y + row * (cell_cy + gap), cell_cx, cell_cy


def _image_box(image: Dict[str, Any], idx: int, total: int) -> tuple[int, int, int, int]:
    auto_x, auto_y, auto_cx, auto_cy = _auto_image_box(idx, total)
    x = _fraction_emu(image.get("x"), SLIDE_W, auto_x)
    y = _fraction_emu(image.get("y"), SLIDE_H, auto_y)
    cx = _fraction_emu(image.get("w"), SLIDE_W, auto_cx)
    cy = _fraction_emu(image.get("h"), SLIDE_H, auto_cy)
    cx = max(20_000, min(cx, SLIDE_W))
    cy = max(20_000, min(cy, SLIDE_H))
    x = max(0, min(x, SLIDE_W - cx))
    y = max(0, min(y, SLIDE_H - cy))
    return x, y, cx, cy


def _fit_image_box(image: Dict[str, Any], x: int, y: int, cx: int, cy: int) -> tuple[int, int, int, int]:
    if image.get("fit") == "stretch":
        return x, y, cx, cy
    width = int(image.get("width_px") or 0)
    height = int(image.get("height_px") or 0)
    if width <= 0 or height <= 0:
        return x, y, cx, cy
    scale = min(cx / width, cy / height)
    fit_cx = max(20_000, int(width * scale))
    fit_cy = max(20_000, int(height * scale))
    return x + (cx - fit_cx) // 2, y + (cy - fit_cy) // 2, fit_cx, fit_cy


def image_bounds(image: Dict[str, Any], idx: int, total: int) -> tuple[int, int, int, int]:
    x, y, cx, cy = _image_box(image, idx, total)
    return _fit_image_box(image, x, y, cx, cy)


def picture(shape_id: int, image: Dict[str, Any], idx: int, total: int) -> str:
    x, y, cx, cy = image_bounds(image, idx, total)
    name = _x(image.get("name") or pathlib.PurePosixPath(str(image.get("path") or "image")).name)
    descr = _x(image.get("alt_text") or name)
    return (
        "<p:pic><p:nvPicPr>"
        f'<p:cNvPr id="{shape_id}" name="{name}" descr="{descr}"/>'
        "<p:cNvPicPr/><p:nvPr/></p:nvPicPr>"
        f'<p:blipFill><a:blip r:embed="{image["rel_id"]}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        "<p:spPr>"
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln><a:noFill/></a:ln>'
        "</p:spPr></p:pic>"
    )
