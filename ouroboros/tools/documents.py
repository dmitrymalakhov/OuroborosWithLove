"""Document analysis tools.

Extracts text and structure from user-provided documents so the agent can
summarize, critique, answer questions, or turn them into tasks.
"""

from __future__ import annotations

import base64, datetime, hashlib, ipaddress, json, mimetypes, os, re, socket, subprocess, tempfile, urllib.parse, zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import clip_text, safe_relpath

DEFAULT_MAX_CHARS = 30_000
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_SLIDES = 80
DEFAULT_MAX_XLSX_SHEETS = 30
PDF_TOC_SCAN_PAGES = 8
PDF_TOC_MAX_ENTRIES = 80
PDF_TOC_CALIBRATION_MAX_ENTRIES = 12
DEFAULT_SEARCH_MAX_PAGES = 500
MAX_SEARCH_PAGES = 1000
DEFAULT_SEARCH_MAX_RESULTS = 12
MAX_SEARCH_RESULTS = 50
DEFAULT_SEARCH_CONTEXT_CHARS = 360
MAX_SEARCH_CONTEXT_CHARS = 1200
DOCUMENT_INDEX_VERSION = 1
DOCUMENT_INDEX_DIR = "document_indexes"
DEFAULT_INDEX_MAX_PAGES = 1000
DEFAULT_INDEX_MAX_SLIDES = 500
MAX_INDEX_SLIDES = 1000
INDEX_MAX_TEXT_CHARS_PER_UNIT = 20_000
INDEX_ENTITY_LIMIT = 80
MAX_XLSX_ROWS_PER_SHEET = 200
MAX_XLSX_CELLS_PER_SHEET = 2_000
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_XML_PART_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_FILES = 80
MAX_ARCHIVE_MEMBER_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 150 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".xml", ".html", ".htm", ".log", ".py", ".js", ".ts", ".css",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}
ANALYSIS_TYPES = {"summary", "critique", "extract_tasks", "answer_question", "raw"}
ARCHIVE_EXTENSIONS = {".zip"}
DOCUMENT_EXTENSIONS = {".pdf", ".pptx", ".docx", ".ppt", ".xlsx"} | TEXT_EXTENSIONS | IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS
PDF_OCR_RENDER_ZOOM = 2.0

IMAGE_OCR_PROMPT = """Extract the visible content from this user-uploaded image for document analysis.

Focus on OCR accuracy, not interpretation:
- preserve all visible text, labels, dates, currencies, percentages, totals, formulas, and footnotes;
- preserve tables as Markdown tables when possible, keeping row/column labels and units;
- keep numbers exactly as shown, including signs, separators, decimal commas/dots, and empty cells;
- mark uncertain values as [unclear] and do not invent missing values;
- if this is a financial statement, forecast, balance sheet, P&L, or cash flow model screenshot/photo, extract enough structure for numerical cross-checks.

Return only the extracted content."""

PDF_PAGE_OCR_PROMPT = IMAGE_OCR_PROMPT + """

This image is one rendered page from a scanned PDF. Preserve the page structure, table rows and columns, queue/stage labels, engineering parameters, units, footnotes, and handwritten/printed notes when visible."""

SEARCH_STOPWORDS = {
    "and", "are", "for", "from", "has", "have", "the", "this", "that", "what", "where", "with",
    "без", "был", "где", "для", "его", "есть", "как", "или", "над", "при", "про", "что", "это",
}

FINANCIAL_TERMS = {
    "amortization", "bond", "breach", "call option", "coupon", "covenant", "cross-default",
    "debt", "default", "dividend", "dscr", "ebitda", "event of default", "financial statements",
    "guarantee", "guarantor", "interest", "issuer", "leverage", "ltv", "maturity", "redemption",
    "restricted payments", "risk factors", "security", "use of proceeds",
    "дивиденды", "дефолт", "долг", "залог", "ковенант", "обеспечение", "оферта", "погашение",
    "процент", "риск", "эмитент",
}

@dataclass
class ExtractedDocument:
    kind: str
    metadata: Dict[str, str] = field(default_factory=dict)
    sections: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


ProgressFn = Callable[[str], None]

def _emit_progress(progress: ProgressFn | None, text: str) -> None:
    if progress is None:
        return
    try:
        progress(text)
    except Exception:
        pass

def _clean_limit(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except Exception:
        return default
    return max(minimum, min(value, maximum))


def _normalize_analysis_type(value: str) -> str:
    value = (value or "summary").strip().lower()
    if value not in ANALYSIS_TYPES:
        return "summary"
    return value


def _format_page_selection(pages: List[int]) -> str:
    if not pages:
        return ""
    ranges: List[str] = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _parse_page_ranges(
    page_ranges: str,
    total_pages: int | None,
    max_pages: int,
) -> Tuple[List[int], List[str]]:
    warnings: List[str] = []
    raw = str(page_ranges or "").strip()
    if not raw:
        if total_pages is None:
            return list(range(1, max_pages + 1)), warnings
        pages = list(range(1, min(total_pages, max_pages) + 1))
        if total_pages > max_pages:
            warnings.append(f"Only the first {max_pages} of {total_pages} pages were extracted.")
        return pages, warnings

    normalized = raw.replace("—", "-").replace("–", "-")
    tokens = [token for token in re.split(r"[,;\s]+", normalized) if token]
    pages: List[int] = []
    seen = set()
    skipped_out_of_range: List[int] = []

    for token in tokens:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(f"Invalid page_ranges token: {token!r}. Use format like '15-21,48-55'.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1:
            raise ValueError("page_ranges must use 1-based positive page numbers")
        if end < start:
            raise ValueError(f"Invalid page range: {token!r}")
        for page in range(start, end + 1):
            if total_pages is not None and page > total_pages:
                skipped_out_of_range.append(page)
                continue
            if page not in seen:
                pages.append(page)
                seen.add(page)

    if not pages:
        raise ValueError("page_ranges did not select any pages")
    if len(pages) > max_pages:
        warnings.append(f"Only the first {max_pages} selected pages were extracted from page_ranges={raw!r}.")
        pages = pages[:max_pages]
    if skipped_out_of_range:
        warnings.append(
            f"Skipped pages outside document length: {_format_page_selection(skipped_out_of_range)}."
        )
    return pages, warnings


def _parse_slide_ranges(
    slide_ranges: str,
    total_slides: int | None,
    max_slides: int,
    action: str = "extracted",
) -> Tuple[List[int], List[str]]:
    warnings: List[str] = []
    raw = str(slide_ranges or "").strip()
    if not raw:
        if total_slides is None:
            return list(range(1, max_slides + 1)), warnings
        slides = list(range(1, min(total_slides, max_slides) + 1))
        if total_slides > max_slides:
            warnings.append(f"Only the first {max_slides} of {total_slides} slides were {action}.")
        return slides, warnings

    normalized = raw.replace("—", "-").replace("–", "-")
    tokens = [token for token in re.split(r"[,;\s]+", normalized) if token]
    slides: List[int] = []
    seen = set()
    skipped_out_of_range: List[int] = []

    for token in tokens:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(f"Invalid slide_ranges token: {token!r}. Use format like '15-21,48-55'.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1:
            raise ValueError("slide_ranges must use 1-based positive slide numbers")
        if end < start:
            raise ValueError(f"Invalid slide range: {token!r}")
        for slide in range(start, end + 1):
            if total_slides is not None and slide > total_slides:
                skipped_out_of_range.append(slide)
                continue
            if slide not in seen:
                slides.append(slide)
                seen.add(slide)

    if not slides:
        raise ValueError("slide_ranges did not select any slides")
    if len(slides) > max_slides:
        warnings.append(f"Only the first {max_slides} selected slides were {action} from slide_ranges={raw!r}.")
        slides = slides[:max_slides]
    if skipped_out_of_range:
        warnings.append(
            f"Skipped slides outside deck length: {_format_page_selection(skipped_out_of_range)}."
        )
    return slides, warnings


def _resolve_document_path(ctx: ToolContext, path: str, source: str) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty string")

    source = (source or "drive").strip().lower()
    if source not in {"drive", "repo"}:
        raise ValueError("source must be 'drive' or 'repo'")

    if source == "repo" and str(ctx.user_role or "user").lower() != "admin":
        raise PermissionError("source='repo' is admin-only in multi-user mode")

    root = ctx.repo_dir if source == "repo" else ctx.drive_root
    root = root.resolve()
    resolved = (root / safe_relpath(path)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal is not allowed")

    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not resolved.is_file():
        raise ValueError(f"Document path is not a file: {path}")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Document is too large: {resolved.stat().st_size} bytes")
    return resolved


def _safe_output_path(root: Path, rel: str) -> Path:
    rel = safe_relpath(rel)
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Path traversal is not allowed")
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _document_index_path(ctx: ToolContext, file_hash: str) -> Path:
    safe_hash = re.sub(r"[^a-fA-F0-9]+", "", str(file_hash or ""))[:64]
    if len(safe_hash) < 16:
        raise ValueError("Invalid document hash")
    return _safe_output_path(ctx.drive_root, str(Path(DOCUMENT_INDEX_DIR) / f"{safe_hash}.json"))


def _index_rel_path(ctx: ToolContext, index_path: Path) -> str:
    return str(index_path.relative_to(ctx.drive_root.resolve()))


def _load_document_index(ctx: ToolContext, document_path: Path, source: str, tool_path: str) -> Tuple[Dict[str, Any] | None, str, Path]:
    file_hash = _file_sha256(document_path)
    index_path = _document_index_path(ctx, file_hash)
    if not index_path.exists():
        return None, file_hash, index_path
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None, file_hash, index_path
    if int(index.get("version") or 0) != DOCUMENT_INDEX_VERSION:
        return None, file_hash, index_path
    if str(index.get("sha256") or "") != file_hash:
        return None, file_hash, index_path
    if str(index.get("source") or "") != str(source or ""):
        return None, file_hash, index_path
    if str(index.get("path") or "") != str(tool_path or ""):
        return None, file_hash, index_path
    return index, file_hash, index_path


def _write_document_index(ctx: ToolContext, index: Dict[str, Any]) -> Path:
    index_path = _document_index_path(ctx, str(index.get("sha256") or ""))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_path


def _limited_text(text: str, max_chars: int = INDEX_MAX_TEXT_CHARS_PER_UNIT) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[Unit text clipped for index]"


def _add_limited(values: Dict[str, List[str]], key: str, candidates: Iterable[Any], limit: int = INDEX_ENTITY_LIMIT) -> None:
    bucket = values.setdefault(key, [])
    seen = {item.casefold() for item in bucket}
    for candidate in candidates:
        value = re.sub(r"\s+", " ", str(candidate or "")).strip(" .,:;")
        if not value:
            continue
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        bucket.append(value)
        if len(bucket) >= limit:
            return


def _extract_financial_entities(text: str) -> Dict[str, List[str]]:
    raw = str(text or "")
    entities: Dict[str, List[str]] = {}
    _add_limited(entities, "isin", re.findall(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b", raw))
    _add_limited(entities, "rates", re.findall(r"\b\d+(?:[.,]\d+)?\s*%", raw))
    _add_limited(
        entities,
        "amounts",
        re.findall(
            r"\b(?:[$€£₽]\s*\d[\d\s.,]*|\d[\d\s.,]*(?:RUB|USD|EUR|GBP|руб\.?|млн|млрд|тыс\.?))\b",
            raw,
            flags=re.I,
        ),
    )
    _add_limited(entities, "dates", re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", raw))
    term_hits = []
    folded = raw.casefold()
    for term in sorted(FINANCIAL_TERMS):
        if term.casefold() in folded:
            term_hits.append(term)
    _add_limited(entities, "financial_terms", term_hits)
    _add_limited(entities, "ratios", re.findall(r"\b(?:DSCR|LTV|EBITDA|EBIT|Debt/EBITDA|Net Debt/EBITDA)\b", raw, flags=re.I))
    return {key: values for key, values in entities.items() if values}


def _merge_entity_maps(items: Iterable[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for item in items:
        for key, values in (item or {}).items():
            _add_limited(merged, key, values)
    return merged


def _safe_filename(name: str, fallback: str = "download") -> str:
    name = str(name or "").replace("\\", "_").replace("/", "_").replace(":", "_")
    name = "".join(ch for ch in name if ord(ch) >= 32)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback
    if len(name) > 180:
        suffix = Path(name).suffix[:20]
        stem = Path(name).stem[:140].strip(" .") or fallback
        name = stem + suffix
    return name


def _zip_member_safe_name(name: str) -> str:
    parts = []
    for part in Path(str(name).replace("\\", "/")).parts:
        if part in ("", ".", "..") or part.startswith("/"):
            continue
        parts.append(_safe_filename(part, fallback="entry"))
    return "/".join(parts)


def _read_plain_text(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDocument(
        kind="text",
        metadata={"encoding": "utf-8"},
        sections=[("Text", text)],
    )


def _detect_image_mime(path: Path) -> str:
    detected = mimetypes.guess_type(path.name)[0] or ""
    if detected == "image/jpg":
        return "image/jpeg"
    return detected


def _get_image_ocr_model() -> str:
    from ouroboros.llm import default_main_model

    return os.environ.get("OUROBOROS_MODEL", "") or default_main_model()


def _get_llm_client():
    from ouroboros.llm import LLMClient

    return LLMClient()


def _emit_llm_usage(ctx: ToolContext | None, usage: Dict[str, Any], model: str) -> None:
    if ctx is None or ctx.event_queue is None:
        return
    try:
        ctx.event_queue.put_nowait({
            "type": "llm_usage",
            "model": model,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
            "cost": float(usage.get("cost", 0.0) or 0.0),
            "task_id": ctx.task_id,
            "task_type": ctx.current_task_type or "task",
            **ctx.event_scope(),
        })
    except Exception:
        pass


def _extract_image(path: Path, progress: ProgressFn | None = None, ctx: ToolContext | None = None) -> ExtractedDocument:
    image_mime = _detect_image_mime(path)
    suffix = path.suffix.lower()
    if image_mime not in IMAGE_MIME_TYPES or suffix not in IMAGE_EXTENSIONS:
        return ExtractedDocument(
            kind=suffix.lstrip(".") or "image",
            metadata={"extractor": "vlm_ocr", "mime_type": image_mime or "unknown"},
            warnings=["Unsupported image type. Supported: PNG, JPEG, WEBP, GIF, BMP."],
        )

    _emit_progress(progress, f"Открыл изображение `{path.name}`. Запускаю OCR через vision-модель.")
    model = _get_image_ocr_model()
    try:
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        text, usage = _get_llm_client().vision_query(
            prompt=IMAGE_OCR_PROMPT,
            images=[{"base64": image_b64, "mime": image_mime}],
            model=model,
            max_tokens=4096,
            reasoning_effort="low",
        )
        _emit_llm_usage(ctx, usage, model)
    except Exception as exc:
        return ExtractedDocument(
            kind=suffix.lstrip(".") or "image",
            metadata={"extractor": "vlm_ocr", "mime_type": image_mime, "model": model},
            warnings=[f"Image OCR failed: {type(exc).__name__}: {exc}"],
        )

    return ExtractedDocument(
        kind=suffix.lstrip(".") or "image",
        metadata={"extractor": "vlm_ocr", "mime_type": image_mime, "model": model},
        sections=[("Image OCR", text or "[No text extracted]")],
    )


def _try_import_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader, "pypdf"
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return PdfReader, "PyPDF2"
    except Exception:
        return None, ""


def _try_import_fitz():
    try:
        import fitz  # type: ignore

        return fitz
    except Exception:
        return None


def _pdf_extraction_has_text(extracted: ExtractedDocument) -> bool:
    if extracted.kind != "pdf":
        return True
    for title, text in extracted.sections:
        if title == "PDF navigation map":
            continue
        stripped = str(text or "").strip()
        if not stripped or stripped == "[No extractable text]":
            continue
        if stripped.startswith("[Could not extract page text:"):
            continue
        if any(ch.isalnum() for ch in stripped):
            return True
    return False


def _clean_pdf_toc_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip(" .\t")
    return title[:180]


def _extract_pdf_outline_entries(reader: Any, total_pages: int, max_entries: int = PDF_TOC_MAX_ENTRIES) -> List[Dict[str, Any]]:
    outline = None
    for attr in ("outline", "outlines"):
        try:
            outline = getattr(reader, attr)
        except Exception:
            outline = None
        if outline:
            break
    if not outline:
        return []

    entries: List[Dict[str, Any]] = []

    def page_for(item: Any) -> int | None:
        for method_name in ("get_destination_page_number", "getDestinationPageNumber"):
            method = getattr(reader, method_name, None)
            if not method:
                continue
            try:
                index = int(method(item))
            except Exception:
                continue
            page_num = index + 1
            if 1 <= page_num <= total_pages:
                return page_num
        return None

    def title_for(item: Any) -> str:
        title = getattr(item, "title", "")
        if not title and hasattr(item, "get"):
            try:
                title = item.get("/Title") or item.get("Title") or ""
            except Exception:
                title = ""
        return _clean_pdf_toc_title(title)

    def walk(items: Any, level: int = 1) -> None:
        if len(entries) >= max_entries:
            return
        if isinstance(items, (list, tuple)):
            for item in items:
                if len(entries) >= max_entries:
                    return
                walk(item, level + 1 if isinstance(item, (list, tuple)) else level)
            return

        title = title_for(items)
        if not title:
            return
        entries.append({
            "title": title,
            "page": page_for(items),
            "level": max(1, min(level, 6)),
        })

    walk(outline)
    return entries


def _pdf_toc_header_present(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*(table of contents|contents|содержание|оглавление)\s*$", text or ""))


def _parse_pdf_toc_line(line: str, total_pages: int | None) -> Dict[str, Any] | None:
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    if len(line) < 5 or len(line) > 220:
        return None
    if re.fullmatch(r"(?i)table of contents|contents|содержание|оглавление", line):
        return None

    match = re.fullmatch(r"(?P<title>.+?)[\s._=-]{2,}(?P<page>\d{1,4})", line)
    leader = bool(match)
    if not match:
        match = re.fullmatch(
            r"(?P<title>(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+.+?)\s+(?P<page>\d{1,4})",
            line,
            flags=re.I,
        )
    if not match:
        return None

    title = _clean_pdf_toc_title(match.group("title"))
    if len(title) < 3 or not re.search(r"[A-Za-zА-Яа-я]", title):
        return None

    page = int(match.group("page"))
    if page < 1:
        return None
    if total_pages is not None and page > total_pages:
        return None
    if total_pages is None and page > 5000:
        return None

    return {"title": title, "page": page, "level": 1, "leader": leader}


def _extract_pdf_text_toc_entries(
    page_texts: List[Tuple[int, str]],
    total_pages: int | None,
    max_entries: int = PDF_TOC_MAX_ENTRIES,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen = set()
    for page_num, text in page_texts[:PDF_TOC_SCAN_PAGES]:
        header_present = _pdf_toc_header_present(text)
        page_entries: List[Dict[str, Any]] = []
        for raw_line in str(text or "").splitlines()[:220]:
            entry = _parse_pdf_toc_line(raw_line, total_pages)
            if entry is None:
                continue
            key = (entry["title"].casefold(), entry["page"])
            if key in seen:
                continue
            entry["found_on_page"] = page_num
            page_entries.append(entry)

        leader_count = sum(1 for entry in page_entries if entry.get("leader"))
        if not header_present and (len(page_entries) < 4 or leader_count < 2):
            continue

        for entry in page_entries:
            key = (entry["title"].casefold(), entry["page"])
            if key in seen:
                continue
            seen.add(key)
            entry.pop("leader", None)
            entries.append(entry)
            if len(entries) >= max_entries:
                return entries
    return entries


def _format_pdf_navigation_map(entries: List[Dict[str, Any]], source: str) -> str:
    if source == "outline":
        lines = [
            "Detected PDF navigation map from document outline/bookmarks.",
            "Page numbers below are PDF page numbers. Use likely headings to decide whether the document contains the needed information, then call analyze_document again with page_ranges for the promising sections.",
        ]
    else:
        lines = [
            "Detected PDF navigation map from front-matter table of contents text.",
            "Page references below come from the printed table of contents and may need a small offset from PDF page numbers. Use likely headings to decide whether the document contains the needed information, then call analyze_document again with page_ranges for the promising sections.",
        ]

    for entry in entries[:PDF_TOC_MAX_ENTRIES]:
        level = int(entry.get("level") or 1)
        prefix = "  " * max(0, min(level - 1, 5)) + "- "
        page = entry.get("page")
        if source == "outline":
            loc = f"p. {page}" if page else "p. ?"
        else:
            loc = f"ref p. {page}" if page else "ref p. ?"
        lines.append(f"{prefix}{loc}: {entry['title']}")
    if len(entries) >= PDF_TOC_MAX_ENTRIES:
        lines.append(f"[Navigation map clipped to first {PDF_TOC_MAX_ENTRIES} entries.]")
    return "\n".join(lines)


def _heading_match_key(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.?)\s+", "", text, flags=re.I)
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_heading_page(title: str, page_texts: List[Tuple[int, str]], min_page: int = 1) -> int | None:
    needle = _heading_match_key(title)
    if len(needle) < 4:
        return None
    for page_num, text in page_texts:
        if page_num < min_page:
            continue
        lines = [line for line in str(text or "").splitlines()[:80] if line.strip()]
        for line in lines:
            hay = _heading_match_key(line)
            if not hay:
                continue
            if needle == hay or needle in hay or hay in needle:
                return page_num
        page_key = _heading_match_key(text[:4000])
        if needle and needle in page_key:
            return page_num
    return None


def _calibrate_toc_offset(
    navigation: List[Dict[str, Any]],
    page_texts: List[Tuple[int, str]],
    max_entries: int = PDF_TOC_CALIBRATION_MAX_ENTRIES,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for entry in navigation:
        printed_page = entry.get("printed_page", entry.get("page"))
        if not isinstance(printed_page, int):
            continue
        found_on_page = entry.get("found_on_page")
        min_page = int(found_on_page) + 1 if isinstance(found_on_page, int) else 1
        found_page = _find_heading_page(str(entry.get("title") or ""), page_texts, min_page=min_page)
        if found_page is None:
            continue
        samples.append({
            "title": entry.get("title"),
            "printed_page": printed_page,
            "pdf_page": found_page,
            "offset": found_page - printed_page,
        })
        if len(samples) >= max_entries:
            break

    if not samples:
        return {"offset": 0, "confidence": "none", "samples": []}

    counts: Dict[int, int] = {}
    for sample in samples:
        offset = int(sample["offset"])
        counts[offset] = counts.get(offset, 0) + 1
    best_offset = sorted(counts.items(), key=lambda item: (-item[1], abs(item[0])))[0][0]
    matching = [sample for sample in samples if int(sample["offset"]) == best_offset]
    if len(matching) >= 3:
        confidence = "high"
    elif len(matching) >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "offset": best_offset,
        "confidence": confidence,
        "samples": matching[:6],
        "sample_count": len(samples),
        "matching_sample_count": len(matching),
    }


def _apply_pdf_navigation_calibration(
    navigation: List[Dict[str, Any]],
    calibration: Dict[str, Any],
    total_pages: int | None,
) -> None:
    confidence = str(calibration.get("confidence") or "none")
    offset = int(calibration.get("offset") or 0)
    for entry in navigation:
        source = str(entry.get("source") or "")
        page = entry.get("page")
        if not isinstance(page, int):
            continue
        if source == "outline":
            entry["pdf_page"] = page
            continue
        entry["printed_page"] = page
        if confidence == "none":
            continue
        pdf_page = page + offset
        if total_pages is not None and not (1 <= pdf_page <= total_pages):
            continue
        entry["pdf_page"] = pdf_page
        entry["toc_offset"] = offset


def _entry_location_value(entry: Dict[str, Any], kind: str) -> int | None:
    if kind == "pdf":
        pdf_page = entry.get("pdf_page")
        if isinstance(pdf_page, int):
            return pdf_page
        if str(entry.get("source") or "") == "outline" and isinstance(entry.get("page"), int):
            return int(entry["page"])
        return None
    keys = {
        "docx": ("paragraph",),
        "pptx": ("slide",),
        "text": ("line",),
    }.get(kind, ("page", "paragraph", "line"))
    for key in keys:
        value = entry.get(key)
        if isinstance(value, int):
            return value
    return None


def _assign_section_paths(units: List[Dict[str, Any]], navigation: List[Dict[str, Any]], kind: str) -> None:
    unit_key = "page" if kind == "pdf" else "paragraph" if kind == "docx" else "slide" if kind == "pptx" else "line"
    entries = [
        (int(loc), idx, entry)
        for idx, entry in enumerate(navigation)
        for loc in [_entry_location_value(entry, kind)]
        if isinstance(loc, int)
    ]
    entries.sort(key=lambda item: (item[0], item[1]))
    if not entries:
        return

    path: List[str] = []
    entry_index = 0
    for unit in sorted(units, key=lambda item: int(item.get(unit_key) or 10**9)):
        location = unit.get(unit_key)
        if not isinstance(location, int):
            continue
        while entry_index < len(entries) and entries[entry_index][0] <= location:
            _loc, _idx, entry = entries[entry_index]
            title = _clean_pdf_toc_title(entry.get("title"))
            level = max(1, min(int(entry.get("level") or 1), 6))
            if title:
                if len(path) >= level:
                    path = path[:level - 1]
                while len(path) < level - 1:
                    path.append("")
                path.append(title)
                path = [part for part in path if part]
            entry_index += 1
        if path:
            unit["section_path"] = list(path)
            unit["section"] = path[-1]


def _search_terms(query: str) -> List[str]:
    raw_terms = re.findall(r"[0-9A-Za-zА-Яа-яЁё]{2,}", str(query or "").casefold())
    terms: List[str] = []
    seen = set()
    for term in raw_terms:
        if term in SEARCH_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _score_search_text(text: str, query: str, terms: List[str]) -> int:
    haystack = str(text or "").casefold()
    phrase = re.sub(r"\s+", " ", str(query or "").casefold()).strip()
    score = 0
    if phrase and phrase in re.sub(r"\s+", " ", haystack):
        score += 60 + 10 * min(5, haystack.count(phrase))

    term_hits = 0
    for term in terms:
        count = haystack.count(term)
        if count:
            term_hits += 1
            score += 8 * min(count, 5)

    if terms and term_hits == len(terms):
        score += 25
    elif terms and term_hits >= max(1, int(len(terms) * 0.6)):
        score += 10
    return score


def _snippet_for_search_match(text: str, query: str, terms: List[str], context_chars: int) -> str:
    raw = str(text or "")
    if not raw.strip():
        return "[No extractable text]"

    haystack = raw.casefold()
    phrase = str(query or "").casefold().strip()
    positions: List[int] = []
    if phrase:
        pos = haystack.find(phrase)
        if pos >= 0:
            positions.append(pos)
    for term in terms:
        pos = haystack.find(term)
        if pos >= 0:
            positions.append(pos)

    center = min(positions) if positions else 0
    half = max(60, context_chars // 2)
    start = max(0, center - half)
    end = min(len(raw), center + half)
    snippet = raw[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(raw):
        snippet += " ..."
    return snippet or "[Empty]"


def _rank_search_hits(hits: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    return sorted(
        hits,
        key=lambda item: (
            -int(item.get("score") or 0),
            int(item.get("pdf_page") or item.get("page") or item.get("paragraph") or item.get("slide") or item.get("line") or 10**9),
            str(item.get("location") or ""),
        ),
    )[:max_results]


def _navigation_hits(entries: List[Dict[str, Any]], query: str, terms: List[str], max_results: int) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for entry in entries:
        score = _score_search_text(str(entry.get("title") or ""), query, terms)
        if score <= 0:
            continue
        hit = dict(entry)
        hit["score"] = score
        hits.append(hit)
    return _rank_search_hits(hits, max_results)


def _search_pdf_page_texts(
    page_texts: List[Tuple[int, str]],
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for page_num, text in page_texts:
        score = _score_search_text(text, query, terms)
        if score <= 0:
            continue
        hits.append({
            "location": f"Page {page_num}",
            "page": page_num,
            "score": score,
            "snippet": _snippet_for_search_match(text, query, terms, context_chars),
        })
    return _rank_search_hits(hits, max_results)


def _suggested_pdf_page_ranges(
    hits: List[Dict[str, Any]],
    max_pages_each: int = 2,
    total_pages: int | None = None,
    max_hits: int = 5,
    min_score_ratio: float = 0.5,
) -> str:
    pages: List[int] = []
    seen = set()
    if not hits:
        return ""
    max_score = max(int(hit.get("score") or 0) for hit in hits)
    min_score = int(max_score * min_score_ratio)
    strong_hits = [hit for hit in hits if int(hit.get("score") or 0) >= min_score][:max_hits]
    for hit in strong_hits:
        page = hit.get("page")
        if not isinstance(page, int):
            continue
        for candidate in range(page, page + max_pages_each):
            if total_pages is not None and candidate > total_pages:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            pages.append(candidate)
    return _format_page_selection(sorted(pages))


def _suggested_pptx_slide_ranges(
    hits: List[Dict[str, Any]],
    max_slides_each: int = 2,
    total_slides: int | None = None,
    max_hits: int = 5,
    min_score_ratio: float = 0.5,
) -> str:
    slides: List[int] = []
    seen = set()
    if not hits:
        return ""
    max_score = max(int(hit.get("score") or 0) for hit in hits)
    min_score = int(max_score * min_score_ratio)
    strong_hits = [hit for hit in hits if int(hit.get("score") or 0) >= min_score][:max_hits]
    for hit in strong_hits:
        slide = hit.get("slide")
        if not isinstance(slide, int):
            continue
        for candidate in range(slide, slide + max_slides_each):
            if total_slides is not None and candidate > total_slides:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            slides.append(candidate)
    return _format_page_selection(sorted(slides))


def _docx_paragraphs(path: Path) -> List[str]:
    with zipfile.ZipFile(path) as zf:
        try:
            return _paragraphs_from_xml(_read_zip_xml(zf, "word/document.xml"))
        except KeyError:
            return []


def _search_docx(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    progress: ProgressFn | None = None,
) -> Dict[str, Any]:
    paragraphs = _docx_paragraphs(path)
    _emit_progress(progress, f"Открыл DOCX `{path.name}`: {len(paragraphs)} абзацев. Ищу совпадения по запросу.")
    hits: List[Dict[str, Any]] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        score = _score_search_text(paragraph, query, terms)
        if score <= 0:
            continue
        start = max(0, idx - 2)
        end = min(len(paragraphs), idx + 1)
        context = "\n".join(paragraphs[start:end])
        hits.append({
            "location": f"Paragraph {idx}",
            "paragraph": idx,
            "score": score,
            "snippet": _snippet_for_search_match(context, query, terms, context_chars),
        })

    return {
        "kind": "docx",
        "metadata": {
            "paragraphs": str(len(paragraphs)),
            "extractor": "zip+xml",
            "searched": "all paragraphs",
        },
        "warnings": [
            "DOCX does not store final rendered page numbers; search results use paragraph numbers."
        ],
        "navigation_hits": [],
        "hits": _rank_search_hits(hits, max_results),
        "suggested_page_ranges": "",
    }


def _search_text_file(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    progress: ProgressFn | None = None,
) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [(idx, line) for idx, line in enumerate(text.splitlines(), start=1) if line.strip()]
    if not chunks and text.strip():
        chunks = [(1, text)]
    _emit_progress(progress, f"Открыл текстовый файл `{path.name}`: {len(chunks)} строк/фрагментов. Ищу совпадения.")
    hits: List[Dict[str, Any]] = []
    for line_num, chunk in chunks:
        score = _score_search_text(chunk, query, terms)
        if score <= 0:
            continue
        hits.append({
            "location": f"Line {line_num}",
            "line": line_num,
            "score": score,
            "snippet": _snippet_for_search_match(chunk, query, terms, context_chars),
        })
    return {
        "kind": "text",
        "metadata": {"encoding": "utf-8", "searched": "all non-empty lines"},
        "warnings": [],
        "navigation_hits": [],
        "hits": _rank_search_hits(hits, max_results),
        "suggested_page_ranges": "",
    }


def _extract_pdf_with_python(
    path: Path,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
) -> ExtractedDocument | None:
    PdfReader, library = _try_import_pdf_reader()
    if PdfReader is None:
        return None

    warnings: List[str] = []
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            warnings.append("PDF is encrypted and could not be decrypted with an empty password.")

    total_pages = len(reader.pages)
    pages, range_warnings = _parse_page_ranges(page_ranges, total_pages, max_pages)
    warnings.extend(range_warnings)
    selection = _format_page_selection(pages)
    outline_entries = _extract_pdf_outline_entries(reader, total_pages)
    _emit_progress(
        progress,
        f"Открыл PDF `{path.name}`: {total_pages} стр. Извлекаю текст со страниц {selection}; если текста много, это может занять пару минут.",
    )
    sections: List[Tuple[str, str]] = []
    page_texts: List[Tuple[int, str]] = []
    total_selected = len(pages)
    for position, page_num in enumerate(pages, start=1):
        if position == 1 or position == total_selected or position % 10 == 0:
            _emit_progress(progress, f"Читаю PDF `{path.name}`: страница {page_num} ({position}/{total_selected}).")
        page = reader.pages[page_num - 1]
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[Could not extract page text: {type(exc).__name__}: {exc}]"
        text = text.strip() or "[No extractable text]"
        page_texts.append((page_num, text))
        sections.append((f"Page {page_num}", text))

    metadata = {
        "pages": str(total_pages),
        "pages_extracted": selection,
        "extractor": library,
    }
    navigation_source = ""
    navigation_entries = outline_entries
    if navigation_entries:
        navigation_source = "outline"
    elif not str(page_ranges or "").strip():
        navigation_entries = _extract_pdf_text_toc_entries(page_texts, total_pages)
        if navigation_entries:
            navigation_source = "front_matter_text"

    if navigation_entries and navigation_source:
        metadata["toc_source"] = navigation_source
        metadata["toc_entries"] = str(len(navigation_entries))
        sections.insert(0, ("PDF navigation map", _format_pdf_navigation_map(navigation_entries, navigation_source)))
    elif total_pages > max_pages and not str(page_ranges or "").strip():
        warnings.append(
            f"No PDF outline or front-matter table of contents was detected in the first {min(total_pages, PDF_TOC_SCAN_PAGES)} pages."
        )

    return ExtractedDocument(
        kind="pdf",
        metadata=metadata,
        sections=sections,
        warnings=warnings,
    )


def _extract_pdf_with_pdftotext(
    path: Path,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
) -> ExtractedDocument | None:
    if not str(page_ranges or "").strip():
        _emit_progress(progress, f"Извлекаю текст из PDF `{path.name}` через pdftotext, до {max_pages} стр.")
        try:
            proc = subprocess.run(
                ["pdftotext", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"],
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if proc.returncode != 0:
            return ExtractedDocument(
                kind="pdf",
                metadata={"extractor": "pdftotext"},
                warnings=[proc.stderr.strip() or "pdftotext failed"],
            )

        text = proc.stdout.strip() or "[No extractable text]"
        sections = [("PDF text", text)]
        metadata = {"extractor": "pdftotext", "pages": f"first {max_pages}"}
        navigation_entries = _extract_pdf_text_toc_entries([(0, text)], None)
        if navigation_entries:
            metadata["toc_source"] = "front_matter_text"
            metadata["toc_entries"] = str(len(navigation_entries))
            sections.insert(0, ("PDF navigation map", _format_pdf_navigation_map(navigation_entries, "front_matter_text")))

        return ExtractedDocument(
            kind="pdf",
            metadata=metadata,
            sections=sections,
        )

    pages, warnings = _parse_page_ranges(page_ranges, None, max_pages)
    selection = _format_page_selection(pages)
    _emit_progress(progress, f"Извлекаю текст из PDF `{path.name}` через pdftotext, страницы {selection}.")
    sections: List[Tuple[str, str]] = []
    try:
        for page_num in pages:
            proc = subprocess.run(
                ["pdftotext", "-layout", "-f", str(page_num), "-l", str(page_num), str(path), "-"],
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )
            if proc.returncode != 0:
                return ExtractedDocument(
                    kind="pdf",
                    metadata={"extractor": "pdftotext"},
                    warnings=[proc.stderr.strip() or "pdftotext failed"],
                )
            sections.append((f"Page {page_num}", proc.stdout.strip() or "[No extractable text]"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if not sections:
        return ExtractedDocument(
            kind="pdf",
            metadata={"extractor": "pdftotext"},
            warnings=["pdftotext did not return any pages"],
        )

    return ExtractedDocument(
        kind="pdf",
        metadata={"extractor": "pdftotext", "pages_extracted": selection},
        sections=sections,
        warnings=warnings,
    )


def _extract_pdf_with_vlm_ocr(
    path: Path,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
    ctx: ToolContext | None = None,
) -> ExtractedDocument | None:
    fitz = _try_import_fitz()
    if fitz is None:
        return None

    warnings: List[str] = ["Text extraction produced no usable text; used VLM OCR over rendered PDF pages."]
    sections: List[Tuple[str, str]] = []
    model = _get_image_ocr_model()

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return ExtractedDocument(
            kind="pdf",
            metadata={"extractor": "vlm_pdf_ocr", "model": model},
            warnings=[f"Could not open PDF for OCR rendering: {type(exc).__name__}: {exc}"],
        )

    try:
        total_pages = len(doc)
        pages, range_warnings = _parse_page_ranges(page_ranges, total_pages, max_pages)
        warnings.extend(range_warnings)
        selection = _format_page_selection(pages)
        _emit_progress(
            progress,
            f"В PDF `{path.name}` не найден текстовый слой. Рендерю страницы {selection} в изображения и запускаю OCR через vision-модель.",
        )

        try:
            client = _get_llm_client()
        except Exception as exc:
            return ExtractedDocument(
                kind="pdf",
                metadata={
                    "pages": str(total_pages),
                    "pages_extracted": selection,
                    "extractor": "vlm_pdf_ocr",
                    "model": model,
                },
                warnings=warnings + [f"PDF OCR failed before rendering pages: {type(exc).__name__}: {exc}"],
            )

        matrix = fitz.Matrix(PDF_OCR_RENDER_ZOOM, PDF_OCR_RENDER_ZOOM)
        total_selected = len(pages)
        for position, page_num in enumerate(pages, start=1):
            if position == 1 or position == total_selected or position % 5 == 0:
                _emit_progress(progress, f"OCR PDF `{path.name}`: страница {page_num} ({position}/{total_selected}).")
            try:
                page = doc.load_page(page_num - 1)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
                text, usage = client.vision_query(
                    prompt=PDF_PAGE_OCR_PROMPT,
                    images=[{"base64": image_b64, "mime": "image/png"}],
                    model=model,
                    max_tokens=4096,
                    reasoning_effort="low",
                )
                _emit_llm_usage(ctx, usage, model)
                sections.append((f"Page {page_num} OCR", text or "[No text extracted]"))
            except Exception as exc:
                warnings.append(f"Page {page_num}: OCR failed: {type(exc).__name__}: {exc}")
                sections.append((f"Page {page_num} OCR", "[OCR failed]"))

        return ExtractedDocument(
            kind="pdf",
            metadata={
                "pages": str(total_pages),
                "pages_extracted": selection,
                "extractor": "vlm_pdf_ocr",
                "model": model,
                "render_zoom": str(PDF_OCR_RENDER_ZOOM),
            },
            sections=sections,
            warnings=warnings,
        )
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _search_pdf_with_python(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
) -> Dict[str, Any] | None:
    PdfReader, library = _try_import_pdf_reader()
    if PdfReader is None:
        return None

    warnings: List[str] = []
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            warnings.append("PDF is encrypted and could not be decrypted with an empty password.")

    total_pages = len(reader.pages)
    if str(page_ranges or "").strip():
        pages, range_warnings = _parse_page_ranges(page_ranges, total_pages, max_pages)
        warnings.extend(range_warnings)
    else:
        pages = list(range(1, min(total_pages, max_pages) + 1))
        if total_pages > max_pages:
            warnings.append(f"Only the first {max_pages} of {total_pages} pages were searched.")
    selection = _format_page_selection(pages)

    _emit_progress(
        progress,
        f"Ищу в PDF `{path.name}` по страницам {selection}: запрос `{query}`.",
    )

    page_texts: List[Tuple[int, str]] = []
    first_page_texts: List[Tuple[int, str]] = []
    total_selected = len(pages)
    for position, page_num in enumerate(pages, start=1):
        if position == 1 or position == total_selected or position % 25 == 0:
            _emit_progress(progress, f"Поиск в PDF `{path.name}`: страница {page_num} ({position}/{total_selected}).")
        page = reader.pages[page_num - 1]
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[Could not extract page text: {type(exc).__name__}: {exc}]"
        text = text.strip()
        page_texts.append((page_num, text))
        if page_num <= PDF_TOC_SCAN_PAGES:
            first_page_texts.append((page_num, text))

    navigation_entries = _extract_pdf_outline_entries(reader, total_pages)
    navigation_source = "outline" if navigation_entries else ""
    if not navigation_entries and first_page_texts:
        navigation_entries = _extract_pdf_text_toc_entries(first_page_texts, total_pages)
        navigation_source = "front_matter_text" if navigation_entries else ""

    metadata = {
        "pages": str(total_pages),
        "pages_searched": selection,
        "extractor": library,
    }
    if navigation_entries and navigation_source:
        metadata["toc_source"] = navigation_source
        metadata["toc_entries"] = str(len(navigation_entries))

    hits = _search_pdf_page_texts(page_texts, query, terms, max_results, context_chars)
    return {
        "kind": "pdf",
        "metadata": metadata,
        "warnings": warnings,
        "navigation_hits": _navigation_hits(navigation_entries, query, terms, max_results),
        "hits": hits,
        "suggested_page_ranges": _suggested_pdf_page_ranges(hits, total_pages=total_pages),
    }


def _search_pdf_with_pdftotext(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
) -> Dict[str, Any] | None:
    warnings: List[str] = []
    page_texts: List[Tuple[int, str]] = []
    try:
        if not str(page_ranges or "").strip():
            _emit_progress(progress, f"Ищу в PDF `{path.name}` через pdftotext, до {max_pages} стр.")
            proc = subprocess.run(
                ["pdftotext", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"],
                text=True,
                capture_output=True,
                timeout=90,
                check=False,
            )
            if proc.returncode != 0:
                return None
            for idx, text in enumerate(proc.stdout.split("\f")[:max_pages], start=1):
                page_texts.append((idx, text.strip()))
        else:
            pages, warnings = _parse_page_ranges(page_ranges, None, max_pages)
            selection = _format_page_selection(pages)
            _emit_progress(progress, f"Ищу в PDF `{path.name}` через pdftotext, страницы {selection}.")
            for page_num in pages:
                proc = subprocess.run(
                    ["pdftotext", "-layout", "-f", str(page_num), "-l", str(page_num), str(path), "-"],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode != 0:
                    continue
                page_texts.append((page_num, proc.stdout.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if not page_texts:
        return None

    selection = _format_page_selection([page_num for page_num, _text in page_texts])
    navigation_entries = _extract_pdf_text_toc_entries(page_texts[:PDF_TOC_SCAN_PAGES], None)
    metadata = {
        "pages_searched": selection,
        "extractor": "pdftotext",
    }
    if navigation_entries:
        metadata["toc_source"] = "front_matter_text"
        metadata["toc_entries"] = str(len(navigation_entries))
    hits = _search_pdf_page_texts(page_texts, query, terms, max_results, context_chars)
    return {
        "kind": "pdf",
        "metadata": metadata,
        "warnings": warnings,
        "navigation_hits": _navigation_hits(navigation_entries, query, terms, max_results),
        "hits": hits,
        "suggested_page_ranges": _suggested_pdf_page_ranges(hits),
    }


def _search_pdf(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
) -> Dict[str, Any]:
    result = _search_pdf_with_python(
        path,
        query=query,
        terms=terms,
        max_results=max_results,
        context_chars=context_chars,
        max_pages=max_pages,
        page_ranges=page_ranges,
        progress=progress,
    )
    if result is not None:
        return result

    result = _search_pdf_with_pdftotext(
        path,
        query=query,
        terms=terms,
        max_results=max_results,
        context_chars=context_chars,
        max_pages=max_pages,
        page_ranges=page_ranges,
        progress=progress,
    )
    if result is not None:
        return result

    return {
        "kind": "pdf",
        "metadata": {},
        "warnings": ["No PDF extractor is available. Install pypdf or pdftotext to search PDF files."],
        "navigation_hits": [],
        "hits": [],
        "suggested_page_ranges": "",
    }


def _xml_attr(node: ET.Element, name: str) -> str:
    for key, value in node.attrib.items():
        if key == name or key.endswith("}" + name):
            return value
    return ""


def _is_likely_document_heading(text: str, style: str = "") -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    style_folded = str(style or "").casefold()
    if style_folded.startswith("heading") or style_folded.startswith("заголовок"):
        return True
    if not value or len(value) > 180 or not re.search(r"[A-Za-zА-Яа-я]", value):
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+\S", value, flags=re.I):
        return True
    folded = value.casefold().strip(" .:")
    return folded in FINANCIAL_TERMS


def _docx_paragraph_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        try:
            root = ET.fromstring(_read_zip_xml(zf, "word/document.xml"))
        except KeyError:
            return []
    for para in root.iter():
        if not para.tag.endswith("}p"):
            continue
        parts = [node.text or "" for node in para.iter() if node.tag.endswith("}t")]
        text = "".join(parts).strip()
        if not text:
            continue
        style = ""
        for node in para.iter():
            if node.tag.endswith("}pStyle"):
                style = _xml_attr(node, "val")
                break
        records.append({
            "paragraph": len(records) + 1,
            "text": text,
            "style": style,
        })
    return records


def _index_units_result(
    *,
    kind: str,
    metadata: Dict[str, Any],
    navigation: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "metadata": metadata,
        "navigation": navigation,
        "units": units,
        "warnings": warnings,
        "entities": _merge_entity_maps(unit.get("entities") or {} for unit in units),
    }


def _index_pdf_with_python(path: Path, max_pages: int, progress: ProgressFn | None = None) -> Dict[str, Any] | None:
    PdfReader, library = _try_import_pdf_reader()
    if PdfReader is None:
        return None

    warnings: List[str] = []
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            warnings.append("PDF is encrypted and could not be decrypted with an empty password.")

    total_pages = len(reader.pages)
    pages = list(range(1, min(total_pages, max_pages) + 1))
    if total_pages > max_pages:
        warnings.append(f"Only the first {max_pages} of {total_pages} pages were indexed.")
    _emit_progress(progress, f"Индексирую PDF `{path.name}`: {len(pages)} из {total_pages} страниц.")

    units: List[Dict[str, Any]] = []
    page_texts: List[Tuple[int, str]] = []
    first_page_texts: List[Tuple[int, str]] = []
    for position, page_num in enumerate(pages, start=1):
        if position == 1 or position == len(pages) or position % 25 == 0:
            _emit_progress(progress, f"Индекс PDF `{path.name}`: страница {page_num} ({position}/{len(pages)}).")
        page = reader.pages[page_num - 1]
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[Could not extract page text: {type(exc).__name__}: {exc}]"
        text = text.strip()
        page_texts.append((page_num, text))
        if page_num <= PDF_TOC_SCAN_PAGES:
            first_page_texts.append((page_num, text))
        units.append({
            "location": f"Page {page_num}",
            "page": page_num,
            "text": _limited_text(text),
            "entities": _extract_financial_entities(text),
        })

    navigation = _extract_pdf_outline_entries(reader, total_pages)
    navigation_source = "outline" if navigation else ""
    if not navigation:
        navigation = _extract_pdf_text_toc_entries(first_page_texts, total_pages)
        navigation_source = "front_matter_text" if navigation else ""
    for item in navigation:
        item["source"] = navigation_source
    calibration = {"offset": 0, "confidence": "none", "samples": []}
    if navigation_source == "front_matter_text":
        calibration = _calibrate_toc_offset(navigation, page_texts)
    _apply_pdf_navigation_calibration(navigation, calibration, total_pages)
    _assign_section_paths(units, navigation, "pdf")

    metadata: Dict[str, Any] = {
        "pages": total_pages,
        "pages_indexed": len(pages),
        "extractor": library,
    }
    if navigation_source:
        metadata["toc_source"] = navigation_source
        metadata["toc_entries"] = len(navigation)
    if calibration.get("confidence") != "none":
        metadata["toc_calibration"] = {
            "offset": calibration.get("offset"),
            "confidence": calibration.get("confidence"),
            "sample_count": calibration.get("sample_count"),
            "matching_sample_count": calibration.get("matching_sample_count"),
        }
    return _index_units_result(kind="pdf", metadata=metadata, navigation=navigation, units=units, warnings=warnings)


def _index_pdf_with_pdftotext(path: Path, max_pages: int, progress: ProgressFn | None = None) -> Dict[str, Any] | None:
    try:
        _emit_progress(progress, f"Индексирую PDF `{path.name}` через pdftotext, до {max_pages} стр.")
        proc = subprocess.run(
            ["pdftotext", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    units: List[Dict[str, Any]] = []
    page_texts: List[Tuple[int, str]] = []
    for idx, text in enumerate(proc.stdout.split("\f")[:max_pages], start=1):
        text = text.strip()
        if not text:
            continue
        page_texts.append((idx, text))
        units.append({
            "location": f"Page {idx}",
            "page": idx,
            "text": _limited_text(text),
            "entities": _extract_financial_entities(text),
        })
    if not units:
        return None
    navigation = _extract_pdf_text_toc_entries(page_texts[:PDF_TOC_SCAN_PAGES], None)
    for item in navigation:
        item["source"] = "front_matter_text"
    calibration = {"offset": 0, "confidence": "none", "samples": []}
    if navigation:
        calibration = _calibrate_toc_offset(navigation, page_texts)
    _apply_pdf_navigation_calibration(navigation, calibration, len(units))
    _assign_section_paths(units, navigation, "pdf")
    metadata: Dict[str, Any] = {
        "pages_indexed": len(units),
        "extractor": "pdftotext",
    }
    if navigation:
        metadata["toc_source"] = "front_matter_text"
        metadata["toc_entries"] = len(navigation)
    if calibration.get("confidence") != "none":
        metadata["toc_calibration"] = {
            "offset": calibration.get("offset"),
            "confidence": calibration.get("confidence"),
            "sample_count": calibration.get("sample_count"),
            "matching_sample_count": calibration.get("matching_sample_count"),
        }
    return _index_units_result(kind="pdf", metadata=metadata, navigation=navigation, units=units, warnings=[])


def _index_pdf(path: Path, max_pages: int, progress: ProgressFn | None = None) -> Dict[str, Any]:
    result = _index_pdf_with_python(path, max_pages=max_pages, progress=progress)
    if result is not None:
        return result
    result = _index_pdf_with_pdftotext(path, max_pages=max_pages, progress=progress)
    if result is not None:
        return result
    return _index_units_result(
        kind="pdf",
        metadata={},
        navigation=[],
        units=[],
        warnings=["No PDF extractor is available. Install pypdf or pdftotext to index PDF files."],
    )


def _index_docx(path: Path, progress: ProgressFn | None = None) -> Dict[str, Any]:
    records = _docx_paragraph_records(path)
    _emit_progress(progress, f"Индексирую DOCX `{path.name}`: {len(records)} абзацев.")
    units: List[Dict[str, Any]] = []
    navigation: List[Dict[str, Any]] = []
    for record in records:
        paragraph = int(record["paragraph"])
        text = str(record.get("text") or "")
        style = str(record.get("style") or "")
        units.append({
            "location": f"Paragraph {paragraph}",
            "paragraph": paragraph,
            "text": _limited_text(text),
            "style": style,
            "entities": _extract_financial_entities(text),
        })
        if _is_likely_document_heading(text, style):
            navigation.append({
                "title": _clean_pdf_toc_title(text),
                "paragraph": paragraph,
                "level": 1,
                "source": "docx_heading",
            })
    _assign_section_paths(units, navigation, "docx")
    metadata = {
        "paragraphs": len(records),
        "paragraphs_indexed": len(units),
        "extractor": "zip+xml",
        "heading_entries": len(navigation),
    }
    warnings = ["DOCX does not store stable rendered page numbers; index uses paragraph numbers."]
    return _index_units_result(kind="docx", metadata=metadata, navigation=navigation, units=units, warnings=warnings)


def _index_text(path: Path, progress: ProgressFn | None = None) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [(idx, line.strip()) for idx, line in enumerate(text.splitlines(), start=1) if line.strip()]
    if not lines and text.strip():
        lines = [(1, text.strip())]
    _emit_progress(progress, f"Индексирую текстовый файл `{path.name}`: {len(lines)} строк/фрагментов.")
    units = [{
        "location": f"Line {line_num}",
        "line": line_num,
        "text": _limited_text(line),
        "entities": _extract_financial_entities(line),
    } for line_num, line in lines]
    navigation = [{
        "title": _clean_pdf_toc_title(text),
        "line": line_num,
        "level": 1,
        "source": "text_heading",
    } for line_num, text in lines if _is_likely_document_heading(text)]
    _assign_section_paths(units, navigation, "text")
    return _index_units_result(
        kind="text",
        metadata={"lines_indexed": len(units), "encoding": "utf-8", "heading_entries": len(navigation)},
        navigation=navigation,
        units=units,
        warnings=[],
    )


def _index_pptx(
    path: Path,
    max_slides: int,
    progress: ProgressFn | None = None,
    slide_ranges: str = "",
) -> Dict[str, Any]:
    warnings: List[str] = []
    units: List[Dict[str, Any]] = []
    navigation: List[Dict[str, Any]] = []
    selected_slide_numbers: List[int] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        slides = sorted(
            [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)],
            key=_number_from_name,
        )
        notes = {
            _number_from_name(name): name
            for name in names
            if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
        }

        if str(slide_ranges or "").strip():
            selected_slide_numbers, range_warnings = _parse_slide_ranges(
                slide_ranges,
                len(slides),
                max_slides,
                action="indexed",
            )
            warnings.extend(range_warnings)
        else:
            selected_slide_numbers = list(range(1, min(len(slides), max_slides) + 1))
            if len(slides) > max_slides:
                warnings.append(f"Only the first {max_slides} of {len(slides)} slides were indexed.")

        slides_to_index = len(selected_slide_numbers)
        selection = _format_page_selection(selected_slide_numbers)
        _emit_progress(progress, f"Индексирую PPTX `{path.name}`: слайды {selection} ({slides_to_index} из {len(slides)}).")
        for position, selected_slide in enumerate(selected_slide_numbers, start=1):
            if position == 1 or position == slides_to_index or position % 25 == 0:
                _emit_progress(progress, f"Индекс PPTX `{path.name}`: слайд {selected_slide} ({position}/{slides_to_index}).")
            slide_name = slides[selected_slide - 1]
            slide_num = _number_from_name(slide_name)
            try:
                slide_lines = _paragraphs_from_xml(_read_zip_xml(zf, slide_name))
            except Exception as exc:
                slide_lines = [f"[Could not extract slide text: {type(exc).__name__}: {exc}]"]
            slide_lines = [line for line in slide_lines if str(line or "").strip()]

            notes_lines: List[str] = []
            notes_name = notes.get(slide_num)
            if notes_name:
                try:
                    notes_lines = _paragraphs_from_xml(_read_zip_xml(zf, notes_name))
                except Exception as exc:
                    notes_lines = [f"[Could not extract speaker notes: {type(exc).__name__}: {exc}]"]
                notes_lines = [line for line in notes_lines if str(line or "").strip()]

            title = _clean_pdf_toc_title(slide_lines[0] if slide_lines else f"Slide {selected_slide}")
            parts = []
            if slide_lines:
                parts.append("\n".join(slide_lines))
            else:
                parts.append("[No extractable slide text]")
            if notes_lines:
                parts.append("Speaker notes:\n" + "\n".join(notes_lines))
            text = "\n\n".join(parts)
            units.append({
                "location": f"Slide {selected_slide}",
                "slide": selected_slide,
                "title": title,
                "text": _limited_text(text),
                "entities": _extract_financial_entities(text),
            })
            if title and title != f"Slide {selected_slide}":
                navigation.append({
                    "title": title,
                    "slide": selected_slide,
                    "level": 1,
                    "source": "pptx_slide_title",
                })

    _assign_section_paths(units, navigation, "pptx")
    return _index_units_result(
        kind="pptx",
        metadata={
            "slides": len(slides),
            "slides_indexed": len(units),
            "slides_indexed_selection": _format_page_selection(selected_slide_numbers),
            "extractor": "zip+xml",
            "slide_title_entries": len(navigation),
        },
        navigation=navigation,
        units=units,
        warnings=warnings,
    )


def _build_document_index_payload(
    path: Path,
    source: str,
    tool_path: str,
    file_hash: str,
    max_pages: int,
    max_slides: int,
    progress: ProgressFn | None = None,
) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        body = _index_pdf(path, max_pages=max_pages, progress=progress)
    elif suffix == ".pptx":
        body = _index_pptx(path, max_slides=max_slides, progress=progress)
    elif suffix == ".docx":
        body = _index_docx(path, progress=progress)
    elif suffix in TEXT_EXTENSIONS:
        body = _index_text(path, progress=progress)
    else:
        body = _index_units_result(
            kind=suffix.lstrip(".") or "unknown",
            metadata={},
            navigation=[],
            units=[],
            warnings=[
                "Unsupported document index type. Supported: PDF, PPTX, DOCX, TXT, MD, CSV, JSON, HTML, XML, and code-like files.",
            ],
        )

    return {
        "version": DOCUMENT_INDEX_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source,
        "path": tool_path,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_hash,
        **body,
    }


def _search_document_index(
    index: Dict[str, Any],
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    page_ranges: str = "",
    slide_ranges: str = "",
) -> Dict[str, Any]:
    warnings = list(index.get("warnings") or [])
    units = list(index.get("units") or [])
    navigation = list(index.get("navigation") or [])
    kind = str(index.get("kind") or "unknown")
    metadata = dict(index.get("metadata") or {})
    metadata["index_used"] = "true"
    metadata["indexed_at"] = str(index.get("created_at") or "")

    if page_ranges and kind == "pdf":
        total_pages = metadata.get("pages")
        try:
            total_pages_int = int(total_pages) if total_pages is not None else None
        except Exception:
            total_pages_int = None
        selected, range_warnings = _parse_page_ranges(page_ranges, total_pages_int, MAX_SEARCH_PAGES)
        selected_set = set(selected)
        warnings.extend(range_warnings)
        units = [unit for unit in units if int(unit.get("page") or 0) in selected_set]
        navigation = [
            entry for entry in navigation
            if int(entry.get("pdf_page") or entry.get("page") or 0) in selected_set
        ]
        metadata["pages_searched"] = _format_page_selection(selected)

    if slide_ranges and kind == "pptx":
        total_slides = metadata.get("slides")
        try:
            total_slides_int = int(total_slides) if total_slides is not None else None
        except Exception:
            total_slides_int = None
        selected, range_warnings = _parse_slide_ranges(
            slide_ranges,
            total_slides_int,
            MAX_INDEX_SLIDES,
            action="searched",
        )
        selected_set = set(selected)
        warnings.extend(range_warnings)
        units = [unit for unit in units if int(unit.get("slide") or 0) in selected_set]
        navigation = [entry for entry in navigation if int(entry.get("slide") or 0) in selected_set]
        metadata["slides_searched"] = _format_page_selection(selected)

    hits: List[Dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("text") or "")
        section_path = [str(part) for part in (unit.get("section_path") or []) if str(part).strip()]
        section_text = " > ".join(section_path)
        score = _score_search_text(text, query, terms)
        if section_text:
            score += 2 * _score_search_text(section_text, query, terms)
        if score <= 0:
            continue
        hit = {
            "location": unit.get("location") or "Indexed unit",
            "score": score,
            "snippet": _snippet_for_search_match(text, query, terms, context_chars),
        }
        if section_path:
            hit["section_path"] = section_path
            hit["section"] = section_path[-1]
        for key in ("pdf_page", "page", "paragraph", "slide", "line"):
            if key in unit:
                hit[key] = unit[key]
        hits.append(hit)
    hits = _rank_search_hits(hits, max_results)
    total_pages_value = metadata.get("pages")
    try:
        total_pages_int = int(total_pages_value) if total_pages_value is not None else None
    except Exception:
        total_pages_int = None
    total_slides_value = metadata.get("slides")
    try:
        total_slides_int = int(total_slides_value) if total_slides_value is not None else None
    except Exception:
        total_slides_int = None
    return {
        "kind": kind,
        "metadata": metadata,
        "warnings": warnings,
        "navigation_hits": _navigation_hits(navigation, query, terms, max_results),
        "hits": hits,
        "suggested_page_ranges": _suggested_pdf_page_ranges(hits, total_pages=total_pages_int) if kind == "pdf" else "",
        "suggested_slide_ranges": _suggested_pptx_slide_ranges(hits, total_slides=total_slides_int) if kind == "pptx" else "",
    }


def _search_pptx(
    path: Path,
    query: str,
    terms: List[str],
    max_results: int,
    context_chars: int,
    max_slides: int,
    slide_ranges: str = "",
    progress: ProgressFn | None = None,
) -> Dict[str, Any]:
    index = {
        "version": DOCUMENT_INDEX_VERSION,
        "created_at": "",
        **_index_pptx(path, max_slides=max_slides, progress=progress, slide_ranges=slide_ranges),
    }
    result = _search_document_index(
        index,
        query=query,
        terms=terms,
        max_results=max_results,
        context_chars=context_chars,
    )
    result.setdefault("metadata", {})
    result["metadata"]["index_used"] = "temporary"
    if str(slide_ranges or "").strip():
        result["metadata"]["slides_searched"] = str(index.get("metadata", {}).get("slides_indexed_selection") or "")
    return result


def _extract_pdf(
    path: Path,
    max_pages: int,
    page_ranges: str = "",
    progress: ProgressFn | None = None,
    ctx: ToolContext | None = None,
) -> ExtractedDocument:
    extracted = _extract_pdf_with_python(path, max_pages, page_ranges=page_ranges, progress=progress)
    if extracted is not None:
        if _pdf_extraction_has_text(extracted):
            return extracted
        ocr = _extract_pdf_with_vlm_ocr(path, max_pages, page_ranges=page_ranges, progress=progress, ctx=ctx)
        if ocr is not None:
            return ocr
        return extracted

    extracted = _extract_pdf_with_pdftotext(path, max_pages, page_ranges=page_ranges, progress=progress)
    if extracted is not None:
        if _pdf_extraction_has_text(extracted):
            return extracted
        ocr = _extract_pdf_with_vlm_ocr(path, max_pages, page_ranges=page_ranges, progress=progress, ctx=ctx)
        if ocr is not None:
            return ocr
        return extracted

    extracted = _extract_pdf_with_vlm_ocr(path, max_pages, page_ranges=page_ranges, progress=progress, ctx=ctx)
    if extracted is not None:
        return extracted

    return ExtractedDocument(
        kind="pdf",
        warnings=[
            "No PDF extractor is available. Install pypdf, pdftotext, or PyMuPDF to analyze PDF files.",
        ],
    )


def _paragraphs_from_xml(xml_bytes: bytes) -> List[str]:
    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for para in root.iter():
        if not para.tag.endswith("}p"):
            continue
        parts = []
        for node in para.iter():
            if node.tag.endswith("}t") and node.text:
                parts.append(node.text)
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _read_zip_xml(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > MAX_XML_PART_BYTES:
        raise ValueError(f"XML part is too large: {name} ({info.file_size} bytes)")
    return zf.read(name)


def _number_from_name(name: str) -> int:
    match = re.search(r"(\d+)", name)
    if not match:
        return 0
    return int(match.group(1))


def _extract_pptx(
    path: Path,
    max_slides: int,
    slide_ranges: str = "",
    progress: ProgressFn | None = None,
) -> ExtractedDocument:
    warnings: List[str] = []
    sections: List[Tuple[str, str]] = []
    selected_slide_numbers: List[int] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        slides = sorted(
            [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)],
            key=_number_from_name,
        )
        notes = {
            _number_from_name(name): name
            for name in names
            if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
        }

        if str(slide_ranges or "").strip():
            selected_slide_numbers, range_warnings = _parse_slide_ranges(
                slide_ranges,
                len(slides),
                max_slides,
                action="extracted",
            )
            warnings.extend(range_warnings)
        else:
            selected_slide_numbers = list(range(1, min(len(slides), max_slides) + 1))
            if len(slides) > max_slides:
                warnings.append(f"Only the first {max_slides} of {len(slides)} slides were extracted.")

        slides_to_extract = len(selected_slide_numbers)
        selection = _format_page_selection(selected_slide_numbers)
        _emit_progress(
            progress,
            f"Открыл презентацию `{path.name}`: {len(slides)} слайдов. Извлекаю текст со слайдов {selection}.",
        )
        for position, selected_slide in enumerate(selected_slide_numbers, start=1):
            if position == 1 or position == slides_to_extract or position % 20 == 0:
                _emit_progress(progress, f"Читаю презентацию `{path.name}`: слайд {selected_slide} ({position}/{slides_to_extract}).")
            slide_name = slides[selected_slide - 1]
            slide_num = _number_from_name(slide_name)
            try:
                slide_lines = _paragraphs_from_xml(_read_zip_xml(zf, slide_name))
            except Exception as exc:
                slide_lines = [f"[Could not extract slide text: {type(exc).__name__}: {exc}]"]

            parts = []
            if slide_lines:
                parts.append("\n".join(slide_lines))
            else:
                parts.append("[No extractable slide text]")

            notes_name = notes.get(slide_num)
            if notes_name:
                try:
                    note_lines = _paragraphs_from_xml(_read_zip_xml(zf, notes_name))
                except Exception as exc:
                    note_lines = [f"[Could not extract speaker notes: {type(exc).__name__}: {exc}]"]
                if note_lines:
                    parts.append("Speaker notes:\n" + "\n".join(note_lines))

            sections.append((f"Slide {selected_slide}", "\n\n".join(parts)))

    return ExtractedDocument(
        kind="pptx",
        metadata={"slides": str(len(slides)), "slides_extracted": _format_page_selection(selected_slide_numbers), "extractor": "zip+xml"},
        sections=sections,
        warnings=warnings,
    )


def _extract_docx(path: Path, progress: ProgressFn | None = None) -> ExtractedDocument:
    _emit_progress(progress, f"Открыл DOCX `{path.name}`. Извлекаю основной текст документа.")
    with zipfile.ZipFile(path) as zf:
        try:
            paragraphs = _paragraphs_from_xml(_read_zip_xml(zf, "word/document.xml"))
        except KeyError:
            paragraphs = []
    return ExtractedDocument(
        kind="docx",
        metadata={"extractor": "zip+xml"},
        sections=[("Document text", "\n".join(paragraphs) or "[No extractable text]")],
    )


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        xml_bytes = _read_zip_xml(zf, "xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml_bytes)
    strings: List[str] = []
    for item in root:
        if not item.tag.endswith("}si"):
            continue
        parts = [node.text or "" for node in item.iter() if node.tag.endswith("}t")]
        strings.append("".join(parts))
    return strings


def _xlsx_sheet_targets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook = ET.fromstring(_read_zip_xml(zf, "xl/workbook.xml"))
    rels: Dict[str, str] = {}
    try:
        rel_root = ET.fromstring(_read_zip_xml(zf, "xl/_rels/workbook.xml.rels"))
        for rel in rel_root:
            rel_id = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if not rel_id or not target:
                continue
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            rels[rel_id] = target
    except KeyError:
        pass

    sheets: List[Tuple[str, str]] = []
    for sheet in workbook.iter():
        if not sheet.tag.endswith("}sheet"):
            continue
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
        target = rels.get(rel_id)
        if target:
            sheets.append((name, target))
    return sheets


def _xlsx_cell_text(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    formula = ""
    value = ""
    for node in cell:
        if node.tag.endswith("}f") and node.text:
            formula = node.text.strip()
        elif node.tag.endswith("}v") and node.text is not None:
            value = node.text.strip()
        elif cell_type == "inlineStr" and node.tag.endswith("}is"):
            value = "".join(t.text or "" for t in node.iter() if t.tag.endswith("}t")).strip()

    if cell_type == "s" and value.isdigit():
        idx = int(value)
        value = shared_strings[idx] if 0 <= idx < len(shared_strings) else value
    elif cell_type == "b":
        value = "TRUE" if value == "1" else "FALSE" if value == "0" else value

    if formula:
        if value:
            return f"={formula} -> {value}"
        return f"={formula}"
    return value


def _extract_xlsx(path: Path, max_sheets: int = DEFAULT_MAX_XLSX_SHEETS, progress: ProgressFn | None = None) -> ExtractedDocument:
    warnings: List[str] = []
    sections: List[Tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        shared_strings = _xlsx_shared_strings(zf)
        sheets = _xlsx_sheet_targets(zf)
        if not sheets:
            return ExtractedDocument(
                kind="xlsx",
                metadata={"extractor": "zip+xml"},
                warnings=["Workbook did not expose any worksheets."],
            )

        sheets_to_extract = min(len(sheets), max_sheets)
        _emit_progress(
            progress,
            f"Открыл XLSX `{path.name}`: {len(sheets)} листов. Извлекаю видимые значения и формулы из {sheets_to_extract} листов.",
        )

        for idx, (sheet_name, target) in enumerate(sheets[:max_sheets], start=1):
            if idx == 1 or idx == sheets_to_extract or idx % 10 == 0:
                _emit_progress(progress, f"Читаю XLSX `{path.name}`: лист {idx}/{sheets_to_extract}.")
            try:
                root = ET.fromstring(_read_zip_xml(zf, target))
            except Exception as exc:
                warnings.append(f"{sheet_name}: could not read worksheet: {type(exc).__name__}: {exc}")
                continue

            lines: List[str] = []
            cells_seen = 0
            rows_seen = 0
            truncated = False
            for row in root.iter():
                if not row.tag.endswith("}row"):
                    continue
                rows_seen += 1
                if rows_seen > MAX_XLSX_ROWS_PER_SHEET:
                    truncated = True
                    break
                row_values: List[str] = []
                for cell in row:
                    if not cell.tag.endswith("}c"):
                        continue
                    cells_seen += 1
                    if cells_seen > MAX_XLSX_CELLS_PER_SHEET:
                        truncated = True
                        break
                    ref = cell.attrib.get("r", "")
                    text = _xlsx_cell_text(cell, shared_strings)
                    if text:
                        row_values.append(f"{ref}={text}" if ref else text)
                if row_values:
                    lines.append(" | ".join(row_values))
                if truncated:
                    break
            if truncated:
                warnings.append(
                    f"{sheet_name}: only first {MAX_XLSX_ROWS_PER_SHEET} rows / {MAX_XLSX_CELLS_PER_SHEET} cells were extracted."
                )
            sections.append((f"Sheet {idx}: {sheet_name}", "\n".join(lines) or "[No visible cell values]"))

    if len(sheets) > max_sheets:
        warnings.append(f"Only the first {max_sheets} of {len(sheets)} sheets were extracted.")

    return ExtractedDocument(
        kind="xlsx",
        metadata={"sheets": str(len(sheets)), "extractor": "zip+xml"},
        sections=sections,
        warnings=warnings,
    )


def _extract_zip(
    path: Path,
    max_pages: int,
    max_slides: int,
    max_archive_files: int,
    page_ranges: str = "",
    slide_ranges: str = "",
    progress: ProgressFn | None = None,
    ctx: ToolContext | None = None,
) -> ExtractedDocument:
    warnings: List[str] = []
    sections: List[Tuple[str, str]] = []
    analyzed = 0
    skipped = 0
    total_uncompressed = 0
    listed: List[str] = []
    scan_infos: List[zipfile.ZipInfo] = []

    _emit_progress(progress, f"Открыл ZIP `{path.name}`. Смотрю, какие документы лежат внутри.")

    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            safe_name = _zip_member_safe_name(info.filename)
            if not safe_name:
                skipped += 1
                warnings.append(f"Skipped unsafe archive entry: {info.filename}")
                continue
            total_uncompressed += int(info.file_size or 0)
            if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                warnings.append(f"Stopped after archive total exceeded {MAX_ARCHIVE_TOTAL_BYTES} bytes.")
                break
            listed.append(f"{safe_name} ({info.file_size} bytes)")
            scan_infos.append(info)

        supported_count = sum(
            1
            for info in scan_infos
            if Path(_zip_member_safe_name(info.filename)).suffix.lower() in DOCUMENT_EXTENSIONS - ARCHIVE_EXTENSIONS
        )
        _emit_progress(
            progress,
            f"В архиве `{path.name}`: {len(infos)} файлов, для анализа подходит {supported_count}. Начинаю извлекать текст.",
        )

        with tempfile.TemporaryDirectory(prefix="ouroboros_zip_") as tmp:
            tmp_root = Path(tmp).resolve()
            for info in scan_infos:
                if analyzed >= max_archive_files:
                    warnings.append(f"Only the first {max_archive_files} supported archive files were analyzed.")
                    break
                safe_name = _zip_member_safe_name(info.filename)
                if not safe_name:
                    continue
                suffix = Path(safe_name).suffix.lower()
                if suffix in ARCHIVE_EXTENSIONS:
                    skipped += 1
                    warnings.append(f"Nested archive skipped: {safe_name}")
                    continue
                if suffix not in DOCUMENT_EXTENSIONS:
                    skipped += 1
                    continue
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    skipped += 1
                    warnings.append(f"Skipped oversized archive entry: {safe_name} ({info.file_size} bytes)")
                    continue

                if analyzed < 5:
                    _emit_progress(progress, f"Нашёл в ZIP `{safe_name}`. Распаковываю и читаю содержимое.")
                elif analyzed == 5:
                    _emit_progress(progress, "В архиве ещё есть подходящие документы; продолжаю разбор без лишних сообщений.")

                member_path = (tmp_root / safe_name).resolve()
                try:
                    member_path.relative_to(tmp_root)
                except ValueError:
                    skipped += 1
                    warnings.append(f"Skipped unsafe archive entry: {info.filename}")
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                member_path.write_bytes(zf.read(info))

                extracted = _extract_document(
                    member_path,
                    max_pages=max_pages,
                    max_slides=max_slides,
                    max_archive_files=max_archive_files,
                    allow_archives=False,
                    page_ranges=page_ranges,
                    slide_ranges=slide_ranges,
                    progress=progress,
                    ctx=ctx,
                )
                analyzed += 1
                for warning in extracted.warnings:
                    warnings.append(f"{safe_name}: {warning}")
                if extracted.sections:
                    for title, text in extracted.sections:
                        sections.append((f"{safe_name} / {title}", text))
                else:
                    sections.append((safe_name, "[No extractable content]"))

    if listed:
        sections.insert(0, ("Archive contents", "\n".join(listed[:200])))
    if skipped:
        warnings.append(f"Skipped {skipped} unsupported, nested, or unsafe archive entries.")

    _emit_progress(progress, f"Разбор ZIP `{path.name}` завершён: проанализировано файлов {analyzed}, пропущено {skipped}.")

    return ExtractedDocument(
        kind="zip",
        metadata={
            "entries": str(len(listed)),
            "analyzed_files": str(analyzed),
            "extractor": "zip",
        },
        sections=sections,
        warnings=warnings,
    )


def _extract_document(
    path: Path,
    max_pages: int,
    max_slides: int,
    max_archive_files: int = MAX_ARCHIVE_FILES,
    allow_archives: bool = True,
    page_ranges: str = "",
    slide_ranges: str = "",
    progress: ProgressFn | None = None,
    ctx: ToolContext | None = None,
) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, max_pages, page_ranges=page_ranges, progress=progress, ctx=ctx)
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image(path, progress=progress, ctx=ctx)
    if suffix == ".pptx":
        return _extract_pptx(path, max_slides, slide_ranges=slide_ranges, progress=progress)
    if suffix == ".docx":
        return _extract_docx(path, progress=progress)
    if suffix == ".xlsx":
        return _extract_xlsx(path, max_sheets=min(max_slides, DEFAULT_MAX_XLSX_SHEETS), progress=progress)
    if suffix == ".zip" and allow_archives:
        return _extract_zip(
            path,
            max_pages=max_pages,
            max_slides=max_slides,
            max_archive_files=max_archive_files,
            page_ranges=page_ranges,
            slide_ranges=slide_ranges,
            progress=progress,
            ctx=ctx,
        )
    if suffix in TEXT_EXTENSIONS:
        _emit_progress(progress, f"Открыл текстовый файл `{path.name}`. Читаю содержимое.")
        return _read_plain_text(path)
    if suffix == ".ppt":
        return ExtractedDocument(
            kind="ppt",
            warnings=["Legacy .ppt files are not supported. Convert the file to .pptx first."],
        )
    return ExtractedDocument(
        kind=suffix.lstrip(".") or "unknown",
        warnings=[
            "Unsupported document type. Supported: PDF, ZIP, PPTX, DOCX, XLSX, images (PNG/JPEG/WEBP/GIF/BMP), TXT, MD, CSV, JSON, HTML, XML, code files.",
        ],
    )


def _analysis_instruction(analysis_type: str, question: str) -> str:
    if analysis_type == "summary":
        return "Summarize the document: main thesis, key points, important details, and gaps."
    if analysis_type == "critique":
        return "Critique the document: strengths, weak claims, missing evidence, contradictions, and concrete improvements."
    if analysis_type == "extract_tasks":
        return "Extract actionable tasks, owners if mentioned, deadlines if mentioned, risks, and dependencies."
    if analysis_type == "answer_question":
        if question:
            return f"Answer this question using only the extracted document content: {question}"
        return "Answer the user's question using only the extracted document content."
    return "Return the extracted document content without additional interpretation."


def _format_result(
    path: Path,
    source: str,
    result: ExtractedDocument,
    analysis_type: str,
    question: str,
    max_chars: int,
) -> str:
    lines = [
        "# Document Analysis",
        "",
        f"- source: {source}",
        f"- file: {path.name}",
        f"- type: {result.kind}",
        f"- size_bytes: {path.stat().st_size}",
    ]
    for key, value in sorted(result.metadata.items()):
        lines.append(f"- {key}: {value}")
    if result.warnings:
        lines.append("- warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.extend([
        "",
        "## Requested analysis",
        _analysis_instruction(analysis_type, question),
        "",
        "## Extracted content",
    ])

    if result.sections:
        for title, text in result.sections:
            lines.extend(["", f"### {title}", text.strip() or "[Empty]"])
    else:
        lines.append("[No extractable content]")

    raw = "\n".join(lines).strip() + "\n"
    clipped = clip_text(raw, max_chars)
    if clipped != raw:
        clipped += f"\n\n[Output clipped to {max_chars} characters. Increase max_chars or narrow page_ranges/slide_ranges/max_pages/max_slides for more content.]\n"
    return clipped


def _format_search_result(path: Path, source: str, query: str, result: Dict[str, Any], tool_path: str = "") -> str:
    hits = list(result.get("hits") or [])
    navigation_hits = list(result.get("navigation_hits") or [])
    suggested_page_ranges = str(result.get("suggested_page_ranges") or "").strip()
    suggested_slide_ranges = str(result.get("suggested_slide_ranges") or "").strip()
    safe_query = re.sub(r"\s+", " ", str(query or "")).strip()
    tool_path = str(tool_path or path.name)
    lines = [
        "# Document Search",
        "",
        f"- source: {source}",
        f"- file: {path.name}",
        f"- type: {result.get('kind') or path.suffix.lower().lstrip('.') or 'unknown'}",
        f"- query: {safe_query}",
        f"- size_bytes: {path.stat().st_size}",
        f"- results: {len(hits)}",
    ]
    for key, value in sorted(dict(result.get("metadata") or {}).items()):
        lines.append(f"- {key}: {value}")
    warnings = list(result.get("warnings") or [])
    if warnings:
        lines.append("- warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    if navigation_hits:
        lines.extend(["", "## Navigation hits"])
        for hit in navigation_hits:
            pdf_page = hit.get("pdf_page")
            printed_page = hit.get("printed_page")
            page = hit.get("page")
            slide = hit.get("slide")
            if pdf_page and printed_page and pdf_page != printed_page:
                loc = f"pdf p. {pdf_page} / printed p. {printed_page}"
            elif pdf_page:
                loc = f"pdf p. {pdf_page}"
            elif page:
                loc = f"p. {page}"
            elif slide:
                loc = f"slide {slide}"
            else:
                loc = "p. ?"
            lines.append(f"- {loc}: {hit.get('title')} (score {hit.get('score')})")

    lines.extend(["", "## Search results"])
    if hits:
        for hit in hits:
            section_path = hit.get("section_path") or []
            section_note = f" - {' > '.join(section_path)}" if section_path else ""
            lines.extend([
                "",
                f"### {hit.get('location')}{section_note} (score {hit.get('score')})",
                str(hit.get("snippet") or "[Empty]"),
            ])
    else:
        lines.append("[No lexical matches found. Try synonyms, ticker/issuer names, contract terms, or a narrower query.]")

    if suggested_page_ranges:
        lines.extend([
            "",
            "## Suggested next step",
            (
                "For a PDF answer, call "
                f"`analyze_document(path={tool_path!r}, source={source!r}, analysis_type='answer_question', "
                f"question={str(query)!r}, page_ranges={suggested_page_ranges!r}, max_chars=60000)`."
            ),
        ])
    elif suggested_slide_ranges:
        lines.extend([
            "",
            "## Suggested next step",
            (
                "For a PPTX answer, call "
                f"`analyze_document(path={tool_path!r}, source={source!r}, analysis_type='answer_question', "
                f"question={str(query)!r}, slide_ranges={suggested_slide_ranges!r}, max_chars=60000)`."
            ),
        ])
    elif str(result.get("kind") or "") == "docx":
        lines.extend([
            "",
            "## Suggested next step",
            "Use the paragraph-numbered snippets above as the working evidence. DOCX page numbers are not stable until rendered.",
        ])

    return "\n".join(lines).strip() + "\n"


def _format_index_result(index: Dict[str, Any], index_path_rel: str) -> str:
    units = list(index.get("units") or [])
    navigation = list(index.get("navigation") or [])
    entities = dict(index.get("entities") or {})
    lines = [
        "# Document Index",
        "",
        f"- source: {index.get('source')}",
        f"- file: {index.get('file_name')}",
        f"- path: {index.get('path')}",
        f"- type: {index.get('kind')}",
        f"- size_bytes: {index.get('size_bytes')}",
        f"- sha256: {str(index.get('sha256') or '')[:16]}...",
        f"- index_path: {index_path_rel}",
        f"- indexed_units: {len(units)}",
        f"- navigation_entries: {len(navigation)}",
    ]
    for key, value in sorted(dict(index.get("metadata") or {}).items()):
        lines.append(f"- {key}: {value}")

    warnings = list(index.get("warnings") or [])
    if warnings:
        lines.append("- warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    if navigation:
        lines.extend(["", "## Navigation preview"])
        for item in navigation[:40]:
            loc = ""
            if item.get("pdf_page") and item.get("printed_page") and item.get("pdf_page") != item.get("printed_page"):
                loc = f"pdf p. {item.get('pdf_page')} / printed p. {item.get('printed_page')}: "
            elif item.get("pdf_page"):
                loc = f"pdf p. {item.get('pdf_page')}: "
            elif item.get("printed_page"):
                loc = f"printed p. {item.get('printed_page')}: "
            elif item.get("page"):
                loc = f"p. {item.get('page')}: "
            elif item.get("paragraph"):
                loc = f"paragraph {item.get('paragraph')}: "
            elif item.get("slide"):
                loc = f"slide {item.get('slide')}: "
            elif item.get("line"):
                loc = f"line {item.get('line')}: "
            lines.append(f"- {loc}{item.get('title')}")
        if len(navigation) > 40:
            lines.append(f"[Navigation preview clipped to 40 of {len(navigation)} entries.]")

    if entities:
        lines.extend(["", "## Financial/entity hints"])
        for key, values in sorted(entities.items()):
            preview = ", ".join(str(value) for value in list(values)[:20])
            if len(values) > 20:
                preview += f", ... ({len(values)} total)"
            lines.append(f"- {key}: {preview}")

    lines.extend([
        "",
        "## Suggested next step",
        f"Use `search_document(path={str(index.get('path') or '')!r}, source={str(index.get('source') or 'drive')!r}, query='<keywords or question terms>')` to search this cached index.",
    ])
    return "\n".join(lines).strip() + "\n"


def _validate_download_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(f"Refusing to download from private or local address: {ip}")
    return parsed


def _filename_from_download(url: str, content_disposition: str, content_type: str) -> str:
    cd = content_disposition or ""
    match = re.search(r"filename\*=([^']*)''([^;]+)", cd, flags=re.I)
    if match:
        return _safe_filename(urllib.parse.unquote(match.group(2)))
    match = re.search(r'filename="?([^";]+)"?', cd, flags=re.I)
    if match:
        return _safe_filename(match.group(1))

    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path or "")).name
    if not name:
        name = "download"
    safe = _safe_filename(name)
    if "." not in safe:
        if "zip" in content_type:
            safe += ".zip"
        elif "pdf" in content_type:
            safe += ".pdf"
    return safe


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find free filename for {path.name}")


def _download_url_to_drive(
    ctx: ToolContext,
    url: str,
    output_path: str = "",
    max_bytes: int = MAX_FILE_BYTES,
) -> str:
    import requests

    _emit_progress(ctx.emit_progress_fn, "Скачиваю файл по ссылке в рабочую папку. Если сервер отдаёт загрузку вместо страницы, я заберу сам файл.")
    parsed = _validate_download_url(url)
    max_bytes = _clean_limit(max_bytes, MAX_FILE_BYTES, 1_000, MAX_FILE_BYTES)
    with requests.get(parsed.geturl(), stream=True, allow_redirects=True, timeout=(10, 60)) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        filename = _filename_from_download(parsed.geturl(), resp.headers.get("content-disposition", ""), content_type)
        content_length = resp.headers.get("content-length")
        size_note = ""
        if content_length and str(content_length).isdigit():
            size_mb = int(content_length) / (1024 * 1024)
            size_note = f" Размер около {size_mb:.1f} MB."
        _emit_progress(ctx.emit_progress_fn, f"Файл найден: `{filename}`.{size_note} Сохраняю в workspace.")
        if output_path and output_path.strip():
            rel = safe_relpath(output_path)
            if rel.endswith("/"):
                rel = rel + filename
            elif not Path(rel).suffix:
                rel = str(Path(rel) / filename)
        else:
            day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            rel = str(Path("downloads") / day / filename)

        out_path = _dedupe_path(_safe_output_path(ctx.drive_root, rel))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    out_path.unlink(missing_ok=True)
                    raise ValueError(f"Download exceeded max_bytes={max_bytes}")
                f.write(chunk)

    rel_saved = str(out_path.relative_to(ctx.drive_root.resolve()))
    _emit_progress(ctx.emit_progress_fn, f"Файл сохранён: `{rel_saved}`. Теперь его можно открыть через `analyze_document`.")
    return (
        "OK: downloaded file\n"
        f"- path: {rel_saved}\n"
        f"- size_bytes: {total}\n"
        f"- content_type: {content_type or 'unknown'}\n"
        "Next: call analyze_document(path='<path>', source='drive') to inspect it."
    )


def _extract_archive(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    output_dir: str = "",
    max_files: int = MAX_ARCHIVE_FILES,
    overwrite: bool = False,
) -> str:
    _emit_progress(ctx.emit_progress_fn, f"Распаковываю архив `{path}` в рабочую папку.")
    archive_path = _resolve_document_path(ctx, path, source)
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("Only .zip archives are supported")
    max_files = _clean_limit(max_files, MAX_ARCHIVE_FILES, 1, 500)

    base_name = _safe_filename(archive_path.stem, fallback="archive")
    rel_dir = safe_relpath(output_dir) if output_dir else str(Path("extracted") / base_name)
    out_root = _safe_output_path(ctx.drive_root, rel_dir)
    if out_root.exists() and not overwrite:
        out_root = _dedupe_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    extracted: List[str] = []
    skipped: List[str] = []
    total = 0
    with zipfile.ZipFile(archive_path) as zf:
        file_count = sum(1 for info in zf.infolist() if not info.is_dir())
        _emit_progress(ctx.emit_progress_fn, f"Открыл ZIP: внутри {file_count} файлов. Безопасно извлекаю до {max_files}.")
        for info in zf.infolist():
            if info.is_dir():
                continue
            if len(extracted) >= max_files:
                skipped.append(f"... stopped after max_files={max_files}")
                break
            safe_name = _zip_member_safe_name(info.filename)
            if not safe_name:
                skipped.append(f"unsafe: {info.filename}")
                continue
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                skipped.append(f"oversized: {safe_name}")
                continue
            total += int(info.file_size or 0)
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                skipped.append(f"archive total exceeded {MAX_ARCHIVE_TOTAL_BYTES} bytes")
                break
            target = (out_root / safe_name).resolve()
            try:
                target.relative_to(out_root.resolve())
            except ValueError:
                skipped.append(f"unsafe: {info.filename}")
                continue
            if target.exists() and not overwrite:
                target = _dedupe_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            extracted.append(str(target.relative_to(ctx.drive_root.resolve())))

    _emit_progress(ctx.emit_progress_fn, f"Архив распакован: извлечено файлов {len(extracted)}, пропущено {len(skipped)}.")
    lines = [
        "OK: archive extracted",
        f"- archive: {path}",
        f"- output_dir: {out_root.relative_to(ctx.drive_root.resolve())}",
        f"- extracted_files: {len(extracted)}",
    ]
    if extracted:
        lines.append("")
        lines.append("## Extracted files")
        lines.extend(f"- {item}" for item in extracted[:100])
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        lines.extend(f"- {item}" for item in skipped[:50])
    return "\n".join(lines)


def _analyze_document(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    analysis_type: str = "summary",
    question: str = "",
    page_ranges: str = "",
    slide_ranges: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    analysis_type = _normalize_analysis_type(analysis_type)
    source = (source or "drive").strip().lower()
    max_chars = _clean_limit(max_chars, DEFAULT_MAX_CHARS, 2_000, 100_000)
    max_pages = _clean_limit(max_pages, DEFAULT_MAX_PAGES, 1, 200)
    max_slides = _clean_limit(max_slides, DEFAULT_MAX_SLIDES, 1, 300)
    max_archive_files = _clean_limit(max_archive_files, MAX_ARCHIVE_FILES, 1, 200)

    document_path = _resolve_document_path(ctx, path, source)
    _emit_progress(ctx.emit_progress_fn, f"Открыл файл `{document_path.name}`. Определяю тип и готовлю извлечение текста.")
    extracted = _extract_document(
        document_path,
        max_pages=max_pages,
        max_slides=max_slides,
        max_archive_files=max_archive_files,
        page_ranges=page_ranges,
        slide_ranges=slide_ranges,
        progress=ctx.emit_progress_fn,
        ctx=ctx,
    )
    _emit_progress(ctx.emit_progress_fn, f"Извлечение текста из `{document_path.name}` завершено. Собираю ответ по твоему запросу.")
    return _format_result(document_path, source, extracted, analysis_type, question or "", max_chars)


def _index_document(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    max_pages: int = DEFAULT_INDEX_MAX_PAGES,
    max_slides: int = DEFAULT_INDEX_MAX_SLIDES,
    force_rebuild: bool = False,
) -> str:
    source = (source or "drive").strip().lower()
    max_pages = _clean_limit(max_pages, DEFAULT_INDEX_MAX_PAGES, 1, MAX_SEARCH_PAGES)
    max_slides = _clean_limit(max_slides, DEFAULT_INDEX_MAX_SLIDES, 1, MAX_INDEX_SLIDES)
    document_path = _resolve_document_path(ctx, path, source)
    cached, file_hash, index_path = _load_document_index(ctx, document_path, source, path)
    if cached is not None and not force_rebuild:
        _emit_progress(ctx.emit_progress_fn, f"Индекс `{document_path.name}` уже есть. Использую сохранённую карту документа.")
        return _format_index_result(cached, _index_rel_path(ctx, index_path))

    _emit_progress(ctx.emit_progress_fn, f"Строю индекс большого документа `{document_path.name}`. Это может занять пару минут.")
    index = _build_document_index_payload(
        document_path,
        source=source,
        tool_path=path,
        file_hash=file_hash,
        max_pages=max_pages,
        max_slides=max_slides,
        progress=ctx.emit_progress_fn,
    )
    index_path = _write_document_index(ctx, index)
    _emit_progress(ctx.emit_progress_fn, f"Индекс документа `{document_path.name}` сохранён: `{_index_rel_path(ctx, index_path)}`.")
    return _format_index_result(index, _index_rel_path(ctx, index_path))


def _search_document(
    ctx: ToolContext,
    path: str,
    query: str,
    source: str = "drive",
    page_ranges: str = "",
    slide_ranges: str = "",
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    max_pages: int = DEFAULT_SEARCH_MAX_PAGES,
    max_slides: int = DEFAULT_INDEX_MAX_SLIDES,
    context_chars: int = DEFAULT_SEARCH_CONTEXT_CHARS,
) -> str:
    query = str(query or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")
    source = (source or "drive").strip().lower()
    max_results = _clean_limit(max_results, DEFAULT_SEARCH_MAX_RESULTS, 1, MAX_SEARCH_RESULTS)
    max_pages = _clean_limit(max_pages, DEFAULT_SEARCH_MAX_PAGES, 1, MAX_SEARCH_PAGES)
    max_slides = _clean_limit(max_slides, DEFAULT_INDEX_MAX_SLIDES, 1, MAX_INDEX_SLIDES)
    context_chars = _clean_limit(context_chars, DEFAULT_SEARCH_CONTEXT_CHARS, 120, MAX_SEARCH_CONTEXT_CHARS)
    terms = _search_terms(query)

    document_path = _resolve_document_path(ctx, path, source)
    suffix = document_path.suffix.lower()
    _emit_progress(ctx.emit_progress_fn, f"Открыл файл `{document_path.name}`. Запускаю поиск по документу: `{query}`.")
    cached, _file_hash, index_path = _load_document_index(ctx, document_path, source, path)
    if cached is not None:
        _emit_progress(ctx.emit_progress_fn, f"Нашёл сохранённый индекс `{_index_rel_path(ctx, index_path)}`. Ищу по нему без повторного разбора файла.")
        result = _search_document_index(
            cached,
            query=query,
            terms=terms,
            max_results=max_results,
            context_chars=context_chars,
            page_ranges=page_ranges,
            slide_ranges=slide_ranges,
        )
        result.setdefault("metadata", {})
        result["metadata"]["index_path"] = _index_rel_path(ctx, index_path)
        _emit_progress(ctx.emit_progress_fn, f"Поиск в индексе `{document_path.name}` завершён. Возвращаю лучшие совпадения.")
        return _format_search_result(document_path, source, query, result, tool_path=path)

    if suffix == ".pdf":
        result = _search_pdf(
            document_path,
            query=query,
            terms=terms,
            max_results=max_results,
            context_chars=context_chars,
            max_pages=max_pages,
            page_ranges=page_ranges,
            progress=ctx.emit_progress_fn,
        )
    elif suffix == ".pptx":
        result = _search_pptx(
            document_path,
            query=query,
            terms=terms,
            max_results=max_results,
            context_chars=context_chars,
            max_slides=max_slides,
            slide_ranges=slide_ranges,
            progress=ctx.emit_progress_fn,
        )
    elif suffix == ".docx":
        result = _search_docx(
            document_path,
            query=query,
            terms=terms,
            max_results=max_results,
            context_chars=context_chars,
            progress=ctx.emit_progress_fn,
        )
    elif suffix in TEXT_EXTENSIONS:
        result = _search_text_file(
            document_path,
            query=query,
            terms=terms,
            max_results=max_results,
            context_chars=context_chars,
            progress=ctx.emit_progress_fn,
        )
    else:
        result = {
            "kind": suffix.lstrip(".") or "unknown",
            "metadata": {},
            "warnings": [
                "Unsupported document search type. Supported: PDF, PPTX, DOCX, TXT, MD, CSV, JSON, HTML, XML, and code-like files.",
            ],
            "navigation_hits": [],
            "hits": [],
            "suggested_page_ranges": "",
        }

    _emit_progress(ctx.emit_progress_fn, f"Поиск в `{document_path.name}` завершён. Возвращаю лучшие совпадения.")
    return _format_search_result(document_path, source, query, result, tool_path=path)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("analyze_document", {
            "name": "analyze_document",
            "description": (
                "Extract text and structure from PDF, scanned PDF via rendered-page VLM OCR, ZIP, PPTX, DOCX, images, and text-like files for analysis. "
                "Use it before summarizing documents, critiquing presentations, answering questions "
                "about uploaded files, OCR-reading photos/screenshots of tables, or extracting action items. "
                "For long PDFs, it returns a navigation "
                "map from bookmarks or a table of contents when available; use page_ranges to read specific "
                "later sections. For large PPTX decks, use slide_ranges after search_document finds relevant slides."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Document path relative to the selected source root."},
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read from user Drive workspace by default. Repo source is admin-only.",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "critique", "extract_tasks", "answer_question", "raw"],
                    "default": "summary",
                    "description": "How the caller intends to analyze the extracted content.",
                },
                "question": {
                    "type": "string",
                    "description": "Question to answer when analysis_type is answer_question.",
                },
                "page_ranges": {
                    "type": "string",
                    "description": (
                        "Optional 1-based PDF page ranges to extract, e.g. '15-21,48-55'. "
                        "Use this after a table of contents points to later sections, instead of asking the user to split the PDF."
                    ),
                },
                "slide_ranges": {
                    "type": "string",
                    "description": (
                        "Optional 1-based PPTX slide ranges to extract, e.g. '12-15,31'. "
                        "Use this after search_document finds relevant slides in a large presentation."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "default": DEFAULT_MAX_CHARS,
                    "description": "Maximum characters returned to the LLM.",
                },
                "max_pages": {
                    "type": "integer",
                    "default": DEFAULT_MAX_PAGES,
                    "description": "Maximum PDF pages to extract, including pages selected by page_ranges.",
                },
                "max_slides": {
                    "type": "integer",
                    "default": DEFAULT_MAX_SLIDES,
                    "description": "Maximum PPTX slides to extract.",
                },
                "max_archive_files": {
                    "type": "integer",
                    "default": MAX_ARCHIVE_FILES,
                    "description": "Maximum supported files to analyze inside a ZIP archive.",
                },
            }, "required": ["path"]},
        }, _analyze_document, timeout_sec=180),
        ToolEntry("index_document", {
            "name": "index_document",
            "description": (
                "Build and cache a navigation/search index for a large PDF, PPTX, DOCX, or text-like document. "
                "Use it before repeated questions over long financial/legal documents or large presentations. "
                "The index stores page/slide/paragraph text units, bookmarks/table-of-contents/headings/slide titles, "
                "speaker notes, and financial entity hints."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Document path relative to the selected source root."},
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read from user Drive workspace by default. Repo source is admin-only.",
                },
                "max_pages": {
                    "type": "integer",
                    "default": DEFAULT_INDEX_MAX_PAGES,
                    "description": "Maximum PDF pages to index.",
                },
                "max_slides": {
                    "type": "integer",
                    "default": DEFAULT_INDEX_MAX_SLIDES,
                    "description": "Maximum PPTX slides to index.",
                },
                "force_rebuild": {
                    "type": "boolean",
                    "default": False,
                    "description": "Rebuild even if a matching hash-based index already exists.",
                },
            }, "required": ["path"]},
        }, _index_document, timeout_sec=120),
        ToolEntry("search_document", {
            "name": "search_document",
            "description": (
                "Search inside a large PDF, PPTX, DOCX, or text-like document and return ranked snippets with locations. "
                "Use it when the user asks to find whether a long financial/legal document or presentation contains specific information. "
                "If index_document has already cached an index, this searches that index. For PDFs, it scans pages, "
                "also checks bookmarks/table-of-contents hits, and returns suggested page_ranges. For PPTX, it returns "
                "suggested slide_ranges for a follow-up analyze_document call."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Document path relative to the selected source root."},
                "query": {
                    "type": "string",
                    "description": "Search query, keywords, issuer names, ticker, covenant term, section title, or question keywords.",
                },
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read from user Drive workspace by default. Repo source is admin-only.",
                },
                "page_ranges": {
                    "type": "string",
                    "description": "Optional 1-based PDF page ranges to search, e.g. '15-21,48-55'. Leave empty to scan from the beginning up to max_pages.",
                },
                "slide_ranges": {
                    "type": "string",
                    "description": "Optional 1-based PPTX slide ranges to search, e.g. '12-15,31'. Leave empty to scan from the beginning up to max_slides when no cached index exists.",
                },
                "max_results": {
                    "type": "integer",
                    "default": DEFAULT_SEARCH_MAX_RESULTS,
                    "description": "Maximum ranked matches to return.",
                },
                "max_pages": {
                    "type": "integer",
                    "default": DEFAULT_SEARCH_MAX_PAGES,
                    "description": "Maximum PDF pages to search when page_ranges is empty.",
                },
                "max_slides": {
                    "type": "integer",
                    "default": DEFAULT_INDEX_MAX_SLIDES,
                    "description": "Maximum PPTX slides to search when no cached index exists.",
                },
                "context_chars": {
                    "type": "integer",
                    "default": DEFAULT_SEARCH_CONTEXT_CHARS,
                    "description": "Approximate snippet size around each match.",
                },
            }, "required": ["path", "query"]},
        }, _search_document, timeout_sec=90),
        ToolEntry("extract_archive", {
            "name": "extract_archive",
            "description": (
                "Safely extract a ZIP archive from the user's Drive workspace. "
                "Use when the user needs files unpacked for later inspection or when a ZIP contains multiple reports."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "ZIP path relative to the selected source root."},
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read archive from user Drive by default. Repo source is admin-only.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory relative to user Drive. Defaults to extracted/<archive-name>.",
                },
                "max_files": {
                    "type": "integer",
                    "default": MAX_ARCHIVE_FILES,
                    "description": "Maximum archive entries to extract.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite existing extracted files instead of creating deduplicated names.",
                },
            }, "required": ["path"]},
        }, _extract_archive, timeout_sec=60),
        ToolEntry("download_url_to_drive", {
            "name": "download_url_to_drive",
            "description": (
                "Download a public HTTP/HTTPS file URL into the user's Drive workspace. "
                "Use for direct PDF/ZIP/DOCX/PPTX links that browser automation cannot read because they start a download."
            ),
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "Public HTTP/HTTPS URL to download."},
                "output_path": {
                    "type": "string",
                    "description": "Optional relative output path or directory in user Drive. Defaults to downloads/YYYY-MM-DD/<filename>.",
                },
                "max_bytes": {
                    "type": "integer",
                    "default": MAX_FILE_BYTES,
                    "description": "Maximum download size in bytes.",
                },
            }, "required": ["url"]},
        }, _download_url_to_drive, timeout_sec=90),
    ]
