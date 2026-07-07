"""Presentation export tools."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import safe_relpath


PDF_MIME_TYPE = "application/pdf"
MAX_CONVERT_FILE_BYTES = 100 * 1024 * 1024
LIBREOFFICE_TIMEOUT_SEC = 180


def _scope(ctx: ToolContext) -> dict:
    return {
        "user_id": ctx.current_user_id,
        "user_role": ctx.user_role,
        "drive_root": str(ctx.drive_root),
        "shared_drive_root": str(ctx.shared_drive_root or ctx.drive_root),
    }


def _clean_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def _resolve_pptx_input_path(ctx: ToolContext, path: str) -> tuple[pathlib.Path, str]:
    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty .pptx path in Drive")

    target = ctx.drive_path(path)
    root = ctx.drive_root.resolve()
    try:
        rel = str(target.relative_to(root))
    except ValueError:
        raise ValueError("Path traversal is not allowed.")

    if target.suffix.lower() != ".pptx":
        raise ValueError("Only .pptx files can be converted to PDF.")
    if not target.exists():
        raise FileNotFoundError(f"Presentation not found: {path}")
    if not target.is_file():
        raise ValueError(f"Presentation path is not a file: {path}")
    size = target.stat().st_size
    if size > MAX_CONVERT_FILE_BYTES:
        raise ValueError(f"Presentation is too large to convert: {size} bytes")
    return target, rel


def _resolve_pdf_output_path(
    ctx: ToolContext,
    output_path: str,
    source_rel: str,
    overwrite: bool,
) -> tuple[pathlib.Path, str]:
    if output_path:
        rel = safe_relpath(output_path)
        if not rel.lower().endswith(".pdf"):
            rel += ".pdf"
    else:
        rel = str(pathlib.PurePosixPath(source_rel).with_suffix(".pdf"))

    target = ctx.drive_path(rel)
    root = ctx.drive_root.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal is not allowed.")

    if target.exists() and target.is_dir():
        raise ValueError("output_path points to a directory")

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


def _find_libreoffice() -> str:
    env_path = os.environ.get("LIBREOFFICE_BIN", "").strip()
    candidates = [
        env_path,
        shutil.which("libreoffice") or "",
        shutil.which("soffice") or "",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    return ""


def _run_libreoffice_pdf_export(binary: str, source: pathlib.Path, out_dir: pathlib.Path) -> subprocess.CompletedProcess:
    profile_dir = out_dir / "lo-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source),
    ]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=LIBREOFFICE_TIMEOUT_SEC,
        check=False,
    )


def _process_output(proc: subprocess.CompletedProcess) -> str:
    output = "\n".join(part.strip() for part in [proc.stdout or "", proc.stderr or ""] if part.strip())
    return _clean_text(output, 1200)


def _convert_pptx_to_pdf(
    ctx: ToolContext,
    path: str,
    output_path: str = "",
    overwrite: bool = False,
    send_to_chat: bool = True,
) -> str:
    pptx_path, rel_source = _resolve_pptx_input_path(ctx, path)
    output, rel_output = _resolve_pdf_output_path(ctx, output_path, rel_source, bool(overwrite))

    binary = _find_libreoffice()
    if not binary:
        return "\n".join([
            "⚠️ LibreOffice is not available for PPTX to PDF conversion",
            f"- source_path: {rel_source}",
            "- output_path: not_created",
            "- install_hint: install LibreOffice or set LIBREOFFICE_BIN to the soffice/libreoffice executable",
            "- telegram_delivery: skipped",
        ])

    ctx.emit_progress_fn(f"Converting presentation `{pptx_path.name}` to PDF via LibreOffice.")
    tmp_root = ctx.drive_root / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix=".pptx2pdf-", dir=str(tmp_root)) as tmp_name:
            tmp_dir = pathlib.Path(tmp_name)
            proc = _run_libreoffice_pdf_export(binary, pptx_path, tmp_dir)
            converter_output = _process_output(proc)
            if proc.returncode != 0:
                lines = [
                    "⚠️ LibreOffice failed to convert PPTX to PDF",
                    f"- source_path: {rel_source}",
                    "- output_path: not_created",
                    f"- returncode: {proc.returncode}",
                    "- telegram_delivery: skipped",
                ]
                if converter_output:
                    lines.append(f"- converter_output: {converter_output}")
                return "\n".join(lines)

            generated = tmp_dir / f"{pptx_path.stem}.pdf"
            if not generated.exists():
                pdfs = [candidate for candidate in tmp_dir.glob("*.pdf") if candidate.is_file()]
                if len(pdfs) == 1:
                    generated = pdfs[0]
                else:
                    lines = [
                        "⚠️ LibreOffice did not produce a PDF file",
                        f"- source_path: {rel_source}",
                        "- output_path: not_created",
                        "- telegram_delivery: skipped",
                    ]
                    if converter_output:
                        lines.append(f"- converter_output: {converter_output}")
                    return "\n".join(lines)

            if output.exists():
                output.unlink()
            shutil.move(str(generated), str(output))
    except OSError as exc:
        return "\n".join([
            "⚠️ LibreOffice could not be started for PPTX to PDF conversion",
            f"- source_path: {rel_source}",
            "- output_path: not_created",
            f"- error: {_clean_text(exc, 300)}",
            "- telegram_delivery: skipped",
        ])
    except subprocess.TimeoutExpired:
        return "\n".join([
            "⚠️ LibreOffice timed out while converting PPTX to PDF",
            f"- source_path: {rel_source}",
            "- output_path: not_created",
            "- telegram_delivery: skipped",
        ])

    size = output.stat().st_size
    queued = False
    if send_to_chat and ctx.current_chat_id:
        ctx.pending_events.append({
            "type": "send_document",
            "chat_id": ctx.current_chat_id,
            "path": rel_output,
            "caption": output.stem,
            "filename": output.name,
            "mime_type": PDF_MIME_TYPE,
            **_scope(ctx),
        })
        queued = True

    return "\n".join([
        "OK: presentation converted to PDF",
        f"- source_path: {rel_source}",
        f"- output_path: {rel_output}",
        f"- size_bytes: {size}",
        f"- converter: {pathlib.Path(binary).name}",
        f"- telegram_delivery: {'queued' if queued else 'skipped'}",
    ])


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("convert_pptx_to_pdf", {
            "name": "convert_pptx_to_pdf",
            "description": (
                "Convert an existing .pptx presentation from the user's Drive workspace to PDF using "
                "LibreOffice headless, then optionally queue the PDF for Telegram delivery. Use when the "
                "user asks for a PDF version/export of a PowerPoint deck or of a presentation just created."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {
                    "type": "string",
                    "description": "Source .pptx path in the user's Drive workspace.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output .pdf path in Drive. Defaults to the source path with .pdf extension.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite output_path if it exists; otherwise create a deduplicated filename.",
                },
                "send_to_chat": {
                    "type": "boolean",
                    "default": True,
                    "description": "Queue the converted PDF for Telegram delivery when the task has an active chat.",
                },
            }, "required": ["path"]},
        }, _convert_pptx_to_pdf, timeout_sec=240),
    ]
