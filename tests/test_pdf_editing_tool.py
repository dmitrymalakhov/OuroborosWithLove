import pathlib

import fitz
import pytest

from ouroboros.tools.pdf_editing import PDF_MIME_TYPE, _edit_pdf, _inspect_pdf_for_edit
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _ctx(repo_dir: pathlib.Path, drive_root: pathlib.Path) -> ToolContext:
    repo_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        repo_dir=repo_dir,
        drive_root=drive_root,
        current_chat_id=123,
        current_user_id=456,
        user_role="user",
    )


def _write_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Invoice amount: 100 RUB", fontsize=12)
    page.insert_text((72, 104), "Date: 2026-06-03", fontsize=12)
    doc.save(path)
    doc.close()


def _pdf_text(path: pathlib.Path) -> str:
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def test_inspect_pdf_for_edit_reports_pages_matches_and_form_section(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_pdf(drive / "invoice.pdf")

    result = _inspect_pdf_for_edit(_ctx(repo, drive), path="invoice.pdf", search_text="100 RUB")

    assert "# PDF Edit Inspection" in result
    assert "- pages: 1" in result
    assert "Invoice amount" in result
    assert "page 1, occurrence 1" in result
    assert "rect=[" in result
    assert "## Form Fields" in result


def test_edit_pdf_replaces_text_adds_overlay_preserves_original_and_queues_delivery(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_pdf(drive / "invoice.pdf")
    ctx = _ctx(repo, drive)

    result = _edit_pdf(
        ctx,
        path="invoice.pdf",
        output_path="exports/invoice-edited.pdf",
        operations=[
            {
                "type": "replace_text",
                "page": 1,
                "search_text": "100 RUB",
                "replacement_text": "125 RUB",
                "font_size": 12,
                "confirmed": True,
            },
            {
                "type": "add_text",
                "page": 1,
                "text": "Approved",
                "x": 72,
                "y": 136,
                "font_size": 12,
                "confidence": 0.95,
            },
        ],
    )

    assert "OK: PDF edited" in result
    assert "- applied: 2" in result
    assert (drive / "exports" / "invoice-edited.pdf").exists()
    assert "100 RUB" in _pdf_text(drive / "invoice.pdf")
    edited_text = _pdf_text(drive / "exports" / "invoice-edited.pdf")
    assert "125 RUB" in edited_text
    assert "Approved" in edited_text

    assert len(ctx.pending_events) == 1
    event = ctx.pending_events[0]
    assert event["type"] == "send_document"
    assert event["path"] == "exports/invoice-edited.pdf"
    assert event["mime_type"] == PDF_MIME_TYPE


def test_edit_pdf_does_not_create_output_without_confirmation(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_pdf(drive / "invoice.pdf")
    ctx = _ctx(repo, drive)

    result = _edit_pdf(
        ctx,
        path="invoice.pdf",
        output_path="exports/unconfirmed.pdf",
        operations=[
            {"type": "add_text", "page": 1, "text": "Unconfirmed", "x": 72, "y": 160},
        ],
    )

    assert "No PDF edits were applied" in result
    assert "missing confirmation" in result
    assert not (drive / "exports" / "unconfirmed.pdf").exists()
    assert ctx.pending_events == []


def test_pdf_editing_tools_reject_path_traversal(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_pdf(drive / "invoice.pdf")

    with pytest.raises(ValueError, match="Path traversal"):
        _inspect_pdf_for_edit(_ctx(repo, drive), path="../invoice.pdf")


def test_pdf_editing_pack_is_registered(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "drive")

    tools = registry.get_tools_by_pack("pdf_edit", include_dependencies=True)

    assert "inspect_pdf_for_edit" in tools
    assert "edit_pdf" in tools
    assert "analyze_document" in tools
