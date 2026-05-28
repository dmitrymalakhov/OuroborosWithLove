import pathlib

from ouroboros.tools.core import _send_file
from ouroboros.tools.registry import ToolContext


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


def test_send_file_queues_telegram_document(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "exports").mkdir()
    (drive / "exports" / "table.csv").write_text("name;inn\nACME;123\n", encoding="utf-8")
    ctx = _ctx(repo, drive)

    result = _send_file(
        ctx,
        path="exports/table.csv",
        caption="Excel-ready CSV",
        mime_type="text/csv",
    )

    assert "OK: file queued" in result
    assert len(ctx.pending_events) == 1
    event = ctx.pending_events[0]
    assert event["type"] == "send_document"
    assert event["chat_id"] == 123
    assert event["path"] == "exports/table.csv"
    assert event["filename"] == "table.csv"
    assert event["mime_type"] == "text/csv"
    assert event["drive_root"] == str(drive)


def test_send_file_rejects_missing_file(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    ctx = _ctx(repo, drive)

    result = _send_file(ctx, path="missing.csv")

    assert "File not found" in result
    assert ctx.pending_events == []
