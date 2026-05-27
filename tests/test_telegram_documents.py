import json

from supervisor.telegram import save_incoming_document


def test_save_incoming_document_writes_to_uploads_and_logs(tmp_path):
    meta = save_incoming_document(
        tmp_path,
        file_bytes=b"%PDF-1.4 test",
        original_name="Проход_к_поездам.pdf",
        mime_type="application/pdf",
        telegram_file_id="file-123",
        telegram_file_unique_id="unique-123",
        caption="podskazhi",
        message_id=42,
    )

    saved = tmp_path / meta["path"]
    assert saved.exists()
    assert saved.read_bytes() == b"%PDF-1.4 test"
    assert str(meta["path"]).startswith("uploads/")
    assert str(meta["path"]).endswith("_Проход_к_поездам.pdf")
    assert meta["mime_type"] == "application/pdf"

    upload_logs = (tmp_path / "logs" / "uploads.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(upload_logs) == 1
    logged = json.loads(upload_logs[0])
    assert logged["path"] == meta["path"]
    assert logged["telegram_file_unique_id"] == "unique-123"


def test_save_incoming_document_deduplicates_same_message_filename(tmp_path):
    first = save_incoming_document(
        tmp_path,
        file_bytes=b"first",
        original_name="../bad/name.pdf",
        mime_type="application/pdf",
        telegram_file_id="file-1",
        message_id=7,
    )
    second = save_incoming_document(
        tmp_path,
        file_bytes=b"second",
        original_name="../bad/name.pdf",
        mime_type="application/pdf",
        telegram_file_id="file-2",
        message_id=7,
    )

    assert first["path"] != second["path"]
    assert ".." not in first["path"]
    assert "/" in first["path"]
    assert (tmp_path / first["path"]).read_bytes() == b"first"
    assert (tmp_path / second["path"]).read_bytes() == b"second"
