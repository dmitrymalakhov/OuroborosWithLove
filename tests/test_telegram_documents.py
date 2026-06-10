import json

from supervisor.telegram import save_incoming_audio, save_incoming_document, save_incoming_image
from supervisor.telegram_images import (
    format_image_task_text,
    image_attachment_metadata,
    save_telegram_image_upload,
)


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


def test_save_incoming_image_writes_to_uploads_and_logs(tmp_path):
    meta = save_incoming_image(
        tmp_path,
        file_bytes=b"\x89PNG image bytes",
        original_name="telegram_photo_12",
        mime_type="image/png",
        telegram_file_id="image-file",
        telegram_file_unique_id="image-unique",
        caption="убери фон",
        message_id=12,
    )

    saved = tmp_path / meta["path"]
    assert saved.exists()
    assert saved.read_bytes() == b"\x89PNG image bytes"
    assert meta["type"] == "telegram_image_saved"
    assert meta["mime_type"] == "image/png"
    assert str(meta["path"]).startswith("uploads/")
    assert str(meta["path"]).endswith("_telegram_photo_12.png")

    upload_logs = (tmp_path / "logs" / "uploads.jsonl").read_text(encoding="utf-8").strip().splitlines()
    logged = json.loads(upload_logs[0])
    assert logged["type"] == "telegram_image_saved"
    assert logged["telegram_file_unique_id"] == "image-unique"


def test_image_attachment_metadata_extracts_photo_and_image_document():
    photo = {
        "message_id": 31,
        "photo": [
            {"file_id": "small", "file_unique_id": "small-u"},
            {"file_id": "large", "file_unique_id": "large-u"},
        ],
    }
    assert image_attachment_metadata(photo) == {
        "file_id": "large",
        "original_name": "telegram_photo_31",
        "unique_id": "large-u",
        "mime_type": "",
    }

    image_document = {
        "message_id": 32,
        "document": {
            "file_id": "doc-image",
            "file_unique_id": "doc-u",
            "file_name": "input.webp",
            "mime_type": "image/webp",
        },
    }
    assert image_attachment_metadata(image_document)["original_name"] == "input.webp"
    assert image_attachment_metadata(image_document)["mime_type"] == "image/webp"


def test_save_telegram_image_upload_returns_image_data_and_saved_meta(tmp_path):
    class FakeTelegram:
        def download_file_bytes(self, file_id):
            assert file_id == "image-file"
            return b"\xff\xd8jpeg bytes", "image/jpeg", "photos/file_1.jpg"

    image_data, saved_image, failed = save_telegram_image_upload(
        FakeTelegram(),
        tmp_path,
        image_file_id="image-file",
        original_name="telegram_photo_77",
        telegram_file_unique_id="image-unique",
        caption="сделай ярче",
        message_id=77,
        now_iso="2026-06-03T00:00:00+00:00",
    )

    assert failed is False
    assert saved_image is not None
    assert image_data is not None
    assert image_data[1] == "image/jpeg"
    assert image_data[2] == "сделай ярче"
    assert str(saved_image["path"]).endswith("_telegram_photo_77.jpg")
    assert (tmp_path / saved_image["path"]).read_bytes() == b"\xff\xd8jpeg bytes"


def test_format_image_task_text_can_avoid_caption_duplication():
    saved_image = {
        "path": "uploads/2026-06-03/1_photo.jpg",
        "filename": "photo.jpg",
        "mime_type": "image/jpeg",
        "size_bytes": 100,
    }

    text = format_image_task_text(
        "",
        "убери фон",
        saved_image,
        fallback="Please inspect the attached image.",
        include_caption=False,
    )

    assert text.startswith("Please inspect the attached image.")
    assert "убери фон" not in text
    assert "analyze_document" in text
    assert "vlm_query" in text
    assert "edit_image" in text
    assert "uploads/2026-06-03/1_photo.jpg" in text


def test_save_incoming_audio_writes_upload_and_logs_audio_metadata(tmp_path):
    meta = save_incoming_audio(
        tmp_path,
        file_bytes=b"OggS voice bytes",
        original_name="telegram_voice_99.ogg",
        mime_type="audio/ogg",
        telegram_file_id="voice-file",
        telegram_file_unique_id="voice-unique",
        message_id=99,
        attachment_type="voice",
        duration_sec=12,
    )

    saved = tmp_path / meta["path"]
    assert saved.exists()
    assert saved.read_bytes() == b"OggS voice bytes"
    assert meta["type"] == "telegram_audio_saved"
    assert meta["attachment_type"] == "voice"
    assert meta["duration_sec"] == 12
    assert str(meta["path"]).startswith("uploads/")
    assert str(meta["path"]).endswith("_telegram_voice_99.ogg")

    upload_logs = (tmp_path / "logs" / "uploads.jsonl").read_text(encoding="utf-8").strip().splitlines()
    logged = json.loads(upload_logs[0])
    assert logged["type"] == "telegram_audio_saved"
    assert logged["telegram_file_unique_id"] == "voice-unique"
