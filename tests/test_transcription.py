from types import SimpleNamespace

import pytest

from supervisor.transcription import (
    audio_upload_name,
    format_audio_task_text,
    is_audio_attachment,
    transcribe_audio_file,
)


def test_is_audio_attachment_detects_mime_type_and_supported_suffix():
    assert is_audio_attachment({"mime_type": "audio/ogg"})
    assert is_audio_attachment({"file_name": "meeting.m4a"})
    assert not is_audio_attachment({"mime_type": "application/pdf", "file_name": "report.pdf"})


def test_audio_upload_name_uses_openai_friendly_ogg_extension_for_voice():
    name = audio_upload_name(
        {"mime_type": "audio/ogg"},
        attachment_type="voice",
        message_id=42,
        telegram_path="voice/file_123.oga",
    )

    assert name == "telegram_voice_42.ogg"


def test_transcribe_audio_file_calls_openai_audio_transcriptions(tmp_path):
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"OggS test")

    class FakeTranscriptions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            assert kwargs["file"].read() == b"OggS test"
            return SimpleNamespace(text="Привет, это голосовое")

    fake_transcriptions = FakeTranscriptions()
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))

    transcript = transcribe_audio_file(
        audio_path,
        model="whisper-1",
        language="ru",
        prompt="Распознай русскую речь.",
        client=fake_client,
    )

    assert transcript == "Привет, это голосовое"
    assert fake_transcriptions.kwargs["model"] == "whisper-1"
    assert fake_transcriptions.kwargs["language"] == "ru"
    assert fake_transcriptions.kwargs["prompt"] == "Распознай русскую речь."


def test_transcribe_audio_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe_audio_file(tmp_path / "missing.ogg", client=SimpleNamespace())


def test_format_audio_task_text_includes_metadata_and_transcript():
    text = format_audio_task_text(
        "Что там в голосовом?",
        "",
        {
            "path": "uploads/2026-06-02/42_voice.ogg",
            "filename": "42_voice.ogg",
            "mime_type": "audio/ogg",
            "size_bytes": 123,
            "duration_sec": 3,
            "transcript_path": "uploads/2026-06-02/42_voice.ogg.transcript.txt",
        },
        "Нужно созвониться сегодня.",
    )

    assert "Что там в голосовом?" in text
    assert "[Telegram audio transcribed]" in text
    assert "- duration_sec: 3" in text
    assert "Нужно созвониться сегодня." in text
