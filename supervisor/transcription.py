"""OpenAI audio transcription helpers for Telegram voice/audio uploads."""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Optional


DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
OPENAI_AUDIO_MAX_BYTES = 25_000_000

SUPPORTED_AUDIO_SUFFIXES = {
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".oga",
    ".ogg",
    ".wav",
    ".webm",
}

_EXT_BY_MIME = {
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/oga": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def is_audio_attachment(attachment: Dict[str, Any]) -> bool:
    """Return True when Telegram attachment metadata points to transcribable audio."""
    mime_type = str(attachment.get("mime_type") or "").strip().lower()
    if mime_type.startswith("audio/"):
        return True
    file_name = str(attachment.get("file_name") or "").strip()
    suffix = pathlib.Path(file_name).suffix.lower()
    return suffix in SUPPORTED_AUDIO_SUFFIXES


def _extension_for_audio(attachment: Dict[str, Any], telegram_path: str = "", default: str = ".ogg") -> str:
    for candidate in (str(attachment.get("file_name") or ""), str(telegram_path or "")):
        suffix = pathlib.Path(candidate).suffix.lower()
        if suffix in SUPPORTED_AUDIO_SUFFIXES:
            return ".ogg" if suffix == ".oga" else suffix
    mime_type = str(attachment.get("mime_type") or "").strip().lower()
    return _EXT_BY_MIME.get(mime_type, default)


def audio_upload_name(
    attachment: Dict[str, Any],
    *,
    attachment_type: str,
    message_id: int = 0,
    telegram_path: str = "",
) -> str:
    """Build a stable OpenAI-friendly filename for a Telegram audio attachment."""
    file_name = str(attachment.get("file_name") or "").strip()
    if file_name:
        suffix = pathlib.Path(file_name).suffix.lower()
        if suffix == ".oga":
            return str(pathlib.Path(file_name).with_suffix(".ogg"))
        return file_name

    ext = _extension_for_audio(attachment, telegram_path=telegram_path)
    kind = str(attachment_type or "audio").strip().lower() or "audio"
    prefix = "telegram_voice" if kind == "voice" else "telegram_audio"
    if message_id:
        return f"{prefix}_{int(message_id)}{ext}"
    return f"{prefix}{ext}"


def transcribe_audio_file(
    path: pathlib.Path,
    *,
    model: str = DEFAULT_TRANSCRIPTION_MODEL,
    language: str = "",
    prompt: str = "",
    client: Optional[Any] = None,
) -> str:
    """Transcribe a local audio file using OpenAI's Audio Transcriptions API."""
    audio_path = pathlib.Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size > OPENAI_AUDIO_MAX_BYTES:
        raise ValueError(f"Audio file is too large for OpenAI transcription: {audio_path.stat().st_size} bytes")

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; audio transcription is unavailable")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    request: Dict[str, Any] = {
        "model": str(model or DEFAULT_TRANSCRIPTION_MODEL),
    }
    if language:
        request["language"] = str(language)
    if prompt:
        request["prompt"] = str(prompt)

    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **request)

    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict) and isinstance(dumped.get("text"), str):
            return dumped["text"].strip()
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"].strip()
    return str(response or "").strip()


def format_audio_task_text(
    text: str,
    caption: str,
    audio_meta: Dict[str, Any],
    transcript: str,
) -> str:
    """Build the message sent to the agent after a Telegram audio transcription."""
    base_text = str(text or caption or "").strip()
    lines = []
    if base_text:
        lines.append(base_text)
        lines.append("")
    lines.extend([
        "[Telegram audio transcribed]",
        f"- path: {audio_meta.get('path', '')}",
        f"- filename: {audio_meta.get('filename', '')}",
        f"- mime_type: {audio_meta.get('mime_type', '')}",
        f"- size_bytes: {audio_meta.get('size_bytes', '')}",
    ])
    duration = int(audio_meta.get("duration_sec") or 0)
    if duration:
        lines.append(f"- duration_sec: {duration}")
    transcript_path = str(audio_meta.get("transcript_path") or "").strip()
    if transcript_path:
        lines.append(f"- transcript_path: {transcript_path}")
    lines.extend(["", "Transcription:", str(transcript or "").strip()])
    return "\n".join(lines).strip()
