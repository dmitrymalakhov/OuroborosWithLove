"""Telegram image upload helpers."""

from __future__ import annotations

import base64
import datetime
import pathlib
from typing import Any, Dict, Optional, Tuple

from supervisor.state import append_jsonl
from supervisor.telegram import TelegramClient, save_incoming_image, send_with_budget


def image_attachment_metadata(msg: Dict[str, Any]) -> Dict[str, str]:
    """Extract Telegram photo/image-document metadata without downloading it."""
    message_id = int(msg.get("message_id") or 0)
    if msg.get("photo"):
        best_photo = msg["photo"][-1]
        return {
            "file_id": str(best_photo.get("file_id") or ""),
            "original_name": f"telegram_photo_{message_id}",
            "unique_id": str(best_photo.get("file_unique_id") or ""),
            "mime_type": "",
        }

    doc = msg.get("document") or {}
    mime_type = str(doc.get("mime_type") or "")
    if mime_type.startswith("image/"):
        return {
            "file_id": str(doc.get("file_id") or ""),
            "original_name": str(doc.get("file_name") or "telegram_image"),
            "unique_id": str(doc.get("file_unique_id") or ""),
            "mime_type": mime_type,
        }

    return {"file_id": "", "original_name": "", "unique_id": "", "mime_type": ""}


def format_image_task_text(
    text: str,
    caption: str,
    saved_image: Dict[str, Any],
    fallback: str = "Photo attached.",
    include_caption: bool = True,
) -> str:
    """Build the agent-facing note for a saved Telegram image."""
    base = str(text or (caption if include_caption else "") or fallback)
    return (
        f"{base}\n\n"
        "[Telegram image saved]\n"
        f"- path: {saved_image['path']}\n"
        f"- filename: {saved_image['filename']}\n"
        f"- mime_type: {saved_image['mime_type']}\n"
        f"- size_bytes: {saved_image['size_bytes']}\n"
        "Use analyze_document(path='<path>', source='drive') to OCR/read text or numbers from this image. "
        "Use vlm_query(path='<path>', prompt='<question>') for visual analysis. "
        "Use edit_image(path='<path>', prompt='<requested edit>') only when the user asks to modify this image."
    ).strip()


def save_telegram_image_upload(
    tg_client: TelegramClient,
    drive_root: pathlib.Path,
    *,
    image_meta: Optional[Dict[str, str]] = None,
    image_file_id: str = "",
    original_name: str = "",
    mime_type: str = "",
    telegram_file_unique_id: str = "",
    caption: str = "",
    message_id: int = 0,
    now_iso: str = "",
) -> Tuple[Optional[Tuple[str, str, str]], Optional[Dict[str, Any]], bool]:
    """Download, persist, and log a Telegram image upload.

    Returns (image_data_for_vlm, saved_image_meta, download_failed).
    """
    if image_meta:
        image_file_id = image_meta.get("file_id", image_file_id)
        original_name = image_meta.get("original_name", original_name)
        mime_type = image_meta.get("mime_type", mime_type)
        telegram_file_unique_id = image_meta.get("unique_id", telegram_file_unique_id)
    file_bytes, detected_mime, telegram_path = tg_client.download_file_bytes(image_file_id)
    if file_bytes is None:
        append_jsonl(drive_root / "logs" / "events.jsonl", {
            "ts": now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "telegram_image_download_failed",
            "mime_type": mime_type,
        })
        return None, None, True

    image_mime = str(mime_type or detected_mime or "")
    if not image_mime.startswith("image/"):
        image_mime = "image/jpeg"
    clean_name = original_name
    if not clean_name:
        suffix = pathlib.PurePosixPath(telegram_path).suffix
        clean_name = f"telegram_photo_{int(message_id or 0)}{suffix}"
    saved_image = save_incoming_image(
        drive_root,
        file_bytes=file_bytes,
        original_name=clean_name,
        mime_type=image_mime,
        telegram_file_id=image_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        caption=caption,
        message_id=int(message_id or 0),
    )
    append_jsonl(drive_root / "logs" / "events.jsonl", {
        "ts": now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": "telegram_image_saved",
        "path": saved_image.get("path"),
        "mime_type": saved_image.get("mime_type"),
        "size_bytes": saved_image.get("size_bytes"),
    })
    image_b64 = base64.b64encode(file_bytes).decode("ascii")
    return (image_b64, image_mime, caption), saved_image, False


def image_log_text(text: str, caption: str, saved_image: Optional[Dict[str, Any]], download_failed: bool) -> str:
    if saved_image:
        return (text or caption or "") + f"\n[attached image saved: {saved_image['path']}]"
    if download_failed:
        return (text or caption or "") + "\n[attached image download failed]"
    return ""


def send_image_download_failed(chat_id: int, drive_root: pathlib.Path, user_id: int) -> None:
    send_with_budget(
        chat_id,
        "⚠️ Не смог скачать фото из Telegram. Пришли его ещё раз.",
        log_drive_root=drive_root,
        log_user_id=user_id,
    )


def inject_busy_image(agent: Any, chat_id: int, text: str, caption: str, saved_image: Dict[str, Any],
                      drive_root: pathlib.Path, user_id: int) -> None:
    agent.inject_message(format_image_task_text(text, caption, saved_image))
    send_with_budget(
        chat_id,
        "📎 Фото получил. Сохранил в workspace и добавил к текущей задаче.",
        is_progress=True,
        log_drive_root=drive_root,
        log_user_id=user_id,
    )


def prepare_free_image_task(chat_id: int, text: str, caption: str, saved_image: Dict[str, Any],
                            drive_root: pathlib.Path, user_id: int) -> str:
    send_with_budget(
        chat_id,
        f"🖼️ Фото получил: {saved_image['filename']}. Сохранил в workspace, сейчас обработаю запрос.",
        is_progress=True,
        log_drive_root=drive_root,
        log_user_id=user_id,
    )
    return format_image_task_text(
        text,
        caption,
        saved_image,
        fallback="Please inspect the attached image.",
        include_caption=False,
    )
