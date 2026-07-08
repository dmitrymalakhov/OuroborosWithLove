"""
Supervisor — Telegram client + formatting.

TelegramClient, message splitting, markdown→HTML conversion, send_with_budget.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests

from supervisor.state import load_state, save_state, append_jsonl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config (set via init())
# ---------------------------------------------------------------------------
DRIVE_ROOT = None  # pathlib.Path
TOTAL_BUDGET_LIMIT: float = 0.0
BUDGET_REPORT_EVERY_MESSAGES: int = 10
_TG: Optional["TelegramClient"] = None

_MIME_BY_EXTENSION = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "pdf": "application/pdf",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "m4a": "audio/mp4",
    "oga": "audio/ogg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}

_IMAGE_EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


def redact_telegram_token(text: str, token: str = "") -> str:
    safe = str(text or "")
    if token:
        safe = safe.replace(str(token), "<telegram-token>")
    safe = re.sub(r"/bot[^/\s'\")]+", "/bot<telegram-token>", safe)
    safe = re.sub(r"file/bot[^/\s'\")]+", "file/bot<telegram-token>", safe)
    return safe


def init(drive_root, total_budget_limit: float, budget_report_every: int,
         tg_client: "TelegramClient") -> None:
    global DRIVE_ROOT, TOTAL_BUDGET_LIMIT, BUDGET_REPORT_EVERY_MESSAGES, _TG
    DRIVE_ROOT = drive_root
    TOTAL_BUDGET_LIMIT = total_budget_limit
    BUDGET_REPORT_EVERY_MESSAGES = budget_report_every
    _TG = tg_client


def get_tg() -> "TelegramClient":
    assert _TG is not None, "telegram.init() not called"
    return _TG


# ---------------------------------------------------------------------------
# TelegramClient
# ---------------------------------------------------------------------------

class TelegramClient:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self._token = token
        self._me_cache: Optional[Dict[str, Any]] = None

    def _format_error(self, exc: Exception) -> str:
        return redact_telegram_token(repr(exc), self._token)

    def _format_api_error(self, data: Any) -> str:
        return redact_telegram_token(f"telegram_api_error: {data}", self._token)

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                allowed_updates = [
                    "message",
                    "edited_message",
                    "my_chat_member",
                    "callback_query",
                    "poll",
                    "poll_answer",
                ]
                r = requests.get(
                    f"{self.base}/getUpdates",
                    params={"offset": offset, "timeout": timeout,
                            "allowed_updates": json.dumps(allowed_updates)},
                    timeout=timeout + 5,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is not True:
                    raise RuntimeError(f"Telegram getUpdates failed: {data}")
                return data.get("result") or []
            except Exception as e:
                last_err = self._format_error(e)
                if attempt < 2:
                    import time
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Telegram getUpdates failed after retries: {last_err}")

    def get_me(self) -> Dict[str, Any]:
        if self._me_cache is not None:
            return dict(self._me_cache)
        try:
            r = requests.get(f"{self.base}/getMe", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Telegram getMe failed: {self._format_error(e)}") from None
        if data.get("ok") is not True:
            raise RuntimeError(f"Telegram getMe failed: {data}")
        result = data.get("result") or {}
        self._me_cache = dict(result)
        return dict(result)

    def send_message(self, chat_id: int, text: str, parse_mode: str = "") -> Tuple[bool, str]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                payload: Dict[str, Any] = {"chat_id": chat_id, "text": text,
                                           "disable_web_page_preview": True}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = requests.post(f"{self.base}/sendMessage", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok"
                last_err = self._format_api_error(data)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def send_message_with_markup(
        self,
        chat_id: int,
        text: str,
        reply_markup: Dict[str, Any],
        parse_mode: str = "",
    ) -> Tuple[bool, str, int]:
        """Send a message with inline keyboard markup and return message_id."""
        last_err = "unknown"
        for attempt in range(3):
            try:
                payload: Dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                    "reply_markup": json.dumps(reply_markup, ensure_ascii=False),
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = requests.post(f"{self.base}/sendMessage", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    result = data.get("result") or {}
                    return True, "ok", int(result.get("message_id") or 0)
                last_err = self._format_api_error(data)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err, 0

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: str = "",
    ) -> Tuple[bool, str]:
        """Best-effort edit of an already sent Telegram message."""
        last_err = "unknown"
        for attempt in range(2):
            try:
                payload: Dict[str, Any] = {
                    "chat_id": int(chat_id),
                    "message_id": int(message_id),
                    "text": text,
                    "disable_web_page_preview": True,
                }
                if reply_markup is not None:
                    payload["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = requests.post(f"{self.base}/editMessageText", data=payload, timeout=15)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok"
                last_err = self._format_api_error(data)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt == 0:
                import time
                time.sleep(0.4)
        return False, last_err

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
        show_alert: bool = False,
    ) -> Tuple[bool, str]:
        last_err = "unknown"
        try:
            payload: Dict[str, Any] = {
                "callback_query_id": callback_query_id,
                "show_alert": bool(show_alert),
            }
            if text:
                payload["text"] = text[:200]
            r = requests.post(f"{self.base}/answerCallbackQuery", data=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("ok") is True:
                return True, "ok"
            last_err = self._format_api_error(data)
        except Exception as e:
            last_err = self._format_error(e)
        return False, last_err

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        """Send chat action (typing indicator). Best-effort, no retries."""
        try:
            r = requests.post(
                f"{self.base}/sendChatAction",
                data={"chat_id": chat_id, "action": action},
                timeout=5,
            )
            return r.status_code == 200
        except Exception as e:
            log.debug("Failed to send chat action to chat_id=%d: %s", chat_id, self._format_error(e))
            return False

    def send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: str = "",
        filename: str = "image.png",
        mime_type: str = "image/png",
    ) -> Tuple[bool, str]:
        """Send a photo to a chat. photo_bytes is raw PNG/JPEG data."""
        last_err = "unknown"
        display_name = filename or "image.png"
        detected_mime = mime_type or "image/png"
        for attempt in range(3):
            try:
                files = {"photo": (display_name, photo_bytes, detected_mime)}
                data: Dict[str, Any] = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                r = requests.post(
                    f"{self.base}/sendPhoto",
                    data=data, files=files, timeout=30,
                )
                r.raise_for_status()
                resp = r.json()
                if resp.get("ok") is True:
                    return True, "ok"
                last_err = self._format_api_error(resp)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def send_document(
        self,
        chat_id: int,
        file_path: pathlib.Path,
        caption: str = "",
        filename: str = "",
        mime_type: str = "application/octet-stream",
    ) -> Tuple[bool, str]:
        """Send a local file to a chat as a Telegram document."""
        last_err = "unknown"
        display_name = filename or file_path.name
        for attempt in range(3):
            try:
                with file_path.open("rb") as f:
                    files = {"document": (display_name, f, mime_type or "application/octet-stream")}
                    data: Dict[str, Any] = {"chat_id": chat_id}
                    if caption:
                        data["caption"] = caption[:1024]
                    r = requests.post(
                        f"{self.base}/sendDocument",
                        data=data, files=files, timeout=90,
                    )
                r.raise_for_status()
                resp = r.json()
                if resp.get("ok") is True:
                    return True, "ok"
                last_err = self._format_api_error(resp)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def send_poll(
        self,
        chat_id: int,
        question: str,
        options: List[str],
        *,
        is_anonymous: bool = False,
        allows_multiple_answers: bool = False,
        open_period_seconds: int = 0,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Send a native Telegram poll and return the sent Message payload."""
        last_err = "unknown"
        option_payload = [{"text": str(opt)} for opt in options]
        for attempt in range(3):
            try:
                payload: Dict[str, Any] = {
                    "chat_id": int(chat_id),
                    "question": str(question or ""),
                    "options": json.dumps(option_payload, ensure_ascii=False),
                    "is_anonymous": bool(is_anonymous),
                    "allows_multiple_answers": bool(allows_multiple_answers),
                    "type": "regular",
                }
                if open_period_seconds:
                    payload["open_period"] = int(open_period_seconds)
                r = requests.post(f"{self.base}/sendPoll", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok", dict(data.get("result") or {})
                last_err = self._format_api_error(data)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err, {}

    def stop_poll(self, chat_id: int, message_id: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Stop a Telegram poll sent by the bot and return the stopped Poll payload."""
        last_err = "unknown"
        for attempt in range(3):
            try:
                payload = {"chat_id": int(chat_id), "message_id": int(message_id)}
                r = requests.post(f"{self.base}/stopPoll", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok", dict(data.get("result") or {})
                last_err = self._format_api_error(data)
            except Exception as e:
                last_err = self._format_error(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err, {}

    def download_file_bytes(self, file_id: str, max_bytes: int = 50_000_000) -> Tuple[Optional[bytes], str, str]:
        """Download a Telegram file and return (bytes, mime_type, telegram_file_path)."""
        try:
            r = requests.get(f"{self.base}/getFile", params={"file_id": file_id}, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                return None, "", ""
            file_path = data["result"].get("file_path", "")
            file_size = int(data["result"].get("file_size") or 0)
            if file_size > max_bytes:
                return None, "", file_path

            download_url = f"https://api.telegram.org/file/bot{self._token}/{file_path}"
            r2 = requests.get(download_url, timeout=30)
            r2.raise_for_status()

            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            mime = _MIME_BY_EXTENSION.get(ext, "application/octet-stream")
            if len(r2.content) > max_bytes:
                return None, mime, file_path

            return r2.content, mime, file_path
        except Exception as e:
            log.warning("Failed to download file_id=%s from Telegram: %s", file_id, self._format_error(e))
            return None, "", ""

    def download_file_base64(self, file_id: str, max_bytes: int = 10_000_000) -> Tuple[Optional[str], str]:
        """Download a file from Telegram and return (base64_data, mime_type). Returns (None, "") on failure."""
        content, mime, _file_path = self.download_file_bytes(file_id, max_bytes=max_bytes)
        if content is None:
            return None, ""
        import base64
        return base64.b64encode(content).decode("ascii"), mime if mime.startswith("image/") else "image/jpeg"


# ---------------------------------------------------------------------------
# Incoming file persistence
# ---------------------------------------------------------------------------

def _safe_upload_filename(name: str, fallback: str = "telegram_file") -> str:
    name = unicodedata.normalize("NFC", str(name or "")).strip()
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = "".join(c for c in name if ord(c) >= 32)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback
    if len(name) > 180:
        stem = pathlib.Path(name).stem[:140].strip(" .") or fallback
        suffix = pathlib.Path(name).suffix[:20]
        name = stem + suffix
    return name


def _save_incoming_upload(
    drive_root: pathlib.Path,
    *,
    file_bytes: bytes,
    original_name: str,
    mime_type: str,
    telegram_file_id: str,
    telegram_file_unique_id: str = "",
    caption: str = "",
    message_id: int = 0,
    event_type: str,
    fallback_name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a Telegram upload in the user's workspace and return metadata."""
    root = pathlib.Path(drive_root)
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    upload_dir = root / "uploads" / day
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_upload_filename(original_name, fallback=fallback_name)
    prefix = str(int(message_id)) if message_id else datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
    rel_path = pathlib.PurePosixPath("uploads") / day / f"{prefix}_{safe_name}"
    path = root / rel_path
    counter = 2
    while path.exists():
        stem = pathlib.Path(safe_name).stem
        suffix = pathlib.Path(safe_name).suffix
        rel_path = pathlib.PurePosixPath("uploads") / day / f"{prefix}_{stem}_{counter}{suffix}"
        path = root / rel_path
        counter += 1

    path.write_bytes(file_bytes)
    meta = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": event_type,
        "path": str(rel_path),
        "filename": safe_name,
        "original_name": original_name or safe_name,
        "mime_type": mime_type,
        "size_bytes": len(file_bytes),
        "telegram_file_id": telegram_file_id,
        "telegram_file_unique_id": telegram_file_unique_id,
        "caption": caption,
        "message_id": message_id,
    }
    if extra_meta:
        meta.update(extra_meta)
    append_jsonl(root / "logs" / "uploads.jsonl", meta)
    return meta


def save_incoming_document(
    drive_root: pathlib.Path,
    *,
    file_bytes: bytes,
    original_name: str,
    mime_type: str,
    telegram_file_id: str,
    telegram_file_unique_id: str = "",
    caption: str = "",
    message_id: int = 0,
) -> Dict[str, Any]:
    """Persist a Telegram document in the user's workspace and return metadata."""
    return _save_incoming_upload(
        drive_root,
        file_bytes=file_bytes,
        original_name=original_name,
        mime_type=mime_type,
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        caption=caption,
        message_id=message_id,
        event_type="telegram_document_saved",
        fallback_name="telegram_file",
    )


def save_incoming_image(
    drive_root: pathlib.Path,
    *,
    file_bytes: bytes,
    original_name: str,
    mime_type: str,
    telegram_file_id: str,
    telegram_file_unique_id: str = "",
    caption: str = "",
    message_id: int = 0,
) -> Dict[str, Any]:
    """Persist a Telegram image/photo upload in the user's workspace and return metadata."""
    clean_mime = str(mime_type or "").lower().strip()
    suffix = pathlib.Path(str(original_name or "")).suffix
    fallback_suffix = _IMAGE_EXTENSION_BY_MIME.get(clean_mime, ".jpg")
    fallback_name = f"telegram_image{fallback_suffix}"
    safe_original = original_name
    if safe_original and not suffix and clean_mime in _IMAGE_EXTENSION_BY_MIME:
        safe_original = safe_original + fallback_suffix
    return _save_incoming_upload(
        drive_root,
        file_bytes=file_bytes,
        original_name=safe_original,
        mime_type=mime_type,
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        caption=caption,
        message_id=message_id,
        event_type="telegram_image_saved",
        fallback_name=fallback_name,
    )


def save_incoming_audio(
    drive_root: pathlib.Path,
    *,
    file_bytes: bytes,
    original_name: str,
    mime_type: str,
    telegram_file_id: str,
    telegram_file_unique_id: str = "",
    caption: str = "",
    message_id: int = 0,
    attachment_type: str = "audio",
    duration_sec: int = 0,
) -> Dict[str, Any]:
    """Persist a Telegram voice/audio upload in the user's workspace and return metadata."""
    return _save_incoming_upload(
        drive_root,
        file_bytes=file_bytes,
        original_name=original_name,
        mime_type=mime_type,
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        caption=caption,
        message_id=message_id,
        event_type="telegram_audio_saved",
        fallback_name="telegram_audio.ogg",
        extra_meta={
            "attachment_type": str(attachment_type or "audio"),
            "duration_sec": int(duration_sec or 0),
        },
    )


# ---------------------------------------------------------------------------
# Message splitting + formatting
# ---------------------------------------------------------------------------

def split_telegram(text: str, limit: int = 3800) -> List[str]:
    chunks: List[str] = []
    s = text
    while len(s) > limit:
        cut = s.rfind("\n", 0, limit)
        if cut < 100:
            cut = limit
        chunks.append(s[:cut])
        s = s[cut:]
    chunks.append(s)
    return chunks


def _sanitize_telegram_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        c for c in text
        if (ord(c) >= 32 or c in ("\n", "\t")) and not (0xD800 <= ord(c) <= 0xDFFF)
    )


def _tg_utf16_len(text: str) -> int:
    if not text:
        return 0
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _strip_markdown(text: str) -> str:
    """Strip all markdown formatting markers, leaving only plain text."""
    # Fenced code blocks (keep content)
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    # Inline code (keep content)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold+italic (***text***)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    # Bold (**text**)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Italic (*text* or _text_)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strikethrough (~~text~~)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headers (# text -> text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # List markers (- or * at start of line, keep bullet but remove markdown)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    # Clean up any remaining stray markdown markers
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    return text


def _markdown_to_telegram_html(md: str) -> str:
    """Convert Markdown to Telegram-safe HTML.

    Supported: fenced code, inline code, **bold**, *italic*, _italic_,
    ~~strikethrough~~, [links](url), # headers, list items.
    Handles unmatched markers gracefully. Telegram only allows: b, i, u, s, code, pre, a.
    """
    import html as _html
    md = md or ""

    # --- Step 1: extract fenced code blocks into placeholders ---
    # Match ``` with optional language, then content, then closing ```
    fence_re = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.MULTILINE)
    fenced: list = []

    def _save_fence(m: re.Match) -> str:
        code_content = m.group(1)
        # Remove trailing newline if present
        if code_content.endswith("\n"):
            code_content = code_content[:-1]
        code_esc = _html.escape(code_content, quote=False)
        placeholder = f"\x00FENCE{len(fenced)}\x00"
        fenced.append(f"<pre>{code_esc}</pre>")
        return placeholder

    text = fence_re.sub(_save_fence, md)

    # --- Step 2: extract inline code into placeholders ---
    inline_code_re = re.compile(r"`([^`\n]+)`")
    inlines: list = []

    def _save_inline(m: re.Match) -> str:
        code_esc = _html.escape(m.group(1), quote=False)
        placeholder = f"\x00CODE{len(inlines)}\x00"
        inlines.append(f"<code>{code_esc}</code>")
        return placeholder

    text = inline_code_re.sub(_save_inline, text)

    # --- Step 3: HTML-escape remaining text (before adding HTML tags) ---
    text = _html.escape(text, quote=False)

    # --- Step 4: apply markdown formatting (order matters) ---
    # Headers: # at start of line -> bold with newline
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Links: [text](url) - escape the URL too
    def _replace_link(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2)
        # URL must not contain quotes or special chars that break HTML
        url_safe = url.replace('"', '%22').replace('<', '%3C').replace('>', '%3E')
        return f'<a href="{url_safe}">{link_text}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_link, text)

    # Bold+italic: ***text*** (must come before ** and *)
    # Use non-greedy match, handle line breaks
    text = re.sub(r"\*\*\*([^*\n]+?)\*\*\*", r"<b><i>\1</i></b>", text)

    # Bold: **text** (non-greedy, single line)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", text)

    # Strikethrough: ~~text~~ (non-greedy, single line)
    text = re.sub(r"~~([^~\n]+?)~~", r"<s>\1</s>", text)

    # Italic: *text* (single *, not adjacent to another *, single line)
    # Lookahead/lookbehind to avoid matching ** or *** remnants
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", text)

    # Italic: _text_ (word-boundary to avoid matching snake_case, single line)
    text = re.sub(r"\b_([^_\n]+?)_\b", r"<i>\1</i>", text)

    # List items: convert - or * at line start to •
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)

    # --- Step 5: restore placeholders ---
    for i, code in enumerate(inlines):
        text = text.replace(f"\x00CODE{i}\x00", code)
    for i, block in enumerate(fenced):
        text = text.replace(f"\x00FENCE{i}\x00", block)

    return text


def _chunk_markdown_for_telegram(md: str, max_chars: int = 3500) -> List[str]:
    md = md or ""
    max_chars = max(256, min(4096, int(max_chars)))
    lines = md.splitlines(keepends=True)
    chunks: List[str] = []
    cur = ""
    in_fence = False
    fence_open = "```\n"
    fence_close = "```\n"

    def _flush() -> None:
        nonlocal cur
        if cur and cur.strip():
            chunks.append(cur)
        cur = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if in_fence:
                fence_open = line if line.endswith("\n") else (line + "\n")

        reserve = _tg_utf16_len(fence_close) if in_fence else 0
        if _tg_utf16_len(cur) + _tg_utf16_len(line) > max_chars - reserve:
            if in_fence and cur:
                cur += fence_close
            _flush()
            cur = fence_open if in_fence else ""
        cur += line

    if in_fence:
        cur += fence_close
    _flush()
    return chunks or [md]


def _send_markdown_telegram(chat_id: int, text: str) -> Tuple[bool, str]:
    """Send markdown text as Telegram HTML, with plain-text fallback."""
    tg = get_tg()
    chunks = _chunk_markdown_for_telegram(text or "", max_chars=3200)
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        return False, "empty_chunks"
    last_err = "ok"
    for md_part in chunks:
        html_text = _markdown_to_telegram_html(md_part)
        ok, err = tg.send_message(chat_id, _sanitize_telegram_text(html_text), parse_mode="HTML")
        if not ok:
            plain = _strip_markdown(md_part)
            if not plain.strip():
                return False, err
            ok2, err2 = tg.send_message(chat_id, _sanitize_telegram_text(plain))
            if not ok2:
                return False, err2
        last_err = err
    return True, last_err


# ---------------------------------------------------------------------------
# Budget + logging
# ---------------------------------------------------------------------------

def _format_budget_line(st: Dict[str, Any], public: bool = False) -> str:
    if public:
        return "—\nBudget: shared pool active"
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    pct = (spent / total * 100.0) if total > 0 else 0.0
    sha = (st.get("current_sha") or "")[:8]
    branch = st.get("current_branch") or "?"
    return f"—\nBudget: ${spent:.4f} / ${total:.2f} ({pct:.2f}%) | {branch}@{sha}"


def _is_budget_admin(st: Dict[str, Any], user_id: Optional[int]) -> bool:
    if user_id is None:
        return True
    try:
        uid = int(user_id)
    except Exception:
        return False
    admin_ids = st.get("admin_user_ids")
    if isinstance(admin_ids, list) and admin_ids:
        return uid in {int(x) for x in admin_ids}
    return uid == int(st.get("owner_id") or 0)


def budget_line(force: bool = False, public: bool = False) -> str:
    try:
        st = load_state()
        every = max(1, int(BUDGET_REPORT_EVERY_MESSAGES))
        if force:
            st["budget_messages_since_report"] = 0
            save_state(st)
            return _format_budget_line(st, public=public)

        counter = int(st.get("budget_messages_since_report") or 0) + 1
        if counter < every:
            st["budget_messages_since_report"] = counter
            save_state(st)
            return ""

        st["budget_messages_since_report"] = 0
        save_state(st)
        return _format_budget_line(st, public=public)
    except Exception:
        log.debug("Suppressed exception in budget_line", exc_info=True)
        return ""


def log_chat(direction: str, chat_id: int, user_id: int, text: str,
             drive_root: Optional[pathlib.Path] = None) -> None:
    root = pathlib.Path(drive_root) if drive_root is not None else DRIVE_ROOT
    append_jsonl(root / "logs" / "chat.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_id": load_state().get("session_id"),
        "direction": direction,
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text,
    })


def send_with_budget(chat_id: int, text: str, log_text: Optional[str] = None,
                     force_budget: bool = False, fmt: str = "",
                     is_progress: bool = False,
                     log_drive_root: Optional[pathlib.Path] = None,
                     log_user_id: Optional[int] = None,
                     suppress_log: bool = False) -> None:
    st = load_state()
    owner_id = int(log_user_id if log_user_id is not None else (st.get("owner_id") or 0))
    public_budget = not _is_budget_admin(st, log_user_id if log_user_id is not None else owner_id)
    # Progress messages go to progress.jsonl instead of chat.jsonl
    # This keeps chat history clean for context building
    if not suppress_log:
        if is_progress:
            root = pathlib.Path(log_drive_root) if log_drive_root is not None else DRIVE_ROOT
            append_jsonl(root / "logs" / "progress.jsonl", {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "direction": "out", "chat_id": chat_id, "user_id": owner_id,
                "text": text if log_text is None else log_text,
            })
        else:
            log_chat("out", chat_id, owner_id, text if log_text is None else log_text,
                     drive_root=log_drive_root)
    budget = budget_line(force=force_budget, public=public_budget)
    _text = str(text or "")
    if not budget:
        if _text.strip() in ("", "\u200b"):
            return
        full = _text
    else:
        base = _text.rstrip()
        if base in ("", "\u200b"):
            full = budget
        else:
            full = base + "\n\n" + budget

    if fmt == "markdown":
        ok, err = _send_markdown_telegram(chat_id, full)
        if not ok:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "telegram_send_error",
                    "chat_id": chat_id,
                    "error": err,
                    "format": "markdown",
                },
            )
        return

    tg = get_tg()
    for idx, part in enumerate(split_telegram(full)):
        ok, err = tg.send_message(chat_id, part)
        if not ok:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "telegram_send_error",
                    "chat_id": chat_id,
                    "part_index": idx,
                    "error": err,
                },
            )
            break
