"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter/GigaChat).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
import uuid
import base64
import mimetypes
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper (OpenRouter or GigaChat). All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._provider = str(os.environ.get("OUROBOROS_LLM_PROVIDER", "openrouter")).strip().lower()
        if self._provider not in {"openrouter", "gigachat"}:
            log.warning("Unknown OUROBOROS_LLM_PROVIDER=%r, using openrouter", self._provider)
            self._provider = "openrouter"

        if self._provider == "gigachat":
            self._base_url = os.environ.get("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1").strip()
            self._oauth_url = os.environ.get("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth").strip()
            self._oauth_scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
            self._authorization_key = os.environ.get("GIGACHAT_AUTHORIZATION_KEY", "").strip()
            # Optional pre-issued token (for manual control or external rotation).
            self._api_key = api_key or os.environ.get("GIGACHAT_ACCESS_TOKEN", "").strip()
        else:
            self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            self._base_url = base_url

        self._client = None
        self._token_expires_at = 0

    def _ensure_gigachat_token(self, force_refresh: bool = False) -> None:
        """Ensure GigaChat access token is present and reasonably fresh."""
        if self._provider != "gigachat":
            return

        now = int(time.time())
        if not force_refresh and self._api_key and self._token_expires_at and now < (self._token_expires_at - 30):
            return

        if self._api_key and not self._token_expires_at and not force_refresh:
            # Token was provided manually; assume it is still valid until API says otherwise.
            return

        if not self._authorization_key:
            raise RuntimeError(
                "GigaChat token missing. Set GIGACHAT_ACCESS_TOKEN or GIGACHAT_AUTHORIZATION_KEY."
            )

        import requests

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self._authorization_key}",
        }
        payload = {"scope": self._oauth_scope}
        resp = requests.post(self._oauth_url, data=payload, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        data = resp.json()
        token = str(data.get("access_token") or "").strip()
        expires_at_raw = int(data.get("expires_at") or 0)
        # Some APIs return milliseconds since epoch. Normalize to seconds.
        expires_at = int(expires_at_raw / 1000) if expires_at_raw > 10_000_000_000 else expires_at_raw
        if not token:
            raise RuntimeError("GigaChat OAuth response does not contain access_token")
        self._api_key = token
        self._token_expires_at = expires_at
        self._client = None

    def _gigachat_upload_image(self, image: Dict[str, Any], idx: int = 0) -> str:
        """Upload one image to GigaChat files API and return file id."""
        self._ensure_gigachat_token()
        import requests

        content: bytes
        mime = str(image.get("mime") or "image/png")
        filename = f"vision_{idx}.png"

        if "base64" in image:
            raw_b64 = str(image["base64"] or "").strip()
            # Accept either plain base64 or data URLs.
            if raw_b64.startswith("data:") and "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            content = base64.b64decode(raw_b64)
            ext = mimetypes.guess_extension(mime) or ".png"
            filename = f"vision_{idx}{ext}"
        elif "url" in image:
            url = str(image["url"]).strip()
            parsed = urlparse(url)
            guessed = os.path.basename(parsed.path) or ""
            if guessed:
                filename = guessed
            guessed_mime = mimetypes.guess_type(filename)[0]
            if guessed_mime:
                mime = guessed_mime
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            content = resp.content
        else:
            raise ValueError("image must contain either 'url' or 'base64'")

        upload_url = f"{self._base_url.rstrip('/')}/files"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        files = {"file": (filename, content, mime)}
        data = {"purpose": "general"}
        resp = requests.post(upload_url, headers=headers, files=files, data=data, timeout=30, verify=False)
        if resp.status_code == 401:
            # Token might be stale (especially if user provided pre-issued token).
            self._ensure_gigachat_token(force_refresh=True)
            headers = {"Authorization": f"Bearer {self._api_key}"}
            resp = requests.post(upload_url, headers=headers, files=files, data=data, timeout=30, verify=False)
        resp.raise_for_status()
        payload = resp.json() or {}
        file_id = str(payload.get("id") or "").strip()
        if not file_id:
            raise RuntimeError("GigaChat upload response missing file id")
        return file_id

    def _get_client(self):
        if self._provider == "gigachat":
            self._ensure_gigachat_token()
        if self._client is None:
            from openai import OpenAI
            if self._provider == "gigachat":
                import httpx
                self._client = OpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    http_client=httpx.Client(verify=False),
                )
            else:
                self._client = OpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    default_headers={
                        "HTTP-Referer": "https://colab.research.google.com/",
                        "X-Title": "Ouroboros",
                    },
                )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        if self._provider != "openrouter":
            return None
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None


    @staticmethod
    def _normalize_messages_for_gigachat(messages):
        """Flatten Anthropic-style content blocks and fix message roles for GigaChat."""
        result = []
        system_seen = False
        for msg in messages:
            msg = dict(msg)
            # Flatten content arrays (Anthropic cache blocks) to plain strings
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif isinstance(block, str):
                        parts.append(block)
                msg["content"] = "\n\n".join(p for p in parts if p)
            # GigaChat: system role only allowed as the very first message
            if msg.get("role") == "system":
                if system_seen:
                    msg["role"] = "user"
                    msg["content"] = "[SYSTEM] " + str(msg.get("content") or "")
                else:
                    system_seen = True
            result.append(msg)
        return result

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        if self._provider == "gigachat":
            messages = self._normalize_messages_for_gigachat(messages)
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching
        if self._provider == "openrouter" and model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        if self._provider == "openrouter":
            kwargs["extra_body"] = extra_body
        if tools:
            # Add cache_control to last tool for Anthropic prompt caching
            # This caches all tool schemas (they never change between calls)
            tools_with_cache = [t for t in tools]  # shallow copy
            if tools_with_cache and self._provider == "openrouter":
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # GigaChat access token expires every 30m — refresh once on auth errors.
            if self._provider == "gigachat":
                err_text = str(e).lower()
                if "401" in err_text or "unauthorized" in err_text:
                    self._ensure_gigachat_token(force_refresh=True)
                    client = self._get_client()
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            else:
                raise
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if self._provider == "openrouter" and not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        if self._provider == "gigachat":
            # GigaChat works with file attachments. One image per message.
            messages: List[Dict[str, Any]] = []
            for i, img in enumerate(images):
                try:
                    file_id = self._gigachat_upload_image(img, idx=i)
                except Exception:
                    log.warning("vision_query: failed to upload image #%d to GigaChat", i + 1, exc_info=True)
                    continue
                text = prompt if i == 0 else f"Дополнительное изображение {i + 1}. Учитывай его в анализе."
                messages.append({
                    "role": "user",
                    "content": text,
                    "attachments": [file_id],
                })
            if not messages:
                raise RuntimeError("No images were uploaded to GigaChat for vision query")
        else:
            # OpenRouter/OpenAI-compatible multimodal payload
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                if "url" in img:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img["url"]},
                    })
                elif "base64" in img:
                    mime = img.get("mime", "image/png")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                    })
                else:
                    log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))
            messages = [{"role": "user", "content": content}]

        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
