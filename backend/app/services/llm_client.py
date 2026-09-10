"""Thin, defensive wrapper around the Anthropic Messages API.

Design rules:
  * The API key is read from configuration on the server. It is never returned
    by any endpoint and never reaches the browser.
  * Every failure mode is caught and reported as a typed result rather than an
    exception, so a degraded LLM never takes the application down.
  * Structured output (output_config.format) is used so the caller gets a
    validated JSON object instead of prose it has to parse by hand.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from app.config import settings

log = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


@dataclass
class LLMResult:
    ok: bool
    data: dict = field(default_factory=dict)
    text: str = ""
    model: str | None = None
    error: str | None = None
    error_kind: str | None = None


class LLMUnavailable(Exception):
    pass


def _get_client():
    """Create the Anthropic client lazily and reuse it."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not settings.anthropic_api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not configured")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("anthropic package is not installed") from exc
        _client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        return _client


def reset_client() -> None:
    """Used by tests to force re-creation with patched settings."""
    global _client
    with _client_lock:
        _client = None


def is_available() -> bool:
    return bool(settings.anthropic_api_key)


def complete_json(
    system: str,
    user_content: str,
    schema: dict,
    max_tokens: int | None = None,
) -> LLMResult:
    """Ask Claude for a JSON object matching `schema`.

    Returns an LLMResult; it never raises for API problems. The caller decides
    what to show when ok is False.
    """
    model = settings.model_name
    try:
        client = _get_client()
    except LLMUnavailable as exc:
        return LLMResult(ok=False, error=str(exc), error_kind="not_configured", model=model)

    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return LLMResult(
            ok=False, error="anthropic package is not installed",
            error_kind="not_configured", model=model,
        )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APITimeoutError as exc:
        log.warning("LLM timeout: %s", exc)
        return LLMResult(ok=False, error="The AI service timed out.", error_kind="timeout", model=model)
    except anthropic.RateLimitError as exc:
        log.warning("LLM rate limited: %s", exc)
        return LLMResult(
            ok=False,
            error="The AI service is rate limited. Please try again shortly.",
            error_kind="rate_limit",
            model=model,
        )
    except anthropic.AuthenticationError:
        log.error("LLM authentication failed - check ANTHROPIC_API_KEY")
        return LLMResult(
            ok=False, error="The AI service rejected the configured credentials.",
            error_kind="auth", model=model,
        )
    except anthropic.NotFoundError:
        log.error("LLM model not found: %s", model)
        return LLMResult(
            ok=False, error=f"Configured model {model!r} was not found.",
            error_kind="bad_model", model=model,
        )
    except anthropic.BadRequestError as exc:
        log.error("LLM bad request: %s", exc)
        return LLMResult(ok=False, error="The AI request was rejected.", error_kind="bad_request", model=model)
    except anthropic.APIStatusError as exc:
        log.warning("LLM status error %s: %s", exc.status_code, exc)
        return LLMResult(
            ok=False,
            error="The AI service is temporarily unavailable.",
            error_kind="unavailable" if exc.status_code >= 500 else "api_error",
            model=model,
        )
    except anthropic.APIConnectionError as exc:
        log.warning("LLM connection error: %s", exc)
        return LLMResult(
            ok=False, error="Could not reach the AI service.",
            error_kind="connection", model=model,
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        log.exception("Unexpected LLM failure")
        return LLMResult(ok=False, error=f"Unexpected AI error: {exc}", error_kind="unknown", model=model)

    if getattr(response, "stop_reason", None) == "refusal":
        return LLMResult(
            ok=False,
            error="The AI service declined to answer this request.",
            error_kind="refusal",
            model=model,
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text.strip():
        return LLMResult(ok=False, error="The AI returned an empty response.",
                         error_kind="empty", model=model)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON despite structured output")
        return LLMResult(
            ok=False, text=text, error="The AI returned an unreadable response.",
            error_kind="invalid_json", model=model,
        )
    if not isinstance(data, dict):
        return LLMResult(
            ok=False, text=text, error="The AI returned an unexpected shape.",
            error_kind="invalid_shape", model=model,
        )
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        return LLMResult(
            ok=False,
            text=text,
            error=f"The AI response was missing required fields: {', '.join(missing)}.",
            error_kind="incomplete",
            model=model,
        )
    return LLMResult(ok=True, data=data, text=text, model=model)
