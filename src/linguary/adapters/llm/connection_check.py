"""Connection check for the language model prior to task launch."""

from __future__ import annotations

from typing import Any
from urllib.request import getproxies

import httpx

from linguary.adapters.config.settings import LLMSettings, proxy_secrets
from linguary.adapters.llm.openai_compatible import safe_detail


async def test_llm_connection(
    settings: LLMSettings,
    *,
    timeout_seconds: float = 10.0,
) -> tuple[bool, str]:
    """Validates model availability with a lightweight request without saving settings."""
    model = ""
    for candidate in (settings.model, settings.analysis_model, settings.processing_model):
        if candidate and candidate.strip():
            model = candidate.strip()
            break

    api_key = (settings.api_key or "").strip()
    if not model and not api_key:
        return False, "Не указаны модель и API-ключ"
    if not model:
        return False, "Не указана модель"
    if not api_key:
        return False, "Не указан API-ключ"

    base_url = settings.base_url.strip()
    if not base_url:
        return False, "Не указан базовый адрес провайдера"

    urls = ()
    if settings.proxy_mode == "custom":
        urls = (settings.proxy_url,)
    elif settings.proxy_mode == "auto":
        urls = tuple(value for key, value in getproxies().items() if key in ("http", "https", "all"))
    proxy_secret_tokens = tuple(secret for url in urls for secret in proxy_secrets(url))

    def _clean(text: str) -> str:
        return safe_detail(text, api_key=settings.api_key, secrets=proxy_secret_tokens)

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }

    client_kwargs: dict[str, Any] = {
        "timeout": min(timeout_seconds, 10.0),
        "trust_env": settings.proxy_mode == "auto",
    }
    if settings.proxy_mode == "custom" and settings.proxy_url:
        client_kwargs["proxy"] = settings.proxy_url

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(endpoint, json=body, headers=headers)
            if 200 <= response.status_code < 300:
                return True, "Подключение успешно установлено"
            if response.status_code in (401, 403):
                return False, f"Ошибка авторизации ({response.status_code}): неверный API-ключ или нет доступа к модели"
            if response.status_code == 404:
                return False, "Модель или эндпоинт не найдены (404)"
            return False, f"Провайдер вернул ошибку ({response.status_code})"
    except httpx.TimeoutException:
        return False, f"Превышено время ожидания ответа ({timeout_seconds:g} с)"
    except (httpx.ConnectError, httpx.ConnectTimeout) as error:
        return False, _clean(f"Не удалось подключиться к серверу: {error}")
    except httpx.HTTPError as error:
        return False, _clean(f"Сетевая ошибка при обращении к провайдеру: {error}")
    except Exception as error:
        return False, _clean(f"Ошибка проверки подключения: {error}")


__all__ = ["test_llm_connection"]
