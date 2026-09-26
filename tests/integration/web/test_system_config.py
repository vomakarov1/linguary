"""Состояние конфигурации и окружения (API-01, PRD § 21.6, § 23).

Экран отвечает на вопрос «готово ли приложение работать», поэтому проверяется и обратное
состояние — первый запуск без настроенной LLM и машина без Java. Отдельно проверяется,
что ключ провайдера не попадает в ответ ни при каких настройках.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from linguary.adapters.config import AppSettings, LLMSettings, ServerSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.composition import build_application
from linguary.web import create_app

#: Ключ провайдера, которого не должно быть ни в одном ответе.
SECRET_KEY = "sk-secret-value-must-not-leak"


@pytest.fixture
def configured_client(settings: AppSettings, llm_client: FakeLLMClient):
    """Клиент приложения с настроенной LLM и сервером, открытым в сеть."""
    configured = replace(
        settings,
        llm=LLMSettings(model="gpt-5-mini", api_key=SECRET_KEY),
        server=ServerSettings(host="0.0.0.0", port=9000),
    )
    application = build_application(settings=configured, llm_client=llm_client)
    try:
        with TestClient(
            create_app(application=application, run_background_worker=False)
        ) as client:
            yield client
    finally:
        application.dispose()


def test_config_reports_llm_as_not_ready_on_the_first_run(client: TestClient) -> None:
    """Без модели и ключа приложение честно сообщает, что LLM не настроена (PRD § 23)."""
    llm = client.get("/system/config").json()["llm"]

    assert llm["is_ready"] is False
    assert llm["model"] is None
    assert llm["has_api_key"] is False
    assert "config.toml" in llm["hint"]


def test_config_shows_the_active_model(configured_client: TestClient) -> None:
    """Настроенная LLM показывает читаемое имя активной модели (PRD § 23)."""
    llm = configured_client.get("/system/config").json()["llm"]

    assert llm["is_ready"] is True
    assert llm["model"] == "gpt-5-mini"
    assert llm["has_api_key"] is True
    assert llm["hint"] is None


def test_config_never_exposes_the_api_key(configured_client: TestClient) -> None:
    """Ключ провайдера не попадает в ответ ни одним полем (PRD § 23)."""
    response = configured_client.get("/system/config")

    assert SECRET_KEY not in response.text


def test_config_warns_about_the_external_provider(client: TestClient) -> None:
    """Экран конфигурации предупреждает о передаче текста книги наружу (PRD § 4.1)."""
    warning = client.get("/system/config").json()["external_llm_warning"]

    assert "внешнему провайдеру LLM" in warning


def test_config_reports_epubcheck_availability(client: TestClient) -> None:
    """Доступность внешней проверки показывается с подсказкой, как её включить."""
    epubcheck = client.get("/system/config").json()["epubcheck"]

    assert isinstance(epubcheck["available"], bool)
    if epubcheck["available"]:
        assert epubcheck["jar_path"]
        assert epubcheck["hint"] is None
    else:
        assert epubcheck["reason"]
        assert "EPUBCheck" in epubcheck["hint"]


def test_config_shows_where_data_is_stored(client: TestClient, data_root) -> None:
    """Экран показывает каталог данных и файл базы (PRD § 22, UX notes)."""
    storage = client.get("/system/config").json()["storage"]

    assert storage["data_root"] == str(data_root)
    assert storage["database_path"].endswith("linguary.db")
    assert storage["books_root"].endswith("books")


def test_config_reports_local_only_server(client: TestClient) -> None:
    """По умолчанию приложение слушает только эту машину (PRD § 4.1)."""
    server = client.get("/system/config").json()["server"]

    assert server["host"] == "127.0.0.1"
    assert server["is_local_only"] is True
    assert server["base_url"] == "http://127.0.0.1:8000"
    assert server["warning"] is None


def test_config_warns_about_a_server_open_to_the_network(
    configured_client: TestClient,
) -> None:
    """Адрес, открытый в сеть, сопровождается предупреждением (PRD § 4.1)."""
    server = configured_client.get("/system/config").json()["server"]

    assert server["is_local_only"] is False
    assert "не только с этой машины" in server["warning"]
