"""Configuration state and environment (API-01, PRD § 21.6, § 23).

The screen answers the question "is the application ready to run", so the reverse state
is also verified — first launch without a configured LLM and machine without Java.
Additionally, verifies that provider key does not leak into the response under any configuration.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from linguary.adapters.config import AppSettings, LLMSettings, ServerSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.composition import build_application
from linguary.web import create_app

#: Provider key that must not appear in any response.
SECRET_KEY = "sk-secret-value-must-not-leak"


@pytest.fixture
def configured_client(settings: AppSettings, llm_client: FakeLLMClient):
    """Application client with configured LLM and server exposed to the network."""
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
    """Without model and key, the application reports that LLM is not configured (PRD § 23)."""
    llm = client.get("/system/config").json()["llm"]

    assert llm["is_ready"] is False
    assert llm["model"] is None
    assert llm["has_api_key"] is False
    assert "config.toml" in llm["hint"]


def test_config_shows_the_active_model(configured_client: TestClient) -> None:
    """Configured LLM exposes the human-readable name of the active model (PRD § 23)."""
    llm = configured_client.get("/system/config").json()["llm"]

    assert llm["is_ready"] is True
    assert llm["model"] == "gpt-5-mini"
    assert llm["has_api_key"] is True
    assert llm["hint"] is None


def test_config_never_exposes_the_api_key(configured_client: TestClient) -> None:
    """Provider key does not leak into the response in any field (PRD § 23)."""
    response = configured_client.get("/system/config")

    assert SECRET_KEY not in response.text


def test_config_warns_about_the_external_provider(client: TestClient) -> None:
    """Configuration screen warns about sending book text to an external party (PRD § 4.1)."""
    warning = client.get("/system/config").json()["external_llm_warning"]

    assert "внешнему провайдеру LLM" in warning


def test_config_reports_epubcheck_availability(client: TestClient) -> None:
    """Availability of external verification is reported with a hint on how to enable it."""
    epubcheck = client.get("/system/config").json()["epubcheck"]

    assert isinstance(epubcheck["available"], bool)
    if epubcheck["available"]:
        assert epubcheck["jar_path"]
        assert epubcheck["hint"] is None
    else:
        assert epubcheck["reason"]
        assert "EPUBCheck" in epubcheck["hint"]


def test_config_shows_where_data_is_stored(client: TestClient, data_root) -> None:
    """Screen displays data directory and database file location (PRD § 22, UX notes)."""
    storage = client.get("/system/config").json()["storage"]

    assert storage["data_root"] == str(data_root)
    assert storage["database_path"].endswith("linguary.db")
    assert storage["books_root"].endswith("books")


def test_config_reports_local_only_server(client: TestClient) -> None:
    """By default, the application listens only on localhost (PRD § 4.1)."""
    server = client.get("/system/config").json()["server"]

    assert server["host"] == "127.0.0.1"
    assert server["is_local_only"] is True
    assert server["base_url"] == "http://127.0.0.1:8000"
    assert server["warning"] is None


def test_config_warns_about_a_server_open_to_the_network(
    configured_client: TestClient,
) -> None:
    """Address exposed to the network is accompanied by a warning (PRD § 4.1)."""
    server = configured_client.get("/system/config").json()["server"]

    assert server["is_local_only"] is False
    assert "не только с этой машины" in server["warning"]


def test_config_reports_reading_settings(client: TestClient) -> None:
    """Configuration screen returns reader level settings (PRD § 30)."""
    reading = client.get("/system/config").json()["reading"]

    assert reading["reader_level"] == "b2"
    assert reading["default_level"] == "b2"
    assert reading["supported_levels"] == ["a2", "b1", "b2", "c1"]
    assert reading["name_translation_strategy"] == "sound"


def test_strategy_save_omission_conflict_and_reset(configured_client: TestClient) -> None:
    current = configured_client.get('/system/config').json()
    body = dict(
        provider='openai_compatible', model='gpt-5-mini', base_url=current['llm']['base_url'],
        requests_per_minute=None, concurrency=current['worker']['concurrency'], expected_revision=0,
        name_translation_strategy='meaning',
    )
    saved = configured_client.post('/system/config', json=body)
    assert saved.status_code == 200
    assert saved.json()['reading']['name_translation_strategy'] == 'meaning'
    body.pop('name_translation_strategy')
    assert configured_client.post('/system/config', json=body).status_code == 409
    assert configured_client.get('/system/config').json()['reading']['name_translation_strategy'] == 'meaning'
    body['expected_revision'] = 1
    assert configured_client.post('/system/config', json=body).json()['reading']['name_translation_strategy'] == 'meaning'
    assert configured_client.post('/system/config/reset', json={'expected_revision': 2}).json()['reading']['name_translation_strategy'] == 'sound'

