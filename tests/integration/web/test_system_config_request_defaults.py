"""Verify optional request body handling for the LLM connection check."""

from unittest.mock import AsyncMock

import pytest

from linguary.web.routes import system


@pytest.mark.parametrize("request_kwargs", [{}, {"json": {}}], ids=["missing-body", "empty-object"])
def test_config_test_uses_current_settings_without_overrides(
    configured_client, configured_application, monkeypatch, request_kwargs
):
    check_connection = AsyncMock(return_value=(True, "connected"))
    monkeypatch.setattr(system, "test_llm_connection", check_connection)

    response = configured_client.post("/system/config/test", **request_kwargs)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "connected", "error": None}
    check_connection.assert_awaited_once_with(configured_application.settings.llm)
