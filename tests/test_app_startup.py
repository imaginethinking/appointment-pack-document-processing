"""Smoke tests for the production FastAPI entry point."""

import importlib
import sys

import pytest

from appointment_pack_processing import __version__
from appointment_pack_processing.config import get_settings


def test_production_entry_point_exposes_expected_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm the Uvicorn entry point imports with the required routes."""
    monkeypatch.setenv(
        "APP_INTERNAL_API_KEY",
        "startup-test-internal-api-key",
    )
    monkeypatch.setenv(
        "APP_ENVIRONMENT",
        "test",
    )

    get_settings.cache_clear()
    sys.modules.pop(
        "appointment_pack_processing.main",
        None,
    )

    try:
        main_module = importlib.import_module("appointment_pack_processing.main")

        assert main_module.app.title == ("Appointment Pack Document Processing")
        assert main_module.app.version == __version__

        route_paths = set(main_module.app.openapi()["paths"])

        assert {
            "/health",
            "/internal/v1/documents/extract",
            "/internal/v1/documents/summarise",
        }.issubset(route_paths)
    finally:
        get_settings.cache_clear()
        sys.modules.pop(
            "appointment_pack_processing.main",
            None,
        )
