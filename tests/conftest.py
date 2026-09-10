"""Shared pytest fixtures for the document processing service tests."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from appointment_pack_processing.api.dependencies import get_document_processing_service
from appointment_pack_processing.app import create_app
from appointment_pack_processing.config import get_settings
from appointment_pack_processing.services.document_processing_service import (
    DocumentProcessingService,
)


@pytest.fixture
def internal_api_key() -> str:
    """Return the shared API key used by processing route tests."""
    return "test-internal-api-key"


@pytest.fixture
def processing_service() -> Mock:
    """Provide a mock processing service for API contract tests."""
    return Mock(spec=DocumentProcessingService)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    internal_api_key: str,
    processing_service: Mock,
) -> Iterator[TestClient]:
    """Create a test client with the processing service dependency replaced."""
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    monkeypatch.setenv("APP_INTERNAL_API_KEY", internal_api_key)
    monkeypatch.setenv("APP_MAXIMUM_FILE_SIZE_BYTES", "1024")

    get_settings.cache_clear()

    application = create_app()
    application.dependency_overrides[get_document_processing_service] = lambda: processing_service

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
    get_settings.cache_clear()
