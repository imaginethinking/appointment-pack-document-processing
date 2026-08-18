from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from appointment_pack_processing.config import get_settings
from appointment_pack_processing.document_processing_service import (
    DocumentProcessingService,
)


@pytest.fixture
def internal_api_key() -> str:
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
    """Create the FastAPI test client with processing dependencies replaced."""
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    monkeypatch.setenv("APP_INTERNAL_API_KEY", internal_api_key)
    monkeypatch.setenv("APP_MAXIMUM_FILE_SIZE_BYTES", "1024")

    get_settings.cache_clear()

    import appointment_pack_processing.main as main_module

    monkeypatch.setattr(
        main_module,
        "DocumentProcessingService",
        Mock(return_value=processing_service),
    )

    with TestClient(main_module.create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()