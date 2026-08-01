from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from appointment_pack_processing.config import get_settings


@pytest.fixture
def internal_api_key() -> str:
    return "test-internal-api-key"


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    internal_api_key: str,
) -> Iterator[TestClient]:
    monkeypatch.setenv(
        "APP_ENVIRONMENT",
        "test",
    )
    monkeypatch.setenv(
        "APP_INTERNAL_API_KEY",
        internal_api_key,
    )
    monkeypatch.setenv(
        "APP_MAXIMUM_FILE_SIZE_BYTES",
        "1024",
    )

    get_settings.cache_clear()

    from appointment_pack_processing.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()