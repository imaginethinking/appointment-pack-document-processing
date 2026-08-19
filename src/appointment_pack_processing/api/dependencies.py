from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from appointment_pack_processing.config import Settings, get_settings
from appointment_pack_processing.services.document_processing_service import (
    DocumentProcessingService,
)

SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]


def get_document_processing_service(request: Request) -> DocumentProcessingService:
    """Return the processing service configured for the current application."""
    return request.app.state.document_processing_service


DocumentProcessingServiceDependency = Annotated[
    DocumentProcessingService,
    Depends(get_document_processing_service),
]


def require_internal_api_key(
    settings: SettingsDependency,
    provided_api_key: Annotated[
        str | None,
        Header(alias="X-Internal-Api-Key"),
    ] = None,
) -> None:
    """Reject processing requests that do not provide the shared internal API key."""
    expected_api_key = settings.internal_api_key.get_secret_value()

    # The API key is a shared secret, so use a timing-safe comparison.
    if provided_api_key is None or not compare_digest(
        provided_api_key,
        expected_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )
