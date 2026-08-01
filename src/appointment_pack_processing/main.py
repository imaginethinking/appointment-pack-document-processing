from secrets import compare_digest
from typing import Annotated
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from appointment_pack_processing import __version__
from appointment_pack_processing.config import Settings, get_settings
from appointment_pack_processing.document_processor import (
    DocumentProcessor,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedContentTypeError,
)
from appointment_pack_processing.schemas import (
    DocumentProcessingResponse,
    DocumentType,
    HealthResponse,
)

SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]


def require_internal_api_key(
    settings: SettingsDependency,
    provided_api_key: Annotated[
        str | None,
        Header(alias="X-Internal-Api-Key"),
    ] = None,
) -> None:
    expected_api_key = settings.internal_api_key.get_secret_value()

    if provided_api_key is None or not compare_digest(
        provided_api_key,
        expected_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=("Internal OCR and summarisation service for The Appointment Pack"),
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    def get_health() -> HealthResponse:
        return HealthResponse(
            status="UP",
            service=settings.app_name,
            version=__version__,
            environment=settings.environment,
        )

    @application.post(
        "/internal/v1/documents/process",
        response_model=DocumentProcessingResponse,
        tags=["documents"],
        dependencies=[Depends(require_internal_api_key)],
    )
    async def process_document(
        document_id: Annotated[
            UUID,
            Form(alias="documentId"),
        ],
        document_type: Annotated[
            DocumentType,
            Form(alias="documentType"),
        ],
        file: Annotated[
            UploadFile,
            File(),
        ],
    ) -> DocumentProcessingResponse:
        content_type = (file.content_type or "application/octet-stream")

        processor = DocumentProcessor(settings.maximum_file_size_bytes)

        try:
            content = await file.read(
                settings.maximum_file_size_bytes + 1
            )

            return processor.process(
                document_id=document_id,
                document_type=document_type,
                content_type=content_type,
                content=content,
            )
        except EmptyDocumentError as exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exception),
            ) from exception
        except DocumentTooLargeError as exception:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exception),
            ) from exception
        except UnsupportedContentTypeError as exception:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exception),
            ) from exception
        finally:
            await file.close()

    return application


app = create_app()