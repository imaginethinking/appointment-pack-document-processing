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
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from appointment_pack_processing import __version__
from appointment_pack_processing.appointment_summary_service import (
    AppointmentSummaryService,
)
from appointment_pack_processing.config import (
    Settings,
    get_settings,
)
from appointment_pack_processing.deidentification_service import (
    DeidentificationService,
)
from appointment_pack_processing.document_processing_service import (
    AiInputTooLongError,
    ApprovedTextBlankError,
    DocumentProcessingService,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedContentTypeError,
)
from appointment_pack_processing.image_preprocessing_service import (
    ImagePreprocessingService,
)
from appointment_pack_processing.ocr_service import (
    OcrError,
    OcrService,
)
from appointment_pack_processing.openai_summary_service import (
    AiSummaryResponseError,
    AiSummaryTimeoutError,
    AiSummaryUnavailableError,
    OpenAiSummaryService,
)
from appointment_pack_processing.schemas import (
    DocumentExtractionResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    DocumentType,
    HealthResponse,
    RedactionContext,
)
from appointment_pack_processing.text_extraction_service import (
    TextExtractionError,
    TextExtractionService,
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

    image_preprocessing_service = ImagePreprocessingService(
        enabled=settings.ocr_preprocessing_enabled,
        minimum_image_width=settings.ocr_minimum_image_width,
    )

    ocr_service = OcrService(
        language=settings.ocr_language,
        pdf_dpi=settings.ocr_dpi,
        image_preprocessing_service=image_preprocessing_service,
        tesseract_command=settings.tesseract_command,
    )

    text_extraction_service = TextExtractionService(
        maximum_pdf_pages=settings.maximum_pdf_pages,
        ocr_service=ocr_service,
    )

    deidentification_service = DeidentificationService()

    appointment_summary_service = AppointmentSummaryService()

    openai_api_key = (
        settings.openai_api_key.get_secret_value() if settings.openai_api_key is not None else None
    )

    openai_summary_service = OpenAiSummaryService(
        api_key=openai_api_key,
        model_name=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        maximum_output_tokens=settings.openai_max_output_tokens,
        prompt_version=settings.openai_prompt_version,
    )

    document_processing_service = DocumentProcessingService(
        maximum_file_size_bytes=settings.maximum_file_size_bytes,
        maximum_ai_input_characters=settings.maximum_ai_input_characters,
        text_extraction_service=text_extraction_service,
        deidentification_service=deidentification_service,
        appointment_summary_service=appointment_summary_service,
        openai_summary_service=openai_summary_service,
    )

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Internal document extraction, "
            "de-identification and summarisation "
            "service for The Appointment Pack"
        ),
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
        "/internal/v1/documents/extract",
        response_model=DocumentExtractionResponse,
        tags=["documents"],
        dependencies=[Depends(require_internal_api_key)],
    )
    async def extract_document(
        document_id: Annotated[
            UUID,
            Form(alias="documentId"),
        ],
        document_type: Annotated[
            DocumentType,
            Form(alias="documentType"),
        ],
        redaction_context_json: Annotated[
            str,
            Form(alias="redactionContext"),
        ],
        file: Annotated[
            UploadFile,
            File(),
        ],
    ) -> DocumentExtractionResponse:
        content_type = file.content_type or "application/octet-stream"

        try:
            try:
                redaction_context = RedactionContext.model_validate_json(redaction_context_json)
            except ValidationError as exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Redaction context is invalid",
                ) from exception

            content = await file.read(settings.maximum_file_size_bytes + 1)

            return await run_in_threadpool(
                document_processing_service.extract,
                document_id,
                document_type,
                content_type,
                content,
                redaction_context,
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
        except (
            TextExtractionError,
            OcrError,
        ) as exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exception),
            ) from exception
        finally:
            await file.close()

    @application.post(
        "/internal/v1/documents/summarise",
        response_model=DocumentSummaryResponse,
        tags=["documents"],
        dependencies=[Depends(require_internal_api_key)],
    )
    async def summarise_document(
        request: DocumentSummaryRequest,
    ) -> DocumentSummaryResponse:
        try:
            return await run_in_threadpool(
                document_processing_service.summarise,
                request.document_id,
                request.approved_deidentified_text,
            )
        except ApprovedTextBlankError as exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exception),
            ) from exception
        except AiInputTooLongError as exception:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exception),
            ) from exception
        except AiSummaryTimeoutError as exception:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="External summary generation timed out",
            ) from exception
        except AiSummaryUnavailableError as exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External summary generation is currently unavailable",
            ) from exception
        except AiSummaryResponseError as exception:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="External summary service returned an invalid response",
            ) from exception

    return application


app = create_app()