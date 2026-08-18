from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from appointment_pack_processing.api.dependencies import (
    DocumentProcessingServiceDependency,
    SettingsDependency,
    require_internal_api_key,
)
from appointment_pack_processing.schemas import (
    DocumentExtractionResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    DocumentType,
    RedactionContext,
)
from appointment_pack_processing.services.document_processing_service import (
    AiInputTooLongError,
    ApprovedTextBlankError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedContentTypeError,
)
from appointment_pack_processing.services.ocr_service import OcrError
from appointment_pack_processing.services.openai_summary_service import (
    AiSummaryResponseError,
    AiSummaryTimeoutError,
    AiSummaryUnavailableError,
)
from appointment_pack_processing.services.text_extraction_service import TextExtractionError

router = APIRouter(
    prefix="/internal/v1/documents",
    tags=["documents"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post(
    "/extract",
    response_model=DocumentExtractionResponse,
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
    settings: SettingsDependency,
    document_processing_service: DocumentProcessingServiceDependency,
) -> DocumentExtractionResponse:
    """Extract and process an uploaded document using the current document workflow."""
    content_type = file.content_type or "application/octet-stream"

    try:
        try:
            redaction_context = RedactionContext.model_validate_json(redaction_context_json)
        except ValidationError as exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Redaction context is invalid",
            ) from exception

        # Read one byte beyond the limit so oversized uploads can be detected
        # without reading the entire file into memory.
        content = await file.read(settings.maximum_file_size_bytes + 1)

        # Extraction and OCR are blocking operations, so keep them off the event loop.
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


@router.post(
    "/summarise",
    response_model=DocumentSummaryResponse,
)
async def summarise_document(
    request: DocumentSummaryRequest,
    document_processing_service: DocumentProcessingServiceDependency,
) -> DocumentSummaryResponse:
    """Summarise approved de-identified consultation text."""
    try:
        # The OpenAI client is synchronous, so execute it outside the event loop.
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