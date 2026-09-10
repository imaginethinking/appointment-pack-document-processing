"""Create the FastAPI application and its document processing services."""

from fastapi import FastAPI

from appointment_pack_processing import __version__
from appointment_pack_processing.api.routes.documents import router as documents_router
from appointment_pack_processing.api.routes.health import router as health_router
from appointment_pack_processing.config import Settings, get_settings
from appointment_pack_processing.services.appointment_details_service import (
    AppointmentDetailsService,
)
from appointment_pack_processing.services.deidentification_service import (
    DeidentificationService,
)
from appointment_pack_processing.services.document_processing_service import (
    DocumentProcessingService,
)
from appointment_pack_processing.services.image_preprocessing_service import (
    ImagePreprocessingService,
)
from appointment_pack_processing.services.ocr_service import OcrService
from appointment_pack_processing.services.openai_summary_service import OpenAiSummaryService
from appointment_pack_processing.services.text_extraction_service import TextExtractionService


def build_document_processing_service(settings: Settings) -> DocumentProcessingService:
    """Construct the document-processing service graph for one application instance."""
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
    appointment_details_service = AppointmentDetailsService()

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

    return DocumentProcessingService(
        maximum_file_size_bytes=settings.maximum_file_size_bytes,
        maximum_ai_input_characters=settings.maximum_ai_input_characters,
        text_extraction_service=text_extraction_service,
        deidentification_service=deidentification_service,
        appointment_details_service=appointment_details_service,
        openai_summary_service=openai_summary_service,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Internal document extraction, "
            "de-identification and summarisation "
            "service for The Appointment Pack"
        ),
    )

    application.state.document_processing_service = build_document_processing_service(settings)

    application.include_router(health_router)
    application.include_router(documents_router)

    return application
