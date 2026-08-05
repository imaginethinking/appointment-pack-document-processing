from uuid import UUID

from appointment_pack_processing import __version__
from appointment_pack_processing.appointment_summary_service import (
    AppointmentSummaryService,
)
from appointment_pack_processing.deidentification_service import (
    DeidentificationService,
)
from appointment_pack_processing.openai_summary_service import (
    OpenAiSummaryService,
)
from appointment_pack_processing.schemas import (
    DocumentExtractionResponse,
    DocumentSummaryResponse,
    DocumentType,
    RedactionContext,
)
from appointment_pack_processing.text_extraction_service import (
    TextExtractionService,
)


class EmptyDocumentError(ValueError):
    """Raised when an uploaded document contains no data."""


class DocumentTooLargeError(ValueError):
    """Raised when an uploaded document exceeds the configured size limit."""


class UnsupportedContentTypeError(ValueError):
    """Raised when an uploaded document has an unsupported media type."""


class ApprovedTextBlankError(ValueError):
    """Raised when approved text is blank."""


class AiInputTooLongError(ValueError):
    """Raised when approved text exceeds the AI limit."""


class DocumentProcessingService:
    SUPPORTED_CONTENT_TYPES = frozenset(
        {
            "application/pdf",
            "image/jpeg",
            "image/png",
        }
    )

    def __init__(
        self,
        maximum_file_size_bytes: int,
        maximum_ai_input_characters: int,
        text_extraction_service: TextExtractionService,
        deidentification_service: DeidentificationService,
        appointment_summary_service: AppointmentSummaryService,
        openai_summary_service: OpenAiSummaryService,
    ) -> None:
        self.maximum_file_size_bytes = maximum_file_size_bytes
        self.maximum_ai_input_characters = maximum_ai_input_characters
        self.text_extraction_service = text_extraction_service
        self.deidentification_service = deidentification_service
        self.appointment_summary_service = appointment_summary_service
        self.openai_summary_service = openai_summary_service

    def extract(
        self,
        document_id: UUID,
        document_type: DocumentType,
        content_type: str,
        content: bytes,
        redaction_context: RedactionContext,
    ) -> DocumentExtractionResponse:
        self._validate_document(
            content_type=content_type,
            content=content,
        )

        extracted_text = self.text_extraction_service.extract(
            content_type=content_type,
            content=content,
        )

        if document_type == DocumentType.CONSULTATION_OUTCOME_LETTER:
            deidentification_result = self.deidentification_service.deidentify(
                text=extracted_text,
                context=redaction_context,
            )

            return self._process_consultation_outcome_letter(
                document_id=document_id,
                extracted_text=extracted_text,
                redaction_context=redaction_context,
            )

        return self._process_appointment_letter(
            document_id=document_id,
            extracted_text=extracted_text,
        )

    def summarise(
        self,
        document_id: UUID,
        approved_deidentified_text: str,
    ) -> DocumentSummaryResponse:
        self._validate_approved_text(approved_deidentified_text)

        summary_result = self.openai_summary_service.summarise(approved_deidentified_text)

        return DocumentSummaryResponse(
            document_id=document_id,
            summary=summary_result.summary,
            processor_version=__version__,
            model_name=summary_result.model_name,
            prompt_version=summary_result.prompt_version,
        )

    def _process_appointment_letter(
        self,
        document_id: UUID,
        extracted_text: str,
    ) -> DocumentExtractionResponse:
        summary_result = self.appointment_summary_service.generate(extracted_text)

        return DocumentExtractionResponse(
            document_id=document_id,
            extracted_text=extracted_text,
            deidentified_text=None,
            generated_summary=summary_result.summary,
            processing_warning=summary_result.processing_warning,
            processor_version=__version__,
        )

    def _process_consultation_outcome_letter(
        self,
        document_id: UUID,
        extracted_text: str,
        redaction_context: RedactionContext,
    ) -> DocumentExtractionResponse:
        deidentification_result = self.deidentification_service.deidentify(
            text=extracted_text,
            context=redaction_context,
        )

        return DocumentExtractionResponse(
            document_id=document_id,
            extracted_text=extracted_text,
            deidentified_text=deidentification_result.text,
            generated_summary=None,
            processing_warning=deidentification_result.processing_warning,
            processor_version=__version__,
        )

    def _validate_document(
        self,
        content_type: str,
        content: bytes,
    ) -> None:
        if not content:
            raise EmptyDocumentError("Document file must not be empty")

        if len(content) > self.maximum_file_size_bytes:
            raise DocumentTooLargeError("Document file exceeds the maximum size")

        if content_type not in self.SUPPORTED_CONTENT_TYPES:
            raise UnsupportedContentTypeError("Only PDF, JPEG and PNG documents are supported")

    def _validate_approved_text(
        self,
        approved_deidentified_text: str,
    ) -> None:
        if not approved_deidentified_text.strip():
            raise ApprovedTextBlankError("Approved de-identified text must not be blank")

        if len(approved_deidentified_text) > self.maximum_ai_input_characters:
            raise AiInputTooLongError(
                "Approved de-identified text exceeds the maximum supported length"
            )
