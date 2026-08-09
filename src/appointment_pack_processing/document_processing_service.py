from uuid import UUID

from appointment_pack_processing import __version__
from appointment_pack_processing.appointment_details_service import (
    AppointmentDetailsService,
)
from appointment_pack_processing.deidentification_service import (
    DeidentificationService,
)
from appointment_pack_processing.openai_summary_service import (
    OpenAiSummaryService,
)
from appointment_pack_processing.schemas import (
    AppointmentAddressDetailsResponse,
    AppointmentDetailsResponse,
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
        appointment_details_service: AppointmentDetailsService,
        openai_summary_service: OpenAiSummaryService,
    ) -> None:
        self.maximum_file_size_bytes = maximum_file_size_bytes
        self.maximum_ai_input_characters = maximum_ai_input_characters
        self.text_extraction_service = text_extraction_service
        self.deidentification_service = deidentification_service
        self.appointment_details_service = appointment_details_service
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
        result = self.appointment_details_service.extract(extracted_text)

        address = None

        if result.details.address is not None:
            address = AppointmentAddressDetailsResponse(
                address_line_1=result.details.address.address_line_1,
                address_line_2=result.details.address.address_line_2,
                town_city=result.details.address.town_city,
                county=result.details.address.county,
                postcode=result.details.address.postcode,
                country=result.details.address.country,
            )

        details = AppointmentDetailsResponse(
            date=result.details.date,
            start_time=result.details.start_time,
            end_time=result.details.end_time,
            service=result.details.service,
            appointment_type=result.details.appointment_type,
            clinician_or_team=result.details.clinician_or_team,
            location_name=result.details.location_name,
            address=address,
        )

        return DocumentExtractionResponse(
            document_id=document_id,
            extracted_text=extracted_text,
            deidentified_text=None,
            appointment_details=details,
            processing_warning=result.processing_warning,
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
            appointment_details=None,
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