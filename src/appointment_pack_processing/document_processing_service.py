from uuid import UUID

from appointment_pack_processing import __version__
from appointment_pack_processing.schemas import (
    DocumentExtractionResponse,
    DocumentType,
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
        text_extraction_service: TextExtractionService,
    ) -> None:
        self.maximum_file_size_bytes = maximum_file_size_bytes
        self.text_extraction_service = text_extraction_service

    def extract(
        self,
        document_id: UUID,
        document_type: DocumentType,
        content_type: str,
        content: bytes,
    ) -> DocumentExtractionResponse:
        self._validate_document(
            content_type=content_type,
            content=content,
        )

        extracted_text = self.text_extraction_service.extract(
            content_type=content_type,
            content=content,
        )

        return DocumentExtractionResponse(
            document_id=document_id,
            extracted_text=extracted_text,
            deidentified_text=None,
            generated_summary=None,
            processing_warning=(
                f"Text was extracted from the {document_type.value} "
                "document, but document-specific processing is not yet "
                "implemented."
            ),
            processor_version=__version__,
        )

    def _validate_document(
        self,
        content_type: str,
        content: bytes,
    ) -> None:
        if not content:
            raise EmptyDocumentError(
                "Document file must not be empty"
            )

        if len(content) > self.maximum_file_size_bytes:
            raise DocumentTooLargeError(
                "Document file exceeds the maximum size"
            )

        if content_type not in self.SUPPORTED_CONTENT_TYPES:
            raise UnsupportedContentTypeError(
                "Only PDF, JPEG and PNG documents are supported"
            )