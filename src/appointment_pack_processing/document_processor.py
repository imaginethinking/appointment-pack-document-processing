from uuid import UUID

from appointment_pack_processing import __version__
from appointment_pack_processing.schemas import (
    DocumentProcessingResponse,
    DocumentType,
)


class EmptyDocumentError(ValueError):
    """Raised when an uploaded document contains no data."""

class DocumentTooLargeError(ValueError):
    """Raised when an uploaded document exceeds the configured size limit."""

class UnsupportedContentTypeError(ValueError):
    """Raised when an uploaded document has an unsupported media type."""


class DocumentProcessor:
    SUPPORTED_CONTENT_TYPES = frozenset(
        {
            "application/pdf",
            "image/jpeg",
            "image/png",
        }
    )

    def __init__(self, maximum_file_size_bytes: int) -> None:
        self.maximum_file_size_bytes = maximum_file_size_bytes

    def process(
        self,
        document_id: UUID,
        document_type: DocumentType,
        content_type: str,
        content: bytes,
    ) -> DocumentProcessingResponse:
        self._validate(content_type, content)

        return DocumentProcessingResponse(
            document_id=document_id,
            extracted_text="",
            summary="",
            key_points=[],
            warnings=[
                (
                    f"{document_type.value} was received successfully, "
                    f"but text extraction has not been implemented yet."
                )
            ],
            processor_version=__version__,
            model=None,
        )

    def _validate(
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