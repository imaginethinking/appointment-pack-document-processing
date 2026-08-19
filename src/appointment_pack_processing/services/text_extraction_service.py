"""Document text extraction using embedded PDF text and OCR fallback."""

import pymupdf

from appointment_pack_processing.services.ocr_service import OcrService


class TextExtractionError(ValueError):
    """Base exception for expected document text-extraction failures."""


class UnreadablePdfError(TextExtractionError):
    """Raised when uploaded content cannot be opened as a PDF."""


class PasswordProtectedPdfError(TextExtractionError):
    """Raised when a PDF requires a password."""


class PdfPageLimitExceededError(TextExtractionError):
    """Raised when a PDF exceeds the configured page-count limit."""


class TextNotFoundError(TextExtractionError):
    """Raised when no usable document text can be extracted."""


class TextExtractionService:
    """Extract text from supported PDFs and images without persisting content."""

    PDF_CONTENT_TYPE = "application/pdf"
    IMAGE_CONTENT_TYPES = frozenset(
        {
            "image/jpeg",
            "image/png",
        }
    )

    def __init__(
        self,
        maximum_pdf_pages: int,
        ocr_service: OcrService,
    ) -> None:
        """Configure the PDF page limit and OCR dependency."""
        self.maximum_pdf_pages = maximum_pdf_pages
        self.ocr_service = ocr_service

    def extract(
        self,
        content_type: str,
        content: bytes,
    ) -> str:
        """Extract normalised text from a supported PDF or image."""
        if content_type == self.PDF_CONTENT_TYPE:
            return self._extract_pdf_text(content)

        if content_type in self.IMAGE_CONTENT_TYPES:
            return self.ocr_service.extract_image_text(content)

        raise TextExtractionError("The document content type cannot be processed")

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract usable text from each PDF page and combine the results."""
        try:
            with pymupdf.open(
                stream=content,
                filetype="pdf",
            ) as document:
                self._validate_pdf(document)

                page_texts = [self._extract_pdf_page_text(page) for page in document]
        except TextExtractionError:
            raise
        except Exception as exception:
            raise UnreadablePdfError("The uploaded PDF could not be read") from exception

        extracted_text = "\n\n".join(page_text for page_text in page_texts if page_text)

        if not extracted_text:
            raise TextNotFoundError("No readable text was detected in the PDF")

        return extracted_text

    def _extract_pdf_page_text(
        self,
        page: pymupdf.Page,
    ) -> str:
        """Use embedded page text when available, otherwise fall back to OCR."""
        embedded_text = self._normalise_text(
            page.get_text(
                "text",
                sort=True,
            )
        )

        # Hybrid PDFs are handled page by page so scanned pages still receive OCR.
        if embedded_text:
            return embedded_text

        return self.ocr_service.extract_pdf_page_text(page)

    def _validate_pdf(
        self,
        document: pymupdf.Document,
    ) -> None:
        """Reject password-protected PDFs and documents above the page limit."""
        if document.needs_pass:
            raise PasswordProtectedPdfError("Password-protected PDFs are not supported")

        if document.page_count > self.maximum_pdf_pages:
            raise PdfPageLimitExceededError("PDF exceeds the maximum supported page count")

    def _normalise_text(self, text: str) -> str:
        """Collapse whitespace while preserving line boundaries."""
        normalised_lines = []

        for line in text.splitlines():
            normalised_line = " ".join(line.split())

            if normalised_line:
                normalised_lines.append(normalised_line)

        return "\n".join(normalised_lines)
