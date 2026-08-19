from unittest.mock import Mock

import pymupdf
import pytest

from appointment_pack_processing.services.ocr_service import OcrService
from appointment_pack_processing.services.text_extraction_service import (
    PasswordProtectedPdfError,
    PdfPageLimitExceededError,
    TextExtractionError,
    TextExtractionService,
    TextNotFoundError,
    UnreadablePdfError,
)


def build_pdf(*page_texts: str) -> bytes:
    """Build a small in-memory PDF using synthetic text."""
    document = pymupdf.open()

    try:
        for page_text in page_texts:
            page = document.new_page()

            if page_text:
                page.insert_text((72, 72), page_text)

        return document.tobytes()
    finally:
        document.close()


def build_password_protected_pdf() -> bytes:
    """Build a small encrypted PDF for password-protection testing."""
    document = pymupdf.open()

    try:
        page = document.new_page()
        page.insert_text((72, 72), "Protected document")

        return document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    finally:
        document.close()


@pytest.fixture
def ocr_service() -> Mock:
    """Provide a mocked OCR dependency."""
    return Mock(spec=OcrService)


@pytest.fixture
def service(
    ocr_service: Mock,
) -> TextExtractionService:
    """Provide the text-extraction service with a mocked OCR dependency."""
    return TextExtractionService(
        maximum_pdf_pages=5,
        ocr_service=ocr_service,
    )


def test_extract_dispatches_jpeg_to_image_ocr(
    service: TextExtractionService,
    ocr_service: Mock,
) -> None:
    ocr_service.extract_image_text.return_value = "Image text"
    content = b"jpeg-content"

    result = service.extract(
        "image/jpeg",
        content,
    )

    assert result == "Image text"
    ocr_service.extract_image_text.assert_called_once_with(content)


def test_extract_dispatches_png_to_image_ocr(
    service: TextExtractionService,
    ocr_service: Mock,
) -> None:
    ocr_service.extract_image_text.return_value = "Image text"
    content = b"png-content"

    result = service.extract(
        "image/png",
        content,
    )

    assert result == "Image text"
    ocr_service.extract_image_text.assert_called_once_with(content)


def test_extract_rejects_unsupported_content_type(
    service: TextExtractionService,
) -> None:
    with pytest.raises(
        TextExtractionError,
        match="The document content type cannot be processed",
    ):
        service.extract(
            "text/plain",
            b"text",
        )


def test_extract_pdf_uses_embedded_text_without_ocr(
    service: TextExtractionService,
    ocr_service: Mock,
) -> None:
    result = service.extract(
        "application/pdf",
        build_pdf("Embedded PDF text"),
    )

    assert result == "Embedded PDF text"
    ocr_service.extract_pdf_page_text.assert_not_called()


def test_extract_pdf_uses_ocr_only_for_pages_without_embedded_text(
    service: TextExtractionService,
    ocr_service: Mock,
) -> None:
    ocr_service.extract_pdf_page_text.return_value = "Scanned page text"

    result = service.extract(
        "application/pdf",
        build_pdf(
            "Embedded page text",
            "",
        ),
    )

    assert result == "Embedded page text\n\nScanned page text"
    ocr_service.extract_pdf_page_text.assert_called_once()


def test_extract_pdf_rejects_password_protected_document(
    service: TextExtractionService,
) -> None:
    with pytest.raises(
        PasswordProtectedPdfError,
        match="Password-protected PDFs are not supported",
    ):
        service.extract(
            "application/pdf",
            build_password_protected_pdf(),
        )


def test_extract_pdf_rejects_document_above_page_limit(
    ocr_service: Mock,
) -> None:
    service = TextExtractionService(
        maximum_pdf_pages=1,
        ocr_service=ocr_service,
    )

    with pytest.raises(
        PdfPageLimitExceededError,
        match="PDF exceeds the maximum supported page count",
    ):
        service.extract(
            "application/pdf",
            build_pdf(
                "Page one",
                "Page two",
            ),
        )


def test_extract_pdf_translates_unreadable_content(
    service: TextExtractionService,
) -> None:
    with pytest.raises(
        UnreadablePdfError,
        match="The uploaded PDF could not be read",
    ):
        service.extract(
            "application/pdf",
            b"not-a-pdf",
        )


def test_extract_pdf_rejects_document_with_no_readable_text(
    service: TextExtractionService,
    ocr_service: Mock,
) -> None:
    ocr_service.extract_pdf_page_text.return_value = ""

    with pytest.raises(
        TextNotFoundError,
        match="No readable text was detected in the PDF",
    ):
        service.extract(
            "application/pdf",
            build_pdf(""),
        )


def test_normalise_text_collapses_whitespace_and_blank_lines(
    service: TextExtractionService,
) -> None:
    result = service._normalise_text(
        "  First   line  \n\n Second\tline "
    )

    assert result == "First line\nSecond line"