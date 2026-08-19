from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pytesseract import TesseractError, TesseractNotFoundError

from appointment_pack_processing.services.image_preprocessing_service import (
    ImagePreprocessingService,
)
from appointment_pack_processing.services.ocr_service import (
    OcrProcessingError,
    OcrService,
    OcrTextNotFoundError,
    OcrUnavailableError,
    UnreadableImageError,
)


def build_png_bytes() -> bytes:
    """Build a small synthetic PNG image in memory."""
    image = Image.new(
        "RGB",
        (20, 10),
        color="white",
    )

    output = BytesIO()
    image.save(
        output,
        format="PNG",
    )

    return output.getvalue()


@pytest.fixture
def preprocessing_service() -> Mock:
    """Provide a mocked image-preprocessing dependency."""
    service = Mock(spec=ImagePreprocessingService)
    service.preprocess.side_effect = lambda image: image.convert("RGB")

    return service


@pytest.fixture
def service(
    preprocessing_service: Mock,
) -> OcrService:
    """Provide the OCR service with image preprocessing mocked."""
    return OcrService(
        language="eng",
        pdf_dpi=300,
        image_preprocessing_service=preprocessing_service,
    )


def test_extract_image_text_preprocesses_image_and_returns_ocr_text(
    service: OcrService,
    preprocessing_service: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ocr = Mock(return_value="Extracted text")
    monkeypatch.setattr(
        service,
        "_run_ocr",
        run_ocr,
    )

    result = service.extract_image_text(build_png_bytes())

    assert result == "Extracted text"
    preprocessing_service.preprocess.assert_called_once()
    run_ocr.assert_called_once()


def test_extract_image_text_rejects_unreadable_image(
    service: OcrService,
) -> None:
    with pytest.raises(
        UnreadableImageError,
        match="The uploaded image could not be read",
    ):
        service.extract_image_text(b"not-an-image")


def test_extract_image_text_rejects_empty_ocr_result(
    service: OcrService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service,
        "_run_ocr",
        Mock(return_value=""),
    )

    with pytest.raises(
        OcrTextNotFoundError,
        match="No readable text was detected in the uploaded image",
    ):
        service.extract_image_text(build_png_bytes())


def test_run_ocr_translates_missing_tesseract(
    service: OcrService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_tesseract(
        *args: object,
        **kwargs: object,
    ) -> str:
        raise TesseractNotFoundError()

    monkeypatch.setattr(
        ("appointment_pack_processing.services.ocr_service.pytesseract.image_to_string"),
        raise_missing_tesseract,
    )

    with pytest.raises(
        OcrUnavailableError,
        match="Tesseract OCR is not installed or configured",
    ):
        service._run_ocr(
            Image.new(
                "RGB",
                (10, 10),
                color="white",
            )
        )


def test_run_ocr_translates_tesseract_processing_error(
    service: OcrService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_tesseract_error(
        *args: object,
        **kwargs: object,
    ) -> str:
        raise TesseractError(
            1,
            "synthetic failure",
        )

    monkeypatch.setattr(
        ("appointment_pack_processing.services.ocr_service.pytesseract.image_to_string"),
        raise_tesseract_error,
    )

    with pytest.raises(
        OcrProcessingError,
        match="Tesseract failed to process the document",
    ):
        service._run_ocr(
            Image.new(
                "RGB",
                (10, 10),
                color="white",
            )
        )


def test_run_ocr_normalises_returned_text(
    service: OcrService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ("appointment_pack_processing.services.ocr_service.pytesseract.image_to_string"),
        Mock(return_value=(" First   line \n\n Second\tline ")),
    )

    result = service._run_ocr(
        Image.new(
            "RGB",
            (10, 10),
            color="white",
        )
    )

    assert result == "First line\nSecond line"


def test_extract_pdf_page_renders_at_configured_dpi_and_preprocesses(
    service: OcrService,
    preprocessing_service: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixmap = SimpleNamespace(
        width=2,
        height=1,
        samples=bytes(
            [
                255,
                255,
                255,
                0,
                0,
                0,
            ]
        ),
    )

    page = Mock()
    page.get_pixmap.return_value = pixmap

    run_ocr = Mock(return_value="Page text")
    monkeypatch.setattr(
        service,
        "_run_ocr",
        run_ocr,
    )

    result = service.extract_pdf_page_text(page)

    assert result == "Page text"

    page.get_pixmap.assert_called_once_with(
        dpi=300,
        alpha=False,
    )

    preprocessing_service.preprocess.assert_called_once()
    run_ocr.assert_called_once()
