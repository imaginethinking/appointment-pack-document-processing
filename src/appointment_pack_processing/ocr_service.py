from io import BytesIO

import pymupdf
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import TesseractError, TesseractNotFoundError

from appointment_pack_processing.image_preprocessing_service import (
    ImagePreprocessingService,
)


class OcrError(ValueError):
    """Base exception for expected OCR failures."""


class OcrUnavailableError(OcrError):
    """Raised when the Tesseract executable is unavailable."""


class UnreadableImageError(OcrError):
    """Raised when uploaded image content cannot be decoded."""


class OcrProcessingError(OcrError):
    """Raised when Tesseract fails while processing an image."""


class OcrTextNotFoundError(OcrError):
    """Raised when OCR produces no usable text."""


class OcrService:
    def __init__(
        self,
        language: str,
        pdf_dpi: int,
        image_preprocessing_service: ImagePreprocessingService,
        tesseract_command: str | None = None,
    ) -> None:
        self.language = language
        self.pdf_dpi = pdf_dpi
        self.image_preprocessing_service = (
            image_preprocessing_service
        )

        if tesseract_command:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_command
            )

    def extract_image_text(
        self,
        content: bytes,
    ) -> str:
        try:
            with Image.open(BytesIO(content)) as image:
                oriented_image = ImageOps.exif_transpose(
                    image
                )

                prepared_image = (
                    self.image_preprocessing_service.preprocess(
                        oriented_image
                    )
                )

                extracted_text = self._run_ocr(
                    prepared_image
                )
        except UnidentifiedImageError as exception:
            raise UnreadableImageError(
                "The uploaded image could not be read"
            ) from exception
        except OSError as exception:
            raise UnreadableImageError(
                "The uploaded image could not be read"
            ) from exception

        if not extracted_text:
            raise OcrTextNotFoundError(
                "No readable text was detected in the uploaded image"
            )

        return extracted_text

    def extract_pdf_page_text(
        self,
        page: pymupdf.Page,
    ) -> str:
        pixmap = page.get_pixmap(
            dpi=self.pdf_dpi,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            (
                pixmap.width,
                pixmap.height,
            ),
            pixmap.samples,
        )

        prepared_image = (
            self.image_preprocessing_service.preprocess(
                image
            )
        )

        return self._run_ocr(
            prepared_image
        )

    def _run_ocr(
        self,
        image: Image.Image,
    ) -> str:
        try:
            extracted_text = pytesseract.image_to_string(
                image,
                lang=self.language,
            )
        except TesseractNotFoundError as exception:
            raise OcrUnavailableError(
                "Tesseract OCR is not installed or configured"
            ) from exception
        except TesseractError as exception:
            raise OcrProcessingError(
                "Tesseract failed to process the document"
            ) from exception

        return self._normalise_text(
            extracted_text
        )

    def _normalise_text(
        self,
        text: str,
    ) -> str:
        normalised_lines = []

        for line in text.splitlines():
            normalised_line = " ".join(
                line.split()
            )

            if normalised_line:
                normalised_lines.append(
                    normalised_line
                )

        return "\n".join(
            normalised_lines
        )