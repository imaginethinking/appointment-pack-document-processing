from io import BytesIO

import pymupdf
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import TesseractError, TesseractNotFoundError


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
        tesseract_command: str | None = None,
    ) -> None:
        self.language = language
        self.pdf_dpi = pdf_dpi

        if tesseract_command:
            pytesseract.pytesseract.tesseract_cmd = tesseract_command

    def extract_image_text(self, content: bytes) -> str:
        try:
            with Image.open(BytesIO(content)) as image:
                prepared_image = ImageOps.exif_transpose(image).convert("RGB")
                extracted_text = self._run_ocr(prepared_image)
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
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )

        return self._run_ocr(image)

    def _run_ocr(self, image: Image.Image) -> str:
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

        return self._normalise_text(extracted_text)

    def _normalise_text(self, text: str) -> str:
        normalised_lines = []

        for line in text.splitlines():
            normalised_line = " ".join(line.split())

            if normalised_line:
                normalised_lines.append(normalised_line)

        return "\n".join(normalised_lines)