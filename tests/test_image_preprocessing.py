"""Tests for image preprocessing before OCR."""

from PIL import Image

from appointment_pack_processing.services.image_preprocessing_service import (
    ImagePreprocessingService,
)


def test_preprocess_returns_rgb_image_when_disabled() -> None:
    """Checks that disabled preprocessing returns an RGB image without resizing it."""
    service = ImagePreprocessingService(
        enabled=False,
        minimum_image_width=200,
    )

    source = Image.new(
        "L",
        (100, 50),
        color=128,
    )

    result = service.preprocess(source)

    assert result.mode == "RGB"
    assert result.size == (100, 50)


def test_preprocess_enlarges_small_image_when_enabled() -> None:
    """Checks that small images are enlarged when preprocessing is enabled."""
    service = ImagePreprocessingService(
        enabled=True,
        minimum_image_width=200,
    )

    source = Image.new(
        "RGB",
        (100, 50),
        color="white",
    )

    result = service.preprocess(source)

    assert result.mode == "L"
    assert result.size == (200, 100)


def test_preprocess_keeps_large_image_dimensions_when_enabled() -> None:
    """Checks that images already above the minimum width keep their dimensions."""
    service = ImagePreprocessingService(
        enabled=True,
        minimum_image_width=200,
    )

    source = Image.new(
        "RGB",
        (300, 150),
        color="white",
    )

    result = service.preprocess(source)

    assert result.mode == "L"
    assert result.size == (300, 150)
