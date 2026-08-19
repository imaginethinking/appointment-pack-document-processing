import pytest

from appointment_pack_processing.schemas import RedactionContext
from appointment_pack_processing.services.deidentification_service import (
    DeidentificationService,
)


@pytest.fixture
def service() -> DeidentificationService:
    """Provide the deterministic de-identification service."""
    return DeidentificationService()


def deidentify(
    service: DeidentificationService,
    text: str,
    known_values: list[str] | None = None,
) -> str:
    """Return only the redacted text for concise test assertions."""
    result = service.deidentify(
        text=text,
        context=RedactionContext(
            known_values=known_values or [],
        ),
    )

    return result.text


def test_known_values_are_redacted_case_insensitively(
    service: DeidentificationService,
) -> None:
    result = deidentify(
        service,
        "ALEX EXAMPLE attended clinic. Alex Example was reviewed.",
        ["Alex Example"],
    )

    assert result == ("[REDACTED] attended clinic. [REDACTED] was reviewed.")


def test_known_values_are_normalised_and_deduplicated(
    service: DeidentificationService,
) -> None:
    result = deidentify(
        service,
        "Alex   Example attended clinic.",
        [
            "  Alex   Example  ",
            "alex example",
            "A",
        ],
    )

    assert result == "[REDACTED] attended clinic."


def test_longer_known_values_are_replaced_before_overlapping_shorter_values(
    service: DeidentificationService,
) -> None:
    result = deidentify(
        service,
        "Alex Example attended with Example.",
        [
            "Example",
            "Alex Example",
        ],
    )

    assert result == ("[REDACTED] attended with [REDACTED].")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Email: patient@example.com",
            "Email: [REDACTED]",
        ),
        (
            "Postcode: AB1 2DE",
            "Postcode: [REDACTED]",
        ),
        (
            "Telephone: 07123 456789",
            "Telephone: [REDACTED]",
        ),
        (
            "NHS number: 123 456 7890",
            "NHS number: [REDACTED]",
        ),
        (
            "CHI no: 0101011234",
            "CHI no: [REDACTED]",
        ),
        (
            "MRN: ABC12345",
            "MRN: [REDACTED]",
        ),
        (
            "DOB: 01/01/2000",
            "DOB: [REDACTED]",
        ),
    ],
)
def test_supported_identifier_patterns_are_redacted(
    service: DeidentificationService,
    text: str,
    expected: str,
) -> None:
    assert deidentify(service, text) == expected


def test_unlabelled_clinical_dates_are_not_redacted(
    service: DeidentificationService,
) -> None:
    text = "The consultation occurred on 20/08/2026 and follow-up is on 20/09/2026."

    assert deidentify(service, text) == text


def test_adjacent_repeated_placeholders_are_collapsed(
    service: DeidentificationService,
) -> None:
    result = deidentify(
        service,
        "Alex Example, AB1 2DE",
        ["Alex Example"],
    )

    assert result == "[REDACTED]"


def test_deidentify_returns_human_review_warning(
    service: DeidentificationService,
) -> None:
    result = service.deidentify(
        text="No identifiers in this sentence.",
        context=RedactionContext(),
    )

    assert result.processing_warning == service.REVIEW_WARNING
