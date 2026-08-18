from datetime import date, time

import pytest

from appointment_pack_processing.services.appointment_details_service import (
    AppointmentDetailsService,
)


@pytest.fixture
def service() -> AppointmentDetailsService:
    """Provide the deterministic appointment-details service."""
    return AppointmentDetailsService()


@pytest.mark.parametrize(
    ("document_date", "expected_date"),
    [
        ("2026-08-20", date(2026, 8, 20)),
        ("20/08/2026", date(2026, 8, 20)),
        ("20-08-2026", date(2026, 8, 20)),
        ("20.08.2026", date(2026, 8, 20)),
        ("20 August 2026", date(2026, 8, 20)),
        ("20 Aug 2026", date(2026, 8, 20)),
        ("Thursday 20 August 2026", date(2026, 8, 20)),
        ("20th August 2026", date(2026, 8, 20)),
    ],
)
def test_extract_normalises_supported_dates(
    service: AppointmentDetailsService,
    document_date: str,
    expected_date: date,
) -> None:
    result = service.extract(
        f"Date: {document_date}\n"
        "Time: 09:30\n"
        "Location: Example Hospital"
    )

    assert result.details.date == expected_date


@pytest.mark.parametrize(
    ("document_time", "expected_time"),
    [
        ("09:30", time(9, 30)),
        ("09.30", time(9, 30)),
        ("9:30am", time(9, 30)),
        ("9.30 am", time(9, 30)),
        ("2 PM", time(14, 0)),
        ("2pm", time(14, 0)),
    ],
)
def test_extract_normalises_supported_start_times(
    service: AppointmentDetailsService,
    document_time: str,
    expected_time: time,
) -> None:
    result = service.extract(
        "Date: 20/08/2026\n"
        f"Time: {document_time}\n"
        "Location: Example Hospital"
    )

    assert result.details.start_time == expected_time


def test_extract_reads_supported_labelled_fields(
    service: AppointmentDetailsService,
) -> None:
    text = """Date: 20/08/2026
Start time: 09:30
End time: 10:15
Service: Neurology
Appointment type: Outpatient appointment
With: Dr Smith
Location: Neurology Outpatients
Address line 1: Example Hospital
Address line 2: Example Road
Town/City: Exampletown
County: Exampleshire
Postcode: AB1 2DE
Country: United Kingdom"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.end_time == time(10, 15)
    assert result.details.service == "Neurology"
    assert result.details.appointment_type == "Outpatient appointment"
    assert result.details.clinician_or_team == "Dr Smith"
    assert result.details.location_name == "Neurology Outpatients"

    assert result.details.address is not None
    assert result.details.address.address_line_1 == "Example Hospital"
    assert result.details.address.address_line_2 == "Example Road"
    assert result.details.address.town_city == "Exampletown"
    assert result.details.address.county == "Exampleshire"
    assert result.details.address.postcode == "AB1 2DE"
    assert result.details.address.country == "United Kingdom"

    assert result.processing_warning is None


def test_extract_accepts_labels_with_whitespace_instead_of_separator(
    service: AppointmentDetailsService,
) -> None:
    text = """Date 20/08/2026
Start time 09:30
End time 10:15
Service Neurology
Appointment type Outpatient appointment
With Dr Smith
Location Neurology Outpatients
Address line 1 Example Hospital
Address line 2 Example Road
Town/City Exampletown
County Exampleshire
Postcode AB1 2DE
Country United Kingdom"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.end_time == time(10, 15)
    assert result.details.service == "Neurology"
    assert result.details.appointment_type == "Outpatient appointment"
    assert result.details.clinician_or_team == "Dr Smith"
    assert result.details.location_name == "Neurology Outpatients"

    assert result.details.address is not None
    assert result.details.address.address_line_1 == "Example Hospital"
    assert result.details.address.address_line_2 == "Example Road"
    assert result.details.address.town_city == "Exampletown"
    assert result.details.address.county == "Exampleshire"
    assert result.details.address.postcode == "AB1 2DE"
    assert result.details.address.country == "United Kingdom"

    assert result.processing_warning is None


def test_extract_prefers_separator_labels_over_whitespace_fallback(
    service: AppointmentDetailsService,
) -> None:
    text = """Date 21/08/2026
Time 10:45
Service General Medicine
Location Fallback Clinic
Date: 20/08/2026
Time: 09:30
Service: Neurology
Location: Preferred Clinic"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.service == "Neurology"
    assert result.details.location_name == "Preferred Clinic"


def test_extract_supports_hyphen_separator(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date - 20/08/2026\n"
        "Time - 09:30\n"
        "Location - Example Hospital"
    )

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.location_name == "Example Hospital"


def test_extract_returns_partial_address_without_guessing_missing_fields(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date: 20/08/2026\n"
        "Time: 09:30\n"
        "Address: Example Hospital\n"
        "Postcode: AB1 2DE"
    )

    assert result.details.address is not None
    assert result.details.address.address_line_1 == "Example Hospital"
    assert result.details.address.postcode == "AB1 2DE"

    assert result.details.address.address_line_2 is None
    assert result.details.address.town_city is None
    assert result.details.address.county is None
    assert result.details.address.country is None


def test_extract_leaves_unparseable_values_as_none(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date: sometime next week\n"
        "Time: morning\n"
        "Location: Example Hospital"
    )

    assert result.details.date is None
    assert result.details.start_time is None
    assert result.details.location_name == "Example Hospital"

    assert result.processing_warning is not None
    assert "date and start time" in result.processing_warning


def test_extract_warns_when_no_supported_fields_are_found(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Please attend the hospital for your forthcoming appointment."
    )

    assert result.details.date is None
    assert result.details.start_time is None
    assert result.details.location_name is None
    assert result.details.address is None

    assert result.processing_warning == (
        "No supported labelled appointment fields were identified. "
        "Review the extracted text and enter the appointment details manually."
    )


def test_extract_warns_when_core_fields_are_missing(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Service: Neurology\n"
        "With: Dr Smith"
    )

    assert result.processing_warning == (
        "Some core appointment details could not be identified or normalised: "
        "date, start time, and location. "
        "Review the extracted text before confirming the appointment."
    )