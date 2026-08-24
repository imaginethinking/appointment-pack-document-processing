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
    result = service.extract(f"Date: {document_date}\nTime: 09:30\nLocation: Example Hospital")

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
    result = service.extract(f"Date: 20/08/2026\nTime: {document_time}\nLocation: Example Hospital")

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


def test_extract_supports_values_on_following_lines(
    service: AppointmentDetailsService,
) -> None:
    text = """Date:
20 August 2026
Time:
09:30
Location:
Example Hospital"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.location_name == "Example Hospital"


def test_extract_does_not_take_another_label_as_next_line_value(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract("End time:\nService: Neurology\nLocation: Example Hospital")

    assert result.details.end_time is None
    assert result.details.service == "Neurology"


def test_extract_supports_combined_date_and_time_label(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date and time: Thursday 20 August 2026 at 9:30am\nLocation: Example Hospital"
    )

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)


def test_extract_supports_combined_date_and_time_on_following_line(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Appointment date and time:\n20 August 2026 at 14:15\nLocation: Example Hospital"
    )

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(14, 15)


def test_extract_prefers_separator_labels_when_context_is_equivalent(
    service: AppointmentDetailsService,
) -> None:
    text = """Date 21/08/2026
Time 10:45
Service General Medicine
Location Fallback Clinic
Administrative section
Administrative section
Administrative section
Administrative section
Administrative section
Administrative section
Administrative section
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
    result = service.extract("Date - 20/08/2026\nTime - 09:30\nLocation - Example Hospital")

    assert result.details.date == date(2026, 8, 20)
    assert result.details.start_time == time(9, 30)
    assert result.details.location_name == "Example Hospital"


def test_extract_continues_after_unparseable_date_candidate(
    service: AppointmentDetailsService,
) -> None:
    text = """Date of birth 14/02/1967
Reference REF-001
Date 03 September 2026
Time 10:20
Service Cardiology
Location Cardiology Outpatients"""

    result = service.extract(text)

    assert result.details.date == date(2026, 9, 3)


def test_extract_continues_after_unparseable_time_candidate(
    service: AppointmentDetailsService,
) -> None:
    text = """Time to be confirmed
Date 03 September 2026
Time 10:20
Service Cardiology
Location Cardiology Outpatients"""

    result = service.extract(text)

    assert result.details.start_time == time(10, 20)


def test_extract_prefers_candidate_with_stronger_appointment_context(
    service: AppointmentDetailsService,
) -> None:
    text = """Date 01/08/2026
Reference REF-001
Administrative information
Administrative information
Administrative information
Administrative information
Administrative information
Administrative information
Administrative information
Date 20/08/2026
Time 09:30
Service Neurology
Appointment type Outpatient appointment
With Neurology Team
Location Neurology Outpatients"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)


def test_extract_uses_context_to_select_appointment_address(
    service: AppointmentDetailsService,
) -> None:
    text = """Patient Example Patient
Date of birth 14/02/1967
Address 18 Example Close
Reference REF-001
Date 03 September 2026
Time 10:20
Service Cardiology
Appointment type New outpatient consultation
With Hypertension Clinic
Location Cardiology Outpatients
Address 12 Hospital Way
Town/City Exampletown
County Exampleshire
Postcode AB1 2DE
Country United Kingdom"""

    result = service.extract(text)

    assert result.details.address is not None
    assert result.details.address.address_line_1 == "12 Hospital Way"
    assert result.details.address.town_city == "Exampletown"
    assert result.details.address.county == "Exampleshire"
    assert result.details.address.postcode == "AB1 2DE"
    assert result.details.address.country == "United Kingdom"


def test_extract_handles_town_or_city_label_without_partial_label_match(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date: 20/08/2026\nTime: 09:30\nVenue: Example Unit\nTown or city: Eastmere"
    )

    assert result.details.address is not None
    assert result.details.address.town_city == "Eastmere"


def test_extract_reconstructs_narrative_date_and_time_across_lines(
    service: AppointmentDetailsService,
) -> None:
    text = """Dear Patient, we have arranged an appointment for you on 17 September
2026 at
14:15 with the Cardiac Physiology Team.
Clinic details
Service: Cardiology Diagnostics
Venue: Diagnostic Investigations Unit"""

    result = service.extract(text)

    assert result.details.date == date(2026, 9, 17)
    assert result.details.start_time == time(14, 15)


def test_extract_reconstructs_narrative_date_split_after_month(
    service: AppointmentDetailsService,
) -> None:
    text = """Your appointment has been arranged for 17 September
2026 at 14:15.
Location: Example Hospital"""

    result = service.extract(text)

    assert result.details.date == date(2026, 9, 17)
    assert result.details.start_time == time(14, 15)


def test_extract_narrative_clinician_team(
    service: AppointmentDetailsService,
) -> None:
    text = """We have arranged an appointment for you on 17 September 2026 at
14:15 with the Cardiac Physiology Team. Please bring your medication list.
Location: Example Hospital"""

    result = service.extract(text)

    assert result.details.clinician_or_team == "Cardiac Physiology Team"


def test_extract_narrative_appointment_type(
    service: AppointmentDetailsService,
) -> None:
    text = """Your appointment is for
Ambulatory blood pressure monitor fitting. Please bring your medication list.
Date: 17/09/2026
Time: 14:15
Location: Example Hospital"""

    result = service.extract(text)

    assert result.details.appointment_type == "Ambulatory blood pressure monitor fitting"


def test_extract_narrative_location(
    service: AppointmentDetailsService,
) -> None:
    text = """We have arranged an appointment for you on 17 September 2026 at 14:15.
Please attend the Diagnostic Investigations Unit, Example University Hospital.
Service: Cardiology"""

    result = service.extract(text)

    assert result.details.location_name == (
        "Diagnostic Investigations Unit, Example University Hospital"
    )


def test_extract_labelled_location_outweighs_narrative_location(
    service: AppointmentDetailsService,
) -> None:
    text = """We have arranged an appointment for you on 17 September 2026 at 14:15.
Please attend the Diagnostic Investigations Unit, Example University Hospital.
Service: Cardiology
Venue: Diagnostic Investigations Unit
Address: 12 Hospital Way
Town or city: Exampletown"""

    result = service.extract(text)

    assert result.details.location_name == "Diagnostic Investigations Unit"


def test_extract_narrative_end_time(
    service: AppointmentDetailsService,
) -> None:
    text = """Your appointment is on 17 September 2026 at 14:15.
The appointment is expected to finish
at approximately 14:45.
Location: Example Hospital"""

    result = service.extract(text)

    assert result.details.end_time == time(14, 45)


def test_extract_does_not_use_office_hours_as_appointment_end_time(
    service: AppointmentDetailsService,
) -> None:
    text = """Your appointment is on 17 September 2026 at 14:15.
Location: Example Hospital
The Appointments Office is open from 09:00 to 17:00."""

    result = service.extract(text)

    assert result.details.start_time == time(14, 15)
    assert result.details.end_time is None


def test_extract_does_not_use_phone_number_as_time(
    service: AppointmentDetailsService,
) -> None:
    text = """Your appointment is on 17 September 2026 at 14:15.
Location: Example Hospital
If you cannot attend, call 01632 960500."""

    result = service.extract(text)

    assert result.details.start_time == time(14, 15)


def test_extract_narrative_service_when_strong_service_wording_is_present(
    service: AppointmentDetailsService,
) -> None:
    text = """You are booked for an appointment with the Respiratory Medicine Department.
Date: 20/08/2026
Time: 09:30
Location: Example Hospital"""

    result = service.extract(text)

    assert result.details.service == "Respiratory Medicine Department"


def test_extract_keeps_best_available_candidate_without_confidence_threshold(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract("Date 20/08/2026")

    assert result.details.date == date(2026, 8, 20)
    assert result.processing_warning is not None
    assert "start time and location" in result.processing_warning


def test_extract_warns_when_different_candidates_are_equally_plausible(
    service: AppointmentDetailsService,
) -> None:
    text = """Date: 20/08/2026
Time: 09:30
Location: First Clinic
Administrative text
Administrative text
Administrative text
Administrative text
Administrative text
Administrative text
Administrative text
Date: 21/08/2026
Time: 10:30
Location: Second Clinic"""

    result = service.extract(text)

    assert result.details.date == date(2026, 8, 20)
    assert result.processing_warning is not None
    assert "Multiple plausible values were identified" in result.processing_warning
    assert "date" in result.processing_warning


def test_extract_returns_partial_address_without_guessing_missing_fields(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract(
        "Date: 20/08/2026\nTime: 09:30\nAddress: Example Hospital\nPostcode: AB1 2DE"
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
    result = service.extract("Date: sometime next week\nTime: morning\nLocation: Example Hospital")

    assert result.details.date is None
    assert result.details.start_time is None
    assert result.details.location_name == "Example Hospital"

    assert result.processing_warning is not None
    assert "date and start time" in result.processing_warning


def test_extract_warns_when_no_supported_fields_are_found(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract("Please bring your medication list with you.")

    assert result.details.date is None
    assert result.details.start_time is None
    assert result.details.location_name is None
    assert result.details.address is None

    assert result.processing_warning == (
        "No supported appointment details were identified. "
        "Review the extracted text and enter the appointment details manually."
    )


def test_extract_warns_when_core_fields_are_missing(
    service: AppointmentDetailsService,
) -> None:
    result = service.extract("Service: Neurology\nWith: Dr Smith")

    assert result.processing_warning == (
        "Some core appointment details could not be identified or normalised: "
        "date, start time, and location. "
        "Review the extracted text before confirming the appointment."
    )


def test_extract_regression_for_metadata_and_appointment_detail_blocks(
    service: AppointmentDetailsService,
) -> None:
    text = """Patient correspondence
Example University Hospital
Outpatient appointment
Patient Example Patient
Date of birth 14/02/1967
NHS number 9990000001
Reference APT-01-A-2026
Address 18 Example Close, Exampletown
Letter date 19/08/2026
Dear Patient,
Your appointment details are shown below.
Date 03 September 2026
Time 10:20
End time
Service Cardiology
Appointment type New outpatient consultation
With Hypertension Clinic
Location Cardiology Outpatients
Address 12 Hospital Way
Town/City Exampletown
County Exampleshire
Postcode AB1 2DE
Country United Kingdom"""

    result = service.extract(text)

    assert result.details.date == date(2026, 9, 3)
    assert result.details.start_time == time(10, 20)
    assert result.details.service == "Cardiology"
    assert result.details.appointment_type == "New outpatient consultation"
    assert result.details.clinician_or_team == "Hypertension Clinic"
    assert result.details.location_name == "Cardiology Outpatients"
    assert result.details.address is not None
    assert result.details.address.address_line_1 == "12 Hospital Way"


def test_extract_regression_for_wrapped_narrative_letter(
    service: AppointmentDetailsService,
) -> None:
    text = """Example University Hospital
Appointment information
Patient: Example Patient
Date of birth: 14/02/1967
NHS number: 9990000001
Address: 18 Example Close, Exampletown, Exampleshire, AB4 7CD, United Kingdom
Dear Patient, we have arranged an appointment for you on 17 September 2026 at
14:15 with the Cardiac Physiology Team. Please attend the Diagnostic
Investigations Unit, Example University Hospital. The appointment is for
Ambulatory blood pressure monitor fitting. The appointment is expected to finish
at approximately 14:45.
Clinic details
Service: Cardiology Diagnostics
Venue: Diagnostic Investigations Unit
Address: 12 Hospital Way
Town or city: Exampletown
County: Exampleshire
Postcode: AB1 2DE
Country: United Kingdom"""

    result = service.extract(text)

    assert result.details.date == date(2026, 9, 17)
    assert result.details.start_time == time(14, 15)
    assert result.details.end_time == time(14, 45)
    assert result.details.service == "Cardiology Diagnostics"
    assert result.details.appointment_type == "Ambulatory blood pressure monitor fitting"
    assert result.details.clinician_or_team == "Cardiac Physiology Team"
    assert result.details.location_name == "Diagnostic Investigations Unit"
    assert result.details.address is not None
    assert result.details.address.address_line_1 == "12 Hospital Way"
    assert result.details.address.town_city == "Exampletown"
    assert result.details.address.county == "Exampleshire"
    assert result.details.address.postcode == "AB1 2DE"
    assert result.details.address.country == "United Kingdom"
    assert result.processing_warning is None
