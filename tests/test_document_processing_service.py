"""Tests for document processing and summarisation behaviour."""

from datetime import date, time
from unittest.mock import Mock
from uuid import uuid4

import pytest

from appointment_pack_processing import __version__
from appointment_pack_processing.schemas import DocumentType, RedactionContext
from appointment_pack_processing.services.appointment_details_service import (
    AppointmentAddressDetails,
    AppointmentDetails,
    AppointmentDetailsResult,
    AppointmentDetailsService,
)
from appointment_pack_processing.services.deidentification_service import (
    DeidentificationResult,
    DeidentificationService,
)
from appointment_pack_processing.services.document_processing_service import (
    AiInputTooLongError,
    ApprovedTextBlankError,
    DocumentProcessingService,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedContentTypeError,
)
from appointment_pack_processing.services.openai_summary_service import (
    OpenAiSummaryResult,
    OpenAiSummaryService,
)
from appointment_pack_processing.services.text_extraction_service import TextExtractionService


def build_service(
    *,
    maximum_file_size_bytes: int = 1024,
    maximum_ai_input_characters: int = 1000,
) -> tuple[
    DocumentProcessingService,
    Mock,
    Mock,
    Mock,
    Mock,
]:
    """Build the orchestration service with mocked services."""
    text_extraction_service = Mock(spec=TextExtractionService)
    deidentification_service = Mock(spec=DeidentificationService)
    appointment_details_service = Mock(spec=AppointmentDetailsService)
    openai_summary_service = Mock(spec=OpenAiSummaryService)

    service = DocumentProcessingService(
        maximum_file_size_bytes=maximum_file_size_bytes,
        maximum_ai_input_characters=maximum_ai_input_characters,
        text_extraction_service=text_extraction_service,
        deidentification_service=deidentification_service,
        appointment_details_service=appointment_details_service,
        openai_summary_service=openai_summary_service,
    )

    return (
        service,
        text_extraction_service,
        deidentification_service,
        appointment_details_service,
        openai_summary_service,
    )


def test_extract_rejects_empty_document_before_text_extraction() -> None:
    """Checks that an empty document is rejected before text extraction begins."""
    service, text_extraction_service, _, _, _ = build_service()

    with pytest.raises(
        EmptyDocumentError,
        match="Document file must not be empty",
    ):
        service.extract(
            document_id=uuid4(),
            document_type=DocumentType.APPOINTMENT_LETTER,
            content_type="application/pdf",
            content=b"",
            redaction_context=RedactionContext(),
        )

    text_extraction_service.extract.assert_not_called()


def test_extract_rejects_document_above_size_limit() -> None:
    """Checks that documents above the configured size limit are rejected."""
    service, text_extraction_service, _, _, _ = build_service(
        maximum_file_size_bytes=4,
    )

    with pytest.raises(
        DocumentTooLargeError,
        match="Document file exceeds the maximum size",
    ):
        service.extract(
            document_id=uuid4(),
            document_type=DocumentType.APPOINTMENT_LETTER,
            content_type="application/pdf",
            content=b"12345",
            redaction_context=RedactionContext(),
        )

    text_extraction_service.extract.assert_not_called()


def test_extract_rejects_unsupported_media_type() -> None:
    """Checks that unsupported document media types are rejected."""
    service, text_extraction_service, _, _, _ = build_service()

    with pytest.raises(
        UnsupportedContentTypeError,
        match="Only PDF, JPEG and PNG documents are supported",
    ):
        service.extract(
            document_id=uuid4(),
            document_type=DocumentType.APPOINTMENT_LETTER,
            content_type="text/plain",
            content=b"document",
            redaction_context=RedactionContext(),
        )

    text_extraction_service.extract.assert_not_called()


def test_extract_processes_appointment_letter_without_deidentification() -> None:
    """Checks that appointment letters use appointment extraction without deidentification."""
    (
        service,
        text_extraction_service,
        deidentification_service,
        appointment_details_service,
        _,
    ) = build_service()

    document_id = uuid4()
    content = b"appointment"
    extracted_text = "Date: 20/08/2026\nTime: 09:30\nLocation: Example Hospital"

    text_extraction_service.extract.return_value = extracted_text
    appointment_details_service.extract.return_value = AppointmentDetailsResult(
        details=AppointmentDetails(
            date=date(2026, 8, 20),
            start_time=time(9, 30),
            end_time=None,
            service="Neurology",
            appointment_type="Outpatient appointment",
            clinician_or_team="Dr Smith",
            location_name="Example Hospital",
            address=AppointmentAddressDetails(
                address_line_1="1 Example Street",
                address_line_2=None,
                town_city="Exampletown",
                county=None,
                postcode="AB1 2DE",
                country="United Kingdom",
            ),
        ),
        processing_warning=None,
    )

    response = service.extract(
        document_id=document_id,
        document_type=DocumentType.APPOINTMENT_LETTER,
        content_type="application/pdf",
        content=content,
        redaction_context=RedactionContext(
            known_values=["Example Patient"],
        ),
    )

    text_extraction_service.extract.assert_called_once_with(
        content_type="application/pdf",
        content=content,
    )
    appointment_details_service.extract.assert_called_once_with(extracted_text)
    deidentification_service.deidentify.assert_not_called()

    assert response.document_id == document_id
    assert response.extracted_text == extracted_text
    assert response.deidentified_text is None
    assert response.processing_warning is None
    assert response.processor_version == __version__

    assert response.appointment_details is not None
    assert response.appointment_details.date == date(2026, 8, 20)
    assert response.appointment_details.start_time == time(9, 30)
    assert response.appointment_details.service == "Neurology"

    assert response.appointment_details.address is not None
    assert response.appointment_details.address.postcode == "AB1 2DE"


def test_extract_processes_consultation_without_appointment_extraction() -> None:
    """Checks that consultation letters use deidentification without appointment extraction."""
    (
        service,
        text_extraction_service,
        deidentification_service,
        appointment_details_service,
        _,
    ) = build_service()

    document_id = uuid4()
    context = RedactionContext(
        known_values=["Example Patient"],
    )
    extracted_text = "Example Patient attended the clinic."

    text_extraction_service.extract.return_value = extracted_text
    deidentification_service.deidentify.return_value = DeidentificationResult(
        text="[REDACTED] attended the clinic.",
        processing_warning="Review the de-identified text.",
    )

    response = service.extract(
        document_id=document_id,
        document_type=DocumentType.CONSULTATION_OUTCOME_LETTER,
        content_type="image/png",
        content=b"image",
        redaction_context=context,
    )

    text_extraction_service.extract.assert_called_once_with(
        content_type="image/png",
        content=b"image",
    )
    deidentification_service.deidentify.assert_called_once_with(
        text=extracted_text,
        context=context,
    )
    appointment_details_service.extract.assert_not_called()

    assert response.document_id == document_id
    assert response.extracted_text == extracted_text
    assert response.deidentified_text == "[REDACTED] attended the clinic."
    assert response.appointment_details is None
    assert response.processing_warning == "Review the de-identified text."
    assert response.processor_version == __version__


def test_summarise_rejects_blank_approved_text() -> None:
    """Checks that blank approved consultation text is rejected."""
    service, _, _, _, openai_summary_service = build_service()

    with pytest.raises(
        ApprovedTextBlankError,
        match="Approved de-identified text must not be blank",
    ):
        service.summarise(
            document_id=uuid4(),
            approved_deidentified_text="   ",
        )

    openai_summary_service.summarise.assert_not_called()


def test_summarise_rejects_approved_text_above_character_limit() -> None:
    """Checks that approved consultation text above the character limit is rejected."""
    service, _, _, _, openai_summary_service = build_service(
        maximum_ai_input_characters=5,
    )

    with pytest.raises(
        AiInputTooLongError,
        match="Approved de-identified text exceeds the maximum supported length",
    ):
        service.summarise(
            document_id=uuid4(),
            approved_deidentified_text="123456",
        )

    openai_summary_service.summarise.assert_not_called()


def test_summarise_maps_openai_result_to_api_response() -> None:
    """Checks that a generated summary and its metadata are mapped into the API response."""
    service, _, _, _, openai_summary_service = build_service()

    document_id = uuid4()
    approved_text = "The patient reported improved symptoms."

    openai_summary_service.summarise.return_value = OpenAiSummaryResult(
        summary="The patient reported improved symptoms.",
        model_name="gpt-5-nano",
        prompt_version="consultation-summary-v1",
    )

    response = service.summarise(
        document_id=document_id,
        approved_deidentified_text=approved_text,
    )

    openai_summary_service.summarise.assert_called_once_with(approved_text)

    assert response.document_id == document_id
    assert response.summary == "The patient reported improved symptoms."
    assert response.processor_version == __version__
    assert response.model_name == "gpt-5-nano"
    assert response.prompt_version == "consultation-summary-v1"
