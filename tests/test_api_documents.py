from datetime import date, time
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from starlette.datastructures import UploadFile as StarletteUploadFile

from appointment_pack_processing.schemas import (
    AppointmentAddressDetailsResponse,
    AppointmentDetailsResponse,
    DocumentExtractionResponse,
    DocumentSummaryResponse,
    DocumentType,
)
from appointment_pack_processing.services.document_processing_service import (
    AiInputTooLongError,
    ApprovedTextBlankError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedContentTypeError,
)
from appointment_pack_processing.services.ocr_service import OcrError
from appointment_pack_processing.services.openai_summary_service import (
    AiSummaryResponseError,
    AiSummaryTimeoutError,
    AiSummaryUnavailableError,
)
from appointment_pack_processing.services.text_extraction_service import TextExtractionError


def build_extraction_response(
    *,
    document_id: UUID | None = None,
) -> DocumentExtractionResponse:
    """Build a complete appointment extraction response for route tests."""
    resolved_document_id = document_id or uuid4()

    return DocumentExtractionResponse(
        document_id=resolved_document_id,
        extracted_text="Appointment letter text",
        deidentified_text=None,
        appointment_details=AppointmentDetailsResponse(
            date=date(2026, 8, 20),
            start_time=time(9, 30),
            end_time=None,
            service="Neurology",
            appointment_type="Outpatient appointment",
            clinician_or_team="Dr Smith",
            location_name="Neurology Outpatients",
            address=AppointmentAddressDetailsResponse(
                address_line_1="Example Hospital",
                address_line_2=None,
                town_city="Exampletown",
                county=None,
                postcode="AB1 2CD",
                country="United Kingdom",
            ),
        ),
        processing_warning=None,
        processor_version="0.6.2",
    )


def post_extract(
    client: TestClient,
    internal_api_key: str,
    *,
    document_id: UUID | None = None,
    document_type: str = "APPOINTMENT_LETTER",
    redaction_context: str = '{"knownValues": []}',
    content: bytes = b"document-bytes",
    content_type: str = "application/pdf",
) -> Response:
    """Submit a document extraction request using the current multipart contract."""
    resolved_document_id = document_id or uuid4()

    return client.post(
        "/internal/v1/documents/extract",
        headers={"X-Internal-Api-Key": internal_api_key},
        data={
            "documentId": str(resolved_document_id),
            "documentType": document_type,
            "redactionContext": redaction_context,
        },
        files={
            "file": (
                "document.pdf",
                content,
                content_type,
            )
        },
    )


def test_extract_requires_internal_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/documents/extract",
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
            "redactionContext": '{"knownValues": []}',
        },
        files={
            "file": (
                "appointment.pdf",
                b"document-bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal API key"}


def test_extract_rejects_incorrect_internal_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/documents/extract",
        headers={"X-Internal-Api-Key": "incorrect-key"},
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
            "redactionContext": '{"knownValues": []}',
        },
        files={
            "file": (
                "appointment.pdf",
                b"document-bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal API key"}


def test_extract_uses_current_multipart_contract_and_calls_service(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    document_id = uuid4()
    processing_service.extract.return_value = build_extraction_response(document_id=document_id)

    response = post_extract(
        client,
        internal_api_key,
        document_id=document_id,
        document_type="CONSULTATION_OUTCOME_LETTER",
        redaction_context=('{"knownValues": ["Example Patient", "AB1 2CD"]}'),
        content=b"consultation-content",
    )

    assert response.status_code == 200

    processing_service.extract.assert_called_once()
    call_args = processing_service.extract.call_args.args

    assert call_args[0] == document_id
    assert call_args[1] == DocumentType.CONSULTATION_OUTCOME_LETTER
    assert call_args[2] == "application/pdf"
    assert call_args[3] == b"consultation-content"
    assert call_args[4].known_values == ["Example Patient", "AB1 2CD"]


def test_extract_returns_current_response_shape(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    document_id = uuid4()
    processing_service.extract.return_value = build_extraction_response(document_id=document_id)

    response = post_extract(
        client,
        internal_api_key,
        document_id=document_id,
    )

    assert response.status_code == 200
    assert response.json() == {
        "documentId": str(document_id),
        "extractedText": "Appointment letter text",
        "deidentifiedText": None,
        "appointmentDetails": {
            "date": "2026-08-20",
            "startTime": "09:30:00",
            "endTime": None,
            "service": "Neurology",
            "appointmentType": "Outpatient appointment",
            "clinicianOrTeam": "Dr Smith",
            "locationName": "Neurology Outpatients",
            "address": {
                "addressLine1": "Example Hospital",
                "addressLine2": None,
                "townCity": "Exampletown",
                "county": None,
                "postcode": "AB1 2CD",
                "country": "United Kingdom",
            },
        },
        "processingWarning": None,
        "processorVersion": "0.6.2",
    }


@pytest.mark.parametrize(
    "redaction_context",
    [
        "not-json",
        '{"knownValues": "not-a-list"}',
    ],
)
def test_extract_rejects_invalid_redaction_context(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
    redaction_context: str,
) -> None:
    response = post_extract(
        client,
        internal_api_key,
        redaction_context=redaction_context,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Redaction context is invalid"}
    processing_service.extract.assert_not_called()


def test_extract_reads_only_maximum_size_plus_one_byte(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    processing_service.extract.side_effect = DocumentTooLargeError(
        "Document file exceeds the maximum size"
    )

    response = post_extract(
        client,
        internal_api_key,
        content=b"x" * 4096,
    )

    assert response.status_code == 413

    passed_content = processing_service.extract.call_args.args[3]
    assert passed_content == b"x" * 1025


def test_extract_closes_uploaded_file(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processing_service.extract.return_value = build_extraction_response()
    closed_filenames: list[str | None] = []
    original_close = StarletteUploadFile.close

    async def track_close(upload_file: StarletteUploadFile) -> None:
        """Record file closure while preserving Starlette cleanup behaviour."""
        closed_filenames.append(upload_file.filename)
        await original_close(upload_file)

    monkeypatch.setattr(StarletteUploadFile, "close", track_close)

    response = post_extract(
        client,
        internal_api_key,
    )

    assert response.status_code == 200
    assert "document.pdf" in closed_filenames


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_detail"),
    [
        (
            EmptyDocumentError("Document file must not be empty"),
            400,
            "Document file must not be empty",
        ),
        (
            DocumentTooLargeError("Document file exceeds the maximum size"),
            413,
            "Document file exceeds the maximum size",
        ),
        (
            UnsupportedContentTypeError("Only PDF, JPEG and PNG documents are supported"),
            415,
            "Only PDF, JPEG and PNG documents are supported",
        ),
        (
            TextExtractionError("The uploaded PDF could not be read"),
            422,
            "The uploaded PDF could not be read",
        ),
        (
            OcrError("The uploaded image could not be read"),
            422,
            "The uploaded image could not be read",
        ),
    ],
)
def test_extract_preserves_processing_error_mapping(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
    exception: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    processing_service.extract.side_effect = exception

    response = post_extract(
        client,
        internal_api_key,
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_summarise_requires_internal_api_key(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/documents/summarise",
        json={
            "documentId": str(uuid4()),
            "approvedDeidentifiedText": "approved text",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal API key"}


def test_summarise_uses_current_request_and_response_contract(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    document_id = uuid4()
    approved_text = "The patient reported improved symptoms."
    processing_service.summarise.return_value = DocumentSummaryResponse(
        document_id=document_id,
        summary="The patient reported improved symptoms.",
        processor_version="0.6.2",
        model_name="gpt-5-nano",
        prompt_version="consultation-summary-v1",
    )

    response = client.post(
        "/internal/v1/documents/summarise",
        headers={"X-Internal-Api-Key": internal_api_key},
        json={
            "documentId": str(document_id),
            "approvedDeidentifiedText": approved_text,
        },
    )

    assert response.status_code == 200
    processing_service.summarise.assert_called_once_with(
        document_id,
        approved_text,
    )
    assert response.json() == {
        "documentId": str(document_id),
        "summary": "The patient reported improved symptoms.",
        "processorVersion": "0.6.2",
        "modelName": "gpt-5-nano",
        "promptVersion": "consultation-summary-v1",
    }


def test_summarise_maps_blank_approved_text_to_bad_request(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    processing_service.summarise.side_effect = ApprovedTextBlankError(
        "Approved de-identified text must not be blank"
    )

    response = client.post(
        "/internal/v1/documents/summarise",
        headers={"X-Internal-Api-Key": internal_api_key},
        json={
            "documentId": str(uuid4()),
            "approvedDeidentifiedText": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Approved de-identified text must not be blank"}


def test_summarise_currently_rejects_empty_string_during_request_validation(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    response = client.post(
        "/internal/v1/documents/summarise",
        headers={"X-Internal-Api-Key": internal_api_key},
        json={
            "documentId": str(uuid4()),
            "approvedDeidentifiedText": "",
        },
    )

    assert response.status_code == 422
    processing_service.summarise.assert_not_called()


def test_summarise_maps_oversized_approved_text_to_content_too_large(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
) -> None:
    processing_service.summarise.side_effect = AiInputTooLongError(
        "Approved de-identified text exceeds the maximum supported length"
    )

    response = client.post(
        "/internal/v1/documents/summarise",
        headers={"X-Internal-Api-Key": internal_api_key},
        json={
            "documentId": str(uuid4()),
            "approvedDeidentifiedText": "approved text",
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Approved de-identified text exceeds the maximum supported length"
    }


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_detail"),
    [
        (
            AiSummaryResponseError("provider detail must not leak"),
            502,
            "External summary service returned an invalid response",
        ),
        (
            AiSummaryUnavailableError("provider detail must not leak"),
            503,
            "External summary generation is currently unavailable",
        ),
        (
            AiSummaryTimeoutError("provider detail must not leak"),
            504,
            "External summary generation timed out",
        ),
    ],
)
def test_summarise_preserves_sanitised_ai_error_mapping(
    client: TestClient,
    internal_api_key: str,
    processing_service: Mock,
    exception: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    processing_service.summarise.side_effect = exception

    response = client.post(
        "/internal/v1/documents/summarise",
        headers={"X-Internal-Api-Key": internal_api_key},
        json={
            "documentId": str(uuid4()),
            "approvedDeidentifiedText": "approved text",
        },
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
