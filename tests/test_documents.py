from uuid import UUID, uuid4

import pymupdf
from fastapi.testclient import TestClient

# TODO Add tests using valid OCR images and scanned PDFs containing readable text

def create_pdf_with_text(text: str) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            text,
        )

        return document.tobytes()


def create_pdf_without_text() -> bytes:
    with pymupdf.open() as document:
        document.new_page()

        return document.tobytes()


def test_process_document_extracts_embedded_pdf_text(
    client: TestClient,
    internal_api_key: str,
) -> None:
    document_id = uuid4()
    pdf_content = create_pdf_with_text(
        "Your cardiology appointment is on 15 August 2026 at 10:30."
    )

    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(document_id),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "appointment.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert UUID(response_body["documentId"]) == document_id
    assert (
        "Your cardiology appointment is on 15 August 2026 at 10:30."
        in response_body["extractedText"]
    )
    assert response_body["summary"] == ""
    assert response_body["keyPoints"] == []
    assert response_body["processorVersion"] == "0.2.0"
    assert response_body["model"] is None
    assert len(response_body["warnings"]) == 1


def test_process_document_requires_internal_api_key(
    client: TestClient,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "appointment.pdf",
                create_pdf_with_text("Appointment letter"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Invalid internal API key"
    )


def test_process_document_rejects_unsupported_content_type(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "CONSULTATION_OUTCOME_LETTER",
        },
        files={
            "file": (
                "letter.txt",
                b"Unsupported plain-text document",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == (
        "Only PDF, JPEG and PNG documents are supported"
    )


def test_process_document_rejects_empty_file(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "appointment.pdf",
                b"",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Document file must not be empty"
    )


def test_process_document_rejects_oversized_file(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "appointment.pdf",
                b"x" * 4097,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Document file exceeds the maximum size"
    )


def test_process_document_rejects_invalid_image(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "appointment.png",
                b"\x89PNG\r\n\x1a\n",
                "image/png",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "The uploaded image could not be read"
    )


def test_process_document_rejects_pdf_without_readable_text(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "CONSULTATION_OUTCOME_LETTER",
        },
        files={
            "file": (
                "consultation-letter.pdf",
                create_pdf_without_text(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "No readable text was detected in the PDF"
    )


def test_process_document_rejects_unreadable_pdf(
    client: TestClient,
    internal_api_key: str,
) -> None:
    response = client.post(
        "/internal/v1/documents/process",
        headers={
            "X-Internal-Api-Key": internal_api_key,
        },
        data={
            "documentId": str(uuid4()),
            "documentType": "APPOINTMENT_LETTER",
        },
        files={
            "file": (
                "invalid.pdf",
                b"%PDF-1.7 invalid document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "The uploaded PDF could not be read"
    )