from uuid import UUID, uuid4

from fastapi.testclient import TestClient


def test_process_document_returns_placeholder_result(
    client: TestClient,
    internal_api_key: str,
) -> None:
    document_id = uuid4()

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
                b"%PDF-1.7 placeholder document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert UUID(response_body["documentId"]) == document_id
    assert response_body["extractedText"] == ""
    assert response_body["summary"] == ""
    assert response_body["keyPoints"] == []
    assert response_body["processorVersion"] == "0.1.0"
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
                b"%PDF-1.7 placeholder document",
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
                b"x" * 1025,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Document file exceeds the maximum size"
    )