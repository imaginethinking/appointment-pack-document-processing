"""Tests for the service health route."""

from fastapi.testclient import TestClient


def test_health_returns_service_status(client: TestClient) -> None:
    """Checks that the health route returns the current service status and metadata."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UP",
        "service": "Appointment Pack Document Processing",
        "version": "0.6.2",
        "environment": "test",
    }
