"""Pydantic request and response contracts used by the internal API."""

# Alias the datetime types so fields named date/time do not shadow their type names.
from datetime import date as Date
from datetime import time as Time
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base API model with Spring-compatible alias serialisation."""

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
    )


class DocumentType(StrEnum):
    """Document types supported by the processing service."""

    APPOINTMENT_LETTER = "APPOINTMENT_LETTER"
    CONSULTATION_OUTCOME_LETTER = "CONSULTATION_OUTCOME_LETTER"


class HealthResponse(ApiModel):
    """Health information returned by the service."""

    status: Literal["UP"]
    service: str
    version: str
    environment: str


class ModelMetadataResponse(ApiModel):
    """Model name and revision metadata."""

    name: str
    revision: str


class RedactionContext(ApiModel):
    """Known patient values supplied for deterministic local redaction."""

    known_values: list[str] = Field(
        default_factory=list,
        alias="knownValues",
        max_length=100,
    )


class AppointmentAddressDetailsResponse(ApiModel):
    """Partial appointment address extracted from an appointment letter."""

    address_line_1: str | None = Field(
        default=None,
        alias="addressLine1",
    )

    address_line_2: str | None = Field(
        default=None,
        alias="addressLine2",
    )

    town_city: str | None = Field(
        default=None,
        alias="townCity",
    )

    county: str | None = None

    postcode: str | None = None

    country: str | None = None


class AppointmentDetailsResponse(ApiModel):
    """Structured appointment suggestions extracted from an appointment letter."""

    date: Date | None = None

    start_time: Time | None = Field(
        default=None,
        alias="startTime",
    )

    end_time: Time | None = Field(
        default=None,
        alias="endTime",
    )

    service: str | None = None

    appointment_type: str | None = Field(
        default=None,
        alias="appointmentType",
    )

    clinician_or_team: str | None = Field(
        default=None,
        alias="clinicianOrTeam",
    )

    location_name: str | None = Field(
        default=None,
        alias="locationName",
    )

    address: AppointmentAddressDetailsResponse | None = None


class DocumentExtractionResponse(ApiModel):
    """Result returned after local extraction and document-specific processing."""

    document_id: UUID = Field(
        alias="documentId",
    )

    extracted_text: str = Field(
        alias="extractedText",
    )

    deidentified_text: str | None = Field(
        default=None,
        alias="deidentifiedText",
    )

    appointment_details: AppointmentDetailsResponse | None = Field(
        default=None,
        alias="appointmentDetails",
    )

    processing_warning: str | None = Field(
        default=None,
        alias="processingWarning",
    )

    processor_version: str = Field(
        alias="processorVersion",
    )


class DocumentSummaryRequest(ApiModel):
    """Approved de-identified consultation text submitted for summarisation."""

    document_id: UUID = Field(
        alias="documentId",
    )

    approved_deidentified_text: str = Field(
        alias="approvedDeidentifiedText",
        min_length=1,
    )


class DocumentSummaryResponse(ApiModel):
    """Generated consultation summary and its model provenance."""

    document_id: UUID = Field(
        alias="documentId",
    )

    summary: str = Field(
        min_length=1,
    )

    processor_version: str = Field(
        alias="processorVersion",
    )

    model_name: str = Field(
        alias="modelName",
        min_length=1,
    )

    prompt_version: str = Field(
        alias="promptVersion",
        min_length=1,
    )