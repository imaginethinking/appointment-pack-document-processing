from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
    )


class DocumentType(StrEnum):
    APPOINTMENT_LETTER = "APPOINTMENT_LETTER"
    CONSULTATION_OUTCOME_LETTER = "CONSULTATION_OUTCOME_LETTER"


class HealthResponse(ApiModel):
    status: Literal["UP"]
    service: str
    version: str
    environment: str


class ModelMetadataResponse(ApiModel):
    name: str
    revision: str

class RedactionContext(ApiModel):
    known_values: list[str] = Field(
        default_factory=list,
        alias="knownValues",
        max_length=100,
    )


class DocumentExtractionResponse(ApiModel):
    document_id: UUID = Field(alias="documentId")
    extracted_text: str = Field(alias="extractedText")
    deidentified_text: str | None = Field(
        default=None,
        alias="deidentifiedText",
    )
    generated_summary: str | None = Field(
        default=None,
        alias="generatedSummary",
    )
    processing_warning: str | None = Field(
        default=None,
        alias="processingWarning",
    )
    processor_version: str = Field(alias="processorVersion")

class DocumentSummaryRequest(ApiModel):
    document_id: UUID = Field(alias="documentId")

    approved_deidentified_text: str = Field(
        alias="approvedDeidentifiedText",
        min_length=1,
    )


class DocumentSummaryResponse(ApiModel):
    document_id: UUID = Field(alias="documentId")
    summary: str = Field(min_length=1)

    processor_version: str = Field(
        alias="processorVersion"
    )

    model_name: str = Field(
        alias="modelName",
        min_length=1,
    )

    prompt_version: str = Field(
        alias="promptVersion",
        min_length=1,
    )