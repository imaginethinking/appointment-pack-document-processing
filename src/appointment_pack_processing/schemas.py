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


class DocumentProcessingResponse(ApiModel):
    document_id: UUID = Field(alias="documentId")
    extracted_text: str = Field(alias="extractedText")
    summary: str
    warnings: list[str] = Field(default_factory=list)
    processor_version: str = Field(alias="processorVersion")
    model: ModelMetadataResponse | None = None