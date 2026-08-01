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


class KeyPointType(StrEnum):
    APPOINTMENT = "APPOINTMENT"
    OUTCOME = "OUTCOME"
    FOLLOW_UP = "FOLLOW_UP"
    MEDICATION = "MEDICATION"
    OTHER = "OTHER"


class HealthResponse(ApiModel):
    status: Literal["UP"]
    service: str
    version: str
    environment: str


class KeyPointResponse(ApiModel):
    type: KeyPointType
    text: str = Field(min_length=1)
    source_page: int | None = Field(
        default=None,
        alias="sourcePage",
        ge=1,
    )


class ModelMetadataResponse(ApiModel):
    name: str
    revision: str


class DocumentProcessingResponse(ApiModel):
    document_id: UUID = Field(alias="documentId")
    extracted_text: str = Field(alias="extractedText")
    summary: str
    key_points: list[KeyPointResponse] = Field(
        default_factory=list,
        alias="keyPoints",
    )
    warnings: list[str] = Field(default_factory=list)
    processor_version: str = Field(alias="processorVersion")
    model: ModelMetadataResponse | None = None