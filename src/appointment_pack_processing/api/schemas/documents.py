from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

def to_camel_case(value: str) -> str:
    components = value.split("_")
    return components[0] + "".join(component.title() for component in components[1:])

class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel_case,
        populate_by_name=True,
        serialize_by_alias=True
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

class KeyPointResponse(ApiModel):
    type: KeyPointType
    text: str = Field(min_length=1)
    source_page: int | None = Field(default=None, ge=1)

class ModelMetadataResponse(ApiModel):
    name: str
    revision: str

class DocumentProcessingResponse(ApiModel):
    document_id: UUID
    extracted_text: str
    summary: str
    key_points: list[KeyPointResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processor_version: str
    model: ModelMetadataResponse | None = None