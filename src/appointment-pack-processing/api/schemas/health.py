from typing import Literal

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: Literal["UP"]
    service: str
    version: str
    environment: str