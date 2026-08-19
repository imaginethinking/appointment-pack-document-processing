from fastapi import APIRouter

from appointment_pack_processing import __version__
from appointment_pack_processing.api.dependencies import SettingsDependency
from appointment_pack_processing.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
)
def get_health(settings: SettingsDependency) -> HealthResponse:
    """Return the current service health and runtime metadata."""
    return HealthResponse(
        status="UP",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
