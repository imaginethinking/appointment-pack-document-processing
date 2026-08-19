"""Typed environment configuration for the document-processing service."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from APP_ environment variables."""

    app_name: str = "Appointment Pack Document Processing"
    environment: Literal["development", "test", "production"] = "development"
    internal_api_key: SecretStr

    maximum_file_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
    )

    maximum_pdf_pages: int = Field(
        default=50,
        gt=0,
    )

    ocr_language: str = Field(
        default="eng",
        min_length=1,
    )

    ocr_dpi: int = Field(
        default=300,
        ge=150,
        le=600,
    )

    ocr_preprocessing_enabled: bool = True

    ocr_minimum_image_width: int = Field(
        default=1600,
        ge=800,
        le=4000,
    )

    tesseract_command: str | None = None

    openai_api_key: SecretStr | None = None

    openai_model: str = Field(
        default="gpt-5-nano",
        min_length=1,
    )

    openai_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
    )

    openai_max_output_tokens: int = Field(
        default=1200,
        ge=100,
        le=10_000,
    )

    openai_prompt_version: str = Field(
        default="consultation-summary-v1",
        min_length=1,
        max_length=100,
    )

    maximum_ai_input_characters: int = Field(
        default=100_000,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings for the current process."""
    return Settings()
