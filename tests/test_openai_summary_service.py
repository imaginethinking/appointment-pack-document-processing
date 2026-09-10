"""Tests for approved consultation summarisation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import appointment_pack_processing.services.openai_summary_service as summary_module
from appointment_pack_processing.services.openai_summary_service import (
    AiSummaryResponseError,
    AiSummaryTimeoutError,
    AiSummaryUnavailableError,
    OpenAiSummaryPayload,
    OpenAiSummaryService,
)


def build_service() -> OpenAiSummaryService:
    """Build the summary service with a mocked client and fixed configuration."""
    service = OpenAiSummaryService(
        api_key=None,
        model_name="gpt-5-nano",
        timeout_seconds=15.0,
        maximum_output_tokens=500,
        prompt_version="consultation-summary-v1",
    )

    service.client = Mock()

    return service


def build_completion(
    *,
    parsed: OpenAiSummaryPayload | None,
    model: str = "gpt-5-nano-2026-08-01",
) -> SimpleNamespace:
    """Build a minimal structured completion returned by the mocked client."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=parsed,
                ),
            )
        ],
    )


def test_constructor_configures_openai_without_automatic_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that the OpenAI client is created with automatic retries disabled."""
    constructor = Mock(return_value=Mock())

    monkeypatch.setattr(
        summary_module,
        "OpenAI",
        constructor,
    )

    service = OpenAiSummaryService(
        api_key="test-api-key",
        model_name="gpt-5-nano",
        timeout_seconds=12.5,
        maximum_output_tokens=400,
        prompt_version="prompt-v1",
    )

    constructor.assert_called_once_with(
        api_key="test-api-key",
        timeout=12.5,
        max_retries=0,
    )

    assert service.client is constructor.return_value


def test_summarise_rejects_missing_openai_configuration() -> None:
    """Checks that summarisation fails clearly when OpenAI is not configured."""
    service = OpenAiSummaryService(
        api_key=None,
        model_name="gpt-5-nano",
        timeout_seconds=15.0,
        maximum_output_tokens=500,
        prompt_version="consultation-summary-v1",
    )

    with pytest.raises(
        AiSummaryUnavailableError,
        match="OpenAI is not configured",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_returns_structured_summary_and_metadata() -> None:
    """Checks that a successful response returns the summary and model metadata."""
    service = build_service()

    service.client.chat.completions.parse.return_value = build_completion(
        parsed=OpenAiSummaryPayload(summary="  The patient improved.  "),
    )

    result = service.summarise("The patient reported that symptoms had improved.")

    assert result.summary == "The patient improved."
    assert result.model_name == "gpt-5-nano-2026-08-01"
    assert result.prompt_version == "consultation-summary-v1"


def test_summarise_uses_current_prompt_and_completion_configuration() -> None:
    """Checks that summarisation uses the configured prompt model and output settings."""
    service = build_service()
    approved_text = "The patient reported improved symptoms."

    service.client.chat.completions.parse.return_value = build_completion(
        parsed=OpenAiSummaryPayload(summary="The patient improved."),
    )

    service.summarise(approved_text)

    service.client.chat.completions.parse.assert_called_once()

    call_kwargs = service.client.chat.completions.parse.call_args.kwargs

    assert call_kwargs["model"] == "gpt-5-nano"
    assert call_kwargs["response_format"] is OpenAiSummaryPayload
    assert call_kwargs["reasoning_effort"] == "minimal"
    assert call_kwargs["max_completion_tokens"] == 500

    assert call_kwargs["messages"][0] == {
        "role": "system",
        "content": service.SYSTEM_PROMPT,
    }

    assert call_kwargs["messages"][1] == {
        "role": "user",
        "content": (
            "Summarise the following approved de-identified "
            "consultation letter.\n\n"
            "<consultation_letter>\n"
            f"{approved_text}\n"
            "</consultation_letter>"
        ),
    }


def test_summarise_rejects_completion_without_choice() -> None:
    """Checks that a completion without a summary choice is rejected."""
    service = build_service()

    service.client.chat.completions.parse.return_value = SimpleNamespace(
        model="gpt-5-nano",
        choices=[],
    )

    with pytest.raises(
        AiSummaryResponseError,
        match="OpenAI returned no summary choice",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_rejects_completion_without_parsed_response() -> None:
    """Checks that a completion without a parsed summary is rejected."""
    service = build_service()

    service.client.chat.completions.parse.return_value = build_completion(
        parsed=None,
    )

    with pytest.raises(
        AiSummaryResponseError,
        match="OpenAI returned no valid summary",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summary_payload_rejects_whitespace_only_summary() -> None:
    """Checks that a structured summary containing only spaces is rejected."""
    with pytest.raises(ValidationError):
        OpenAiSummaryPayload(summary="   ")


def test_summarise_translates_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that an OpenAI timeout is translated into the summary timeout error."""
    class SyntheticTimeoutError(Exception):
        """Synthetic provider timeout used without a live SDK request."""

    monkeypatch.setattr(
        summary_module,
        "APITimeoutError",
        SyntheticTimeoutError,
    )

    service = build_service()
    service.client.chat.completions.parse.side_effect = SyntheticTimeoutError()

    with pytest.raises(
        AiSummaryTimeoutError,
        match="OpenAI summary request timed out",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_translates_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that an OpenAI connection failure is translated into the unavailable error."""
    class SyntheticConnectionError(Exception):
        """Synthetic provider connection failure."""

    monkeypatch.setattr(
        summary_module,
        "APIConnectionError",
        SyntheticConnectionError,
    )

    service = build_service()
    service.client.chat.completions.parse.side_effect = SyntheticConnectionError()

    with pytest.raises(
        AiSummaryUnavailableError,
        match="OpenAI could not be reached",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_translates_completion_length_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that a completion stopped by the token limit is treated as an invalid response."""
    class SyntheticLengthError(Exception):
        """Synthetic completion-length failure."""

    monkeypatch.setattr(
        summary_module,
        "LengthFinishReasonError",
        SyntheticLengthError,
    )

    service = build_service()
    service.client.chat.completions.parse.side_effect = SyntheticLengthError()

    with pytest.raises(
        AiSummaryResponseError,
        match="OpenAI reached the completion token limit",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_translates_structured_response_validation_failure() -> None:
    """Checks that invalid structured response data is translated into the expected summary error."""
    service = build_service()

    with pytest.raises(ValidationError) as validation_error:
        OpenAiSummaryPayload(summary="   ")

    service.client.chat.completions.parse.side_effect = validation_error.value

    with pytest.raises(
        AiSummaryResponseError,
        match="OpenAI returned an invalid summary",
    ):
        service.summarise("Approved de-identified consultation text.")


def test_summarise_translates_other_openai_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that other OpenAI failures are translated into the unavailable error."""
    class SyntheticOpenAiError(Exception):
        """Synthetic generic OpenAI SDK failure."""

    monkeypatch.setattr(
        summary_module,
        "OpenAIError",
        SyntheticOpenAiError,
    )

    service = build_service()
    service.client.chat.completions.parse.side_effect = SyntheticOpenAiError()

    with pytest.raises(
        AiSummaryUnavailableError,
        match="OpenAI summary request failed",
    ):
        service.summarise("Approved de-identified consultation text.")
