"""Deterministic local de-identification of consultation text."""

import re
from dataclasses import dataclass

from appointment_pack_processing.schemas import RedactionContext


@dataclass(frozen=True, slots=True)
class DeidentificationResult:
    """De-identified text and the warning shown for patient review."""

    text: str
    processing_warning: str


class DeidentificationService:
    """Redact known patient values and supported identifier patterns locally."""

    REDACTION_PLACEHOLDER = "[REDACTED]"

    REVIEW_WARNING = (
        "Automated de-identification may not remove every personal "
        "identifier. Review the text carefully before approving "
        "external transmission."
    )

    EMAIL_PATTERN = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    UK_POSTCODE_PATTERN = re.compile(
        (
            r"\b(?:GIR[ \t]?0AA|"
            r"(?:[A-PR-UWYZ][0-9][0-9A-HJKSTUW]?|"
            r"[A-PR-UWYZ][A-HK-Y][0-9][0-9ABEHMNPRV-Y]?)"
            r"[ \t]?[0-9][ABD-HJLNP-UW-Z]{2})\b"
        ),
        re.IGNORECASE,
    )

    TELEPHONE_PATTERN = re.compile(
        r"(?<!\w)"
        r"(?:\+44(?:[ \t]?\(0\))?|0)"
        r"(?:[ \t().-]?\d){9,10}"
        r"(?!\w)"
    )

    TEN_DIGIT_IDENTIFIER_PATTERN = re.compile(r"(?<!\d)(?:\d[ \t-]?){9}\d(?!\d)")

    LABELLED_HEALTHCARE_NUMBER_PATTERN = re.compile(
        (
            r"(?P<label>"
            r"(?:"
            r"\bNHS(?:[ \t]+(?:number|no\.?))?"
            r"|\bCHI(?:[ \t]+(?:number|no\.?))?"
            r"|\bH[ \t]*&[ \t]*C(?:[ \t]+(?:number|no\.?))?"
            r")"
            r"[ \t]*[:#-]?[ \t]*"
            r")"
            r"(?P<value>[0-9][0-9 \t-]{5,20})"
        ),
        re.IGNORECASE,
    )

    LABELLED_RECORD_NUMBER_PATTERN = re.compile(
        (
            r"(?P<label>"
            r"\b(?:"
            r"MRN"
            r"|medical[ \t]+record[ \t]+number"
            r"|hospital[ \t]+number"
            r"|patient[ \t]+number"
            r"|case[ \t]+number"
            r")"
            r"[ \t]*[:#-]?[ \t]*"
            r")"
            r"(?P<value>[A-Z0-9][A-Z0-9/-]{2,24})"
        ),
        re.IGNORECASE,
    )

    LABELLED_DATE_OF_BIRTH_PATTERN = re.compile(
        (
            r"(?P<label>"
            r"\b(?:date[ \t]+of[ \t]+birth|DOB|D\.O\.B\.)"
            r"\b[ \t]*[:#-]?[ \t]*"
            r")"
            r"(?P<value>"
            r"\d{4}-\d{1,2}-\d{1,2}"
            r"|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
            r"|\d{1,2}[ \t]+"
            r"(?:"
            r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
            r")"
            r"[ \t]+\d{4}"
            r")"
        ),
        re.IGNORECASE,
    )

    REPEATED_PLACEHOLDER_PATTERN = re.compile(
        r"\[REDACTED\]"
        r"(?:[ \t,;/|-]+\[REDACTED\])+"
    )

    def deidentify(
        self,
        text: str,
        context: RedactionContext,
    ) -> DeidentificationResult:
        """Redact supported identifiers from extracted consultation text."""
        redacted_text = text

        for known_value in self._prepare_known_values(context.known_values):
            redacted_text = self._replace_literal(
                redacted_text,
                known_value,
            )

        # Only labelled dates of birth are targeted so ordinary clinical dates remain intact.
        redacted_text = self._redact_labelled_value(
            redacted_text,
            self.LABELLED_DATE_OF_BIRTH_PATTERN,
        )

        redacted_text = self._redact_labelled_value(
            redacted_text,
            self.LABELLED_HEALTHCARE_NUMBER_PATTERN,
        )

        redacted_text = self._redact_labelled_value(
            redacted_text,
            self.LABELLED_RECORD_NUMBER_PATTERN,
        )

        for pattern in (
            self.EMAIL_PATTERN,
            self.UK_POSTCODE_PATTERN,
            self.TELEPHONE_PATTERN,
            self.TEN_DIGIT_IDENTIFIER_PATTERN,
        ):
            redacted_text = pattern.sub(
                self.REDACTION_PLACEHOLDER,
                redacted_text,
            )

        redacted_text = self.REPEATED_PLACEHOLDER_PATTERN.sub(
            self.REDACTION_PLACEHOLDER,
            redacted_text,
        )

        return DeidentificationResult(
            text=redacted_text,
            processing_warning=self.REVIEW_WARNING,
        )

    def _prepare_known_values(
        self,
        values: list[str],
    ) -> list[str]:
        """Normalise, deduplicate and order known values for safe replacement."""
        unique_values: dict[str, str] = {}

        for value in values:
            normalised_value = " ".join(value.split())

            if len(normalised_value) < 2:
                continue

            unique_values.setdefault(
                normalised_value.casefold(),
                normalised_value,
            )

        # Replace longer values first so shorter overlapping names do not fragment them.
        return sorted(
            unique_values.values(),
            key=len,
            reverse=True,
        )

    def _replace_literal(
        self,
        text: str,
        value: str,
    ) -> str:
        """Replace a known value case-insensitively while allowing flexible whitespace."""
        value_parts = value.split()

        escaped_value = r"[ \t]+".join(re.escape(part) for part in value_parts)

        pattern = re.compile(
            rf"(?<!\w){escaped_value}(?!\w)",
            re.IGNORECASE,
        )

        return pattern.sub(
            self.REDACTION_PLACEHOLDER,
            text,
        )

    def _redact_labelled_value(
        self,
        text: str,
        pattern: re.Pattern[str],
    ) -> str:
        """Redact only the value captured after a recognised identifier label."""
        return pattern.sub(
            self._replace_labelled_value,
            text,
        )

    def _replace_labelled_value(
        self,
        match: re.Match[str],
    ) -> str:
        """Keep an identifier label while replacing its captured value."""
        return f"{match.group('label')}{self.REDACTION_PLACEHOLDER}"
