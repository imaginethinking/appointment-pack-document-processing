import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppointmentSummaryResult:
    summary: str
    processing_warning: str | None


@dataclass(frozen=True, slots=True)
class AppointmentDetails:
    date: str | None
    time: str | None
    clinician: str | None
    service: str | None
    clinic: str | None
    location: str | None
    appointment_type: str | None


class AppointmentSummaryService:
    MAXIMUM_FIELD_LENGTH = 250

    DATE_PATTERNS = (
        re.compile(
            r"^\s*On\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*Date\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    TIME_PATTERNS = (
        re.compile(
            r"^\s*Time\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    CLINICIAN_PATTERNS = (
        re.compile(
            r"^\s*With\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    SERVICE_PATTERNS = (
        re.compile(
            r"^\s*Service\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    CLINIC_PATTERNS = (
        re.compile(
            r"^\s*Clinic\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    LOCATION_PATTERNS = (
        re.compile(
            r"^\s*Location\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    APPOINTMENT_TYPE_PATTERNS = (
        re.compile(
            (
                r"^\s*Appointment\s+type\s*[:\-]\s*"
                r"(?P<value>.+?)\s*$"
            ),
            re.IGNORECASE,
        ),
    )

    def generate(
        self,
        text: str,
    ) -> AppointmentSummaryResult:
        lines = text.splitlines()

        details = AppointmentDetails(
            date=self._find_value(
                lines,
                self.DATE_PATTERNS,
            ),
            time=self._find_value(
                lines,
                self.TIME_PATTERNS,
            ),
            clinician=self._find_value(
                lines,
                self.CLINICIAN_PATTERNS,
            ),
            service=self._find_value(
                lines,
                self.SERVICE_PATTERNS,
            ),
            clinic=self._find_value(
                lines,
                self.CLINIC_PATTERNS,
            ),
            location=self._find_value(
                lines,
                self.LOCATION_PATTERNS,
            ),
            appointment_type=self._find_value(
                lines,
                self.APPOINTMENT_TYPE_PATTERNS,
            ),
        )

        summary = self._build_summary(details)
        processing_warning = self._build_warning(details)

        return AppointmentSummaryResult(
            summary=summary,
            processing_warning=processing_warning,
        )

    def _find_value(
        self,
        lines: list[str],
        patterns: tuple[re.Pattern[str], ...],
    ) -> str | None:
        for line in lines:
            for pattern in patterns:
                match = pattern.match(line)

                if match is None:
                    continue

                value = self._normalise_value(
                    match.group("value")
                )

                if value is not None:
                    return value

        return None

    def _normalise_value(
        self,
        value: str,
    ) -> str | None:
        normalised_value = " ".join(
            value.split()
        ).strip(" |")

        if not normalised_value:
            return None

        if len(normalised_value) > self.MAXIMUM_FIELD_LENGTH:
            return None

        return normalised_value

    def _build_summary(
        self,
        details: AppointmentDetails,
    ) -> str:
        summary_parts: list[str] = []

        if details.date and details.time:
            summary_parts.append(
                (
                    f"Appointment scheduled for {details.date} "
                    f"at {details.time}."
                )
            )
        elif details.date:
            summary_parts.append(
                f"Appointment scheduled for {details.date}."
            )
        elif details.time:
            summary_parts.append(
                f"Appointment time: {details.time}."
            )

        if details.service:
            summary_parts.append(
                f"Service: {details.service}."
            )

        if details.clinic:
            summary_parts.append(
                f"Clinic: {details.clinic}."
            )

        if details.clinician:
            summary_parts.append(
                f"With: {details.clinician}."
            )

        if details.location:
            summary_parts.append(
                f"Location: {details.location}."
            )

        if details.appointment_type:
            summary_parts.append(
                (
                    "Appointment type: "
                    f"{details.appointment_type}."
                )
            )

        if not summary_parts:
            return "No appointment details could be identified from the supported labelled fields."

        return " ".join(summary_parts)

    def _build_warning(
        self,
        details: AppointmentDetails,
    ) -> str | None:
        identified_values = (
            details.date,
            details.time,
            details.clinician,
            details.service,
            details.clinic,
            details.location,
            details.appointment_type,
        )

        if not any(identified_values):
            return (
                "No supported labelled appointment fields were "
                "identified. Review the extracted text and enter "
                "the summary manually if required."
            )

        missing_core_fields: list[str] = []

        if details.date is None:
            missing_core_fields.append("date")

        if details.time is None:
            missing_core_fields.append("time")

        if details.location is None:
            missing_core_fields.append("location")

        if not missing_core_fields:
            return None

        formatted_fields = self._format_field_names(
            missing_core_fields
        )

        return (
            "The deterministic summary could not identify the "
            f"appointment {formatted_fields}. Review the extracted "
            "text before accepting the summary."
        )

    def _format_field_names(
        self,
        field_names: list[str],
    ) -> str:
        if len(field_names) == 1:
            return field_names[0]

        if len(field_names) == 2:
            return " and ".join(field_names)

        return (
            ", ".join(field_names[:-1])
            + f", and {field_names[-1]}"
        )