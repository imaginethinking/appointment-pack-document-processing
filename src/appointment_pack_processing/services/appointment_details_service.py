"""Deterministic extraction of structured appointment details."""

import re
from dataclasses import dataclass
from datetime import date, datetime, time

type PatternGroups = tuple[
    tuple[re.Pattern[str], ...],
    tuple[re.Pattern[str], ...],
]


def _build_label_patterns(*labels: str) -> PatternGroups:
    """Build preferred separator patterns and whitespace-only fallback patterns."""
    separator_patterns = tuple(
        re.compile(
            rf"^\s*{label}\s*[:\-]\s*(?P<value>.+?)\s*$",
            re.IGNORECASE,
        )
        for label in labels
    )
    whitespace_patterns = tuple(
        re.compile(
            rf"^\s*{label}\s+(?P<value>.+?)\s*$",
            re.IGNORECASE,
        )
        for label in labels
    )

    return separator_patterns, whitespace_patterns


@dataclass(frozen=True, slots=True)
class AppointmentAddressDetails:
    """Structured appointment address values extracted from a document."""

    address_line_1: str | None
    address_line_2: str | None
    town_city: str | None
    county: str | None
    postcode: str | None
    country: str | None


@dataclass(frozen=True, slots=True)
class AppointmentDetails:
    """Structured appointment values extracted from a document."""

    date: date | None
    start_time: time | None
    end_time: time | None
    service: str | None
    appointment_type: str | None
    clinician_or_team: str | None
    location_name: str | None
    address: AppointmentAddressDetails | None


@dataclass(frozen=True, slots=True)
class AppointmentDetailsResult:
    """Appointment details together with any review warning."""

    details: AppointmentDetails
    processing_warning: str | None


class AppointmentDetailsService:
    """Extract conservative appointment suggestions from labelled document lines."""

    MAXIMUM_FIELD_LENGTH = 250

    DATE_PATTERNS = _build_label_patterns(
        r"On",
        r"Date",
    )

    START_TIME_PATTERNS = _build_label_patterns(
        r"Time",
        r"Start\s+time",
    )

    END_TIME_PATTERNS = _build_label_patterns(
        r"End\s+time",
        r"Finish\s+time",
    )

    CLINICIAN_PATTERNS = _build_label_patterns(
        r"With",
        r"Clinician",
        r"Consultant",
        r"Team",
    )

    SERVICE_PATTERNS = _build_label_patterns(
        r"Service\s+type",
        r"Service",
    )

    APPOINTMENT_TYPE_PATTERNS = _build_label_patterns(
        r"Appointment\s+type",
    )

    LOCATION_PATTERNS = _build_label_patterns(
        r"Location",
        r"Venue",
        r"Clinic",
    )

    ADDRESS_LINE_1_PATTERNS = _build_label_patterns(
        r"Address\s+line\s+1",
        r"Address",
    )

    ADDRESS_LINE_2_PATTERNS = _build_label_patterns(
        r"Address\s+line\s+2",
    )

    TOWN_CITY_PATTERNS = _build_label_patterns(
        r"Town\s*/\s*City",
        r"Town",
        r"City",
    )

    COUNTY_PATTERNS = _build_label_patterns(
        r"County",
    )

    POSTCODE_PATTERNS = _build_label_patterns(
        r"Postcode",
    )

    COUNTRY_PATTERNS = _build_label_patterns(
        r"Country",
    )

    DATE_FORMATS = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%A %d %B %Y",
        "%A %d %b %Y",
    )

    TIME_FORMATS = (
        "%H:%M",
        "%H.%M",
        "%I:%M %p",
        "%I.%M %p",
        "%I %p",
        "%I%p",
    )

    def extract(
        self,
        text: str,
    ) -> AppointmentDetailsResult:
        """Extract supported appointment fields from labelled document lines."""
        lines = text.splitlines()

        address = self._extract_address(lines)

        details = AppointmentDetails(
            date=self._extract_date(lines),
            start_time=self._extract_time(
                lines,
                self.START_TIME_PATTERNS,
            ),
            end_time=self._extract_time(
                lines,
                self.END_TIME_PATTERNS,
            ),
            service=self._find_value(
                lines,
                self.SERVICE_PATTERNS,
            ),
            appointment_type=self._find_value(
                lines,
                self.APPOINTMENT_TYPE_PATTERNS,
            ),
            clinician_or_team=self._find_value(
                lines,
                self.CLINICIAN_PATTERNS,
            ),
            location_name=self._find_value(
                lines,
                self.LOCATION_PATTERNS,
            ),
            address=address,
        )

        return AppointmentDetailsResult(
            details=details,
            processing_warning=self._build_warning(details),
        )

    def _extract_date(
        self,
        lines: list[str],
    ) -> date | None:
        """Extract and normalise a supported labelled appointment date."""
        value = self._find_value(
            lines,
            self.DATE_PATTERNS,
        )

        if value is None:
            return None

        normalised_value = re.sub(
            r"(\d{1,2})(st|nd|rd|th)\b",
            r"\1",
            value,
            flags=re.IGNORECASE,
        )

        normalised_value = normalised_value.replace(
            ",",
            "",
        )

        # Unsupported or ambiguous date formats remain None rather than being guessed.
        for date_format in self.DATE_FORMATS:
            try:
                return datetime.strptime(
                    normalised_value,
                    date_format,
                ).date()
            except ValueError:
                continue

        return None

    def _extract_time(
        self,
        lines: list[str],
        patterns: PatternGroups,
    ) -> time | None:
        """Extract and normalise a supported labelled appointment time."""
        value = self._find_value(
            lines,
            patterns,
        )

        if value is None:
            return None

        normalised_value = re.sub(
            r"\s*(am|pm)$",
            r" \1",
            value.strip(),
            flags=re.IGNORECASE,
        )

        normalised_value = " ".join(normalised_value.split()).upper()

        for time_format in self.TIME_FORMATS:
            try:
                return datetime.strptime(
                    normalised_value,
                    time_format,
                ).time()
            except ValueError:
                continue

        return None

    def _extract_address(
        self,
        lines: list[str],
    ) -> AppointmentAddressDetails | None:
        """Extract any supported labelled appointment address fields."""
        address = AppointmentAddressDetails(
            address_line_1=self._find_value(
                lines,
                self.ADDRESS_LINE_1_PATTERNS,
                maximum_length=150,
            ),
            address_line_2=self._find_value(
                lines,
                self.ADDRESS_LINE_2_PATTERNS,
                maximum_length=150,
            ),
            town_city=self._find_value(
                lines,
                self.TOWN_CITY_PATTERNS,
                maximum_length=100,
            ),
            county=self._find_value(
                lines,
                self.COUNTY_PATTERNS,
                maximum_length=100,
            ),
            postcode=self._find_value(
                lines,
                self.POSTCODE_PATTERNS,
                maximum_length=20,
            ),
            country=self._find_value(
                lines,
                self.COUNTRY_PATTERNS,
                maximum_length=100,
            ),
        )

        if not any(
            (
                address.address_line_1,
                address.address_line_2,
                address.town_city,
                address.county,
                address.postcode,
                address.country,
            )
        ):
            return None

        return address

    def _find_value(
        self,
        lines: list[str],
        patterns: PatternGroups,
        maximum_length: int = MAXIMUM_FIELD_LENGTH,
    ) -> str | None:
        """Return the first labelled value, preferring explicit separators."""
        # Search the whole document for separator labels before using whitespace fallback.
        for pattern_group in patterns:
            for line in lines:
                for pattern in pattern_group:
                    match = pattern.match(line)

                    if match is None:
                        continue

                    value = self._normalise_value(
                        match.group("value"),
                        maximum_length,
                    )

                    if value is not None:
                        return value

        return None

    def _normalise_value(
        self,
        value: str,
        maximum_length: int,
    ) -> str | None:
        """Normalise a labelled value and reject blank or oversized content."""
        normalised_value = " ".join(value.split()).strip(" |")

        if not normalised_value:
            return None

        if len(normalised_value) > maximum_length:
            return None

        return normalised_value

    def _build_warning(
        self,
        details: AppointmentDetails,
    ) -> str | None:
        """Build a review warning when supported appointment values are missing."""
        address_present = details.address is not None

        identified_values = (
            details.date,
            details.start_time,
            details.end_time,
            details.service,
            details.appointment_type,
            details.clinician_or_team,
            details.location_name,
            address_present,
        )

        if not any(identified_values):
            return (
                "No supported labelled appointment fields were "
                "identified. Review the extracted text and enter "
                "the appointment details manually."
            )

        missing_core_fields: list[str] = []

        if details.date is None:
            missing_core_fields.append("date")

        if details.start_time is None:
            missing_core_fields.append("start time")

        if details.location_name is None and details.address is None:
            missing_core_fields.append("location")

        if not missing_core_fields:
            return None

        return (
            "Some core appointment details could not be identified "
            "or normalised: "
            + self._format_field_names(missing_core_fields)
            + ". Review the extracted text before confirming the appointment."
        )

    def _format_field_names(
        self,
        field_names: list[str],
    ) -> str:
        """Format missing field names for a readable warning message."""
        if len(field_names) == 1:
            return field_names[0]

        if len(field_names) == 2:
            return " and ".join(field_names)

        return ", ".join(field_names[:-1]) + f", and {field_names[-1]}"