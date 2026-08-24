"""Deterministic extraction of structured appointment details."""

import re
from dataclasses import dataclass
from datetime import date, datetime, time

type PatternGroups = tuple[
    tuple[re.Pattern[str], ...],
    tuple[re.Pattern[str], ...],
]
type CandidateMap = dict[str, list["_FieldCandidate"]]


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
class _FieldCandidate:
    """Internal labelled field candidate together with its document position."""

    value: str
    line_index: int
    uses_separator: bool


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
    CONTEXT_WINDOW_LINES = 6

    DATE_FIELD = "date"
    START_TIME_FIELD = "start_time"
    END_TIME_FIELD = "end_time"
    SERVICE_FIELD = "service"
    APPOINTMENT_TYPE_FIELD = "appointment_type"
    CLINICIAN_FIELD = "clinician_or_team"
    LOCATION_FIELD = "location_name"
    ADDRESS_LINE_1_FIELD = "address_line_1"
    ADDRESS_LINE_2_FIELD = "address_line_2"
    TOWN_CITY_FIELD = "town_city"
    COUNTY_FIELD = "county"
    POSTCODE_FIELD = "postcode"
    COUNTRY_FIELD = "country"

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
        candidates = self._collect_candidates(lines)
        self._remove_unparseable_temporal_candidates(candidates)

        address = self._extract_address(candidates)

        details = AppointmentDetails(
            date=self._extract_date(candidates),
            start_time=self._extract_time(
                candidates,
                self.START_TIME_FIELD,
            ),
            end_time=self._extract_time(
                candidates,
                self.END_TIME_FIELD,
            ),
            service=self._select_value(
                candidates,
                self.SERVICE_FIELD,
            ),
            appointment_type=self._select_value(
                candidates,
                self.APPOINTMENT_TYPE_FIELD,
            ),
            clinician_or_team=self._select_value(
                candidates,
                self.CLINICIAN_FIELD,
            ),
            location_name=self._select_value(
                candidates,
                self.LOCATION_FIELD,
            ),
            address=address,
        )

        return AppointmentDetailsResult(
            details=details,
            processing_warning=self._build_warning(details),
        )

    def _collect_candidates(
        self,
        lines: list[str],
    ) -> CandidateMap:
        """Collect all supported labelled field candidates from the document."""
        return {
            self.DATE_FIELD: self._find_candidates(
                lines,
                self.DATE_PATTERNS,
            ),
            self.START_TIME_FIELD: self._find_candidates(
                lines,
                self.START_TIME_PATTERNS,
            ),
            self.END_TIME_FIELD: self._find_candidates(
                lines,
                self.END_TIME_PATTERNS,
            ),
            self.SERVICE_FIELD: self._find_candidates(
                lines,
                self.SERVICE_PATTERNS,
            ),
            self.APPOINTMENT_TYPE_FIELD: self._find_candidates(
                lines,
                self.APPOINTMENT_TYPE_PATTERNS,
            ),
            self.CLINICIAN_FIELD: self._find_candidates(
                lines,
                self.CLINICIAN_PATTERNS,
            ),
            self.LOCATION_FIELD: self._find_candidates(
                lines,
                self.LOCATION_PATTERNS,
            ),
            self.ADDRESS_LINE_1_FIELD: self._find_candidates(
                lines,
                self.ADDRESS_LINE_1_PATTERNS,
                maximum_length=150,
            ),
            self.ADDRESS_LINE_2_FIELD: self._find_candidates(
                lines,
                self.ADDRESS_LINE_2_PATTERNS,
                maximum_length=150,
            ),
            self.TOWN_CITY_FIELD: self._find_candidates(
                lines,
                self.TOWN_CITY_PATTERNS,
                maximum_length=100,
            ),
            self.COUNTY_FIELD: self._find_candidates(
                lines,
                self.COUNTY_PATTERNS,
                maximum_length=100,
            ),
            self.POSTCODE_FIELD: self._find_candidates(
                lines,
                self.POSTCODE_PATTERNS,
                maximum_length=20,
            ),
            self.COUNTRY_FIELD: self._find_candidates(
                lines,
                self.COUNTRY_PATTERNS,
                maximum_length=100,
            ),
        }

    def _remove_unparseable_temporal_candidates(
        self,
        candidates: CandidateMap,
    ) -> None:
        """Remove invalid dates and times while preserving later usable candidates."""
        candidates[self.DATE_FIELD] = [
            candidate
            for candidate in candidates[self.DATE_FIELD]
            if self._parse_date(candidate.value) is not None
        ]
        candidates[self.START_TIME_FIELD] = [
            candidate
            for candidate in candidates[self.START_TIME_FIELD]
            if self._parse_time(candidate.value) is not None
        ]
        candidates[self.END_TIME_FIELD] = [
            candidate
            for candidate in candidates[self.END_TIME_FIELD]
            if self._parse_time(candidate.value) is not None
        ]

    def _extract_date(
        self,
        candidates: CandidateMap,
    ) -> date | None:
        """Select and normalise the strongest supported appointment date candidate."""
        candidate = self._select_candidate(
            candidates,
            self.DATE_FIELD,
        )

        if candidate is None:
            return None

        return self._parse_date(candidate.value)

    def _extract_time(
        self,
        candidates: CandidateMap,
        field_name: str,
    ) -> time | None:
        """Select and normalise the strongest supported appointment time candidate."""
        candidate = self._select_candidate(
            candidates,
            field_name,
        )

        if candidate is None:
            return None

        return self._parse_time(candidate.value)

    def _parse_date(
        self,
        value: str,
    ) -> date | None:
        """Normalise one candidate using the supported deterministic date formats."""
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

        for date_format in self.DATE_FORMATS:
            try:
                return datetime.strptime(
                    normalised_value,
                    date_format,
                ).date()
            except ValueError:
                continue

        return None

    def _parse_time(
        self,
        value: str,
    ) -> time | None:
        """Normalise one candidate using the supported deterministic time formats."""
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
        candidates: CandidateMap,
    ) -> AppointmentAddressDetails | None:
        """Extract the strongest supported appointment address field candidates."""
        address = AppointmentAddressDetails(
            address_line_1=self._select_value(
                candidates,
                self.ADDRESS_LINE_1_FIELD,
            ),
            address_line_2=self._select_value(
                candidates,
                self.ADDRESS_LINE_2_FIELD,
            ),
            town_city=self._select_value(
                candidates,
                self.TOWN_CITY_FIELD,
            ),
            county=self._select_value(
                candidates,
                self.COUNTY_FIELD,
            ),
            postcode=self._select_value(
                candidates,
                self.POSTCODE_FIELD,
            ),
            country=self._select_value(
                candidates,
                self.COUNTRY_FIELD,
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

    def _find_candidates(
        self,
        lines: list[str],
        patterns: PatternGroups,
        maximum_length: int = MAXIMUM_FIELD_LENGTH,
    ) -> list[_FieldCandidate]:
        """Return every usable labelled candidate with its line position."""
        candidates: list[_FieldCandidate] = []
        matched_line_indices: set[int] = set()

        for pattern_group_index, pattern_group in enumerate(patterns):
            uses_separator = pattern_group_index == 0

            for line_index, line in enumerate(lines):
                if line_index in matched_line_indices:
                    continue

                for pattern in pattern_group:
                    match = pattern.match(line)

                    if match is None:
                        continue

                    value = self._normalise_value(
                        match.group("value"),
                        maximum_length,
                    )

                    if value is None:
                        continue

                    candidates.append(
                        _FieldCandidate(
                            value=value,
                            line_index=line_index,
                            uses_separator=uses_separator,
                        )
                    )
                    matched_line_indices.add(line_index)
                    break

        return candidates

    def _select_value(
        self,
        candidates: CandidateMap,
        field_name: str,
    ) -> str | None:
        """Return the strongest available string candidate for one field."""
        candidate = self._select_candidate(
            candidates,
            field_name,
        )

        if candidate is None:
            return None

        return candidate.value

    def _select_candidate(
        self,
        candidates: CandidateMap,
        field_name: str,
    ) -> _FieldCandidate | None:
        """Select a candidate using nearby recognised fields as generic context."""
        field_candidates = candidates[field_name]

        if not field_candidates:
            return None

        return max(
            field_candidates,
            key=lambda candidate: (
                self._context_score(
                    candidate,
                    field_name,
                    candidates,
                ),
                candidate.uses_separator,
                -candidate.line_index,
            ),
        )

    def _context_score(
        self,
        candidate: _FieldCandidate,
        field_name: str,
        candidates: CandidateMap,
    ) -> int:
        """Score how densely a candidate sits among other recognised fields."""
        score = 0

        for other_field_name, other_candidates in candidates.items():
            if other_field_name == field_name:
                continue

            strongest_nearby_score = 0

            for other_candidate in other_candidates:
                distance = abs(candidate.line_index - other_candidate.line_index)

                if distance == 0 or distance > self.CONTEXT_WINDOW_LINES:
                    continue

                nearby_score = self.CONTEXT_WINDOW_LINES + 1 - distance

                # Formatting may vary, so mixed styles still contribute. Matching styles
                # receive the full proximity score because they more often form one block.
                if candidate.uses_separator != other_candidate.uses_separator:
                    nearby_score //= 2

                strongest_nearby_score = max(
                    strongest_nearby_score,
                    nearby_score,
                )

            score += strongest_nearby_score

        return score

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
