"""Deterministic extraction of structured appointment details."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any

type PatternGroups = tuple[
    tuple[re.Pattern[str], ...],
    tuple[re.Pattern[str], ...],
    tuple[re.Pattern[str], ...],
]
type CandidateMap = dict[str, list["_FieldCandidate"]]


def _build_label_patterns(*labels: str) -> PatternGroups:
    """Build separator, whitespace and label-only patterns for one field."""
    separator_patterns = tuple(
        re.compile(
            rf"^\s*{label}\s*[:\-]\s*(?P<value>.*?)\s*$",
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
    label_only_patterns = tuple(
        re.compile(
            rf"^\s*{label}\s*:?[\-]?\s*$",
            re.IGNORECASE,
        )
        for label in labels
    )

    return separator_patterns, whitespace_patterns, label_only_patterns


class _CandidateSource(Enum):
    """Internal provenance used when ranking appointment candidates."""

    LABELLED_SEPARATOR = "LABELLED_SEPARATOR"
    LABELLED_NEXT_LINE = "LABELLED_NEXT_LINE"
    LABELLED_WHITESPACE = "LABELLED_WHITESPACE"
    NARRATIVE_PAIRED = "NARRATIVE_PAIRED"
    NARRATIVE = "NARRATIVE"


@dataclass(frozen=True, slots=True)
class _FieldCandidate:
    """Internal field candidate and its source-text position."""

    value: str
    start_line: int
    end_line: int
    source: _CandidateSource
    group_id: str | None = None


@dataclass(frozen=True, slots=True)
class _NarrativeWindow:
    """Small reconstructed text window used for narrative matching."""

    text: str
    start_line: int
    end_line: int


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
    """Extract deterministic appointment suggestions from structured and narrative text."""

    MAXIMUM_FIELD_LENGTH = 250
    CONTEXT_WINDOW_LINES = 6
    NARRATIVE_WINDOW_LINES = 4
    AMBIGUITY_SCORE_MARGIN = 3

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

    FIELD_PATTERNS: dict[str, tuple[PatternGroups, int]] = {
        DATE_FIELD: (
            _build_label_patterns(r"On", r"Appointment\s+date", r"Date"),
            MAXIMUM_FIELD_LENGTH,
        ),
        START_TIME_FIELD: (
            _build_label_patterns(r"Appointment\s+time", r"Start\s+time", r"Time"),
            MAXIMUM_FIELD_LENGTH,
        ),
        END_TIME_FIELD: (
            _build_label_patterns(r"End\s+time", r"Finish\s+time"),
            MAXIMUM_FIELD_LENGTH,
        ),
        SERVICE_FIELD: (
            _build_label_patterns(r"Service\s+type", r"Service"),
            MAXIMUM_FIELD_LENGTH,
        ),
        APPOINTMENT_TYPE_FIELD: (
            _build_label_patterns(r"Appointment\s+type"),
            MAXIMUM_FIELD_LENGTH,
        ),
        CLINICIAN_FIELD: (
            _build_label_patterns(r"Clinician", r"Consultant", r"Team", r"With"),
            MAXIMUM_FIELD_LENGTH,
        ),
        LOCATION_FIELD: (
            _build_label_patterns(r"Location", r"Venue", r"Clinic"),
            MAXIMUM_FIELD_LENGTH,
        ),
        ADDRESS_LINE_1_FIELD: (
            _build_label_patterns(r"Address\s+line\s+1", r"Address(?!\s+line\s+2\b)"),
            150,
        ),
        ADDRESS_LINE_2_FIELD: (
            _build_label_patterns(r"Address\s+line\s+2"),
            150,
        ),
        TOWN_CITY_FIELD: (
            _build_label_patterns(
                r"Town\s+or\s+city",
                r"Town\s*/\s*City",
                r"Town",
                r"City",
            ),
            100,
        ),
        COUNTY_FIELD: (_build_label_patterns(r"County"), 100),
        POSTCODE_FIELD: (_build_label_patterns(r"Postcode"), 20),
        COUNTRY_FIELD: (_build_label_patterns(r"Country"), 100),
    }

    COMBINED_DATE_TIME_PATTERNS = _build_label_patterns(
        r"Appointment\s+date\s+and\s+time",
        r"Date\s+and\s+time",
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

    MONTH_PATTERN = (
        r"(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )
    WEEKDAY_PATTERN = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    DATE_VALUE_PATTERN = (
        rf"(?:{WEEKDAY_PATTERN}\s+)?(?:"
        rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}|"
        rf"\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{4}}|"
        rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_PATTERN},?\s+\d{{4}})"
    )
    TIME_VALUE_PATTERN = r"(?:\d{1,2}(?::|\.)\d{2}(?:\s*(?:am|pm))?|\d{1,2}\s*(?:am|pm))"
    APPOINTMENT_ANCHOR_PATTERN = (
        r"(?:appointment|booked|booking|scheduled|arranged|attend|attendance)"
    )

    NARRATIVE_DATE_TIME_PATTERN = re.compile(
        rf"\b{APPOINTMENT_ANCHOR_PATTERN}\b.{{0,140}}?\b(?:on|for)\s+"
        rf"(?P<date>{DATE_VALUE_PATTERN})\s+(?:at|@)\s+"
        rf"(?P<time>{TIME_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    NARRATIVE_DATE_PATTERN = re.compile(
        rf"\b{APPOINTMENT_ANCHOR_PATTERN}\b.{{0,140}}?\b(?:on|for)\s+"
        rf"(?P<date>{DATE_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    NARRATIVE_START_TIME_PATTERN = re.compile(
        rf"\b{APPOINTMENT_ANCHOR_PATTERN}\b.{{0,140}}?\b(?:at|from)\s+"
        rf"(?P<time>{TIME_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    NARRATIVE_END_TIME_PATTERN = re.compile(
        rf"\b(?:expected\s+to\s+)?(?:finish|finishes|end|ends)\b.{{0,40}}?"
        rf"\b(?:at|around|about)?\s*(?:approximately\s+|around\s+|about\s+)?"
        rf"(?P<time>{TIME_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    NARRATIVE_UNTIL_TIME_PATTERN = re.compile(
        rf"\b(?:appointment|session|visit)\b.{{0,100}}?\buntil\s+"
        rf"(?P<time>{TIME_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    NARRATIVE_CLINICIAN_PATTERN = re.compile(
        r"\b(?:appointment|booked|scheduled|arranged)\b.{0,180}?\bwith\s+(?:the\s+)?"
        r"(?P<value>[A-Za-z][A-Za-z0-9&'() /\-]{1,120}?)"
        r"(?=\s*(?:[.;]|\bplease\b|\bwhere\b|\bwhich\b|$))",
        re.IGNORECASE,
    )
    NARRATIVE_APPOINTMENT_TYPE_PATTERN = re.compile(
        r"\b(?:the\s+)?appointment\s+is\s+for\s+(?P<value>[^.;]{2,180})"
        r"(?=\s*(?:[.;]|$))",
        re.IGNORECASE,
    )
    NARRATIVE_PLEASE_ATTEND_LOCATION_PATTERN = re.compile(
        r"\bplease\s+attend\s+(?:at\s+)?(?:the\s+)?(?P<value>[^.;]{2,180})"
        r"(?=\s*(?:[.;]|$))",
        re.IGNORECASE,
    )
    NARRATIVE_APPOINTMENT_LOCATION_PATTERN = re.compile(
        r"\b(?:your\s+)?appointment\s+(?:will\s+take\s+place\s+at|is\s+at)\s+"
        r"(?:the\s+)?(?P<value>[^.;]{2,180})(?=\s*(?:[.;]|$))",
        re.IGNORECASE,
    )
    NARRATIVE_SERVICE_PATTERN = re.compile(
        r"\b(?:appointment|booked|scheduled|referred)\b.{0,140}?\bwith\s+(?:the\s+)?"
        r"(?P<value>[A-Za-z][A-Za-z0-9&'() /\-]{1,100}?"
        r"(?:department|service|clinic))\b",
        re.IGNORECASE,
    )

    NARRATIVE_FIELD_PATTERNS: tuple[
        tuple[str, re.Pattern[str], str, bool],
        ...,
    ] = (
        (DATE_FIELD, NARRATIVE_DATE_PATTERN, "date", False),
        (START_TIME_FIELD, NARRATIVE_START_TIME_PATTERN, "time", False),
        (END_TIME_FIELD, NARRATIVE_END_TIME_PATTERN, "time", False),
        (END_TIME_FIELD, NARRATIVE_UNTIL_TIME_PATTERN, "time", False),
        (CLINICIAN_FIELD, NARRATIVE_CLINICIAN_PATTERN, "value", False),
        (APPOINTMENT_TYPE_FIELD, NARRATIVE_APPOINTMENT_TYPE_PATTERN, "value", False),
        (LOCATION_FIELD, NARRATIVE_PLEASE_ATTEND_LOCATION_PATTERN, "value", True),
        (LOCATION_FIELD, NARRATIVE_APPOINTMENT_LOCATION_PATTERN, "value", True),
        (SERVICE_FIELD, NARRATIVE_SERVICE_PATTERN, "value", False),
    )

    SOURCE_SCORES = {
        _CandidateSource.LABELLED_SEPARATOR: 24,
        _CandidateSource.LABELLED_NEXT_LINE: 22,
        _CandidateSource.LABELLED_WHITESPACE: 18,
        _CandidateSource.NARRATIVE_PAIRED: 20,
        _CandidateSource.NARRATIVE: 14,
    }

    def extract(self, text: str) -> AppointmentDetailsResult:
        """Extract appointment fields using labelled, reflowed and narrative passes."""
        lines = [" ".join(line.split()) for line in text.splitlines()]
        candidates = self._collect_labelled_candidates(lines)

        self._collect_combined_date_time_candidates(lines, candidates)
        self._collect_narrative_candidates(self._build_narrative_windows(lines), candidates)
        self._remove_unparseable_temporal_candidates(candidates)

        selected_date, date_ambiguous = self._select_parsed(
            candidates,
            self.DATE_FIELD,
            self._parse_date,
        )
        selected_start_time, start_time_ambiguous = self._select_parsed(
            candidates,
            self.START_TIME_FIELD,
            self._parse_time,
        )
        selected_end_time, end_time_ambiguous = self._select_parsed(
            candidates,
            self.END_TIME_FIELD,
            self._parse_time,
        )
        selected_service, service_ambiguous = self._select_string(
            candidates,
            self.SERVICE_FIELD,
        )
        selected_type, type_ambiguous = self._select_string(
            candidates,
            self.APPOINTMENT_TYPE_FIELD,
        )
        selected_clinician, clinician_ambiguous = self._select_string(
            candidates,
            self.CLINICIAN_FIELD,
        )
        selected_location, location_ambiguous = self._select_string(
            candidates,
            self.LOCATION_FIELD,
        )
        address, address_ambiguities = self._extract_address(candidates)

        details = AppointmentDetails(
            date=selected_date,
            start_time=selected_start_time,
            end_time=selected_end_time,
            service=selected_service,
            appointment_type=selected_type,
            clinician_or_team=selected_clinician,
            location_name=selected_location,
            address=address,
        )

        ambiguous_fields = [
            field_name
            for field_name, is_ambiguous in (
                ("date", date_ambiguous),
                ("start time", start_time_ambiguous),
                ("end time", end_time_ambiguous),
                ("service", service_ambiguous),
                ("appointment type", type_ambiguous),
                ("clinician or team", clinician_ambiguous),
                ("location", location_ambiguous),
            )
            if is_ambiguous
        ]
        ambiguous_fields.extend(address_ambiguities)

        return AppointmentDetailsResult(
            details=details,
            processing_warning=self._build_warning(details, ambiguous_fields),
        )

    def _collect_labelled_candidates(self, lines: list[str]) -> CandidateMap:
        """Collect candidates from ordinary labelled appointment fields."""
        return {
            field_name: self._find_labelled_candidates(lines, patterns, maximum_length)
            for field_name, (patterns, maximum_length) in self.FIELD_PATTERNS.items()
        }

    def _find_labelled_candidates(
        self,
        lines: list[str],
        patterns: PatternGroups,
        maximum_length: int,
    ) -> list[_FieldCandidate]:
        """Return all usable labelled candidates, including next-line values."""
        separator_patterns, whitespace_patterns, label_only_patterns = patterns
        candidates: list[_FieldCandidate] = []

        for line_index, line in enumerate(lines):
            if not line:
                continue

            separator_match = self._first_match(line, separator_patterns)

            if separator_match is not None:
                value = self._normalise_value(separator_match.group("value"), maximum_length)

                if value is not None:
                    candidates.append(
                        _FieldCandidate(
                            value=value,
                            start_line=line_index,
                            end_line=line_index,
                            source=_CandidateSource.LABELLED_SEPARATOR,
                        )
                    )
                    continue

                next_value = self._next_line_value(lines, line_index, maximum_length)

                if next_value is not None:
                    value, value_line = next_value
                    candidates.append(
                        _FieldCandidate(
                            value=value,
                            start_line=line_index,
                            end_line=value_line,
                            source=_CandidateSource.LABELLED_NEXT_LINE,
                        )
                    )
                    continue

            whitespace_match = self._first_match(line, whitespace_patterns)

            if whitespace_match is not None:
                value = self._normalise_value(whitespace_match.group("value"), maximum_length)

                if value is not None:
                    candidates.append(
                        _FieldCandidate(
                            value=value,
                            start_line=line_index,
                            end_line=line_index,
                            source=_CandidateSource.LABELLED_WHITESPACE,
                        )
                    )
                    continue

            if self._first_match(line, label_only_patterns) is None:
                continue

            next_value = self._next_line_value(lines, line_index, maximum_length)

            if next_value is not None:
                value, value_line = next_value
                candidates.append(
                    _FieldCandidate(
                        value=value,
                        start_line=line_index,
                        end_line=value_line,
                        source=_CandidateSource.LABELLED_NEXT_LINE,
                    )
                )

        return self._deduplicate_candidates(candidates)

    def _collect_combined_date_time_candidates(
        self,
        lines: list[str],
        candidates: CandidateMap,
    ) -> None:
        """Add paired candidates from combined date-and-time labels."""
        combined_candidates = self._find_labelled_candidates(
            lines,
            self.COMBINED_DATE_TIME_PATTERNS,
            self.MAXIMUM_FIELD_LENGTH,
        )

        for index, candidate in enumerate(combined_candidates):
            pair = self._find_date_time_pair(candidate.value)

            if pair is None:
                continue

            date_value, time_value = pair
            group_id = f"combined:{candidate.start_line}:{index}"

            for field_name, value in (
                (self.DATE_FIELD, date_value),
                (self.START_TIME_FIELD, time_value),
            ):
                candidates[field_name].append(
                    _FieldCandidate(
                        value=value,
                        start_line=candidate.start_line,
                        end_line=candidate.end_line,
                        source=candidate.source,
                        group_id=group_id,
                    )
                )

    def _build_narrative_windows(self, lines: list[str]) -> list[_NarrativeWindow]:
        """Build small adjacent-line windows without changing extracted display text."""
        windows: list[_NarrativeWindow] = []

        for start_line, line in enumerate(lines):
            if not line or self._is_label_line(line):
                continue

            window_lines: list[str] = []

            for end_line in range(
                start_line,
                min(len(lines), start_line + self.NARRATIVE_WINDOW_LINES),
            ):
                current_line = lines[end_line]

                if not current_line:
                    break

                if end_line > start_line and self._is_label_line(current_line):
                    break

                window_lines.append(current_line)
                windows.append(
                    _NarrativeWindow(
                        text=" ".join(window_lines),
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        return windows

    def _collect_narrative_candidates(
        self,
        windows: list[_NarrativeWindow],
        candidates: CandidateMap,
    ) -> None:
        """Add deterministic candidates from common appointment narrative wording."""
        pair_counter = 0

        for window in windows:
            for match in self.NARRATIVE_DATE_TIME_PATTERN.finditer(window.text):
                group_id = f"narrative:{window.start_line}:{window.end_line}:{pair_counter}"
                pair_counter += 1

                self._append_candidate(
                    candidates,
                    self.DATE_FIELD,
                    match.group("date"),
                    window,
                    _CandidateSource.NARRATIVE_PAIRED,
                    group_id,
                )
                self._append_candidate(
                    candidates,
                    self.START_TIME_FIELD,
                    match.group("time"),
                    window,
                    _CandidateSource.NARRATIVE_PAIRED,
                    group_id,
                )

            for field_name, pattern, group_name, reject_temporal in self.NARRATIVE_FIELD_PATTERNS:
                for match in pattern.finditer(window.text):
                    value = self._normalise_value(
                        match.group(group_name),
                        self.MAXIMUM_FIELD_LENGTH,
                    )

                    if value is None:
                        continue

                    if reject_temporal and self._looks_temporal(value):
                        continue

                    self._append_candidate(
                        candidates,
                        field_name,
                        value,
                        window,
                        _CandidateSource.NARRATIVE,
                    )

        for field_name in candidates:
            candidates[field_name] = self._deduplicate_candidates(candidates[field_name])

    def _append_candidate(
        self,
        candidates: CandidateMap,
        field_name: str,
        value: str,
        window: _NarrativeWindow,
        source: _CandidateSource,
        group_id: str | None = None,
    ) -> None:
        """Append one candidate produced from a reconstructed text window."""
        normalised_value = self._normalise_value(value, self.MAXIMUM_FIELD_LENGTH)

        if normalised_value is None:
            return

        candidates[field_name].append(
            _FieldCandidate(
                value=normalised_value,
                start_line=window.start_line,
                end_line=window.end_line,
                source=source,
                group_id=group_id,
            )
        )

    def _remove_unparseable_temporal_candidates(self, candidates: CandidateMap) -> None:
        """Discard invalid temporal candidates while continuing to later valid values."""
        for field_name, parser in (
            (self.DATE_FIELD, self._parse_date),
            (self.START_TIME_FIELD, self._parse_time),
            (self.END_TIME_FIELD, self._parse_time),
        ):
            candidates[field_name] = [
                candidate
                for candidate in candidates[field_name]
                if parser(candidate.value) is not None
            ]

    def _select_parsed(
        self,
        candidates: CandidateMap,
        field_name: str,
        parser: Callable[[str], Any | None],
    ) -> tuple[Any | None, bool]:
        """Select a candidate and return its parsed value."""
        candidate, ambiguous = self._select_candidate(candidates, field_name, parser)

        if candidate is None:
            return None, False

        return parser(candidate.value), ambiguous

    def _select_string(
        self,
        candidates: CandidateMap,
        field_name: str,
    ) -> tuple[str | None, bool]:
        """Select the strongest string candidate for one appointment field."""
        candidate, ambiguous = self._select_candidate(
            candidates,
            field_name,
            lambda value: value.casefold(),
        )

        if candidate is None:
            return None, False

        return candidate.value, ambiguous

    def _select_candidate(
        self,
        candidates: CandidateMap,
        field_name: str,
        comparison_value: Callable[[str], Any],
    ) -> tuple[_FieldCandidate | None, bool]:
        """Rank candidates and flag close competing values for human review."""
        field_candidates = candidates[field_name]

        if not field_candidates:
            return None, False

        ranked = sorted(
            field_candidates,
            key=lambda candidate: (
                self._candidate_score(candidate, field_name, candidates),
                self.SOURCE_SCORES[candidate.source],
                -candidate.start_line,
            ),
            reverse=True,
        )
        selected = ranked[0]
        selected_score = self._candidate_score(selected, field_name, candidates)
        selected_value = comparison_value(selected.value)

        for competing in ranked[1:]:
            competing_value = comparison_value(competing.value)

            if self._comparison_values_equivalent(selected_value, competing_value):
                continue

            competing_score = self._candidate_score(competing, field_name, candidates)
            return selected, selected_score - competing_score <= self.AMBIGUITY_SCORE_MARGIN

        return selected, False

    def _candidate_score(
        self,
        candidate: _FieldCandidate,
        field_name: str,
        candidates: CandidateMap,
    ) -> int:
        """Score provenance, nearby appointment fields and paired date/time evidence."""
        score = self.SOURCE_SCORES[candidate.source]

        for other_field, other_candidates in candidates.items():
            if other_field == field_name:
                continue

            strongest = 0

            for other_candidate in other_candidates:
                distance = self._line_distance(candidate, other_candidate)

                if distance > self.CONTEXT_WINDOW_LINES:
                    continue

                nearby_score = self.CONTEXT_WINDOW_LINES + 1 - distance

                if (
                    candidate.group_id is not None
                    and candidate.group_id == other_candidate.group_id
                    and {field_name, other_field} == {self.DATE_FIELD, self.START_TIME_FIELD}
                ):
                    nearby_score += 12

                strongest = max(strongest, nearby_score)

            score += strongest

        return score

    def _extract_address(
        self,
        candidates: CandidateMap,
    ) -> tuple[AppointmentAddressDetails | None, list[str]]:
        """Extract independently ranked partial address fields."""
        field_names = {
            self.ADDRESS_LINE_1_FIELD: "address line 1",
            self.ADDRESS_LINE_2_FIELD: "address line 2",
            self.TOWN_CITY_FIELD: "town or city",
            self.COUNTY_FIELD: "county",
            self.POSTCODE_FIELD: "postcode",
            self.COUNTRY_FIELD: "country",
        }
        selected: dict[str, str | None] = {}
        ambiguous_fields: list[str] = []

        for field_name, display_name in field_names.items():
            value, ambiguous = self._select_string(candidates, field_name)
            selected[field_name] = value

            if ambiguous:
                ambiguous_fields.append(display_name)

        address = AppointmentAddressDetails(
            address_line_1=selected[self.ADDRESS_LINE_1_FIELD],
            address_line_2=selected[self.ADDRESS_LINE_2_FIELD],
            town_city=selected[self.TOWN_CITY_FIELD],
            county=selected[self.COUNTY_FIELD],
            postcode=selected[self.POSTCODE_FIELD],
            country=selected[self.COUNTRY_FIELD],
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
            return None, []

        return address, ambiguous_fields

    def _parse_date(self, value: str) -> date | None:
        """Normalise a candidate using the supported deterministic date formats."""
        normalised = re.sub(
            r"(\d{1,2})(st|nd|rd|th)\b",
            r"\1",
            value,
            flags=re.IGNORECASE,
        )
        normalised = normalised.replace(",", "")
        normalised = re.sub(r"\bSept\b", "Sep", normalised, flags=re.IGNORECASE)
        normalised = " ".join(normalised.split())

        for date_format in self.DATE_FORMATS:
            try:
                return datetime.strptime(normalised, date_format).date()
            except ValueError:
                continue

        return None

    def _parse_time(self, value: str) -> time | None:
        """Normalise a candidate using the supported deterministic time formats."""
        normalised = value.strip().rstrip(".,;:")
        normalised = re.sub(
            r"\s*(am|pm)$",
            r" \1",
            normalised,
            flags=re.IGNORECASE,
        )
        normalised = " ".join(normalised.split()).upper()

        for time_format in self.TIME_FORMATS:
            try:
                return datetime.strptime(normalised, time_format).time()
            except ValueError:
                continue

        return None

    def _find_date_time_pair(self, value: str) -> tuple[str, str] | None:
        """Find a supported date followed by a supported time in one value."""
        pattern = re.compile(
            rf"(?P<date>{self.DATE_VALUE_PATTERN})\s*(?:at|,)?\s*"
            rf"(?P<time>{self.TIME_VALUE_PATTERN})\b",
            re.IGNORECASE,
        )
        match = pattern.search(value)

        if match is None:
            return None

        return match.group("date"), match.group("time")

    def _next_line_value(
        self,
        lines: list[str],
        line_index: int,
        maximum_length: int,
    ) -> tuple[str, int] | None:
        """Return one following non-empty line when it is not another label."""
        next_index = line_index + 1

        while next_index < len(lines) and not lines[next_index]:
            next_index += 1

        if next_index >= len(lines) or self._is_label_line(lines[next_index]):
            return None

        value = self._normalise_value(lines[next_index], maximum_length)

        if value is None:
            return None

        return value, next_index

    def _is_label_line(self, line: str) -> bool:
        """Return whether a line begins with any supported structured label."""
        pattern_sets = [patterns for patterns, _ in self.FIELD_PATTERNS.values()]
        pattern_sets.append(self.COMBINED_DATE_TIME_PATTERNS)

        return any(
            self._first_match(line, pattern_group) is not None
            for pattern_set in pattern_sets
            for pattern_group in pattern_set
        )

    def _looks_temporal(self, value: str) -> bool:
        """Avoid interpreting an explicit date or time as a narrative location."""
        return self._parse_date(value) is not None or self._parse_time(value) is not None

    def _first_match(
        self,
        line: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> re.Match[str] | None:
        """Return the first matching pattern for a line."""
        return next((match for pattern in patterns if (match := pattern.match(line))), None)

    def _normalise_value(self, value: str, maximum_length: int) -> str | None:
        """Normalise a candidate and reject blank or oversized content."""
        normalised = " ".join(value.split()).strip(" |").strip()

        if not normalised or len(normalised) > maximum_length:
            return None

        return normalised

    def _deduplicate_candidates(
        self,
        candidates: list[_FieldCandidate],
    ) -> list[_FieldCandidate]:
        """Remove exact duplicates created by overlapping narrative windows."""
        seen: set[tuple[str, int, int, _CandidateSource, str | None]] = set()
        deduplicated: list[_FieldCandidate] = []

        for candidate in candidates:
            key = (
                candidate.value.casefold(),
                candidate.start_line,
                candidate.end_line,
                candidate.source,
                candidate.group_id,
            )

            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(candidate)

        return deduplicated

    def _comparison_values_equivalent(self, first: Any, second: Any) -> bool:
        """Treat identical values and closely overlapping text values as equivalent."""
        if first == second:
            return True

        if isinstance(first, str) and isinstance(second, str):
            shorter, longer = sorted((first, second), key=len)
            return len(shorter) >= 4 and shorter in longer

        return False

    def _line_distance(self, first: _FieldCandidate, second: _FieldCandidate) -> int:
        """Measure the minimum source-line distance between candidate spans."""
        if first.end_line < second.start_line:
            return second.start_line - first.end_line

        if second.end_line < first.start_line:
            return first.start_line - second.end_line

        return 0

    def _build_warning(
        self,
        details: AppointmentDetails,
        ambiguous_fields: list[str],
    ) -> str | None:
        """Build review warnings for missing or closely competing values."""
        identified_values = (
            details.date,
            details.start_time,
            details.end_time,
            details.service,
            details.appointment_type,
            details.clinician_or_team,
            details.location_name,
            details.address,
        )

        if not any(identified_values):
            return (
                "No supported appointment details were identified. "
                "Review the extracted text and enter the appointment details manually."
            )

        warnings: list[str] = []
        missing_core_fields: list[str] = []

        if details.date is None:
            missing_core_fields.append("date")

        if details.start_time is None:
            missing_core_fields.append("start time")

        if details.location_name is None and details.address is None:
            missing_core_fields.append("location")

        if missing_core_fields:
            warnings.append(
                "Some core appointment details could not be identified or normalised: "
                + self._format_field_names(missing_core_fields)
                + "."
            )

        unique_ambiguous_fields = list(dict.fromkeys(ambiguous_fields))

        if unique_ambiguous_fields:
            warnings.append(
                "Multiple plausible values were identified for "
                + self._format_field_names(unique_ambiguous_fields)
                + "; the strongest matches have been suggested."
            )

        if not warnings:
            return None

        return " ".join(warnings) + " Review the extracted text before confirming the appointment."

    def _format_field_names(self, field_names: list[str]) -> str:
        """Format field names for a readable warning message."""
        if len(field_names) == 1:
            return field_names[0]

        if len(field_names) == 2:
            return " and ".join(field_names)

        return ", ".join(field_names[:-1]) + f", and {field_names[-1]}"
