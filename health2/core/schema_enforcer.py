"""
schema_enforcer.py — Deterministic post-processing

Runs after structure.py, before validate.py.
Fixes mechanical issues that LLMs consistently get wrong.
No LLM involved — pure Python, fully testable.

Current rules:
1. Time string conversion — "7 hours 20 minutes" → 7.333 (correct float)
2. Invalid time field values — relative phrases → null
3. String "unknown" → null
4. Extra fields not in schema → remove
5. Measurement value type — coerce string to float where possible
"""

import re
from typing import Optional
from datetime import date

# ─────────────────────────────────────────
# VALID SCHEMA FIELDS PER TYPE
# ─────────────────────────────────────────

SCHEMA_FIELDS = {
    "intake":       {"type", "label", "action", "dose", "unit", "time", "dose_timing", "event_date", "entity_date", "category", "is_intervention_dose", "day_of_protocol", "notes"},
    "symptom":      {"type", "label", "status", "onset_time", "severity", "qualifier", "duration", "interval", "source", "event_date", "entity_date", "notes"},
    "activity":     {"type", "label", "start_time", "end_time", "duration", "status", "event_date", "entity_date", "notes"},
    "machine":      {"type", "label", "start_time", "end_time", "duration", "status", "event_date", "entity_date", "notes"},
    "device":       {"type", "label", "start_time", "status"},
    "measurement":  {"type", "metric", "value", "unit", "time", "source", "notes"},
    "meal":         {"type", "label", "time", "eaten_out", "restaurant", "event_date", "entity_date", "description"},
    "intervention": {"type", "label", "start_date", "end_date", "status", "duration_days", "day_of_protocol", "notes"},
    "outcome":      {"type", "linked_to", "what", "onset_time", "qualifier", "direction"},
    "test":         {"type", "label", "time", "status", "result", "notes"},
    "context":      {"type", "raw_text", "related_to"},
    "theory":       {"type", "raw_text", "linked_to_label", "linked_to_type"},
    "outside":      {"type", "raw_text", "subtype", "notes"},
}

# Relative phrases that are not valid time values
INVALID_TIME_PHRASES = [
    "recently", "last week", "last month", "last year",
    "three weeks ago", "two weeks ago", "a few weeks ago",
    "a while ago", "some time ago", "earlier this week",
    "a few days ago", "days ago", "weeks ago", "months ago",
    "earlier timing", "recently changed", "not long ago",
]

# Time field names across all entity types
# Note: start_date is intentionally excluded — it is managed by calculate_day_of_protocol
TIME_FIELDS = {"time", "start_time", "onset_time", "end_date"}


# ─────────────────────────────────────────
# DURATION / TIME CONVERSION
# ─────────────────────────────────────────

def parse_hours_minutes(value_str: str) -> Optional[float]:
    """
    Convert duration strings to decimal hours.

    Examples:
      "7 hours 20 minutes" → 7.333
      "7h20m"              → 7.333
      "1 hour 45 minutes"  → 1.750
      "45 minutes"         → 0.750
      "2.5 hours"          → 2.500
      "7:20"               → 7.333
    """
    if not value_str or not isinstance(value_str, str):
        return None

    s = value_str.strip().lower()

    # Pattern: "X hours Y minutes" or "X hour Y minutes"
    m = re.match(r"(\d+(?:\.\d+)?)\s*hours?\s+(\d+(?:\.\d+)?)\s*minutes?", s)
    if m:
        return round(float(m.group(1)) + float(m.group(2)) / 60, 3)

    # Pattern: "X hours" only
    m = re.match(r"(\d+(?:\.\d+)?)\s*hours?$", s)
    if m:
        return round(float(m.group(1)), 3)

    # Pattern: "X minutes" only
    m = re.match(r"(\d+(?:\.\d+)?)\s*minutes?$", s)
    if m:
        return round(float(m.group(1)) / 60, 3)

    # Pattern: "XhYm"
    m = re.match(r"(\d+)h(\d+)m", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 60, 3)

    # Pattern: "H:MM" (e.g. "7:20")
    m = re.match(r"(\d+):(\d{2})$", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 60, 3)

    # Plain float or int string
    try:
        return round(float(s), 3)
    except ValueError:
        return None


# ─────────────────────────────────────────
# FIELD-LEVEL FIXES
# ─────────────────────────────────────────

def fix_unknown_strings(entity: dict) -> dict:
    """Replace string 'unknown' with None."""
    return {
        k: (None if isinstance(v, str) and v.strip().lower() == "unknown" else v)
        for k, v in entity.items()
    }


# Valid status values per entity type
VALID_STATUSES = {
    "activity": {"completed", "planned", "incomplete", "did_not_complete"},
    "machine":  {"used", "planned", "did_not_use"},
    "device":   {"used"},
    "test":     {"planned", "done"},
    "intervention": {"active", "completed", "unknown"},
}


def fix_invalid_status(entity: dict) -> tuple:
    """
    Remove entities with invalid status values.
    Returns (entity_or_none, violation_message_or_none)
    """
    entity_type = entity.get("type")
    valid = VALID_STATUSES.get(entity_type)
    if not valid:
        return entity, None

    status = entity.get("status")
    if status and status not in valid:
        label = entity.get("label", entity.get("metric", "?"))
        return None, f"Removed [{entity_type}] '{label}': invalid status '{status}' (valid: {sorted(valid)})"

    return entity, None


def fix_invalid_time_fields(entity: dict) -> dict:
    """Set relative/invalid time phrases to None."""
    result = dict(entity)
    for field in TIME_FIELDS:
        if field in result and isinstance(result[field], str):
            val = result[field].strip().lower()
            for phrase in INVALID_TIME_PHRASES:
                if phrase in val:
                    result[field] = None
                    break
    return result


def remove_extra_fields(entity: dict) -> dict:
    """Remove fields not defined in the schema for this entity type."""
    entity_type = entity.get("type")
    allowed = SCHEMA_FIELDS.get(entity_type)
    if not allowed:
        return entity
    return {k: v for k, v in entity.items() if k in allowed}


VAGUE_QUANTITY_PHRASES = [
    "two or three", "a few", "some", "a couple", "several",
    "one or two", "three or four", "handful", "a bit",
]


def fix_dose_field(entity: dict) -> dict:
    """
    For intake entities:
    If dose is a vague or ambiguous expression → set to null.
    Only unambiguous single numeric values are valid.
    """
    if entity.get("type") != "intake":
        return entity
    result = dict(entity)
    dose = result.get("dose")
    if isinstance(dose, str):
        dose_lower = dose.strip().lower()
        for phrase in VAGUE_QUANTITY_PHRASES:
            if phrase in dose_lower:
                result["dose"] = None
                result["unit"] = None
                return result
        # Try to coerce to float
        try:
            result["dose"] = float(dose)
        except (ValueError, TypeError):
            result["dose"] = None
            result["unit"] = None
    return result


def fix_measurement_value(entity: dict) -> dict:
    """
    For measurement entities:
    - If unit is 'hours' and value looks like a duration string → convert correctly
    - If value is a string number → coerce to float
    - If value cannot be converted → mark for removal
    """
    if entity.get("type") != "measurement":
        return entity

    result = dict(entity)
    value = result.get("value")
    unit = result.get("unit", "") or ""

    if value is None:
        # null value means no numeric was stated — per extraction rules, this should
        # not have been extracted. Drop it as a safety net.
        result["_remove"] = True
        result["_remove_reason"] = "measurement value is null — directional observations should not be extracted"
        return result

    if isinstance(value, str):
        # Duration strings like "7h 20m" are valid — leave as string
        import re as _re
        if _re.match(r"^\d+h\s*\d*m?$", value.strip()) or _re.match(r"^\d+m$", value.strip()):
            return result

        # Try duration conversion if unit suggests time
        if "hour" in unit.lower() or "min" in unit.lower():
            converted = parse_hours_minutes(value)
            if converted is not None:
                result["value"] = converted
                return result

        # Try plain numeric coercion
        try:
            result["value"] = float(value)
        except (ValueError, TypeError):
            result["_remove"] = True
            result["_remove_reason"] = f"measurement value not numeric: {value!r}"

    elif isinstance(value, (int, float)):
        result["value"] = float(value)

    return result


def fix_sleep_duration_value(entity: dict) -> dict:
    """
    For duration measurements (sleep, deep sleep, etc.):
    Normalize the string to compact "Xh Ym" format and leave as string.
    Do NOT convert to float — duration precision is lost in decimal conversion.
    """
    if entity.get("type") != "measurement":
        return entity

    result = dict(entity)
    value = result.get("value")
    unit = (result.get("unit", "") or "").lower()

    if not isinstance(value, str):
        return result

    if not any(t in unit for t in ("hour", "min")):
        return result

    # Parse and reformat as "Xh Ym"
    hours_match = __import__("re").match(
        r"(\d+(?:\.\d+)?)\s*hours?\s+(\d+(?:\.\d+)?)\s*minutes?", value.strip().lower()
    )
    if hours_match:
        h = int(float(hours_match.group(1)))
        m = int(float(hours_match.group(2)))
        result["value"] = f"{h}h {m}m"
        result["unit"] = "duration"
        return result

    mins_match = __import__("re").match(r"(\d+(?:\.\d+)?)\s*minutes?$", value.strip().lower())
    if mins_match:
        m = int(float(mins_match.group(1)))
        result["value"] = f"{m}m"
        result["unit"] = "duration"
        return result

    return result




# ─────────────────────────────────────────
# DOSE NUMBER FROM NOTES
# ─────────────────────────────────────────

import re as _re

DOSE_WORD_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
}

def extract_dose_number_from_notes(notes: str) -> int | None:
    """
    Extract dose number from intake notes field.
    "second dose" → 2, "3rd dose" → 3, "dose 4" → 4
    """
    if not notes or not isinstance(notes, str):
        return None
    s = notes.lower()

    # "second dose", "third pill", "first shot" etc.
    for word, num in DOSE_WORD_MAP.items():
        if word in s and any(t in s for t in ("dose", "pill", "shot", "tablet", "injection")):
            return num

    # "2nd dose", "3rd dose", "4th dose"
    m = _re.search(r"(\d+)(?:st|nd|rd|th)?\s+dose", s)
    if m:
        return int(m.group(1))

    # "dose 2", "dose number 3"
    m = _re.search(r"dose\s+(?:number\s+)?(\d+)", s)
    if m:
        return int(m.group(1))

    return None

# ─────────────────────────────────────────
# DAY OF PROTOCOL CALCULATION
# ─────────────────────────────────────────

# Maps relative start_date phrases to day offsets (days before today)
RELATIVE_DATE_OFFSETS = {
    "this morning": 0,
    "today": 0,
    "tonight": 0,
    "last night": 1,
    "yesterday": 1,
    "yesterday morning": 1,
    "yesterday evening": 1,
    "yesterday night": 1,
    "the night before last": 2,
    "two days ago": 2,
    "2 days ago": 2,
    "three days ago": 3,
    "3 days ago": 3,
    "four days ago": 4,
    "4 days ago": 4,
    "five days ago": 5,
    "5 days ago": 5,
    "six days ago": 6,
    "6 days ago": 6,
    "a week ago": 7,
    "one week ago": 7,
    "7 days ago": 7,
    "two weeks ago": 14,
    "2 weeks ago": 14,
    "last week": 7,
    "last month": 30,
}


def _resolve_start_date_to_iso(raw: str, today_date) -> "date | None":
    """
    Convert a start_date string to an ISO date object.

    Handles:
    - RELATIVE_DATE_OFFSETS map entries  ("yesterday", "two days ago", ...)
    - Weekday names                      ("last Monday", "last Friday", "Monday")
    - Already ISO format                 ("2025-04-17") — parsed and returned
    - Unknown phrases                    → None (caller keeps raw string)
    """
    from datetime import date, timedelta
    import re as _re

    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip().lower()

    # Already ISO — parse and return
    if _re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    # Direct map lookup
    offset = RELATIVE_DATE_OFFSETS.get(s)
    if offset is not None:
        return today_date - timedelta(days=offset)

    # "last <weekday>" or bare "<weekday>"
    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    m = _re.match(r"^(?:last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s)
    if m:
        target_wd = WEEKDAYS[m.group(1)]
        days_back = (today_date.weekday() - target_wd) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" on a Monday → 7 days ago
        return today_date - timedelta(days=days_back)

    return None


def _labels_match(label_a: str, label_b: str) -> bool:
    """
    Token-based label matching for intervention/intake pairing.
    Matches if any non-trivial token from one label appears in the other.

    Examples:
      "doxycycline" vs "doxycycline course"  → True
      "FMT" vs "FMT protocol"               → True
      "amoxicillin" vs "vitamin C"           → False
    """
    STOP_TOKENS = {"course", "protocol", "treatment", "therapy", "program",
                   "supplement", "dose", "medication", "drug", "pill", "tablet"}

    tokens_a = {t for t in label_a.lower().split() if t not in STOP_TOKENS and len(t) > 2}
    tokens_b = {t for t in label_b.lower().split() if t not in STOP_TOKENS and len(t) > 2}

    if tokens_a & tokens_b:
        return True
    if label_a.lower() in label_b.lower() or label_b.lower() in label_a.lower():
        return True
    return False


def calculate_day_of_protocol(entity: dict, today_date=None) -> dict:
    """
    For intervention entities: convert start_date to ISO and compute day_of_protocol
    as a fallback — runs before enricher, enricher results will overwrite if present.

    Uses _resolve_start_date_to_iso which handles:
    - Map entries ("yesterday", "two days ago", ...)
    - Weekday names ("last Monday", "Friday", ...)
    - Already-ISO dates (skipped — enricher already ran)
    - Unknown phrases → no-op
    """
    from datetime import date, timedelta

    if today_date is None:
        today_date = date.today()

    if entity.get("type") != "intervention":
        return entity

    result = dict(entity)
    start_date_raw = result.get("start_date")

    # Skip if already ISO — enricher already processed this
    import re as _re
    if start_date_raw and isinstance(start_date_raw, str) and _re.match(r"^\d{4}-\d{2}-\d{2}$", start_date_raw.strip()):
        return result

    start_date = _resolve_start_date_to_iso(start_date_raw, today_date)
    if start_date is None:
        return result  # Unknown phrase — leave as-is

    day = (today_date - start_date).days + 1
    if day < 1:
        print(f"⚠️  schema_enforcer [fallback]: intervention '{result.get('label')}' — "
              f"day_of_protocol={day} is invalid (start_date in the future?), skipping")
        return result

    result["start_date"] = start_date.isoformat()
    result["day_of_protocol"] = day

    duration_days = result.get("duration_days")
    if isinstance(duration_days, int) and not result.get("end_date"):
        end_date = start_date + timedelta(days=duration_days - 1)
        result["end_date"] = end_date.isoformat()

    print(f"⚙️  schema_enforcer [fallback]: intervention '{result.get('label')}' → "
          f"start_date={result['start_date']}, day_of_protocol={result['day_of_protocol']}")

    return result


def enrich_intervention_from_signals(entities: list, signals: list, today_date=None) -> list:
    """
    Deterministic enrichment from intervention_enricher signals.

    For each signal:
    - intervention entity: writes start_date (ISO), day_of_protocol, duration_days, end_date
    - intake entity: sets is_intervention_dose=True and calculates day_of_protocol

    day_of_protocol logic:
      If dose_number + event_date_raw both known:
        dose_date  = resolve(event_date_raw)
        start_date = dose_date - (dose_number - 1)
      If only event_date_raw known (no dose_number):
        start_date = resolve(event_date_raw)   # treat as protocol start
      intervention.day_of_protocol = (today - start_date).days + 1
      intake.day_of_protocol       = (intake_date - start_date).days + 1

    Guards:
    - day_of_protocol < 1 → skip (data inconsistency, do not write garbage)
    - intake event_date as ISO string → handled via date.fromisoformat()
    """
    from datetime import date, timedelta

    if today_date is None:
        today_date = date.today()

    if not signals:
        return entities

    for signal in signals:
        substance = signal.get("substance_label", "").lower()
        dose_number = signal.get("dose_number")
        event_date_raw = signal.get("event_date_raw")
        duration_days = signal.get("duration_days")
        is_completed = signal.get("is_completed", False)

        # ── Compute start_date from enricher signal ──────────────────────
        start_date = None
        if event_date_raw:
            dose_date = _resolve_start_date_to_iso(event_date_raw, today_date)
            if dose_date is not None:
                if isinstance(dose_number, int) and dose_number >= 1:
                    start_date = dose_date - timedelta(days=dose_number - 1)
                else:
                    start_date = dose_date  # no dose_number → event_date IS protocol start
        elif isinstance(dose_number, int) and dose_number >= 1:
            # event_date_raw yok ama dose_number biliniyor → bugün day N kabul et
            start_date = today_date - timedelta(days=dose_number - 1)
            print(f"⚙️  schema_enforcer: dose_number={dose_number}, no event_date_raw → "
                  f"assuming today is day {dose_number}, start_date={start_date.isoformat()}")

        
        # ── Fallback: enricher signal yetersizse intervention entity'nin
        #    mevcut start_date'ini kullan (calculate_day_of_protocol zaten yazmış olabilir)
        if start_date is None:
            for ent in entities:
                if ent.get("type") != "intervention":
                    continue
                if not _labels_match(substance, ent.get("label", "").lower()):
                    continue
                existing = ent.get("start_date")
                if existing and isinstance(existing, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", existing.strip()):
                    from datetime import date as _date_cls
                    try:
                        start_date = _date_cls.fromisoformat(existing.strip())
                        print(f"⚙️  schema_enforcer: intervention '{ent.get('label')}' — "
                              f"using existing start_date={existing} as fallback for intake enrichment")
                    except ValueError:
                        pass
                break

        # ── Update intervention entity ────────────────────────────────────
        for entity in entities:
            if entity.get("type") != "intervention":
                continue
            if not _labels_match(substance, entity.get("label", "").lower()):
                continue

            if start_date:
                day = (today_date - start_date).days + 1
                if day < 1:
                    print(f"⚠️  schema_enforcer: intervention '{entity.get('label')}' — "
                          f"computed day_of_protocol={day} is invalid, skipping date fields")
                else:
                    entity["start_date"] = start_date.isoformat()
                    entity["day_of_protocol"] = day
                    print(f"⚙️  schema_enforcer: intervention '{entity.get('label')}' → "
                          f"start_date={start_date.isoformat()}, day_of_protocol={day}")

                    if isinstance(duration_days, int):
                        entity["duration_days"] = duration_days
                        end_date = start_date + timedelta(days=duration_days - 1)
                        entity["end_date"] = end_date.isoformat()
                        print(f"⚙️  schema_enforcer: intervention '{entity.get('label')}' → "
                              f"end_date={end_date.isoformat()}")
            elif isinstance(duration_days, int):
                entity["duration_days"] = duration_days

            if is_completed:
                entity["status"] = "completed"
                if not entity.get("end_date"):
                    entity["end_date"] = today_date.isoformat()
                print(f"⚙️  schema_enforcer: intervention '{entity.get('label')}' → "
                      f"status=completed, end_date={entity['end_date']}")
            break

        # ── Update matching intake entities ──────────────────────────────
        # is_intervention_dose is written regardless of whether start_date is known.
        # day_of_protocol requires start_date — skipped if not available.
        for entity in entities:
            if entity.get("type") != "intake":
                continue
            if not _labels_match(substance, entity.get("label", "").lower()):
                continue

            entity["is_intervention_dose"] = True

            if start_date is None:
                print(f"⚙️  schema_enforcer: intake '{entity.get('label')}' → "
                      f"is_intervention_dose=True (day_of_protocol skipped — start_date unknown)")
                continue

            intake_event_raw = entity.get("event_date") or ""
            intake_date = _resolve_start_date_to_iso(intake_event_raw, today_date)

            if intake_date is None:
                # event_date missing or unresolvable → assume today
                intake_date = today_date

            day = (intake_date - start_date).days + 1
            if day < 1:
                print(f"⚠️  schema_enforcer: intake '{entity.get('label')}' — "
                      f"computed day_of_protocol={day} is invalid, skipping")
            else:
                entity["day_of_protocol"] = day
                suffix = f" (event_date: {intake_event_raw})" if intake_event_raw else " (today)"
                print(f"⚙️  schema_enforcer: intake '{entity.get('label')}' → "
                      f"is_intervention_dose=True, day_of_protocol={day}{suffix}")

    return entities

# ─────────────────────────────────────────
# MAIN ENFORCER
# ─────────────────────────────────────────

def enforce_schema(entities: list, note_date=None) -> tuple:
    """
    Apply all deterministic fixes to a list of entities.
    Returns (clean_entities, violations_log).

    note_date: the actual date of the voice note (from Created_at).
    If None, falls back to date.today().

    Intervention enrichment is handled separately via apply_enrichment_only()
    after enforce_schema() completes — never pass signals here.
    """
    from datetime import date
    today_date = note_date if note_date is not None else date.today()

    clean = []
    violations = []

    for entity in entities:
        original_type = entity.get("type", "unknown")

        # 1. Remove extra fields
        e = remove_extra_fields(entity)

        # 2. Fix "unknown" strings → None
        e = fix_unknown_strings(e)

        # 3. Fix invalid status values — remove entity if status is not in schema
        e, status_violation = fix_invalid_status(e)
        if e is None:
            violations.append(status_violation)
            continue

        # 4. Fix invalid time field values
        e = fix_invalid_time_fields(e)

        # 5. Fix dose field — vague quantities → null
        e = fix_dose_field(e)

        # 6. Fix measurement values — duration conversion first, then type coercion
        e = fix_sleep_duration_value(e)
        e = fix_measurement_value(e)

        # 7. Calculate day_of_protocol for intervention and intake
        e = calculate_day_of_protocol(e, today_date=today_date)

        # 8. Calculate entity_date — the actual date the event occurred
        #    event_date: null → today (note_date)
        #    event_date: "yesterday" → note_date - 1
        #    event_date: "two days ago" etc → resolved via RELATIVE_DATE_OFFSETS
        etype = e.get("type", "")
        if etype in {"intake", "symptom", "activity", "machine", "meal"}:
            from datetime import timedelta
            event_date_raw = e.get("event_date")
            if not event_date_raw:
                e["entity_date"] = str(today_date)
            else:
                resolved = _resolve_start_date_to_iso(str(event_date_raw), today_date)
                if resolved:
                    e["entity_date"] = str(resolved)
                else:
                    # Unresolvable phrase — fall back to note_date
                    e["entity_date"] = str(today_date)

        # 5. Remove entities flagged for removal
        if e.pop("_remove", False):
            reason = e.pop("_remove_reason", "schema violation")
            violations.append(f"Removed [{original_type}] {e.get('metric', e.get('label', '?'))}: {reason}")
            continue

        # Clean up internal flags
        e.pop("_remove_reason", None)

        clean.append(e)

    return clean, violations



def apply_enrichment_only(entities: list, intervention_signals: list, note_date=None) -> list:
    """
    Apply ONLY intervention enrichment signals to an already-enforced entity list.
    Does NOT re-run full enforce_schema — avoids double-applying field fixes
    and prevents fallback calculations from overwriting enricher results.

    note_date: the actual date of the voice note (from Created_at).
    Call this after enforce_schema() when enricher signals are ready.
    Returns the updated entity list.
    """
    from datetime import date
    today_date = note_date if note_date is not None else date.today()
    if not intervention_signals:
        return entities
    return enrich_intervention_from_signals(entities, intervention_signals, today_date=today_date)

def log_violations(violations: list):
    if violations:
        print(f"⚙️  schema_enforcer: {len(violations)} fix(es):")
        for v in violations:
            print(f"   → {v}")