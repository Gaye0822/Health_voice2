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

# ─────────────────────────────────────────
# VALID SCHEMA FIELDS PER TYPE
# ─────────────────────────────────────────

SCHEMA_FIELDS = {
    "intake":       {"type", "label", "action", "dose", "unit", "time", "category", "notes"},
    "symptom":      {"type", "label", "status", "onset_time", "severity", "qualifier", "duration", "interval", "source", "notes"},
    "activity":     {"type", "label", "start_time", "duration", "status", "notes"},
    "machine":      {"type", "label", "start_time", "duration", "status", "notes"},
    "device":       {"type", "label", "start_time", "status"},
    "measurement":  {"type", "metric", "value", "unit", "time", "source", "notes"},
    "meal":         {"type", "label", "time", "eaten_out", "restaurant", "description"},
    "intervention": {"type", "label", "start_date", "end_date", "status", "notes"},
    "outcome":      {"type", "linked_to", "onset_time", "qualifier", "direction"},
    "test":         {"type", "label", "time", "status", "result", "notes"},
    "context":      {"type", "raw_text", "related_to"},
    "theory":       {"type", "raw_text", "linked_to_label", "linked_to_type"},
    "outside":      {"type", "raw_text"},
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
TIME_FIELDS = {"time", "start_time", "onset_time", "start_date", "end_date"}


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
        # null value is valid — directional observation captured in notes
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
# MAIN ENFORCER
# ─────────────────────────────────────────

def enforce_schema(entities: list) -> tuple:
    """
    Apply all deterministic fixes to a list of entities.

    Returns:
        (clean_entities, violations_log)
    """
    clean = []
    violations = []

    for entity in entities:
        original_type = entity.get("type", "unknown")

        # 1. Remove extra fields
        e = remove_extra_fields(entity)

        # 2. Fix "unknown" strings → None
        e = fix_unknown_strings(e)

        # 3. Fix invalid time field values
        e = fix_invalid_time_fields(e)

        # 4. Fix dose field — vague quantities → null
        e = fix_dose_field(e)

        # 5. Fix measurement values — duration conversion first, then type coercion
        e = fix_sleep_duration_value(e)  # converts string duration → float
        e = fix_measurement_value(e)      # coerces remaining strings → float

        # 5. Remove entities flagged for removal
        if e.pop("_remove", False):
            reason = e.pop("_remove_reason", "schema violation")
            violations.append(f"Removed [{original_type}] {e.get('metric', e.get('label', '?'))}: {reason}")
            continue

        # Clean up internal flags
        e.pop("_remove_reason", None)

        clean.append(e)

    return clean, violations


def log_violations(violations: list):
    if violations:
        print(f"⚙️  schema_enforcer: {len(violations)} fix(es):")
        for v in violations:
            print(f"   → {v}")