"""
intervention_merger.py — Merge layer for intervention + normal pipeline outputs

Runs AFTER both pipelines complete, BEFORE validate.py.

Responsibilities:
  1. Combine intervention entities + normal entities into a single list
  2. For each intervention entity, find matching intake entities by label
  3. Write is_intervention_dose=True on matched intakes
  4. Calculate day_of_protocol on matched intakes using intervention's start_date

Explicitly NOT responsible for:
  - Intervention entity field resolution (done by intervention_pipeline.py)
  - Schema enforcement (done by schema_enforcer.py)
  - Any LLM calls

Design principle:
  - intervention_pipeline produces intervention entities with start_date as ISO string
  - Normal pipeline produces intake entities with event_date (raw or ISO)
  - Merger connects them deterministically — no LLM, no heuristics
"""

import re
from datetime import date, timedelta


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

_STOP_TOKENS = {
    "course", "protocol", "treatment", "therapy", "program",
    "supplement", "dose", "medication", "drug", "pill", "tablet"
}

_RELATIVE_DATE_OFFSETS = {
    "today": 0,
    "this morning": 0,
    "this evening": 0,
    "this afternoon": 0,
    "tonight": 0,
    "just now": 0,
    "yesterday": 1,
    "yesterday morning": 1,
    "yesterday afternoon": 1,
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

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _labels_match(label_a: str, label_b: str) -> bool:
    """Token-based label matching — same logic throughout the codebase."""
    ta = {t for t in label_a.lower().split() if t not in _STOP_TOKENS and len(t) > 2}
    tb = {t for t in label_b.lower().split() if t not in _STOP_TOKENS and len(t) > 2}
    if ta & tb:
        return True
    if label_a.lower() in label_b.lower() or label_b.lower() in label_a.lower():
        return True
    return False


def _resolve_date(raw: str, today: date) -> "date | None":
    """Convert a relative or ISO date string to a date object. Returns None if unresolvable."""
    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip().lower()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    offset = _RELATIVE_DATE_OFFSETS.get(s)
    if offset is not None:
        return today - timedelta(days=offset)

    m = re.match(r"^(?:last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s)
    if m:
        target_wd = _WEEKDAYS[m.group(1)]
        days_back = (today.weekday() - target_wd) % 7
        if days_back == 0:
            days_back = 7
        return today - timedelta(days=days_back)

    return None


# ─────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────

def merge_intervention_entities(
    intervention_entities: list,
    normal_entities: list,
    today_date: "date | None" = None,
) -> list:
    """
    Merge intervention entities with normal pipeline entities.

    For each intervention entity:
      - Find all intake entities whose label matches (token-based)
      - Set is_intervention_dose=True on each
      - If intervention has a resolved start_date (ISO), calculate day_of_protocol
        for each intake using its event_date

    Returns:
        Combined entity list with intervention + enriched intakes.
        Order: normal entities first, then intervention entities.
        (validate.py and schema_enforcer.py are order-agnostic)
    """
    if today_date is None:
        today_date = date.today()

    if not intervention_entities:
        # Nothing to merge — return normal entities as-is
        return normal_entities

    # Work on copies to avoid mutating caller's lists
    normal = [dict(e) for e in normal_entities]
    interventions = [dict(e) for e in intervention_entities]

    for intervention in interventions:
        int_label = intervention.get("label", "")
        start_date_raw = intervention.get("start_date")

        # Resolve start_date to date object (should already be ISO from pipeline,
        # but handle raw string fallback gracefully)
        start_date: "date | None" = _resolve_date(start_date_raw, today_date) if start_date_raw else None

        matched_intakes = [
            e for e in normal
            if e.get("type") == "intake" and _labels_match(int_label, e.get("label", ""))
        ]

        if not matched_intakes:
            print(
                f"⚙️  intervention_merger: '{int_label}' — "
                f"no matching intake entities found"
            )
        else:
            print(
                f"⚙️  intervention_merger: '{int_label}' → "
                f"{len(matched_intakes)} matching intake(s)"
            )

        for intake in matched_intakes:
            intake["is_intervention_dose"] = True

            if start_date is None:
                print(
                    f"⚙️  intervention_merger: intake '{intake.get('label')}' → "
                    f"is_intervention_dose=True (day_of_protocol skipped — start_date unknown)"
                )
                continue

            # Resolve intake's event_date
            intake_event_raw = intake.get("event_date") or ""
            intake_date = _resolve_date(intake_event_raw, today_date)

            if intake_date is None:
                # No event_date → assume today
                intake_date = today_date

            day = (intake_date - start_date).days + 1

            if day < 1:
                print(
                    f"⚠️  intervention_merger: intake '{intake.get('label')}' — "
                    f"computed day_of_protocol={day} invalid "
                    f"(event_date={intake_event_raw!r}, start_date={start_date_raw!r}), skipping"
                )
            else:
                intake["day_of_protocol"] = day
                suffix = f" (event_date: {intake_event_raw})" if intake_event_raw else " (today)"
                print(
                    f"⚙️  intervention_merger: intake '{intake.get('label')}' → "
                    f"is_intervention_dose=True, day_of_protocol={day}{suffix}"
                )

    # Remove internal _substance_label field before returning
    for e in interventions:
        e.pop("_substance_label", None)

    return normal + interventions