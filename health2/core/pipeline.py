"""
core/pipeline.py — Central pipeline runner.

Encapsulates all post-mention structuring logic so that app.py and watcher.py
both run the exact same pipeline without duplication.

Usage:
    from core.pipeline import run_pipeline

    result = run_pipeline(mentions, normalized_text, note_date=None)
    # result["entities"]             → final entity list
    # result["validation_changes"]   → list of validator change strings
    # result["enforcer_violations"]  → list of schema enforcer violation strings
"""

from core.structure import structure_mentions
from core.schema_enforcer import enforce_schema
from core.validate import validate_entities
from core.intervention_pipeline import run_intervention_pipeline
from core.intervention_merger import merge_intervention_entities
from core.flight_pipeline import run_flight_pipeline


# ── Constants ─────────────────────────────────────────────────────────────────

EVENT_DATE_TYPES = {"meal", "symptom", "intake", "activity", "machine"}

_STOP_WORDS = {"course", "protocol", "supplement", "dose", "medication", "meal"}

_DATE_EXPRESSIONS = {
    "last night", "yesterday", "yesterday morning", "yesterday afternoon",
    "yesterday evening", "yesterday night", "the night before last",
    "two days ago", "three days ago", "last week", "last monday",
    "last tuesday", "last wednesday", "last thursday", "last friday",
    "last saturday", "last sunday",
}

_TODAY_TIME_SIGNALS = {
    "this morning", "this afternoon", "this evening", "tonight",
    "today", "just now", "right now",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ed_labels_match(raw: str, entity_label: str) -> bool:
    ta = {t for t in raw.split() if t not in _STOP_WORDS and len(t) > 2}
    tb = {t for t in entity_label.split() if t not in _STOP_WORDS and len(t) > 2}
    if ta & tb:
        return True
    return raw in entity_label or entity_label in raw


def _apply_event_date_hints(normal_entities: list, normal_mentions: list):
    """Apply event_date and dose_timing hints from mentions onto entities."""
    _hint_pool = []
    for m in normal_mentions:
        temporal = m.get("temporal_evidence", "")
        edl = m.get("event_date_label") if temporal == "explicit_past" else None
        dt_hint = m.get("_dose_timing_hint")
        if edl or dt_hint:
            _hint_pool.append({
                "raw_key": m.get("raw_mention", "").lower(),
                "candidate_type": m.get("candidate_type"),
                "event_date_hint": edl,
                "dose_timing_hint": dt_hint,
                "used": False,
            })

    for entity in normal_entities:
        etype = entity.get("type")
        if etype not in EVENT_DATE_TYPES:
            continue
        label = entity.get("label", "").lower()
        for hint in _hint_pool:
            if hint["used"]:
                continue
            if hint["candidate_type"] != etype:
                continue
            if not _ed_labels_match(hint["raw_key"], label):
                continue
            if hint["event_date_hint"] and not entity.get("event_date"):
                entity["event_date"] = hint["event_date_hint"]
                print(f"⚙️  pipeline event_date ({etype}): '{entity.get('label')}' → '{hint['event_date_hint']}'")
            if hint["dose_timing_hint"] and not entity.get("dose_timing"):
                entity["dose_timing"] = hint["dose_timing_hint"]
                print(f"⚙️  pipeline dose_timing ({etype}): '{entity.get('label')}' → '{hint['dose_timing_hint']}'")
            hint["used"] = True
            break


def _apply_time_to_event_date(normal_entities: list):
    """Move historical time expressions to event_date; clear today signals."""
    for entity in normal_entities:
        if entity.get("type") not in EVENT_DATE_TYPES:
            continue
        time_val = (entity.get("time") or "").strip().lower()
        if time_val in _DATE_EXPRESSIONS:
            if not entity.get("event_date"):
                entity["event_date"] = time_val
                print(f"⚙️  pipeline time→event_date: '{entity.get('label')}' '{time_val}' → event_date")
            entity["time"] = None
        elif time_val in _TODAY_TIME_SIGNALS:
            if entity.get("event_date"):
                print(f"⚙️  pipeline today-time override: '{entity.get('label')}' time='{time_val}' → clearing event_date")
                entity["event_date"] = None
            entity["time"] = None


def _build_intervention_shadows(intervention_mentions: list, normal_mentions: list) -> list:
    """
    For each intervention mention, create a shadow intake mention so that
    intervention_merger can find and annotate the corresponding intake entity.
    Skips if an intake mention for the same substance already exists.
    Returns updated normal_mentions list (with shadows appended).
    """
    result = list(normal_mentions)
    for m in intervention_mentions:
        int_raw = m.get("raw_mention", "").lower()
        already_has_intake = any(
            n.get("candidate_type") == "intake" and
            _ed_labels_match(n.get("raw_mention", "").lower(), int_raw)
            for n in result
        )
        if already_has_intake:
            print(
                f"⚙️  pipeline [intervention shadow]: '{m.get('raw_mention', '')}' "
                f"→ intake mention already exists, skipping shadow"
            )
            continue
        shadow = dict(m)
        shadow["candidate_type"] = "intake"
        shadow["_intervention_shadow"] = True
        result.append(shadow)
        print(
            f"⚙️  pipeline [intervention shadow]: '{m.get('raw_mention', '')}' "
            f"→ intake copy added to normal pipeline"
        )
    return result


# ── Main pipeline function ────────────────────────────────────────────────────

def run_pipeline(mentions: list, normalized_text: str, note_date=None) -> dict:
    """
    Run the full structuring pipeline on a list of mentions.

    Args:
        mentions:        Output of extract_mentions() — may include intervention mentions.
        normalized_text: Normalized transcript text.
        note_date:       Date of the voice note (date object or None).

    Returns:
        {
            "entities":            list of structured entity dicts,
            "validation_changes":  list of validator change description strings,
            "enforcer_violations": list of schema enforcer violation strings,
        }
    """

    # ── Split mentions ────────────────────────────────────────────────────────
    intervention_mentions = [
        m for m in mentions if m.get("candidate_type") == "intervention"
    ]
    flight_mentions = [
        m for m in mentions if m.get("candidate_type") == "flight"
    ]
    normal_mentions = [
        m for m in mentions if m.get("candidate_type") not in {"intervention", "flight"}
    ]

    # ── Normal pipeline ───────────────────────────────────────────────────────
    normal_entities = structure_mentions(normal_mentions, normalized_text)

    normal_entities, enforcer_violations = enforce_schema(normal_entities, note_date=note_date)
    if enforcer_violations:
        print(f"⚙️  schema_enforcer: {len(enforcer_violations)} violation(s)")

    _apply_event_date_hints(normal_entities, normal_mentions)
    _apply_time_to_event_date(normal_entities)

    # ── Intervention shadow mentions ──────────────────────────────────────────
    normal_mentions = _build_intervention_shadows(intervention_mentions, normal_mentions)

    # ── Intervention pipeline ─────────────────────────────────────────────────
    intervention_entities = run_intervention_pipeline(
        intervention_mentions, normalized_text,
        today_date=note_date
    )

    # ── Merge ─────────────────────────────────────────────────────────────────
    entities = merge_intervention_entities(
        intervention_entities, normal_entities,
        today_date=note_date
    )

    # ── Flight pipeline ───────────────────────────────────────────────────────
    flight_entities = run_flight_pipeline(flight_mentions, normalized_text, note_date=note_date)
    if flight_entities:
        entities = entities + flight_entities

    # ── Validate ──────────────────────────────────────────────────────────────
    entities, validation_changes = validate_entities(entities, normalized_text)

    # ── entity_date final pass ────────────────────────────────────────────────
    # Calculated here (after all event_date mutations) so it reflects final state.
    # schema_enforcer calculates an early estimate, but event_date may change in
    # _apply_event_date_hints / _apply_time_to_event_date above. This pass wins.
    _ENTITY_DATE_TYPES = {"intake", "symptom", "activity", "machine", "meal"}
    if note_date is not None:
        from core.schema_enforcer import _resolve_start_date_to_iso
        for e in entities:
            if e.get("type") not in _ENTITY_DATE_TYPES:
                continue
            event_date_raw = e.get("event_date")
            if not event_date_raw:
                e["entity_date"] = str(note_date)
            else:
                resolved = _resolve_start_date_to_iso(str(event_date_raw), note_date)
                e["entity_date"] = str(resolved) if resolved else str(note_date)

    return {
        "entities": entities,
        "validation_changes": validation_changes,
        "enforcer_violations": enforcer_violations,
    }