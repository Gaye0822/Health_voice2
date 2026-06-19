"""
envelope.py — Envelope builder for HDS receiving-side integration.

Converts pipeline output into the agreed envelope format:
    {
        "envelope": { ... metadata ... },
        "layers":   { event_layer, nutrition_layer, theory_layer, context_layer, outside_bucket },
        "meta":     { warnings, validation_changes, enforcer_violations }
    }

Usage:
    from core.envelope import emit_envelope

    result = run_pipeline(mentions, normalized_text, note_date=note_date)
    envelope = emit_envelope(
        entities=result["entities"],
        validation_changes=result["validation_changes"],
        enforcer_violations=result["enforcer_violations"],
        normalized_transcript=normalized_text,
        note_date=note_date,
        source_note_id=source_note_id,   # recording_id from voice package, or generated UUID
        audio_file=None,                 # S3 URI when available, None otherwise
    )
"""

import uuid
from datetime import datetime, timezone


# ── Constants ─────────────────────────────────────────────────────────────────

PIPELINE_VERSION = "1.0.0"
EXTRACTION_MODEL = "claude-sonnet-4-20250514"
TRANSCRIPTION_MODEL = "whisper-1"

# Warning vocabulary — closed set, coordinate before adding new values
WARNING_VOCAB = {
    "label not found in KB registry",
    "inferred from transcript",
    "no body token match",
    "raw mention preserved",
    "low confidence mention",
}

# Entity type → layer mapping
LAYER_MAP = {
    "symptom":      "event_layer",
    "intake":       "event_layer",
    "activity":     "event_layer",
    "machine":      "event_layer",
    "device":       "event_layer",
    "measurement":  "event_layer",
    "outcome":      "event_layer",
    "test":         "event_layer",
    "meal":         "nutrition_layer",
    "intervention": "nutrition_layer",
    "theory":       "theory_layer",
    "context":      "context_layer",
    "flight":       "travel_layer",
    "outside":      "outside_bucket",
}

LAYER_KEYS = [
    "event_layer",
    "nutrition_layer",
    "theory_layer",
    "context_layer",
    "travel_layer",
    "outside_bucket",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_warnings(entity: dict) -> list:
    """
    Extract warning signals from an entity dict and return
    as a closed-vocabulary string list.
    """
    warnings = []

    if entity.get("_not_in_kb"):
        warnings.append("label not found in KB registry")

    if entity.get("_raw_mention"):
        warnings.append("inferred from transcript")

    if entity.get("_no_body_token_match"):
        warnings.append("no body token match")

    if entity.get("_raw_mention_preserved"):
        warnings.append("raw mention preserved")

    if entity.get("_low_confidence"):
        warnings.append("low confidence mention")

    return warnings


def _clean_entity(entity: dict) -> dict:
    """
    Remove internal pipeline fields (prefixed with _) from entity
    before including in envelope. Warnings are extracted separately.
    """
    internal_keys = {
        "_not_in_kb",
        "_raw_mention",
        "_no_body_token_match",
        "_raw_mention_preserved",
        "_low_confidence",
        "_recurring",
        "_intervention_shadow",
        "_dose_timing_hint",
    }
    return {k: v for k, v in entity.items() if k not in internal_keys}


def _build_record(entity: dict) -> dict:
    """
    Build a single envelope record from a pipeline entity.
    Adds:
        - warnings: closed-vocabulary list
        - kb_matched: bool
        - recurring: block if present (from _recurring field)
    Removes internal pipeline fields.
    """
    warnings = _build_warnings(entity)
    kb_matched = not entity.get("_not_in_kb", False)

    record = _clean_entity(entity)
    record["kb_matched"] = kb_matched
    record["warnings"] = warnings

    # Preserve recurring block if present
    if "_recurring" in entity:
        recurring = entity["_recurring"]
        record["recurring"] = {
            "frequency_7d": recurring.get("frequency_7d"),
            "first_seen": recurring.get("first_seen"),
            "last_seen": recurring.get("last_seen"),
            "dates": recurring.get("dates", []),
        }

    return record


def _group_by_layer(entities: list) -> dict:
    """
    Group entity records into layers according to LAYER_MAP.
    Unknown types go into a layer_unknown list.
    """
    layers = {key: [] for key in LAYER_KEYS}
    unknown = []

    for entity in entities:
        entity_type = entity.get("type")
        layer = LAYER_MAP.get(entity_type)
        record = _build_record(entity)

        if layer:
            layers[layer].append(record)
        else:
            unknown.append(record)

    if unknown:
        layers["layer_unknown"] = unknown

    return layers


# ── Main function ─────────────────────────────────────────────────────────────

def emit_envelope(
    entities: list,
    validation_changes: list,
    enforcer_violations: list,
    normalized_transcript: str,
    note_date=None,
    source_note_id: str = None,
    audio_file: str = None,
) -> dict:
    """
    Build and return the HDS envelope dict.

    Args:
        entities:              Final entity list from run_pipeline()
        validation_changes:    Validator change descriptions from run_pipeline()
        enforcer_violations:   Schema enforcer violations from run_pipeline()
        normalized_transcript: Normalized transcript text from normalize.py
        note_date:             Date of the voice note (date object, str, or None)
        source_note_id:        recording_id from voice package, or None to auto-generate UUID
        audio_file:            S3 URI of audio file, or None

    Returns:
        envelope dict ready for JSON serialization
    """

    # ── source_note_id ────────────────────────────────────────────────────────
    if not source_note_id:
        source_note_id = str(uuid.uuid4())

    # ── note_date ─────────────────────────────────────────────────────────────
    if note_date is None:
        note_date_str = None
    elif hasattr(note_date, "isoformat"):
        note_date_str = note_date.isoformat()
    else:
        note_date_str = str(note_date)

    # ── created_at ────────────────────────────────────────────────────────────
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── layers ────────────────────────────────────────────────────────────────
    layers = _group_by_layer(entities)

    return {
        "envelope": {
            "source_note_id":        source_note_id,
            "note_date":             note_date_str,
            "created_at":            created_at,
            "pipeline_version":      PIPELINE_VERSION,
            "extraction_model":      EXTRACTION_MODEL,
            "transcription_model":   TRANSCRIPTION_MODEL,
            "audio_file":            audio_file,
            "normalized_transcript": normalized_transcript,
        },
        "layers": layers,
        "meta": {
            "warnings":             [],   # reserved — note-level signals, not per-record
            "validation_changes":   validation_changes,
            "enforcer_violations":  enforcer_violations,
        },
    }