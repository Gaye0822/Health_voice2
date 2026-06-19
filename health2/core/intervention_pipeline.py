"""
intervention_pipeline.py — Self-contained intervention processing layer

Runs AFTER mention_filter, BEFORE structure.py (in parallel or before).
Receives only intervention mentions + transcript.
Produces only intervention entities — no intake, no other types.

Responsibilities:
  1. Gate: decide whether an intervention mention should be dropped or processed
  2. LLM enrichment: extract protocol signals from transcript
  3. Entity construction: build InterventionEntity dict with all available fields

Explicitly NOT responsible for:
  - is_intervention_dose on intake entities (handled by intervention_merger.py)
  - day_of_protocol on intake entities (handled by intervention_merger.py)
  - Any non-intervention entity type

Pipeline caller (app.py) is responsible for:
  - Splitting intervention mentions from the rest before calling this
  - Calling intervention_merger.py after both pipelines complete

What we take from each mention:
  - raw_mention: the substance label (only this)
  - temporal_evidence: gate routing signal (only this)
  Everything else (reasoning, context, confidence) is mention.py's internal state.
  The intervention pipeline makes its own decisions from the transcript directly.
"""

import json
import os
import re
import anthropic
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────
# GATE — drop/keep decision
# ─────────────────────────────────────────

# temporal_evidence values that always drop an intervention mention
_INTERVENTION_DROP_TE = {"future_plan", "consultation_relay"}


def _gate(mention: dict) -> tuple[bool, str]:
    """
    Decide whether an intervention mention should be dropped.

    Only looks at temporal_evidence — nothing else.
    mention.py already made the semantic decision; we trust it.
    The only information we use from the mention is raw_mention (the label)
    and temporal_evidence (pipeline routing signal).

    Returns:
        (should_drop: bool, reason: str)
    """
    raw = mention.get("raw_mention", "").strip()
    if not raw:
        return True, "empty raw_mention"

    te = mention.get("temporal_evidence", "unclear")

    if te in _INTERVENTION_DROP_TE:
        return True, f"temporal_evidence={te}"

    return False, ""


# ─────────────────────────────────────────
# LLM SIGNAL EXTRACTION
# ─────────────────────────────────────────

_ENRICHER_SYSTEM = """You are a protocol signal extractor.

You receive:
1. A list of intervention labels to find signals for
2. The original transcript text

Your job is to find protocol signals for each intervention substance in the transcript.

For each substance, extract:
- dose_number: which day or dose number in the protocol is this? (integer or null)
  Look for: "day five", "day 5", "fifth day", "second dose", "third pill", "dose 2",
  "it's day 3", "I'm on day 4", "today is day X", "that was day X"
- event_date_raw: when did the most recent dose happen? (string or null)
  Use the user's exact words: "last night", "yesterday", "this morning", "today"
  This is the date of the most recent dose, not the start of the protocol.
- duration_days: total protocol length in days (integer or null)
  Look for "10 days", "a 30-day course", "two weeks", "five days", "for X days"
  Only extract if convertible to a single exact integer.
  "a few weeks", "over a month", "not sure" → null
- is_completed: did the user explicitly say the course is finished? (boolean)
  true only if: "I finished", "I'm done", "that's done", "last dose", "completed the course",
               "took the last one", "finished the course"
  false otherwise — default false, NEVER infer from duration alone
- start_date_raw: when did the protocol start? (string or null)
  Use the user's exact words: "two days ago", "last Monday", "three weeks ago"
  Only fill if user explicitly stated the start. Do NOT calculate from dose_number.
- is_historical: was this protocol described as already finished in the past? (boolean)
  true if: user says "I did a course last month", "back in March I took...",
           "a few weeks ago I finished...", "I had a 10-day course last year"
  false otherwise — default false
- - daily_doses: how many times per day is the substance taken? (integer or null)
  Look for: "twice a day", "two times daily", "every 12 hours", "three times a day", "once daily"
  Convert to integer: once=1, twice=2, three times=3, etc.
  Only fill if explicitly stated. Do NOT infer from dose_number or duration.
- notes: protocol deviations or compliance issues only. (string or null)
  Look for: missed doses, skipped days, dose changes, supply issues.
  Write as a SHORT factual note — maximum one sentence, no raw transcript quotes.
  Examples of good notes: "missed one dose", "skipped two days", "halved dose on day 3"
  If nothing notable → null

Return ONLY a JSON array. One object per substance. Example:
[
  {
    "substance_label": "doxycycline",
    "dose_number": 5,
    "event_date_raw": "this morning",
    "start_date_raw": null,
    "duration_days": 10,
    "daily_doses": 2,
    "is_completed": false,
    "is_historical": false,
    "notes": "missed one dose on day 3"
  }
]

Rules:
- Only extract what is explicitly stated in the transcript. Never infer or guess.
- If a field is not stated → null (or false for booleans)
- If no protocol signals exist at all → return empty array []
- Return ONLY the JSON array, no explanation, no markdown
"""


def _extract_signals(intervention_labels: list[str], transcript: str) -> list[dict]:
    """
    Call LLM to extract protocol signals for the given intervention labels.
    Returns list of signal dicts. Empty list on failure.
    """
    if not intervention_labels:
        return []

    labels_text = "\n".join(f"- {label}" for label in intervention_labels)
    user_message = (
        f"INTERVENTION SUBSTANCES TO FIND SIGNALS FOR:\n{labels_text}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"Extract protocol signals for each substance above."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            temperature=0,
            system=_ENRICHER_SYSTEM,
            messages=[{"role": "user", "content": user_message}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        signals = json.loads(raw)
        if isinstance(signals, list):
            print(f"⚙️  intervention_pipeline [enricher]: {len(signals)} signal(s)")
            for s in signals:
                print(
                    f"   → {s.get('substance_label')}: "
                    f"dose={s.get('dose_number')}, "
                    f"event_date={s.get('event_date_raw')}, "
                    f"start_date={s.get('start_date_raw')}, "
                    f"duration={s.get('duration_days')}, "
                    f"completed={s.get('is_completed')}, "
                    f"historical={s.get('is_historical')}"
                )
            return signals
    except Exception as e:
        print(f"⚠️  intervention_pipeline [enricher] error: {e}")

    return []


# ─────────────────────────────────────────
# DATE RESOLUTION
# ─────────────────────────────────────────

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


def _resolve_date(raw: str, today: date) -> "date | None":
    """
    Convert a relative date string to a date object.
    Returns None if unresolvable (caller keeps raw string).
    """
    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip().lower()

    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    offset = _RELATIVE_DATE_OFFSETS.get(s)
    if offset is not None:
        return today - timedelta(days=offset)

    # "last <weekday>" or bare "<weekday>"
    m = re.match(r"^(?:last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", s)
    if m:
        target_wd = _WEEKDAYS[m.group(1)]
        days_back = (today.weekday() - target_wd) % 7
        if days_back == 0:
            days_back = 7
        return today - timedelta(days=days_back)

    return None


# ─────────────────────────────────────────
# ENTITY BUILDER
# ─────────────────────────────────────────

def _labels_match(label_a: str, label_b: str) -> bool:
    """Token-based label matching — same logic as schema_enforcer."""
    STOP = {"course", "protocol", "treatment", "therapy", "program",
            "supplement", "dose", "medication", "drug", "pill", "tablet"}
    ta = {t for t in label_a.lower().split() if t not in STOP and len(t) > 2}
    tb = {t for t in label_b.lower().split() if t not in STOP and len(t) > 2}
    if ta & tb:
        return True
    if label_a.lower() in label_b.lower() or label_b.lower() in label_a.lower():
        return True
    return False


def _build_entity(mention: dict, signal: "dict | None", today: date) -> "dict | None":
    """
    Build a complete intervention entity dict from a mention and its enricher signal.

    Returns None if the entity cannot meet the minimum viability requirement:
      - duration_days must be a known exact integer
      - start_date must be derivable (from any combination of available signals)

    If either cannot be established → return None → caller drops this intervention.

    Valid combinations that produce a start_date:
      1. start_date_raw explicit
      2. event_date_raw + dose_number  → start = event_date - (dose - 1)
      3. event_date_raw only           → start = event_date (day 1)
      4. dose_number only              → start = today - (dose - 1)
    """
    raw = mention.get("raw_mention", "")
    label = raw  # used in log messages

    # ── Signal guard — no signal at all → drop ─────────────────────────
    if signal is None:
        print(f"⚙️  intervention_pipeline [drop]: '{label}' — no enricher signal found")
        return None

    # ── Duration guard — must be exact integer ─────────────────────────
    duration_days = signal.get("duration_days")
    if not isinstance(duration_days, int) or duration_days < 1:
        print(
            f"⚙️  intervention_pipeline [drop]: '{label}' — "
            f"duration_days missing or vague (got: {signal.get('duration_days')!r})"
        )
        return None

    # ── Compute start_date — must resolve to an exact date ─────────────
    start_date: "date | None" = None
    event_date_raw = signal.get("event_date_raw")
    dose_number = signal.get("dose_number")
    start_date_raw = signal.get("start_date_raw")

    # Priority 1: explicit start_date
    if start_date_raw:
        start_date = _resolve_date(start_date_raw, today)

    # Priority 2: is_completed path
    if start_date is None and signal.get("is_completed") and isinstance(duration_days, int) and duration_days >= 1:
        if isinstance(dose_number, int) and dose_number == duration_days and event_date_raw:
            dose_date = _resolve_date(event_date_raw, today)
            if dose_date is not None:
                start_date = dose_date - timedelta(days=duration_days - 1)
                print(
                    f"⚙️  intervention_pipeline: '{label}' is_completed, dose={dose_number}/{duration_days} "
                    f"→ event_date is last dose, start={start_date.isoformat()}"
                )
        if start_date is None:
            assumed_end = today - timedelta(days=1)
            start_date = assumed_end - timedelta(days=duration_days - 1)
            print(
                f"⚙️  intervention_pipeline: '{label}' is_completed, dose_number unclear "
                f"→ assuming end=yesterday, start={start_date.isoformat()}"
            )

    # Priority 3: event_date + dose_number
    if start_date is None and event_date_raw:
        dose_date = _resolve_date(event_date_raw, today)
        if dose_date is not None:
            if isinstance(dose_number, int) and dose_number >= 1:
                start_date = dose_date - timedelta(days=dose_number - 1)
            else:
                start_date = dose_date

    # Priority 4: dose_number only
    if start_date is None and isinstance(dose_number, int) and dose_number >= 1:
        start_date = today - timedelta(days=dose_number - 1)
        print(
            f"⚙️  intervention_pipeline: dose_number={dose_number}, no event_date → "
            f"assuming today is day {dose_number}, start_date={start_date.isoformat()}"
        )
    # ── Start date guard — must be resolvable ──────────────────────────
    if start_date is None:
        print(
            f"⚙️  intervention_pipeline [drop]: '{label}' — "
            f"start_date not derivable from available signals "
            f"(start_date_raw={start_date_raw!r}, event_date_raw={event_date_raw!r}, dose_number={dose_number!r})"
        )
        return None

    # ── Day of protocol validation ─────────────────────────────────────
    day = (today - start_date).days + 1
    if day < 1:
        print(
            f"⚙️  intervention_pipeline [drop]: '{label}' — "
            f"computed day_of_protocol={day} invalid (start_date in the future?)"
        )
        return None

    # ── All guards passed — build entity ──────────────────────────────
    end_date = start_date + timedelta(days=duration_days - 1)

    status = "active"
    if signal.get("is_completed") or signal.get("is_historical"):
        status = "completed"
    elif end_date < today:
        # end_date is in the past → auto-complete
        status = "completed"
        print(f"⚙️  intervention_pipeline: '{label}' → end_date {end_date.isoformat()} is past, status=completed")

    daily_doses = signal.get("daily_doses")
    if not isinstance(daily_doses, int) or daily_doses < 1:
        daily_doses = None

    notes_parts = []
    if daily_doses:
        dose_label = {1: "once", 2: "twice", 3: "three times"}.get(daily_doses, f"{daily_doses}x")
        notes_parts.append(f"{dose_label} daily")
    raw_notes = signal.get("notes")
    if raw_notes:
        notes_parts.append(raw_notes)
    notes = "; ".join(notes_parts) if notes_parts else None

    entity = {
        "type": "intervention",
        "label": raw,
        "status": status,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "duration_days": duration_days,
        "daily_doses": daily_doses,
        "day_of_protocol": day,
        "notes": notes,
        "_substance_label": raw.lower(),
    }

    print(
        f"⚙️  intervention_pipeline: '{label}' → "
        f"start={start_date.isoformat()}, end={end_date.isoformat()}, "
        f"day={day}/{duration_days}, status={status}"
    )
    return entity


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def run_intervention_pipeline(
    intervention_mentions: list,
    transcript: str,
    today_date: "date | None" = None,
) -> list:
    """
    Main entry point. Receives only intervention-type mentions.

    Returns:
        List of intervention entity dicts ready for merge layer.
        Empty list if all mentions were gated out or no mentions given.
    """
    if not intervention_mentions:
        return []

    if today_date is None:
        today_date = date.today()

    # ── Gate ──────────────────────────────────────────────────────────
    kept = []
    for mention in intervention_mentions:
        drop, reason = _gate(mention)
        if drop:
            print(
                f"⚙️  intervention_pipeline [gate]: DROP "
                f"'{mention.get('raw_mention', '')}' — {reason}"
            )
        else:
            kept.append(mention)

    if not kept:
        return []

    # ── LLM enrichment ────────────────────────────────────────────────
    # Only raw_mention is passed to the enricher — transcript is the primary source.
    # We do not forward mention reasoning, context, or confidence to the LLM.
    labels = [m.get("raw_mention", "") for m in kept if m.get("raw_mention")]
    signals = _extract_signals(labels, transcript)

    # Build substance → signal lookup
    signal_map: dict[str, dict] = {}
    for sig in signals:
        substance = sig.get("substance_label", "").lower()
        if substance:
            signal_map[substance] = sig

    # ── Gate: historical drop ──────────────────────────────────────────
    # After enrichment we know which are historical — drop those
    final_kept = []
    for mention in kept:
        raw = mention.get("raw_mention", "").lower()
        sig = next((s for k, s in signal_map.items() if _labels_match(raw, k)), None)
        if sig and sig.get("is_historical"):
            print(
                f"⚙️  intervention_pipeline [gate]: DROP "
                f"'{mention.get('raw_mention', '')}' — is_historical=True"
            )
            continue
        final_kept.append(mention)

    if not final_kept:
        return []

    # ── Deduplication — same substance may appear in multiple mentions ──
    # Keep only one mention per substance label (first occurrence).
    seen_labels: set = set()
    deduped = []
    for mention in final_kept:
        raw = mention.get("raw_mention", "").lower()
        already_seen = any(_labels_match(raw, seen) for seen in seen_labels)
        if already_seen:
            print(
                f"⚙️  intervention_pipeline [dedup]: skipping duplicate mention "
                f"'{mention.get('raw_mention', '')}'"
            )
        else:
            seen_labels.add(raw)
            deduped.append(mention)

   # ── Entity construction ────────────────────────────────────────────
    entities = []
    for mention in deduped:
        raw = mention.get("raw_mention", "").lower()
        sig = next((s for k, s in signal_map.items() if _labels_match(raw, k)), None)
        entity = _build_entity(mention, sig, today_date)
        if entity is None:
            continue
        entities.append(entity)

    # ── Entity-level deduplication — same substance may produce multiple entities
    # (e.g. two mentions with different temporal_evidence but same substance)
    # Keep the one with the most complete data: prefer completed > active, most fields filled.
    deduped_entities = []
    for entity in entities:
        e_label = entity.get("_substance_label", entity.get("label", "")).lower()
        duplicate = next(
            (e for e in deduped_entities
             if _labels_match(e_label, e.get("_substance_label", e.get("label", "")).lower())),
            None
        )
        if duplicate is None:
            deduped_entities.append(entity)
        else:
            # Keep completed over active; if both same status, keep the one with start_date
            existing_status = duplicate.get("status")
            new_status = entity.get("status")
            if existing_status != "completed" and new_status == "completed":
                deduped_entities.remove(duplicate)
                deduped_entities.append(entity)
                print(
                    f"⚙️  intervention_pipeline [entity-dedup]: replacing '{e_label}' "
                    f"({existing_status}) with completed entity"
                )
            elif duplicate.get("start_date") is None and entity.get("start_date") is not None:
                deduped_entities.remove(duplicate)
                deduped_entities.append(entity)
                print(
                    f"⚙️  intervention_pipeline [entity-dedup]: replacing '{e_label}' "
                    f"(no start_date) with entity that has start_date"
                )
            else:
                print(
                    f"⚙️  intervention_pipeline [entity-dedup]: dropping duplicate '{e_label}'"
                )

    return deduped_entities