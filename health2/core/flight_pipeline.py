"""
flight_pipeline.py — Flight/travel entity extraction for the Health Voice System.

Sits alongside intervention_pipeline.py. Called from pipeline.py after the
normal pipeline completes, before validate.

Design:
  - Receives only flight-candidate mentions (candidate_type == "flight")
  - Gates future flights out
  - Calls LLM once with the full transcript to extract structured flight data
  - Returns list of FlightEntity dicts

FlightEntity fields:
  type              : "flight"
  origin            : str          — departure city/airport (IATA if stated, else city)
  destination       : str          — arrival city/airport
  flight_date       : str | None   — user's own words ("yesterday", "last Tuesday", "2025-11-10")
  departure_time    : str | None   — local time at origin ("9 AM", "morning")
  arrival_time      : str | None   — local time at destination
  duration_minutes  : int | None   — only if explicitly stated or calculable from times
  cabin_class       : str | None   — "economy", "business", "first" — only if stated
  airline           : str | None   — carrier name if mentioned
  notes             : str | None   — stopovers, delays, or other details

DB table (suggested):
  CREATE TABLE flights (
      id               SERIAL PRIMARY KEY,
      transcript_id    INTEGER REFERENCES transcripts(id),
      origin           TEXT NOT NULL,
      destination      TEXT NOT NULL,
      flight_date      TEXT,
      departure_time   TEXT,
      arrival_time     TEXT,
      duration_minutes INTEGER,
      cabin_class      TEXT,
      airline          TEXT,
      notes            TEXT,
      created_at       TIMESTAMPTZ DEFAULT now()
  );
"""

import json
import os
import re
from datetime import date
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────
# GATE
# ─────────────────────────────────────────

_DROP_TEMPORAL = {"future_plan", "consultation_relay"}


def _gate(mention: dict) -> tuple[bool, str]:
    """Returns (should_drop, reason)."""
    if not mention.get("raw_mention", "").strip():
        return True, "empty raw_mention"
    te = mention.get("temporal_evidence", "unclear")
    if te in _DROP_TEMPORAL:
        return True, f"temporal_evidence={te}"
    return False, ""


# ─────────────────────────────────────────
# LLM ENRICHMENT
# ─────────────────────────────────────────

_ENRICHER_SYSTEM = """You are a travel data extractor for a health voice note system.

The speaker has mentioned one or more flights they have taken. Extract structured flight data.

RULES:
1. Only extract flights that clearly happened (past tense, confirmed events).
2. Do NOT infer or guess origin/destination — only extract what is explicitly stated.
3. duration_minutes: fill ONLY if Gabriel explicitly states flight duration ("4-hour flight" -> 240),
   OR if departure and arrival times are both stated in the same timezone context.
   Otherwise null.
4. cabin_class: only fill if explicitly mentioned. Values: "economy", "business", "first".
5. flight_date: use Gabriel's own words exactly ("yesterday", "last Tuesday", "2025-11-10").
   Do not convert to ISO date.
6. If multiple flights are mentioned (e.g. outbound + return), return one object per flight.
7. IATA codes (LHR, JFK) preferred for origin/destination if stated; else use city name.

Respond ONLY with a JSON array. No preamble, no markdown fences.
Each object must have these keys (null if unknown):
{
  "origin": string,
  "destination": string,
  "flight_date": string | null,
  "departure_time": string | null,
  "arrival_time": string | null,
  "duration_minutes": integer | null,
  "cabin_class": string | null,
  "airline": string | null,
  "notes": string | null
}

If no flight data can be extracted, respond with: []
"""


def _extract_signals(transcript: str) -> list[dict]:
    """Call LLM to extract structured flight data from transcript."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_ENRICHER_SYSTEM,
            messages=[
                {"role": "user", "content": f"Transcript:\n\n{transcript}"}
            ]
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        signals = json.loads(raw)
        if not isinstance(signals, list):
            print("warning  flight_pipeline [enricher]: response was not a list")
            return []
        return signals

    except json.JSONDecodeError as e:
        print(f"warning  flight_pipeline [enricher]: JSON parse error - {e}")
        return []
    except Exception as e:
        print(f"warning  flight_pipeline [enricher]: LLM call failed - {e}")
        return []


# ─────────────────────────────────────────
# ENTITY CONSTRUCTION
# ─────────────────────────────────────────

def _build_entity(signal: dict) -> Optional[dict]:
    """
    Build a flight entity dict from an enricher signal.
    Returns None if origin or destination is missing.
    """
    origin = (signal.get("origin") or "").strip()
    destination = (signal.get("destination") or "").strip()

    if not origin or not destination:
        print(
            f"warning  flight_pipeline [build]: skipping - missing origin or destination "
            f"(origin={origin!r}, destination={destination!r})"
        )
        return None

    entity = {
        "type": "flight",
        "label": "flight",
        "origin": origin,
        "destination": destination,
        "flight_date": signal.get("flight_date") or None,
        "departure_time": signal.get("departure_time") or None,
        "arrival_time": signal.get("arrival_time") or None,
        "duration_minutes": None,
        "cabin_class": signal.get("cabin_class") or None,
        "airline": signal.get("airline") or None,
        "notes": signal.get("notes") or None,
    }

    raw_duration = signal.get("duration_minutes")
    if raw_duration is not None:
        try:
            entity["duration_minutes"] = int(raw_duration)
        except (TypeError, ValueError):
            print(f"warning  flight_pipeline [build]: invalid duration_minutes={raw_duration!r}, ignoring")

    return entity


# ─────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────

def _deduplicate(entities: list[dict]) -> list[dict]:
    result = []
    for entity in entities:
        duplicate = any(
            e.get("origin", "").lower() == entity.get("origin", "").lower()
            and e.get("destination", "").lower() == entity.get("destination", "").lower()
            and e.get("flight_date") == entity.get("flight_date")
            for e in result
        )
        if duplicate:
            print(
                f"flight_pipeline [dedup]: skipping duplicate "
                f"{entity.get('origin')} -> {entity.get('destination')} on {entity.get('flight_date')}"
            )
        else:
            result.append(entity)
    return result


# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────

def run_flight_pipeline(
    flight_mentions: list[dict],
    transcript: str,
    note_date: Optional[date] = None,
) -> list[dict]:
    """
    Extract flight entities from flight-candidate mentions and transcript.

    Args:
        flight_mentions: Mentions with candidate_type == "flight" (pre-filtered by pipeline.py)
        transcript:      Normalized transcript text
        note_date:       Date of the voice note (reserved for future date resolution)

    Returns:
        List of flight entity dicts. Empty list if no flights found.
    """
    if not flight_mentions:
        return []

    print(f"flight_pipeline: {len(flight_mentions)} flight mention(s) received")

    # Gate - drop future plans
    kept = []
    for mention in flight_mentions:
        should_drop, reason = _gate(mention)
        if should_drop:
            print(f"flight_pipeline [gate]: DROP '{mention.get('raw_mention', '')}' - {reason}")
        else:
            kept.append(mention)

    if not kept:
        return []

    # LLM enrichment - pass full transcript, not just mention snippets
    signals = _extract_signals(transcript)
    if not signals:
        print("flight_pipeline: LLM returned no signals")
        return []

    print(f"flight_pipeline: {len(signals)} signal(s) from LLM")

    # Build entities
    entities = [e for e in (_build_entity(s) for s in signals) if e is not None]

    # Deduplicate
    entities = _deduplicate(entities)

    print(f"flight_pipeline: {len(entities)} flight entity(ies) produced")
    return entities