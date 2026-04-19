"""
intervention_enricher.py — LLM-based intervention signal extractor

Runs after structure.py, before schema_enforcer.py.

Takes the existing intake entities and transcript, and extracts
protocol signals (dose_number, event_date_raw, duration_days) for
each intervention found in the entity list.

Returns enrichment signals as a list of dicts — schema_enforcer
uses these to fill intervention fields deterministically.

No changes to intake or any other entity type.
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


SYSTEM_PROMPT = """You are a protocol signal extractor.

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
- is_completed: did the user explicitly say the course is finished? (boolean)
  true if: "I finished", "I'm done", "that's done", "last dose", "completed the course",
           "took the last one", "finished the course"
  false otherwise — default to false, do NOT infer completion from duration alone

Return ONLY a JSON array. One object per substance. Example:
[
  {
    "substance_label": "doxycycline",
    "dose_number": 5,
    "event_date_raw": "this morning",
    "duration_days": 10,
    "is_intervention_dose": true,
    "is_completed": false
  }
]

Rules:
- Only extract what is explicitly stated in the transcript. Never infer or guess.
- If dose_number or event_date_raw is not stated → null
- is_intervention_dose is always true for any substance in this list
- is_completed defaults to false — only true if user explicitly says they finished
- If no protocol signals exist at all → return empty array []
- Return ONLY the JSON array, no explanation, no markdown
"""


def _labels_match(label_a: str, label_b: str) -> bool:
    """
    Token-based label matching for intervention/intake pairing.
    Matches if any non-trivial token from one label appears in the other.
    More robust than simple substring containment.

    Examples:
      "doxycycline" vs "doxycycline course"  → True  (token match)
      "doxy" vs "doxycycline"                → False (no token overlap — intentional)
      "FMT" vs "FMT protocol"               → True
      "amoxicillin" vs "vitamin C"           → False
    """
    STOP_TOKENS = {"course", "protocol", "treatment", "therapy", "program",
                   "supplement", "dose", "medication", "drug", "pill", "tablet"}

    tokens_a = {t for t in label_a.lower().split() if t not in STOP_TOKENS and len(t) > 2}
    tokens_b = {t for t in label_b.lower().split() if t not in STOP_TOKENS and len(t) > 2}

    # Direct token intersection
    if tokens_a & tokens_b:
        return True

    # Substring fallback: one full label contained in the other
    # (handles "doxycycline" in "doxycycline 100mg course")
    if label_a.lower() in label_b.lower() or label_b.lower() in label_a.lower():
        return True

    return False


def extract_intervention_signals(
    entities: list,
    transcript: str
) -> list:
    """
    Given the full entity list and transcript, extract protocol signals
    for any intervention entity — regardless of whether a matching intake exists.

    Returns a list of signal dicts:
    [
      {
        "substance_label": "doxycycline",
        "dose_number": 5,
        "event_date_raw": "this morning",
        "duration_days": 10,
        "is_completed": true
      },
      ...
    ]
    """
    # Find intervention labels — these are the only anchor we need
    intervention_labels = []
    for e in entities:
        if e.get("type") == "intervention":
            label = e.get("label", "")
            if label:
                intervention_labels.append(label)

    if not intervention_labels:
        return []

    # Build prompt — transcript is the primary source, not intake entities
    labels_text = "\n".join(f"- {label}" for label in intervention_labels)
    user_message = f"""INTERVENTION SUBSTANCES TO FIND SIGNALS FOR:
{labels_text}

TRANSCRIPT:
{transcript}

Extract protocol signals for each substance above."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            system=SYSTEM_PROMPT,
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
            print(f"⚙️  intervention_enricher: {len(signals)} signal(s) extracted")
            for s in signals:
                print(f"   → {s.get('substance_label')}: dose={s.get('dose_number')}, "
                      f"event_date={s.get('event_date_raw')}, duration={s.get('duration_days')}, "
                      f"completed={s.get('is_completed')}")
            return signals
    except Exception as e:
        print(f"⚠️  intervention_enricher error: {e}")

    return []