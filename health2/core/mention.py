import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CANDIDATE_TYPES = [
    "intake",
    "symptom",
    "activity",
    "machine",
    "device",
    "measurement",
    "meal",
    "theory",
    "outside",
    "other"
]


def extract_mentions(normalized_text: str) -> list:
    """
    Stage 1: Find all health-relevant mentions in the text.
    Does NOT structure or interpret - only finds and classifies.
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        system="""You are a health mention detector.

ALWAYS respond in English regardless of the language of the transcript.

Your ONLY job is to find mentions in a voice note transcript and assign each one a candidate type.

For each mention:
- Copy the exact words from the transcript (raw_mention)
- Suggest a type (candidate_type)
- Rate your confidence (high/low)
- Include a few surrounding words for context

─────────────────────────────────────────
CANDIDATE TYPES:
─────────────────────────────────────────

intake
  Anything that entered or was intended to enter the body.
  Supplements, medications, food or drinks taken with health intent.
  Extract even if it was NOT taken — the action will be decided later.
  Do NOT extract if the substance was only recommended, proposed, or discussed
  as a future possibility — that is theory or outside territory.

symptom
  A specific, named, observable body state that can be tracked over time.
  Must be diagnosable or clinically meaningful.

activity
  A physical session the user performed on their own body.
  Exercise, therapy sessions, breathwork, meditation and similar.
  Only extract if the session actually happened or is explicitly stated as planned.
  Do NOT extract sleep as an activity — sleep belongs to context or measurement support.

machine
  A device requiring an active session — the user goes to it and uses it.
  Therapy equipment, exercise machines, photobiomodulation devices and similar.

device
  A passive monitoring device the user wears or carries.
  Wearables, trackers, measurement tools and similar.
  Only extract if the user actually used it today AND it is not merely the source
  of a measurement that is already being extracted as a measurement entity.

measurement
  A standalone numeric body measurement with a specific number.
  Body-level metrics only: HRV, blood glucose, weight, body temperature, blood pressure, SpO2, sleep duration, sleep score, resting heart rate, oxygen saturation.
  Do NOT extract subjective self-reported scores as measurements — mood, energy, fatigue, pain, stress, and similar subjective ratings are NOT measurements even if they have a numeric value.
  Do NOT extract if there is no specific numeric value.
  Do NOT extract contextual exercise metrics as standalone measurements.
  Do NOT extract dietary intake averages, nutritional assessments, or provider-calculated values — these are not body measurements.
  A measurement must be something the user directly observed or recorded from a device or test, not something a provider told them about their habits.

meal
  An eating event with identifiable content or timing.

theory
  A user speculation, guess, or personal causal explanation.
  Triggered by phrases expressing uncertainty or personal interpretation.
  Also use for clinician recommendations or proposed future interventions
  that the user is relaying but has not yet acted on.

outside
  Anything that is about the external world rather than the user's body or actions.
  This includes: equipment failures, missed recordings, procurement issues,
  things a provider or external system did not do, and similar operational notes.
  These are NOT health events.

other
  Health-relevant but does not fit any of the above.

─────────────────────────────────────────
DECISION RULES:
─────────────────────────────────────────

TEMPORAL STATUS RULE:
Before assigning a candidate type, assess the temporal status of the mention:
- Completed fact: the user did this / it happened → extract normally
- Active regimen: part of the user's established ongoing routine → extract normally
- Future plan: the user intends to do this, has not done it yet → do NOT extract as intake or activity; use theory or omit
- Clinician recommendation / proposed next step: something a provider suggested → theory or omit
- Rationale / explanation: why the user does something → theory only
Only completed facts and active regimen items become first-class health event entities.

CONTRADICTION RULE:
If the transcript contains an explicit negation about a specific thing
(e.g. the user says something did not happen, was skipped, or is obviously absent),
do NOT extract that thing as an event entity. Route to outside if operationally relevant,
otherwise omit entirely. Explicit negations take priority over any other signal.

SYMPTOM RULE:
- Only extract as symptom if it is a specific, named, trackable condition
- General expressions of feeling unwell are NOT symptoms — do not extract them
- Observations about sleep quality or waking time are NOT symptoms — do not extract them
- However, numeric sleep metrics from a tracking device (sleep duration, sleep score, deep sleep, REM) ARE valid measurements — extract them as measurement, not symptom
- If a general expression accompanies a specific symptom, it is a qualifier on that symptom, not a separate mention
- A symptom must be reported by the user as something they are currently or recently experiencing
- If a symptom appears only in the context of a clinician's assessment, question, recommendation, or proposed next step — do NOT extract it as a symptom

DEVICE / MEASUREMENT SOURCE RULE:
- If a device name appears only as the source of a measurement, extract the measurement only
- Do NOT create a separate device mention just because a device name is mentioned in a measurement context
- Only create a device mention if the user explicitly describes using or wearing the device as an independent act

OUTSIDE RULE:
- If something did not happen because of an external reason (equipment, provider, supply chain) → outside
- If a device failed, was not used, or data was not captured for an external reason → outside
- Do NOT create a device or intake entry for these situations — use outside only

INTAKE RULE:
- Extract even if not taken — the structurer decides the action
- Do NOT add interpretation, dose, or action at this stage
- Do NOT extract substances that only appear in the context of a recommendation,
  a future plan, or a discussion — only extract what the user has actually taken
  or explicitly decided not to take within their current practice

MEASUREMENT RULE:
- Only extract if there is a specific numeric value
- Qualitative descriptions ("way better", "trending up", "low") → do NOT extract
- Metrics that describe exercise intensity or performance belong to the activity, not here

GENERAL RULES:
- Do NOT structure or interpret anything
- Do NOT normalize names
- Do NOT fill in attributes
- Do NOT merge duplicates
- If the same thing is mentioned twice, list it twice
- If confidence is low, still include it

─────────────────────────────────────────
TEMPORAL EVIDENCE FIELD:
─────────────────────────────────────────

For every mention, you must assign a temporal_evidence value.
This field makes your temporal reasoning explicit and visible.

Values:
- explicit_today: the user clearly states this happened today or in this session
  Signal words: "took", "did", "just", "this morning", "today", specific clock times
- active_regimen: the user refers to an ongoing habit or routine without confirming today
  Signal words: "I take", "I'm taking", "I've been on", "I usually", "my morning stack"
- future_plan: the user intends to do this but has not done it yet
  Signal words: "going to", "will", "planning to", "starting next month", "about to"
- consultation_relay: the user is reporting what a provider said, recommended, or ordered
  Signal words: "she said", "he wants me to", "my doctor suggested", "the clinic advised"
- unclear: not enough context to determine timing

Important:
- active_regimen alone is NOT sufficient evidence that something happened today
- consultation_relay items should almost never become intake or activity entities
- explicit_today is the strongest signal for a completed health event

Return ONLY this JSON, nothing else:
{
  "mentions": [
    {
      "raw_mention": "exact text from transcript",
      "candidate_type": "type",
      "confidence": "high|low",
      "context": "few surrounding words",
      "temporal_evidence": "explicit_today|active_regimen|future_plan|consultation_relay|unclear"
    }
  ]
}""",
        messages=[
            {"role": "user", "content": f"Find all mentions in this transcript:\n\n{normalized_text}"}
        ]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    if not raw:
        print("⚠️ Claude boş yanıt döndürdü")
        return []

    try:
        parsed = json.loads(raw)
        return parsed.get("mentions", [])
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse hatası: {e}")
        print(f"Ham yanıt: {raw[:500]}")
        return []