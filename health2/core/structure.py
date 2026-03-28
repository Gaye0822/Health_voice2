import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ENTITY_SCHEMAS = {
    "intake": {
        "fields": ["label", "action", "dose", "unit", "time", "category", "notes"],
        "actions": ["took", "did_not_take"],
        "categories": ["supplement", "prescription", "OTC", "food"]
    },
    "symptom": {
        "fields": ["label", "onset_time", "severity", "qualifier", "duration"]
    },
    "activity": {
        "fields": ["label", "start_time", "duration", "status", "notes"],
        "statuses": ["completed", "planned", "incomplete"]
    },
    "machine": {
        "fields": ["label", "start_time", "duration", "status", "notes"],
        "statuses": ["used", "planned"]
    },
    "device": {
        "fields": ["label", "start_time", "status"],
        "statuses": ["used"]
    },
    "measurement": {
        "fields": ["metric", "value", "unit", "time", "source"],
        "note": "Numeric values only. If there is no specific number, do NOT create a measurement entity."
    },
    "meal": {
        "fields": ["label", "time", "eaten_out", "items"]
    },
    "intervention": {
        "fields": ["label", "start_date", "end_date", "status", "notes"],
        "statuses": ["active", "completed", "unknown"],
        "note": "Multi-step protocol with defined start/end dates. NOT single procedures or one-time treatments."
    },
    "outcome": {
        "fields": ["linked_to", "onset_time", "qualifier", "direction"],
        "directions": ["positive", "negative", "mixed", "unknown"],
        "note": "Observed change linked to an intervention or procedure. direction: positive = improvement."
    },
    "test": {
        "fields": ["label", "time", "status", "result", "notes"],
        "statuses": ["planned", "done"],
        "note": "A diagnostic test performed by or on the user. NOT a therapy device or machine."
    },
    "context": {
        "fields": ["raw_text", "related_to"],
        "note": "Health-relevant background info that is not a first-class event. Only when genuinely useful."
    },
    "theory": {
        "fields": ["raw_text", "linked_to_label", "linked_to_type"]
    },
    "outside": {
        "fields": ["raw_text"]
    }
}


def structure_mentions(mentions: list, normalized_text: str) -> list:
    mentions_json = json.dumps(mentions, indent=2)
    schemas_json = json.dumps(ENTITY_SCHEMAS, indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        system=f"""You are a health event structurer.

ALWAYS respond in English regardless of the language of the transcript.

You receive a list of mentions. Convert each into a structured record using the schemas below.

Entity schemas:
{schemas_json}

─────────────────────────────────────────
CANDIDATE TYPE RULE:
─────────────────────────────────────────

Each mention has a candidate_type field that was reviewed and confirmed by the user.
ALWAYS use this as the entity type unless it directly violates a schema rule.
Do NOT override a user-confirmed candidate_type based on your own judgment.

─────────────────────────────────────────
TEMPORAL EVIDENCE RULE:
─────────────────────────────────────────

Each mention has a temporal_evidence field. Use it as follows:

- explicit_today → structure normally as a health event
- active_regimen → structure normally as a health event ONLY if the transcript also contains
  clear same-session confirmation (e.g. a specific time, "today", "this morning", "just took")
  Provider approval or continuation advice ("doctor said to continue", "she said keep going") is NOT same-session confirmation
  If no same-session confirmation exists → omit the entity entirely
- future_plan → do NOT create an event entity; omit entirely
- consultation_relay → do NOT create an event entity; omit entirely
- unclear → apply the TEMPORAL STATUS RULE to decide; when in doubt, omit

This field overrides general inference. Trust it.

─────────────────────────────────────────
CORE PRINCIPLE:
─────────────────────────────────────────

A mention becomes a health event only if ALL THREE conditions are met:
1. It is about the user's body
2. It actually happened (or was specifically decided not to happen, for intake)
3. It can be tracked or compared over time

If any condition fails, route to outside or omit entirely.

─────────────────────────────────────────
TEMPORAL STATUS RULE:
─────────────────────────────────────────

Before structuring any entity, determine its temporal status from the transcript:

- Completed fact: the event occurred within the current period → structure normally
- Active regimen: part of the user's established ongoing routine → structure normally
- Future plan: the user intends to do this but has not done it yet → do NOT create an intake or activity record; omit or route to outside if operationally relevant
- Clinician recommendation / proposed next step: something advised but not yet adopted → do NOT create an event record; omit or route to theory
- Rationale / explanation: the user explains why they do something → theory only

The distinction between "not yet done" and "decided not to do" matters:
- Decided not to take (within current practice, same session) → intake with action: "did_not_take"
- Has not happened yet / future plan → do NOT create an intake record
- If the user uses future-pointing language ("going to", "will", "about to", "planning to") for an intake — regardless of where it appears in the transcript — that intake has not happened yet and must be omitted entirely, never set to "took" or "did_not_take"

CONSULTATION CONTEXT RULE:
If the transcript is primarily a recounting of a clinician or provider encounter
(e.g. the user is reporting what was said, recommended, ordered, or discussed in a call or appointment),
apply the following:
- Measurements reported from test results → extract normally (these are real data points)
- Symptoms discussed in the encounter → extract only if the user confirms they are currently present
- Any substance, supplement, or food that was recommended but not yet added to the user's practice → omit entirely, do NOT create a did_not_take record
- Any protocol change, cycling plan, or dosing adjustment described as starting in the future → omit entirely
- Any substance the user is currently taking and merely mentioned in passing → omit (not a today event)
The guiding question is: did the user's body actually receive or do something today, or is this note purely informational?
In a consultation note, the answer is almost always: only the measurements are real events.

─────────────────────────────────────────
CONTRADICTION RULE:
─────────────────────────────────────────

If the original transcript contains an explicit statement that something did NOT happen,
was skipped, is obviously absent, or is being negated, do NOT create an active event
record for that thing. This is a hard rule — explicit negations in the transcript
override all other signals. Route to outside if the information has operational value,
otherwise omit entirely.

─────────────────────────────────────────
FIELD RULES:
─────────────────────────────────────────

NULL RULE:
- Use null for any unknown or missing value
- NEVER use the string "unknown"

TIME FIELD RULE:
- time / start_time fields must contain a clock time, a date, or a named period (morning, evening)
- Relative phrases like "three weeks ago", "recently", "earlier", "last week" are NOT valid time values → use null
- If timing context is relevant, put it in notes

FIXED FIELDS RULE:
- Only use the fields defined in the schema for each type
- Do NOT add fields beyond what is defined
- If something does not fit a defined field, put it in notes where available, or omit it

─────────────────────────────────────────
TYPE RULES:
─────────────────────────────────────────

INTAKE:
- action is always "took" or "did_not_take" — nothing else
- Any reason for not taking goes in notes
- If something was not obtained due to an external reason → route to outside, not intake
- category: supplement / prescription / OTC / food

SYMPTOM:
- label must be a specific, named, trackable condition
- General expressions of feeling unwell are NOT symptoms — omit them
- Observations about sleep quality or waking time are NOT symptoms — omit them
- If a general expression accompanies a specific symptom → add it to qualifier only
- When multiple symptoms describe the same evolving condition → merge into one record

QUALIFIER RULE:
- qualifier must describe the symptom itself — what it feels like or how it presents
- User interpretations, frequency assessments, or personal context are NOT qualifiers — omit them

ACTIVITY:
- status: completed / planned / incomplete only
- All contextual details (speed, distance, modifications, planned vs actual) go in notes
- Do NOT add fields beyond the schema
- Sleep is NOT an activity — do not create an activity record for sleep

MACHINE:
- Only create a record if the user actively used it today
- If it failed, was unavailable, or data was not captured → route to outside

DEVICE:
- Only create a record if the user actually wore or used it today as an independent act
- If a device name appears only as the source of a measurement, do NOT create a device record — the source field on the measurement is sufficient
- If it failed, was unavailable, or data was not captured → route to outside
- status is always "used" — no other values

MEASUREMENT:
- Only body-level metrics with a specific numeric value
- If there is no number → do NOT create a measurement entity, omit entirely
- Qualitative descriptions like "way better", "trending up", "low" are NOT measurements
- Exercise metrics belong in activity notes, not here

TEST:
- Use for any diagnostic test performed by or on the user
- status: planned (about to do it) / done (already done)
- result: the test result if mentioned, null otherwise
- Do NOT use for therapy devices or machines

INTERVENTION:
- Only extract if it is a multi-step protocol with defined start, end, and ongoing structure
- NOT single procedures, surgeries, transplants, or one-time treatments
- Single procedures appear only as linked_to in an outcome entity
- status: active / completed / unknown — never infer "completed" without explicit statement

OUTCOME:
- An observed change in health state the user links to an intervention or procedure
- linked_to: label of the intervention or procedure
- direction: positive / negative / mixed / unknown
- Only extract if there is a clear observed change

CONTEXT:
- Health-relevant background info that is not a first-class event
- Examples: "first time in a week", "usually happens when stressed"
- Do NOT create context for every passing remark

THEORY:
- linked_to_label: label of the entity this theory relates to
- linked_to_type: type of that entity

OUTSIDE:
- Anything about the external world: equipment issues, procurement failures,
  provider actions, missed recordings, operational notes

─────────────────────────────────────────
MERGE RULE:
─────────────────────────────────────────
- Same entity mentioned twice → merge into one record
- Related symptoms describing the same condition → merge with combined qualifier

─────────────────────────────────────────
LABEL RULE:
─────────────────────────────────────────
- Label must be a specific named entity
- Strip generic suffixes unless part of the brand name
- Vague labels with no resolvable name → omit the entity entirely
- Named but unspecified groups → use group name as label, note in notes field
- Generic group labels like "usual supplements", "morning stack", "my vitamins" with no specific names → omit entirely
  Exception: if the user explicitly says they could not take their usual supplements (did_not_take), the group label is acceptable

Return ONLY this JSON, nothing else:
{{
  "entities": [
    {{
      "type": "...",
      "label": "...",
      ...
    }}
  ]
}}""",
        messages=[
            {
                "role": "user",
                "content": f"Structure these mentions.\n\nOriginal transcript for context:\n{normalized_text}\n\nMentions:\n{mentions_json}"
            }
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
        entities = parsed.get("entities", [])
        return resolve_time_references(entities)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse hatası: {e}")
        print(f"Ham yanıt: {raw[:500]}")
        return []


def resolve_time_references(entities: list) -> list:
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities