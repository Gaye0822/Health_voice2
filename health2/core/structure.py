import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Import models — single source of truth for schemas
try:
    from core.models import get_entity_tool_schema, ENTITY_SCHEMAS, EntityOutput
except ImportError:
    from models import get_entity_tool_schema, ENTITY_SCHEMAS, EntityOutput


STRUCTURE_TOOL = {
    "name": "extract_entities",
    "description": (
        "Extract structured health entities from a voice note transcript. "
        "Each entity must conform exactly to its type schema. "
        "Omit any entity that does not meet the extraction criteria."
    ),
    "input_schema": get_entity_tool_schema()
}


SYSTEM_PROMPT = """You are a health event structurer.

ALWAYS respond in English regardless of the language of the transcript.

You receive a list of mentions extracted from a voice note. Convert each into a structured
health entity by calling the extract_entities tool. You MUST use the tool — do not respond
with plain text.

─────────────────────────────────────────
HOW TO STRUCTURE EACH MENTION
─────────────────────────────────────────

For each mention, work through these questions before writing the entity:

1. WHAT TYPE IS THIS?
   Use the candidate_type from the mention unless it clearly violates the schema.
   Do not override a user-confirmed type based on your own judgment.

2. WHAT ACTUALLY HAPPENED?
   Use the temporal_evidence field:
   - explicit_today → structure normally
   - active_regimen → structure normally only if the transcript contains
     same-session confirmation (specific time, "today", "this morning", "just")
     Provider advice to continue is NOT same-session confirmation.
     If no confirmation → omit
   - future_plan → omit entirely
   - consultation_relay → omit entirely, EXCEPT factual test results and
     factual measurements which can be extracted normally
   - unclear → when in doubt, omit

3. IS THIS AN OBSERVED FACT OR AN INTERPRETATION?
   Observed facts → event layer (intake, symptom, activity, machine, measurement etc.)
   User speculation or causal explanation → theory
   Provider recommendation not yet acted on → theory or omit
   Equipment issue, procurement note, missed recording → outside

   For symptoms specifically: ask whether this is a body state the user is currently
   experiencing, or whether it names a pathogen, virus, or infection source.
   "I have sinusitis" → symptom.
   "I caught rhinovirus from X" or "I have the rhinovirus" → context, not symptom.
   The pathogen name is background information. The actual symptoms are what get extracted.

   The symptom bar is clinical: a symptom must be something a clinician could write
   in a chart as a finding. If it is how the user feels in general, omit it.
   General feelings ("feeling unwell", "feel awful", "feel terrible", "feel garbage",
   "not feeling it"), energy/motivation states ("exhausted", "groggy", "low energy"),
   and vague discomfort ("flu-adjacent", "under the weather", "a bit off") do not qualify.
   Named conditions, localized findings, and observable body responses do qualify.

   Sleep difficulty is never a symptom in this system — regardless of how it is labeled.
   "Hard time sleeping", "trouble falling asleep", "woke up early", "tough sleep", and
   even clinical-sounding labels like insomnia derived from user self-report are all omitted.
   Only device-recorded sleep metrics are extracted, as measurements.

   For activities: if a movement or exercise is described as something done inside or
   as part of a machine session, do not create a separate activity entity for it.
   It belongs in the machine's notes field.

4. WHICH FIELDS ARE ACTUALLY KNOWN?
   Only fill fields with information explicitly present in the transcript.
   Use null for anything unknown or missing.
   Never use the string "unknown".
   Never infer or guess field values.

─────────────────────────────────────────
FIELD RULES
─────────────────────────────────────────

NULL RULE:
- null for any unknown or missing value
- NEVER the string "unknown"

DOSE FIELD:
- Only fill if a single unambiguous numeric value is explicitly stated
- Ranges, approximations, or vague quantities ("two or three", "a few", "some", "a couple") → null
- When in doubt → null

TIME FIELDS:
- Clock time, date, or named period (morning, evening) only
- Relative phrases ("recently", "last week", "three weeks ago") → null
- Put timing context in notes if relevant

DURATION:
- If expressed as hours and minutes ("seven hours 20 minutes", "1 hour 45 minutes")
  write as digit string: "7 hours 20 minutes" not "seven hours 20 minutes"
  Do not convert to decimal

MERGE RULE:
- Same entity mentioned twice → one record
- Related symptoms describing the same condition → merge with combined qualifier

LABEL RULE:
- Specific named entity only
- Vague or unresolvable label → omit entirely

MEAL RULE:
- Capture: label, time, eaten_out (true/false), restaurant name if mentioned
- Do NOT populate items with ingredient lists, gram-level breakdown, or full meal content
- If eaten_out is true and a restaurant name is mentioned → put it in notes
- Detailed food content belongs in a downstream food system, not here

─────────────────────────────────────────
EXAMPLES
─────────────────────────────────────────

EXAMPLE 1 — intake actions: took vs did_not_take vs omit
Mentions include:
- supplementA, intake, explicit_today (forgot this morning)
- supplementB, intake, explicit_today (took around 8am)
- medicationC, intake, explicit_today (took around 3:30pm)
- supplementD, intake, active_regimen (been taking last few nights)
- night stack, intake, active_regimen (stopped taking for last few nights, replaced with supplementD)

Thinking:
- supplementA: explicit_today, forgot → did_not_take, notes: forgot
- supplementB: explicit_today, took → took, time: 8:00am
- medicationC: explicit_today, took → took, time: 3:30pm, category: prescription
- supplementD: active_regimen — same-session confirmation?
  "last few nights" is a recent recurring pattern the user is actively reporting →
  close enough to confirm current practice → took, notes: last few nights
- night stack: active_regimen, user explicitly says not taking it currently →
  did_not_take, notes: replaced with supplementD
  This is a deliberate current-practice decision, not a future plan.
  Any active_regimen item the user explicitly states they are NOT doing →
  did_not_take if it is a named substance, omit if it is vague.

Result entities:
[supplementA did_not_take, supplementB took, medicationC took,
supplementD took, night stack did_not_take]

---

EXAMPLE 2 — symptom vs outcome, and outside
Mentions include:
- substanceX, intake, explicit_today (tried, not helping)
- substanceY, intake, explicit_today (not available, not taken)
- conditionA, symptom, explicit_today (ongoing, severe)
- conditionA improving, candidate_type: outcome, explicit_today
  (two consecutive days of improvement, since interventionZ ~6 weeks ago)
- substanceY not available, outside

Thinking:
- substanceX: tried → took, notes: not helping
- substanceY: not available → did_not_take, notes: not available
- conditionA: currently experiencing, specific named condition → symptom, qualifier: severe, ongoing
- conditionA improving: this is NOT a symptom — it is a directional change in a tracked state,
  linked to a prior intervention → outcome, direction: positive,
  notes: two consecutive days, since interventionZ ~6 weeks ago
  Key distinction: a symptom is a named condition present now.
  An outcome is an observed change over time linked to something.
- substanceY not available: procurement/operational note → outside

Result entities:
[substanceX took, substanceY did_not_take, conditionA symptom,
conditionA improvement outcome, substanceY outside]

---

EXAMPLE 3 — theory linking and context
Mentions include:
- substanceA, intake, explicit_today (took last night for symptomB)
- symptomB, symptom, explicit_today
- conditionC improving, outcome, explicit_today (since interventionD)
- interventionD background, context
- substanceA/sleep/mechanism theory, theory

Thinking:
- substanceA: took, last night → intake, category: OTC, notes: taken for symptomB
- symptomB: specific, observable → symptom, onset_time: last night
- conditionC improving: observed positive change, user links to interventionD →
  outcome, direction: positive, notes: since interventionD
- interventionD background: useful health context for understanding the outcome →
  context, related_to: conditionC improvement
- theory: user speculates substanceA affects sleep, wonders about mechanism →
  theory, linked_to_label: substanceA, linked_to_type: intake
  Capture the speculation in raw_text, link it to what it is about

Result entities:
[substanceA took, symptomB symptom, conditionC improvement outcome,
interventionD context, substanceA/mechanism theory]"""


def structure_mentions(mentions: list, normalized_text: str) -> list:
    """
    Stage 2: Structure mentions into typed health entity records.
    Uses tool_use for strict schema-enforced JSON output.
    Returns list of raw entity dicts (backward compatible with validate.py).
    """
    if not mentions:
        return []

    # Deterministic pre-filter — no LLM involved
    try:
        from core.mention_filter import filter_mentions, log_filtered
    except ImportError:
        from mention_filter import filter_mentions, log_filtered

    mentions, dropped = filter_mentions(mentions, normalized_text)
    log_filtered(dropped)

    if not mentions:
        return []

    mentions_json = json.dumps(mentions, indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0,
        system=SYSTEM_PROMPT,
        tools=[STRUCTURE_TOOL],
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Structure these mentions into health entities.\n\n"
                    f"Original transcript for context:\n{normalized_text}\n\n"
                    f"Mentions:\n{mentions_json}"
                )
            }
        ]
    )

    # Extract tool_use block
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None
    )

    if not tool_use_block:
        print("⚠️ structure_mentions: no tool_use block in response")
        return []

    raw_input = tool_use_block.input

    # Validate with Pydantic
    try:
        output = EntityOutput.model_validate(raw_input)
        entities = [e.model_dump() for e in output.entities]
    except Exception as e:
        print(f"⚠️ structure_mentions: Pydantic validation error: {e}")
        # Fallback: return raw dicts if Pydantic validation fails
        entities = raw_input.get("entities", [])

    return _resolve_time_references(entities)


def _resolve_time_references(entities: list) -> list:
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities