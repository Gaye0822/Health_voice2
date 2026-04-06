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
   Use the candidate_type from the mention — this is an ABSOLUTE rule.
   NEVER change a mention's candidate_type to a different entity type.
   Do not override based on transcript context, related entities, or your own judgment.

   If a mention came in as "intake" → it must leave as "intake".
   If a mention came in as "symptom" → it must leave as "symptom".
   If a mention came in as "outcome" → it must leave as "outcome".

   The relationship between entities (e.g. paracetamol reducing nocturia) may already
   be captured as a separate outcome or theory entity in the same mention list.
   Do not merge them — structure each mention independently using its own candidate_type.

   The ONLY exception: if the candidate_type is "other" → use your judgment to
   determine the best fitting schema, or omit if not structurable.

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
   User speculation or causal explanation about own body → theory
   Speculation about devices, data quality, or addressed to others → outside
   THEORY SELF-DIRECTED RULE: theory is ONLY for speculation about the user's
   own body or health outcomes. "Oura data is garbage", "Eight Sleep is more accurate",
   "the team should use Loop" → all outside, never theory.

   OUTCOME vs THEORY — language test:
   Before creating an outcome entity, check the user's exact language.

   Outcome signals — observed, established pattern:
   "tends to", "has been", "I've noticed", "every time X happens Y",
   "for years", "consistently", "it reduces", "it improves", "has been working"

   Theory signals — speculative, uncertain:
   "maybe", "I think", "could be", "possibly", "I wonder",
   "there's a possibility", "might be", "not sure but", "I guess"

   Rule: if ANY theory signal is present in the relevant sentence → theory, not outcome.
   If language is neutral or observational with no hedging → outcome.

   OUTCOME REASONING — required before creating any outcome entity:
   Before structuring an outcome, answer this question internally:
   "What specific language in the transcript confirms this is an observed
   pattern rather than speculation?"
   - If you find clear observational language (no hedging) → outcome
   - If you cannot find such language, or hedging is present → theory instead
   This reasoning step is mandatory — do not skip it.
   Provider recommendation not yet acted on → theory or omit
   Equipment issue, procurement note, missed recording → outside

   INTERVENTION vs INTAKE — critical distinction:
   An intervention is a multi-session treatment protocol with a defined start and expected end.
   A single completed health event is NEVER an intervention — it is intake, activity, or machine.

   These are interventions:
   - "I've been on antibiotics for 5 days" → intervention
   - "Started a peptide course last month" → intervention
   - "Doing a 30-day FMT protocol" → intervention (multi-session explicitly stated)

   These are NOT interventions — use intake, activity, or machine:
   - "Had the FMT treatment yesterday" → intake (single session)
   - "Got an IV" → intake
   - "Did Novothor at 4pm" → machine
   - "Took paracetamol" → intake

   Rule: if the user describes a single completed event → never intervention.
   Intervention requires explicit multi-session or protocol language.

   For symptoms specifically: ask whether this is a body state the user is currently
   experiencing, or whether it names a pathogen, virus, or infection source.
   "I have sinusitis" → symptom.
   "I caught rhinovirus from X" or "I have the rhinovirus" → context, not symptom.
   The pathogen name is background information. The actual symptoms are what get extracted.

   ABSENCE RULE — THREE STATES:
   Symptoms have three possible states. Only two are ever captured:

   explicit present → status: "present" (default, omit the field if present)
     User says it occurred: "had nocturia last night", "woke up with hot flash"

   explicit absent → status: "absent"
     User explicitly negates a recurring tracked phenomenon:
     "no nocturia last night", "did not have nocturia", "nocturia absent"
     Only apply to recurring tracked phenomena — things the user monitors over time.
     Do NOT apply to incidental symptoms: "no headache today" → omit entirely.

   not mentioned → do not create an entity (silence = unknown, not absent)

   RETROSPECTIVE ABSENCE INTERVAL:
   If the user says "I haven't had nocturia for 5-6 days" or similar:
   → status: "absent", interval: "5-6 days", source: "retrospective summary"
   Do not expand this into fake day-by-day records. One entity captures the interval as stated.

   If the user speculates about why something did not occur → also create a theory entity.

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

   CONTEXT-DEPENDENCY TEST — apply before structuring any symptom:
   Ask: would this finding exist on any other day, without the specific activity or
   situation the user just described?
   If yes → structure as symptom.
   If no → it is an expected after-effect of an activity. Put it in the activity's
   notes field if notable. Do NOT create a symptom entity.

   Cold plunge after-effects specifically: cold sensation, feeling cold for hours
   after a plunge — these are expected physiological responses, NOT symptoms.
   Put them in the cold plunge activity's notes if the user mentions them.
   Shivering from a cold room overnight is different — it is recurring, sleep-disrupting,
   and context-independent. Structure it as a symptom.

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

SYMPTOM STATUS RULE:
- status defaults to "present" — omit the field for normal present symptoms
- status: "absent" only when user explicitly negates a recurring tracked phenomenon
- interval: fill only for retrospective absence ("5-6 days", "about a week") — preserve as stated, do not expand
- source: "retrospective summary" only when interval is filled from a retrospective statement
- Never set status: "absent" for incidental symptoms ("no headache today" → omit entirely)

OUTCOME ENTITY RULE:
An outcome captures an observed directional change in a health variable, linked to something.
The outcome entity uses these fields ONLY: linked_to, what, onset_time, qualifier, direction.
NEVER use label, action, dose, unit, category, or any intake field in an outcome entity.

WHAT FIELD — always fill:
The what field captures what specifically changed — the subject of the outcome.
This is the most important field for making outcomes meaningful over time.
- "language recognition improved since selegiline" → what: "language recognition"
- "nocturia reduced since paracetamol" → what: "nocturia"
- "caffeine sensitivity increased" → what: "caffeine sensitivity"
- "gut health improving since FMT" → what: "gut health"
- "HRV trending up since stopping medication" → what: "HRV"
If what is not explicitly stated but clearly implied → fill it from context.
Only leave null if genuinely cannot be determined.

The substance or activity that caused the outcome is NOT part of the outcome entity.
It has its own separate entity (intake, activity, etc.).

"paracetamol 500mg first night" → intake entity (label: paracetamol, action: took, dose: 500)
"paracetamol reduces nocturia by 50%" → outcome entity (linked_to: paracetamol, what: "nocturia", direction: positive)
These are TWO separate entities. Never combine them into one.

If a mention came in as intake → structure it as intake, full stop.
The fact that it is related to an outcome does not change what it is.

QUALIFIER FIELD — STRICT RULE:
- Only fill if the user explicitly used descriptive language about this specific symptom
- NEVER invent or infer a qualifier — if no descriptive word exists in the transcript → null
- Do not use generic words like "present", "reported", "noted", "observed" as qualifiers
- Examples:
  "stools OK" → qualifier: null, notes: "reported as OK"
  "terrible gas" → qualifier: "terrible"
  "headache, massive" → qualifier: "massive"
  "tingling in hands" (no descriptor) → qualifier: null
- If the user's only description is a positive/neutral status word (OK, fine, normal, better)
  → qualifier: null, put it in notes instead: notes: "reported as OK / fine / normal"

DOSE FIELD:
- Only fill if a single unambiguous numeric value is explicitly stated
- Ranges, approximations, or vague quantities ("two or three", "a few", "some", "a couple") → null
- When in doubt → null

TIME FIELDS:
- Clock time, date, or named period (morning, evening) only
- Relative phrases ("recently", "last week", "three weeks ago") → null
- Put timing context in notes if relevant

END_TIME FIELD (activity and machine only):
- Fill only if the user explicitly states when the session ended
- "from 5:30am to 7:00am" → start_time: "5:30am", end_time: "7:00am"
- If only duration is stated → fill duration, leave end_time null
- If only start_time is stated → leave end_time null

EVENT_DATE FIELD (intake, activity, machine):
- Use when the event did NOT happen today — it happened on a specific past day
- Fill with the user's own words: "yesterday", "Friday", "last week"
- Do NOT convert to a date — preserve as stated
- Examples:
  "I had the FMT treatment yesterday" → event_date: "yesterday"
  "took paracetamol on Friday" → event_date: "Friday"
  "did the IV last week" → event_date: "last week"
- If the event happened today → leave event_date null

DURATION:
- If expressed as hours and minutes ("seven hours 20 minutes", "1 hour 45 minutes")
  write as digit string: "7 hours 20 minutes" not "seven hours 20 minutes"
  Do not convert to decimal

MERGE RULE:
- Same entity mentioned twice WITH THE SAME TYPE → one record
- NEVER merge mentions of different candidate_types, even if they share a common word.
  "paracetamol 500 mg first night" (intake) and "paracetamol reduces nocturia" (outcome)
  share the word "paracetamol" but are fundamentally different records — never merge.
  Different types = always separate entities, no exceptions.
- Multiple symptoms describing the same anatomical location and condition → merge into one entity
  Use the most specific label. Put additional detail in qualifier field.
  Do NOT create separate entities for different aspects of the same finding.

  NEVER merge mentions of different candidate_types even if they share the same label.
  "paracetamol" as intake + "paracetamol reduces nocturia" as outcome → TWO separate entities.
  An intake and an outcome about the same substance are fundamentally different records.
  Same rule applies to any type combination: intake+theory, symptom+outcome, etc.
  Different types = different entities, always.

  Examples of what MUST be merged:
  - "left knee swelling" + "left knee pain" + "left knee functional limitation" (cannot bend)
    → ONE entity: label "left knee swelling and pain", qualifier "completely swollen especially
    in back, fluid accumulation, cannot bend knee"
  - "hand numbness" + "hands getting cold during running"
    → ONE entity: label "hand numbness", qualifier "frozen, no feeling, painful; also cold during outdoor running"
  - Two headache entries for today and yesterday
    → ONE entity: label "headache", qualifier "massive, also present yesterday"
  - "right hip to knee pain" + "swelling" + "hip grinding" (all in same hip area)
    → ONE entity: label "right hip pain", qualifier "grinding, swelling, mechanical"

  The test: if two mentions share the same body part, same type, and are part of the same
  ongoing clinical picture → merge. If they are different types → keep separate always.

MEASUREMENT VALUE RULE:
- If a specific number is stated → fill value field with that number
- If no number is stated but a directional observation exists ("HRV is down",
  "heart rate elevated after 2 AM", "HRV has been better") → set value to null,
  write the observation into the notes field
- NEVER use 0 as a substitute for "no value stated" — 0 means the metric was actually zero
- NEVER omit a measurement just because no number exists — preserve it with value: null

THEORY LINKED_TO RULE:
- Always try to fill linked_to_label and linked_to_type.
- Look at what was discussed just before or after the theory — that is almost always
  what the theory is about. Link to it.
- Only leave linked_to_label null if the theory genuinely cannot be connected to
  any specific entity in this note.
- Examples:
  "I don't know what's going on with my stomach" → linked_to_label: "stool consistency",
  linked_to_type: "symptom" (stomach issues were just discussed)
  "Maybe it's the cold plunge affecting my HRV" → linked_to_label: "cold plunge",
  linked_to_type: "activity"
  "I have no idea why I slept so well" → linked_to_label: "sleep quality",
  linked_to_type: "measurement"
  "I don't know, life sucks" → linked_to_label: null (genuinely unconnected)

LABEL RULE:
- Specific named entity only
- Vague or unresolvable label → omit entirely
- For intake specifically: the substance must be named precisely enough to be
  tracked and compared over time. Group labels are never acceptable:
  "lunch supplements", "wake up stack", "morning vitamins", "usual supplements",
  "night stack" → omit entirely, even if action is did_not_take
  Exception: if the label is a registered canonical term in the knowledge base

MEAL RULE:
- Capture: label, time, eaten_out (true/false)
- If eaten_out is true and a restaurant name is mentioned → put it in restaurant field
- description: preserve the user's spoken food description as a single free-text string
  This is a handoff field for the downstream food system — do not parse or structure it
  Copy the relevant spoken content as-is: "150g chicken, 250g white bread, 40g butter"
  If the user describes what they ate in any detail → put it here
  If no food content was described → leave null
- Do NOT attempt to parse ingredients, calculate nutrition, or structure food items
- This layer is not the food system — it only preserves the description for handoff

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

    # Validate with Pydantic — entity by entity so one bad entity doesn't kill the rest
    entities = []
    raw_entities = raw_input.get("entities", [])
    for raw_entity in raw_entities:
        try:
            validated = EntityOutput.model_validate({"entities": [raw_entity]})
            entities.extend([e.model_dump() for e in validated.entities])
        except Exception as e:
            print(f"⚠️ structure_mentions: Pydantic rejected entity {raw_entity.get('type', '?')} / {raw_entity.get('label', raw_entity.get('linked_to', '?'))}: {e}")
            # Drop the malformed entity — do not pass raw dicts through

    return _resolve_time_references(entities)


def _resolve_time_references(entities: list) -> list:
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities