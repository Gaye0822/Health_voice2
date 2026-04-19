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
   If a mention came in as "intervention" -> it must leave as "intervention".

   The relationship between entities (e.g. paracetamol reducing nocturia) may already
   be captured as a separate outcome or theory entity in the same mention list.
   Do not merge them — structure each mention independently using its own candidate_type.

   The ONLY exception: if the candidate_type is "other" → use your judgment to
   determine the best fitting schema, or omit if not structurable.

   ── INTERVENTION CANDIDATE TYPE — MANDATORY SCHEMA BRANCH ──
   Before processing any mention, check: is candidate_type = "intervention"?
   If NO - continue.
   If YES — stop.DO NOT CHANGE THE ENTITI TYPE . Do not read further into this section. Apply these rules immediately:


   STEP 1: Write type: "intervention". No other type is permitted.
   STEP 2: The substance name may sound like something you would put in an intake entity
           (e.g. "doxycycline", "BPC-157", "PHGG"). Ignore that. The schema branch
           is intervention. Fill intervention fields only:
           label, start_date, end_date, status, duration_days, day_of_protocol, notes.
   STEP 3: Do NOT write action, dose, unit, category, or any intake field.
           These fields do not exist on an intervention entity.
   STEP 4: status defaults to "active" unless the transcript says the course is finished.
   STEP 5: start_date — write the user's exact words. Never convert to a date.
           "two days ago" → start_date: "two days ago"
           "last night" → start_date: "last night"
           "last Monday" → start_date: "last Monday"
   STEP 6: duration_days — integer only if explicitly stated. Otherwise null.
           "10 days" → duration_days: 10. "a month" → null.
   STEP 7: day_of_protocol — always null. Schema_enforcer calculates this.

   Correct output for an intervention mention:
     { "type": "intervention", "label": "doxycycline course", "status": "active",
       "start_date": "last night", "duration_days": 10, "day_of_protocol": null }
   Wrong output — NEVER produce this for an intervention candidate_type:
     { "type": "intake", "label": "doxycycline", "action": "took", "category": "prescription" }

2. WHAT ACTUALLY HAPPENED?
   Use the temporal_evidence field:
   - explicit_today → structure normally
   - active_regimen → structure normally only if the transcript contains
     same-session confirmation (specific time, "today", "this morning", "just")
     Provider advice to continue is NOT same-session confirmation.
     If no confirmation → omit
     EXCEPTION: intervention entities are always structured regardless of
     active_regimen — a multi-day protocol does not need same-session confirmation.
     See the INTERVENTION CANDIDATE TYPE block in step 1 for full schema rules.
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
   Before structuring an outcome, answer these two questions internally:

   1. "What specific language in the transcript confirms this is an observed
      pattern rather than speculation?"
      - If you find clear observational language (no hedging) → proceed to question 2
      - If you cannot find such language, or hedging is present → theory instead

   2. "Is the 'what' field a trackable, measurable construct with a clear baseline?"
      - Must be something that can be compared over time: nocturia frequency,
        HRV value, stool consistency, body weight, blood glucose
      - Must NOT be a vague subjective domain: "language recognition",
        "mental stuff", "psychological benefits", "cognitive function",
        "energy", "how I feel overall"
      - If the construct is not yet defined or measurable → theory instead

   Both questions must pass. If either fails → theory, not outcome.
   This reasoning step is mandatory — do not skip it.
   Provider recommendation not yet acted on → theory or omit
   Equipment issue, procurement note, missed recording → outside

   OUTSIDE SUBTYPE RULE — always fill subtype field.
   First answer: "Is there a specific, documented, quantified malfunction?"

   - device_failure: one device clearly wrong in a concrete, measurable way.
     The malfunction is specific and quantified — not a matter of interpretation.
     "Oura has been off by an entire hour — starts recording at midnight when I slept at 10:30" ✅
     "Polar strap jumping all over the place, no idea what my heart rate was" ✅
     These are NOT device_failure:
     "Eight Sleep and Oura diverging since PONS" → two sources disagree → source_discrepancy ❌
     "resting heart rate has been fixed" → observation about a metric, not a device malfunction ❌
     "HRV was awful" → subjective commentary ❌
     "Eight Sleep is more accurate than Oura" → reliability opinion, not confirmed failure ❌
     When in doubt → source_discrepancy, not device_failure.

   - source_discrepancy: two devices report different values, neither clearly wrong.
     "Eight Sleep and Oura showing different deep sleep since PONS" → source_discrepancy ✅
     "WHOOP shows different HRV than Oura" → source_discrepancy ✅
     These are reliability concerns — not confirmed malfunctions.

   - procurement: supply issues, logistics, missed deliveries, device arrivals.
     "couldn't get my supplements" → procurement
     "lactic device delivered, needs calibration" → procurement

   - operational: notes addressed to the team, admin, scheduling, instructions.
     "we should figure out a system for this" → operational
     "teach me how to calibrate it" → operational

   - general: device opinions, general complaints with no specific documented impact.
     "WHOOP sucks" → general



   These are interventions:
   - "I've been on antibiotics for 5 days" → intervention
   - "Started a peptide course last month" → intervention
   - "Doing a 30-day FMT protocol" → intervention (multi-session explicitly stated)

  

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

     TREATMENT RESPONSE — do NOT use absent for symptom relief:
     If a symptom was eliminated or reduced by a treatment, this is an OUTCOME, not absent.
     "immediately got rid of the pain in my left eye" (after doxycycline) → outcome entity,
       linked_to: doxycycline, what: "left eye pain", direction: positive
     "paracetamol reduced my nocturia" → outcome, not absent
     The symptom being absent due to treatment is a health outcome, not a simple negation.
     Ask: did something cause this to go away? If yes → outcome. If just not present → absent.

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

ACTIVITY STATUS RULE:
- status is REQUIRED for every activity and machine entity — never omit it
- status: "completed" → activity happened and is done (default for past events)
- status: "planned" → DO NOT USE. Future plans are omitted entirely before structuring. If a mention arrived here with future intent, omit it.
- status: "incomplete" → activity was started but not finished (e.g. cut short)
- status: "did_not_complete" → activity was explicitly NOT done / skipped entirely
  Use when user says "didn't lift", "didn't do PEMF", "couldn't get a cold plunge in",
  "never got to zone 2", "didn't make it to the gym"
  The user intended to do it but it did not happen at all.
- When in doubt → "completed" (mention_filter already ensures the activity actually happened)

MACHINE STATUS RULE:
- status is REQUIRED for every machine entity — never omit it
- status: "used" → machine was used in this session
- status: "did_not_use" → machine was explicitly NOT used / skipped
  Use when user says "didn't do PEMF", "no Novothor today", "skipped the machine"

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

DURATION_DAYS FIELD (intervention only):
- Fill if the user explicitly states the total length of the protocol
- "10 days at 100 mg" → duration_days: 10
- "a 30-day course" → duration_days: 30
- If not stated → null

START_DATE FIELD (intervention only):
- Write the user's own words exactly as stated — do NOT convert to a date
- "two days ago" → start_date: "two days ago"
- "last Monday" → start_date: "last Monday"
- "last night" → start_date: "last night"
- "this morning" → start_date: "this morning"
- schema_enforcer will convert this to an actual date using today's date
- If not stated → null

DAY_OF_PROTOCOL FIELD (intervention only):
- Leave null — this is calculated by schema_enforcer, never by the LLM

EVENT_DATE FIELD (intake, activity, machine, symptom):
- Use when the event did NOT happen today — it happened on a specific past day
- Fill with the user's own words: "yesterday", "Friday", "last week"
- Do NOT convert to a date — preserve as stated
- Examples:
  "I had the FMT treatment yesterday" → event_date: "yesterday"
  "took paracetamol on Friday" → event_date: "Friday"
  "did the IV last week" → event_date: "last week"
- If the event happened today → leave event_date null
- CRITICAL: When the transcript switches between yesterday and today, track which
  events belong to which day. Look for temporal markers like "yesterday", "this morning",
  "today". Activities described in a "yesterday" narrative get event_date: "yesterday".
  Example: "yesterday I did onsen around 10:15, then breakfast... this morning I had..."
  → onsen (event_date: "yesterday"), breakfast today (event_date: null)

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

- ONSEN / COLD PLUNGE RULE: always extract as separate entities, never merge.
  Even when mentioned together ("onsen and cold plunge", "onsen cold plunge session"),
  these are two distinct activities — create one entity for each.
  If the user did multiple sessions of the same type (e.g. two cold plunges),
  create separate entities for each with different timing in notes.
  "two cycles of onsen cold plunge" → onsen (entity 1) + cold plunge (entity 2),
  notes: "two cycles, alternated"

- OUTSIDE SUBTYPE RULE: every outside entity MUST have a subtype. Ask first:
  "Is there a specific, documented, quantified malfunction?"

  device_failure: ONE device is clearly wrong in a concrete, measurable way.
    "Oura starts recording sleep at midnight when I fell asleep at 10:30" → device_failure ✅
    "Polar strap jumping all over the place, no heart rate reading at all" → device_failure ✅
    The malfunction is specific and quantified — not a matter of interpretation.

  source_discrepancy: two devices report different values but neither is clearly wrong.
    "Eight Sleep and Oura showing different deep sleep since PONS" → source_discrepancy ✅
    "WHOOP shows different HRV than Oura" → source_discrepancy ✅
    "Eight Sleep more accurate than Oura in my experience" → source_discrepancy ✅
    These are reliability concerns or disagreements, not confirmed failures.
    When in doubt → source_discrepancy, not device_failure.

  procurement: supply, logistics, ordering, device arrivals.
    "couldn't get my supplements", "lactic device delivered, needs calibration" → procurement ✅

  operational: notes addressed to the team, admin, scheduling, instructions.
    "we should figure out a system for this", "teach me how to calibrate it" → operational ✅

  general: device opinions or complaints with no specific documented impact.
    "WHOOP sucks", "I don't trust Oura anymore" → general ✅

- OUTSIDE MERGE RULE: multiple outside mentions about the same device or topic → ONE entity.
  Use the most informative raw_text as the primary description.
  Put additional detail, recommendations, or follow-up in the notes field.
  Do NOT create separate outside entities for each sentence about the same issue.
  Examples:
  "Oura off by an hour" + "Oura data is garbage" + "use Loop instead"
  → ONE outside entity: raw_text: "Oura off by 1-1.5 hours", subtype: device_failure,
    notes: "recommends using Loop instead"
  "WHOOP shows different HRV" + "WHOOP has always been unreliable"
  → ONE outside entity: raw_text: "WHOOP vs Oura HRV discrepancy", subtype: source_discrepancy

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

THEORY MERGE RULE:
- If multiple theories share the same linked_to_label, merge them into a single theory.
- Combine their raw_text into one concise sentence covering all speculations.
- Example: two theories both linked_to_label: "abdominal pain" →
  merge into one: "mushrooms or eggplant may be causing digestive issues"
- Never produce two theory entities with the same linked_to_label.

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
- event_date: if the meal did NOT happen today, capture when it happened.
  Use "yesterday" if the user says yesterday, or a specific date if stated.
  Do NOT drop meals just because they happened yesterday — preserve them with event_date.

  CRITICAL: When the transcript switches between yesterday and today, track which
  meals belong to which day. Look for temporal markers like "yesterday", "this morning",
  "today". Meals described in a "yesterday" context get event_date: "yesterday".

  The mention passed to you has a context field — if it contains "yesterday" or
  references past events, set event_date: "yesterday".

  IMPORTANT: If a mention has a "_event_date_hint" field set to "yesterday",
  you MUST set event_date: "yesterday" for that meal entity. This is a pre-computed
  signal from the system — do not override it.

  Example: "yesterday I did [X]... then breakfast right around 10:15, lunch at 2 PM, dinner around 6"
  → breakfast (event_date: "yesterday"), lunch (event_date: "yesterday"), dinner (event_date: "yesterday")
  "this morning had... breakfast" → event_date: null (today)
  - If the meal is clearly from today → event_date: null

- description: preserve the user's spoken food description as a single free-text string
  This is a handoff field for the downstream food system — do not parse or structure it
  Copy the relevant spoken content as-is: "150g chicken, 250g white bread, 40g butter"
  If the user describes what they ate in any detail → put it here
  If no food content was described → leave null
- Do NOT attempt to parse ingredients, calculate nutrition, or structure food items
- This layer is not the food system — it only preserves the description for handoff

RAW_TEXT FIELD (outside, context, theory):
- Keep raw_text short and descriptive — one sentence maximum
- Summarize what it is about, do not copy the full transcript passage
- outside: "Eight Sleep vs Oura divergence since PONS therapy" not the full paragraph
- context: "switched to carbs for easier digestion due to stomach issues" not the full explanation
- theory: state the core speculation in one sentence
- NEVER copy more than ~15 words directly from the transcript into raw_text

CONTEXT ENTITY RULE:
Before creating a context entity, answer both:
1. Is this information necessary to understand another entity in this note?
2. Which specific entity does this explain? → fill related_to with that entity's label.
If related_to cannot be filled → do NOT create a context entity. Route to outside or omit.
A context entity with related_to: null is invalid — omit it.

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
interventionD context, substanceA/mechanism theory]

---

EXAMPLE 4 — outcome vs theory: language test + tracked construct test
Mentions include:
- selegiline → language recognition improvement, outcome candidate
- selegiline → caffeine sensitivity increase, outcome candidate
- nocturia reduction since paracetamol, outcome candidate
- caffeine mechanism question, theory

Thinking:
- "definitely starting to feel the upsweep in language recognition as a result of selegiline"
  Language test: "definitely" → observational. No hedging. Passes test 1.
  Tracked construct test: "language recognition" — is this measurable with a baseline?
  No. There is no defined metric, no way to compare over time, no established construct.
  This is a subjective domain the user perceives but cannot track empirically.
  → FAILS test 2. Make it theory.
  theory: raw_text: "selegiline improving language recognition — user's subjective perception",
  linked_to: selegiline

- "apparently it's made it capable for me to absolutely bomb my brain with caffeine, super sensitive"
  Language test: "apparently" → hedging signal present.
  → FAILS test 1. Theory.
  theory: raw_text: "selegiline may be increasing caffeine sensitivity",
  linked_to: selegiline

- "paracetamol tends to reduce my nocturia, it's been a trend for years, reduces odds by 50%"
  Language test: "tends to", "for years", "reduces" → established pattern, no hedging. Passes test 1.
  Tracked construct test: "nocturia" — tracked across multiple notes, binary occurrence,
  comparable over time. Passes test 2.
  → outcome: linked_to: paracetamol, what: nocturia, direction: positive, qualifier: "reduces odds by 50%"

- "is it good for me to use caffeine... is it bad cause it's like messing with the internal process"
  Question form, speculative → theory.

Result entities:
[selegiline/language theory, selegiline/caffeine theory, paracetamol/nocturia outcome, caffeine mechanism theory]"""


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

    # Deterministic event_date hint injection — all event types, not just meal.
    # Priority: temporal_evidence="explicit_past" + event_date_label (set by mention.py)
    # Fallback: "yesterday" string in reasoning or context
    EVENT_DATE_TYPES_HINT = {"meal", "intake", "symptom", "activity", "machine"}
    for m in mentions:
        ctype = m.get("candidate_type")
        if ctype not in EVENT_DATE_TYPES_HINT:
            continue
        if m.get("temporal_evidence") == "explicit_past":
            edl = m.get("event_date_label")
            if edl:
                m["_event_date_hint"] = edl
                continue
        reasoning = m.get("reasoning", "").lower()
        context_text = m.get("context", "").lower()
        if "yesterday" in reasoning or "yesterday" in context_text:
            m["_event_date_hint"] = "yesterday"

    mentions_json = json.dumps(mentions, indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=6000,
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

    # ── Intervention guard ────────────────────────────────────────────────
    # Every intervention mention MUST produce an intervention entity.
    # Uses token-based matching for robustness across label variations.

    def _guard_labels_match(raw_mention: str, entity_label: str) -> bool:
        STOP = {"course", "protocol", "treatment", "therapy", "program",
                "supplement", "dose", "medication", "drug", "pill", "tablet"}
        ta = {t for t in raw_mention.lower().split() if t not in STOP and len(t) > 2}
        tb = {t for t in entity_label.lower().split() if t not in STOP and len(t) > 2}
        if ta & tb:
            return True
        if raw_mention.lower() in entity_label.lower() or entity_label.lower() in raw_mention.lower():
            return True
        return False

    for mention in mentions:
        if mention.get("candidate_type") != "intervention":
            continue

        raw = mention.get("raw_mention", "")
        raw_lower = raw.lower()

        # 1. Check if a correct intervention entity already exists for this mention
        intervention_exists = any(
            e.get("type") == "intervention" and _guard_labels_match(raw_lower, e.get("label", ""))
            for e in entities
        )

        # 2. Remove spurious non-intervention entities that share this label
        #    intake IS allowed to coexist — it captures the specific dose event.
        ALLOWED_ALONGSIDE = {"intake"}
        spurious = [
            e for e in entities
            if e.get("type") not in {"intervention"} | ALLOWED_ALONGSIDE
            and _guard_labels_match(raw_lower, e.get("label", e.get("linked_to", "")))
        ]
        for s in spurious:
            print(f"⚙️  structure guard: removing spurious {s.get('type')}|"
                  f"{s.get('label', s.get('linked_to', '?'))} — intervention mention '{raw}' exists")
        if spurious:
            entities = [e for e in entities if e not in spurious]

        if intervention_exists:
            continue

        # 3. No intervention entity found — find the wrong entity the LLM produced
        wrong_entity = next(
            (e for e in entities
             if e.get("type") != "intervention"
             and _guard_labels_match(raw_lower, e.get("label", e.get("linked_to", "")))),
            None
        )

        if wrong_entity:
            print(f"⚙️  structure guard: '{raw}' mention was intervention "
                  f"but LLM produced {wrong_entity.get('type')} — fixing")
            entities = [e for e in entities if e is not wrong_entity]
        else:
            print(f"⚙️  structure guard: '{raw}' mention produced no entity at all — injecting intervention")

        # Inject minimal intervention entity — enricher + schema_enforcer will fill the rest
        entities.append({
            "type": "intervention",
            "label": raw,
            "status": "active",
            "start_date": None,
            "end_date": None,
            "duration_days": None,
            "day_of_protocol": None,
            "notes": mention.get("context") or None,
        })

    # Attach raw_mention from mentions list to each entity
    entities = _attach_raw_mentions(entities, mentions)

    # Apply negation flag from mention_filter
    # If mention was marked as negated, set status accordingly
    negated_labels = set()
    for m in mentions:
        if m.get("_negated"):
            negated_labels.add(m.get("raw_mention", "").lower())

    for entity in entities:
        label = entity.get("label", "").lower()
        etype = entity.get("type", "")
        if label in negated_labels:
            if etype == "activity" and entity.get("status") == "completed":
                entity["status"] = "did_not_complete"
                print(f"⚙️  negation applied: activity '{label}' → did_not_complete")
            elif etype == "machine" and entity.get("status") == "used":
                entity["status"] = "did_not_use"
                print(f"⚙️  negation applied: machine '{label}' → did_not_use")

    # Deterministic event_date hint application — all event types, token-based matching.
    EVENT_DATE_TYPES_APPLY = {"meal", "intake", "symptom", "activity", "machine"}

    def _event_date_labels_match(raw: str, entity_label: str) -> bool:
        STOP = {"course", "protocol", "supplement", "dose", "medication",
                "breakfast", "lunch", "dinner", "meal"}
        ta = {t for t in raw.split() if t not in STOP and len(t) > 2}
        tb = {t for t in entity_label.split() if t not in STOP and len(t) > 2}
        if ta & tb:
            return True
        return raw in entity_label or entity_label in raw

    mention_hint_map: dict = {}
    for m in mentions:
        hint = m.get("_event_date_hint")
        if hint:
            raw_key = m.get("raw_mention", "").lower()
            mention_hint_map[raw_key] = hint

    if mention_hint_map:
        for entity in entities:
            etype = entity.get("type")
            if etype not in EVENT_DATE_TYPES_APPLY:
                continue
            if entity.get("event_date"):
                continue
            label = entity.get("label", "").lower()
            for raw_key, hint in mention_hint_map.items():
                if _event_date_labels_match(raw_key, label):
                    entity["event_date"] = hint
                    print(f"⚙️  event_date hint applied: {etype} '{label}' → event_date: '{hint}'")
                    break

    return _resolve_time_references(entities)


def _attach_raw_mentions(entities: list, mentions: list) -> list:
    """
    For each entity, find the closest matching mention and attach:
    - _raw_mention: original transcript text (from inferred_as if LLM changed it)
    This allows save_entities to detect when LLM inferred a different label.
    """
    for entity in entities:
        label = entity.get("label", entity.get("metric", entity.get("linked_to", "")))
        entity_type = entity.get("type", "")

        for m in mentions:
            m_raw = m.get("raw_mention", "")
            m_inferred = m.get("inferred_as")  # original transcript text if LLM changed it

            # Exact match on raw_mention
            if m_raw.lower() == label.lower():
                if m_inferred:
                    # LLM changed the term — record original transcript text
                    entity["_raw_mention"] = m_inferred
                    print(f"⚙️  attach_raw_mention: '{m_inferred}' → '{label}' (inferred by mention LLM)")
                break

            # Match via inferred_as — raw_mention was normalized
            if m_inferred and m_raw.lower() == label.lower():
                entity["_raw_mention"] = m_inferred
                print(f"⚙️  attach_raw_mention: '{m_inferred}' → '{label}' (via inferred_as)")
                break

    return entities


def _resolve_time_references(entities: list) -> list:
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities