import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Import models — single source of truth
try:
    from core.models import get_mention_tool_schema, MentionOutput
except ImportError:
    from models import get_mention_tool_schema, MentionOutput

CANDIDATE_TYPES = [
    "intake", "symptom", "activity", "machine", "device",
    "measurement", "meal", "theory", "outside", "other"
]

MENTION_TOOL = {
    "name": "extract_mentions",
    "description": (
        "Extract all health-relevant mentions from a voice note transcript. "
        "For each mention, assign a candidate type and temporal evidence value."
    ),
    "input_schema": get_mention_tool_schema()
}

SYSTEM_PROMPT = """You are a health mention detector.

ALWAYS respond in English regardless of the language of the transcript.

Your ONLY job is to find mentions in a voice note transcript and assign each one a
candidate type and temporal evidence value. Call the extract_mentions tool — do not
respond with plain text.

─────────────────────────────────────────
HOW TO EVALUATE EACH MENTION
─────────────────────────────────────────

Before assigning a type to anything, ask these three questions in order:

1. DID THIS ACTUALLY HAPPEN?
   Is this a completed fact or active ongoing state — or is it a future plan,
   a provider recommendation, something the user is speculating about, or something
   that was mentioned in passing without confirmation it occurred?
   If it has not happened yet, or if it was only recommended or discussed → do not
   extract as a health event. Route to theory or omit.

2. IS THIS ABOUT THE USER'S OWN BODY OR ACTIONS?
   Is this something the user did, experienced, or measured directly?
   Or is it about another person, about the external world, about equipment,
   about what a provider said or wants?
   If it is not directly about the user's body or actions → outside or omit.

3. IS THE LABEL SPECIFIC AND RESOLVABLE?
   Can this be named precisely enough to be tracked and compared over time?
   Or is it vague, general, or descriptive of a mood or passing feeling?
   If it cannot be named specifically → omit.

Only extract a mention if all three questions pass.

─────────────────────────────────────────
CANDIDATE TYPES
─────────────────────────────────────────

intake
  Something that entered or was confirmed not to have entered the body.
  Supplements, medications, food or drinks taken with health intent.
  Extract even if NOT taken — the action is decided later.
  Do not extract if the substance was only recommended, discussed, or planned.

symptom
  A clinically named, observable body state that a clinician could write in a chart
  as a finding. Must be specific enough to be tracked and compared over time.
  Ask: could a doctor document this as a clinical finding? If yes → symptom.
  If it is how the user feels in general → omit.

  These do NOT qualify:
  - General feelings: "feeling unwell", "feel awful", "feel terrible", "feel garbage"
  - Energy or motivation: "felt groggy", "exhausted", "low energy", "not feeling it"
  - Sleep difficulty: "hard time sleeping", "woke up early", "tough sleep", "trouble falling asleep"
    Sleep difficulty is never a symptom in this system — even if it could be labeled insomnia.
    Only device-recorded sleep metrics (HRV, sleep duration, deep sleep) are extracted, as measurements.
  - Vague discomfort: "flu-adjacent feeling", "under the weather", "a bit off"
  - Anything the user explicitly dismisses: "probably nothing", "just a bit off"

  These DO qualify:
  - Named conditions: sinusitis, nausea, headache, sore throat, chest tightness
  - Localized specific findings: right knee tenderness, muscle fasciculation
  - Observable body responses: flushing, rash, fever, dizziness

activity
  A physical session the user performed — exercise, breathwork, meditation,
  cold exposure and similar.
  Only extract if it actually happened.

machine
  A device the user actively goes to and uses in a session — therapy equipment,
  photobiomodulation devices, treadmills and similar.
  Only if used in this session.
  If an exercise or movement is described as part of a machine session, extract
  the machine only — do not create a separate activity for what was done inside it.
  The sub-exercise detail belongs in the machine's notes, not as its own entity.

device
  A passive monitoring device the user wears or carries.
  Only extract if the user explicitly used it as an independent act today,
  AND it is not merely the source of a measurement already being extracted.

measurement
  A body-level metric with a specific numeric value, recorded from a device or test.
  HRV, weight, blood glucose, blood pressure, SpO2, sleep duration from a tracker,
  and similar objective measurements qualify.
  Subjective scores, exercise performance metrics, and dietary estimates do not.

meal
  An eating event with identifiable content or timing.

theory
  The user's own speculation, causal explanation, or personal interpretation.
  Also use for clinician recommendations or proposed future interventions
  the user is relaying but has not yet acted on.

outside
  Something about the external world — equipment issues, procurement problems,
  missed recordings, provider operational notes.
  Not a health event.

other
  Health-relevant but does not fit the above.

─────────────────────────────────────────
TEMPORAL EVIDENCE
─────────────────────────────────────────

For every mention, assign one of:

explicit_today   — clearly happened today or in this session
                   "took", "did", "just", "this morning", specific clock times

active_regimen   — ongoing habit with no today confirmation
                   "I take", "I've been on", "I usually"

future_plan      — intended but not yet done
                   "going to", "will", "about to", "planning to"

consultation_relay — reporting what a provider said or recommended
                   "he said", "she wants me to", "my doctor suggested"

unclear          — not enough context to determine

─────────────────────────────────────────
EXAMPLES
─────────────────────────────────────────

EXAMPLE 1 — illness context, vague labels, future plans
Transcript:
"[conditionA] is getting worse with a [symptomB]. Tough sleep, woke up way before my alarm.
Did [machineX] at 5:30am. The [scaleY] is broken. Took [supplementA] and a [supplementB].
Going to do an hour of [activityC]. My HRV has been better even though I feel terrible —
maybe [machineX] is affecting it somehow."

Thinking:
- conditionA → happened? yes, currently present. own body? yes. specific? yes → symptom
- symptomB → same → symptom
- "tough sleep, woke up before alarm" → own body? yes. specific trackable label? no,
  this is a sleep observation not a named condition → omit
- machineX at 5:30am → happened? yes. own body/actions? yes. specific? yes → machine, explicit_today
- scaleY broken → own body? no, equipment issue → outside
- supplementA, supplementB → happened? yes. specific? yes → intake, explicit_today
- "going to do activityC" → happened? no, future plan → omit
- "HRV been better, maybe machineX affecting it" → no specific number, user speculation → theory

Result mentions:
[conditionA (symptom), symptomB (symptom), machineX (machine),
scaleY broken (outside), supplementA (intake), supplementB (intake), machineX/HRV theory (theory)]

---

EXAMPLE 2 — general illness feeling, overarching theory, source context
Transcript:
"Starting to wonder if there's two separate things going on because what I had earlier
in the week felt different from this — now it's full-blown [symptomA] and [symptomB].
I caught this from [personX]. The [person] also has it. Sometimes these things can morph
but it just feels awful."

Thinking:
- "starting to wonder if two things" → user speculation → theory
- symptomA → currently present, specific → symptom, explicit_today
- symptomB → same → symptom, explicit_today
- "I caught this from personX" → this is how/from whom the user got sick, not a current
  body state. It is background context, not a trackable symptom → context
- "[person] also has it" → about another person, not the user → omit
- "sometimes these things can morph" → elaboration of theory, already captured → omit
- "just feels awful" → general illness feeling, no specific trackable label → omit

Key distinction: a symptom is a current named body state the user is experiencing.
"I caught X from Y" or "I have X virus" describes the source or cause, not the symptom itself.
The actual symptoms (sneezing, sinusitis etc.) are what get extracted as symptom entities.

Result mentions:
[symptomA (symptom), symptomB (symptom), two-condition theory (theory), caught-from context (context)]

---

EXAMPLE 3 — prodromal symptom, future plan, meta-commentary
Transcript:
"Woke up with that specific sensation right where [anatomical location] — the one that
9 out of 10 times means I'm about to get sick. Note to [name]: we need to think about
how to code something like that. Going to take [testX] today to check. Did the
[activityA] around 6am, first time in a week."

Thinking:
- "sensation at anatomical location" → happened? yes. own body? yes.
  specific? yes — user explicitly describes this as a distinct, personally recognized
  prodromal sign, not a vague feeling → symptom, explicit_today
- "9 out of 10 times means getting sick" → user's own pattern observation → theory,
  linked to symptom
- "Note to [name]: how to code this" → meta-commentary addressed to a third party,
  not a health event → omit
- "going to take testX today" → happened? not yet, future plan → omit
- activityA at 6am → happened? yes. specific? yes → activity, explicit_today

Result mentions:
[prodromal symptom (symptom), predictive pattern (theory), activityA (activity)]"""


def extract_mentions(normalized_text: str) -> list:
    """
    Stage 1: Find all health-relevant mentions in the text.
    Uses tool_use for strict schema-enforced output.
    Does NOT structure or interpret — only finds and classifies.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        system=SYSTEM_PROMPT,
        tools=[MENTION_TOOL],
        tool_choice={"type": "any"},
        messages=[
            {"role": "user", "content": f"Find all mentions in this transcript:\n\n{normalized_text}"}
        ]
    )

    # Extract tool_use block
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None
    )

    if not tool_use_block:
        print("⚠️ extract_mentions: no tool_use block in response")
        return []

    raw_input = tool_use_block.input

    # Validate with Pydantic
    try:
        output = MentionOutput.model_validate(raw_input)
        return [m.model_dump() for m in output.mentions]
    except Exception as e:
        print(f"⚠️ extract_mentions: Pydantic validation error: {e}")
        # Fallback: return raw dicts
        return raw_input.get("mentions", [])