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
    "measurement", "meal", "outcome", "theory", "outside", "other"
]


def _get_knowledge_context() -> str:
    """Inject KB registry into mention prompt so canonical labels are used at extraction time."""
    try:
        from core.db import get_knowledge_for_prompt
        return get_knowledge_for_prompt()
    except Exception:
        try:
            from db import get_knowledge_for_prompt
            return get_knowledge_for_prompt()
        except Exception:
            return ""


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

   EXCEPTION FOR INTAKE OMISSIONS:
   When the user reports NOT taking a specific named substance, a lower confidence
   threshold applies. Phrases like "I don't think I've done X", "I haven't done X",
   "I may not have taken X", "I skipped X" are sufficient to extract as intake
   with action: did_not_take — even if the user's language is slightly uncertain.
   The substance must still be specifically named.
   Examples:
   - "I don't think I've done the tudca in two days" → intake, did_not_take ✅
   - "I may not have taken my HMB" → intake, did_not_take ✅
   - "I haven't done my supplements" → omit (vague group label) ❌

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

  HEALTH INTENT RULE: A substance qualifies as intake only if the user is
  consuming it as a deliberate health decision — a supplement, medication,
  or something chosen specifically for a health effect.
  Everyday food and drink consumed as part of a normal meal or habit do NOT
  qualify as intake, even if mentioned by name.
  Ask: is the user tracking this as a health variable?
  - Supplements, medications, OTC drugs → always intake
  - Caffeine → intake if user is tracking dose or mentions it deliberately
  - Matcha, Hojicha, green tea → intake if consumed with health intent
  - Peppermint tea → intake if user tracks it (e.g. avoids it late for sleep reasons)
  - Mocha, latte, oat milk latte, regular coffee → NOT intake, omit or meal
  - Water, juice, regular food → NOT intake, omit or meal
  The distinction: is this something the user would want in their health record,
  or just something they happened to drink?

  LABEL RULE: The substance must be specifically named.
  Before extracting any intake, apply this test:
  "Does this label refer to a single specific substance that can be named,
  tracked, and compared over time?"
  If yes → extract.
  If no → omit entirely.

  A label fails this test when it describes a collection of unnamed substances,
  regardless of how the user refers to it. Words like "stack", "kit", "pack",
  "bundle", "vitamins", "supplements" combined with a time or purpose descriptor
  are almost always group labels and must be omitted:
  - "bedtime stack", "sleep kit", "morning stack", "wake up stack" → omit
  - "lunch supplements", "evening vitamins", "night pack" → omit
  - "usual supplements", "the stack", "my vitamins" → omit
  This rule applies even when:
  - The user says "took the bedtime stack" (action is clear but label is not resolvable)
  - The user treats it as a defined concept with "the" or "my"
  - The user says they did not take it (did_not_take does not override the label rule)

  Medication category names without a specific drug name also fail this test:
  - "antibiotic" → omit (which antibiotic? not trackable)
  - "painkiller" → omit (which one?)
  - "medication" → omit
  Exception: if the label appears in the knowledge base registry as a canonical
  defined term with known contents, it may be extracted.

symptom
  A clinically named, observable body state that a clinician could write in a chart
  as a finding. Must be specific enough to be tracked and compared over time.
  Ask: could a doctor document this as a clinical finding? If yes → symptom.
  If it is how the user feels in general → omit.

  ABSENCE RULE:
  There are three states for any symptom:
  - explicit present: user says it occurred → extract as symptom, status: present (default)
  - explicit absent: user says it did NOT occur → see below
  - not mentioned: silence → omit entirely (unknown, not captured)

  Explicit absence: extract ONLY if the symptom is a recurring tracked phenomenon —
  something the user monitors regularly across multiple notes (e.g. nocturia, hot flashes,
  a named recurring condition). For these, explicit negation is meaningful data.
  Extract it as a symptom mention with confidence: high.
  Structure.py will assign status: absent.

  Explicit absence for non-recurring, incidental symptoms → omit.
  "No headache today" — headache is not necessarily a tracked recurring phenomenon → omit.
  "No nocturia last night" — nocturia is a recurring tracked phenomenon → extract.

  Retrospective absence interval: if the user says something like
  "I haven't had nocturia for 5-6 days" — extract it. Structure.py will handle the interval.

  If the absence is notable and the user speculates about why → also extract a theory.

  HOW TO IDENTIFY A RECURRING TRACKED PHENOMENON:
  The user treats it as something they monitor over time. Signals include:
  - They mention it across multiple contexts or reference its history
  - They note its absence as meaningful ("first time in weeks", "finally no X")
  - It is a named condition associated with a chronic or recurring health concern
  When in doubt: if the user's phrasing suggests the absence itself is noteworthy → extract.

  CONTEXT-DEPENDENCY TEST — apply this before extracting any symptom:
  Ask: would this finding exist on any other day, in any other context, without the
  specific activity or situation the user just described?
  If yes → symptom (it stands on its own, it is a real body state).
  If no → it is an expected after-effect of an activity, not a symptom. Put it in
  the activity's notes if notable, or omit entirely.

  Examples of this test:
  - Shivering all night from a cold room → would shivering occur without the cold room?
    Yes — shivering is a body response that can occur independently → symptom ✅
  - Feeling cold for 2 hours after a cold plunge → would this occur without the cold plunge?
    No — this is a direct, expected physiological after-effect → NOT a symptom, activity notes ❌
  - Headache in the morning → would a headache occur without any specific trigger?
    Yes — headache is an independent clinical finding → symptom ✅
  - Gas and bloating after a fatty meal → would this occur without the meal?
    Gas itself is a clinical finding that can occur independently → symptom ✅
    (The meal is the suspected cause, not the defining context)

  These do NOT qualify:
  - General feelings: "feeling unwell", "feel awful", "feel terrible", "feel garbage"
  - Energy or motivation: "felt groggy", "exhausted", "low energy", "not feeling it"
  - Sleep difficulty: "hard time sleeping", "woke up early", "tough sleep", "trouble falling asleep"
    Sleep difficulty is never a symptom in this system — even if it could be labeled insomnia.
    Only device-recorded sleep metrics (HRV, sleep duration, deep sleep) are extracted, as measurements.
  - Vague discomfort: "flu-adjacent feeling", "under the weather", "a bit off"
  - Anything the user explicitly dismisses: "probably nothing", "just a bit off"
  - Expected after-effects of activities: cold sensation after cold plunge, muscle burn during exercise

  These DO qualify:
  - Named conditions: sinusitis, nausea, headache, sore throat, chest tightness
  - Localized specific findings actively reported as current complaints: right knee tenderness, muscle fasciculation
  - Observable body responses: flushing, rash, fever, dizziness, shivering (when not activity-induced)
  - Recurring overnight findings: shivering from cold room, nocturia

  Wounds, cuts, and injuries — apply this test:
  Is the user actively reporting this as a current clinical complaint?
  Or are they mentioning it in passing, describing care being done to it, or referencing it as background?
  - "My knee is really painful right now" → symptom ✅
  - "Cut on my foot is still messed up" mentioned while describing something else → context ❌
  - "Someone came and put gauze on my toe" → outside (someone else's action) ❌
  The difference is whether the user is flagging it as a problem they want tracked,
  or simply acknowledging it exists as background information.

  GUT HEALTH TRACKING — always keep in scope:
  Gut-related symptoms are high-value tracking items even when described informally.
  Do not require clinical precision. Preserve the user's own language.
  Always extract as symptom when the user reports any of the following:
  - Bowel frequency ("going 10 times a day", "bathroom every hour")
    → use canonical label: "bowel frequency"
  - Stool quality or consistency ("soft mess", "complete disaster", "not great")
    → use canonical label: "stool consistency"
  - Gas severity or frequency ("terrible gas", "gas all night", "gassy after dinner")
    → use canonical label: "gas"
  - Abdominal or stomach pain ("stomach in so much pain", "gut hurts")
    → use canonical label: "abdominal pain"
  - Bowel urgency ("running to the toilet", "had to go urgently")
    → use canonical label: "bowel urgency"
  These are never throwaway noise — always extract them.

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
  A body-level metric recorded from a device or test.
  HRV, weight, blood glucose, blood pressure, SpO2, sleep duration from a tracker,
  and similar objective measurements qualify.
  Subjective scores, exercise performance metrics, and dietary estimates do not.

  Extract as measurement if:
  - A specific numeric value is stated → value field will be filled
  - Only a directional observation is stated ("HRV is down", "heart rate elevated after 2 AM")
    → still extract as measurement, value will be null, observation goes in notes

  Do NOT invent a value of 0 when no number is stated.
  Do NOT skip the measurement entirely just because no number was given —
  the directional observation is still worth preserving.

intervention
  A multi-session treatment protocol with a defined start and an expected end.
  An intervention spans days or weeks — it is not a single event.

  Ask these two questions:
  1. Does this have a beginning AND an expected end (even if vague)?
  2. Is this a protocol or course — not just a one-time action?
  If both yes → intervention.
  If either no → intake (single dose or substance) or activity (single session).

  These ARE interventions:
  - "I've been on antibiotics for 5 days" → intervention
  - "Started a peptide course" → intervention
  - "Doing a 30-day elimination diet" → intervention
  - "FMT protocol" → intervention (only if multi-session series is implied)

  These are NOT interventions:
  - "Had the FMT treatment yesterday" → intake (single session, no protocol implied)
  - "Took paracetamol" → intake
  - "Did Novothor" → machine (single session)
  - "Got an IV" → intake (single dose)

  When in doubt: if the user describes a single completed event → intake or activity.
  Intervention requires explicit multi-session or protocol language.

outcome
  An observed directional change in a tracked health variable over time,
  linked to an intervention, substance, or activity.
  Ask: is the user reporting that something got better or worse as a result of something else?
  If yes → outcome.

  An outcome is NOT a symptom — it is a change, not a current state.
  "My knee pain is better" alone → symptom (current state, qualifier: better)
  "My knee pain has been improving since I started the peptides" → outcome (directional change linked to intervention)
  "Intestinal health improved since FMT" → outcome
  "HRV has been trending up since I stopped the medication" → outcome

  Only extract if:
  - There is a clear direction (better / worse / improving / declining)
  - It is linked to something (intervention, substance, activity, period of time)
  - The user is reporting a change over time, not just a current state

  Do not extract vague general wellbeing statements as outcomes.
  "Feeling better overall" → omit (not linked to anything specific)

meal
  An eating event. Capture timing and whether it was eaten out.
  If eaten out and a restaurant name is mentioned, capture it.
  If the user describes food content in any detail → still extract as meal.
  The spoken food description will be preserved as a handoff field for the downstream food system.
  Do not attempt to parse or structure the food content here — just extract the meal event.

theory
  The user's own speculation, causal explanation, or personal interpretation
  about their OWN body or health.

  SELF-DIRECTED RULE: Theory must be about the user's own body, symptoms,
  or health outcomes. If the speculation is about a device, system, data quality,
  or is addressed to someone else → outside, not theory.

  These ARE theories:
  - "Maybe the cold plunge affected my HRV" → own body ✅
  - "I think the paracetamol is reducing my nocturia" → own body ✅
  - "Probably stress caused it" → own body ✅
  - "Could be the stem cells" → own body ✅

  These are NOT theories → outside:
  - "Oura data is complete garbage" → device quality, not own body ❌
  - "The team should use Loop instead" → addressed to others ❌
  - "Eight Sleep is more accurate than Oura" → device comparison ❌
  - "The system needs to handle this better" → meta-commentary ❌

  Also use for clinician recommendations or proposed future interventions
  the user is relaying but has not yet acted on.

outside
  Something about the external world — equipment issues, procurement problems,
  missed recordings, provider operational notes.
  Also includes: data quality observations about devices, notes addressed to
  the team or another person, meta-commentary about the system itself.
  Not a health event.

other
  Health-relevant but does not fit the above.

─────────────────────────────────────────
TEMPORAL EVIDENCE
─────────────────────────────────────────

For every mention, assign one of:

explicit_today   — clearly happened today or in this session
                   "took", "did", "just", "this morning", specific clock times
                   IMPORTANT: "explicit_today" means THIS recording session only.
                   "Last night", "the past few nights", "on Tuesday", "last week"
                   are NOT explicit_today — they are past events.
                   Only use explicit_today if the action happened today or
                   is currently happening right now.

active_regimen   — ongoing habit with no today confirmation
                   "I take", "I've been on", "I usually", "every night"
                   Use this when the user describes a pattern without confirming
                   today's specific instance.

future_plan      — intended but not yet done
                   "going to", "will", "about to", "planning to"

consultation_relay — reporting what a provider said or recommended
                   "he said", "she wants me to", "my doctor suggested"

unclear          — not enough context to determine

IMPORTANT: Temporal evidence must be assessed from the mention's own sentence
and immediate context only. It cannot be inherited from a nearby mention.
"The cold plunge and the meditation" — if only cold plunge has "I did", meditation
must be assessed independently. If no signal exists for meditation → unclear.

Past events reported in today's note:
"I took paracetamol the first two nights" → these are past completed facts,
not today's events. Use explicit_today only if it happened today.
If the past event is specific and factual, it can still be extracted —
but temporal_evidence should reflect when it actually happened, not when it was reported.

─────────────────────────────────────────
REASONING FIELD — REQUIRED FOR EVERY MENTION
─────────────────────────────────────────

For every mention you extract, you must fill the reasoning field.
This is your chain of thought — write it before finalising the candidate_type.

For each mention, answer these questions in the reasoning field:
1. Did this actually happen, or is it a plan, recommendation, or reference?
2. Is this about the user's own body or actions?
3. Is the label specific and resolvable enough to track over time?
4. For symptoms: is the user currently experiencing this, or referencing/dismissing it?
   Did the user explicitly say it is "probably nothing", "not really", "not necessarily"?
   Is this an expected after-effect of an activity rather than an independent finding?
5. For intake: is the substance specifically named, or is it a vague group label?
6. What is the temporal evidence, assessed from this mention's own context only?

Example reasoning for a symptom that should be omitted:
"User says 'a little bit sniffly but doesn't feel like sick necessarily, just the air
is super dry' — user explicitly dismisses this as environmental, not a clinical finding.
Omitting."

Example reasoning for a symptom that should be kept:
"User says 'had a mini hot flash in the middle of the night' — specific, named,
clinically recognisable finding. User does not dismiss it. Keeping as symptom."

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
[prodromal symptom (symptom), predictive pattern (theory), activityA (activity)]

---

EXAMPLE 4 — symptom vs activity after-effect vs recurring overnight finding
Transcript:
"Did a cold plunge around 2:45pm, left me really cold for the next couple hours.
Spent most of the night shivering — I don't understand why, room temperature is
usually the same as Tokyo. Did not sleep well, kept waking up cold.
Also had a splitting headache this morning and elevated heart rate.
Gas all night, still worse in the morning."

Thinking:
- cold plunge → happened? yes → activity, explicit_today
- "really cold for next couple hours" → context-dependency test: would cold sensation
  occur without the cold plunge? No — direct expected after-effect → NOT a symptom.
  Put in activity notes if notable.
- "shivering most of the night" → context-dependency test: would shivering occur
  without a specific activity? Yes — shivering is a body response that stands on its own.
  Is it recurring and sleep-disrupting? Yes → symptom, explicit_today
- "did not sleep well, kept waking up cold" → sleep difficulty, never a symptom → omit
- "splitting headache this morning" → named, specific, independent clinical finding → symptom
- "elevated heart rate" → observable body state, clinically documentable → symptom
- "gas all night, still worse in the morning" → specific, clinically named,
  independent of any activity → symptom

Result mentions:
[cold plunge (activity), shivering (symptom), headache (symptom),
elevated heart rate (symptom), gas (symptom)]

---

EXAMPLE 4 — symptom vs activity after-effect vs recurring overnight finding
Transcript:
"Did a cold plunge around 2:45pm, left me really cold for the next couple hours.
Spent most of the night shivering — I don't understand why, room temperature is
usually the same as Tokyo. Did not sleep well, kept waking up cold.
Also had a splitting headache this morning and elevated heart rate.
Gas all night, still worse in the morning."

Thinking:
- cold plunge → happened? yes → activity, explicit_today
- "really cold for next couple hours" → context-dependency test: would cold sensation
  occur without the cold plunge? No — direct expected after-effect → NOT a symptom.
  Put in activity notes if notable.
  reasoning: "Expected physiological after-effect of cold plunge. Context-dependent. Omitting."
- "shivering most of the night" → context-dependency test: would shivering occur
  without a specific activity? Yes — shivering is a body response that stands on its own.
  Is it recurring and sleep-disrupting? Yes → symptom, explicit_today
  reasoning: "Overnight shivering from cold room. Independent of any activity. Clinically trackable. Keeping."
- "did not sleep well, kept waking up cold" → sleep difficulty, never a symptom → omit
  reasoning: "Sleep difficulty observation. Never a symptom in this system. Omitting."
- "splitting headache this morning" → named, specific, independent clinical finding → symptom
  reasoning: "Named condition. Clinically documentable. Not dismissed. Keeping."
- "elevated heart rate" → observable body state, clinically documentable → symptom
  reasoning: "Observable clinical finding. Not dismissed. Keeping."
- "gas all night, still worse in the morning" → specific, clinically named,
  independent of any activity → symptom
  reasoning: "Named clinical finding, ongoing. Not dismissed. Keeping."

Result mentions:
[cold plunge (activity), shivering (symptom), headache (symptom),
elevated heart rate (symptom), gas (symptom)]

---

EXAMPLE 5 — symptom dismissal and absence rule
Transcript:
"Woke up feeling really tired. A little bit sniffly but again doesn't feel like sick
necessarily, just the air is super dry. Had a mini hot flash in the middle of the night.
Was trying to work on some sore muscles with the massage. No nocturia last night."

Thinking:
- "really tired" → general energy state, not a clinical finding → omit
  reasoning: "General fatigue/energy state. Not clinically nameable. Omitting."
- "a little bit sniffly but doesn't feel like sick necessarily, just the air is super dry"
  → user explicitly dismisses this as environmental, not a clinical symptom → omit
  reasoning: "User says 'doesn't feel like sick necessarily, just the dry air'. Explicitly dismissed. Omitting."
- "mini hot flash in the middle of the night" → specific, named, clinically recognisable,
  not dismissed → symptom, explicit_today
  reasoning: "Named clinical finding. Specific timing. User does not dismiss it. Keeping."
- "sore muscles" → this is the reason for the massage, not a currently reported symptom.
  User is referencing a pre-existing state as context for an action → omit
  reasoning: "Mentioned as reason for massage, not as a current active complaint. Not a direct symptom report. Omitting."
- "no nocturia last night" → absence of a symptom is not a symptom → omit
  reasoning: "Absence statement. The non-occurrence of nocturia is not a symptom entity. Omitting."

Result mentions:
[hot flash (symptom)]

---

EXAMPLE 6 — vague group labels, specific vs unresolvable intake
Transcript:
"Did not take the lunch supplements today. All other supplements were taken.
Oh actually no HMB today — I took the morning one but definitely did not get
the dinner one. Late lunch around 2:30pm, late dinner that didn't end until 8:30pm."

Thinking:
- "lunch supplements" → intake? Label test: can this be named specifically?
  No — "lunch supplements" is a group label with no resolvable substance name.
  Even though the user explicitly says did_not_take → omit entirely.
- "all other supplements were taken" → intake? No specific name → omit entirely.
  Cannot create an entity without knowing what substance it refers to.
- HMB morning → specific named substance, took → intake, explicit_today
- HMB dinner → specific named substance, did_not_take → intake, explicit_today
- lunch → meal, explicit_today
- dinner → meal, explicit_today

Result mentions:
[HMB morning (intake), HMB dinner (intake), lunch (meal), dinner (meal)]

Key rule: a vague group label is never acceptable as an intake label,
even when action is did_not_take and even when the user clearly means something real.
If the substance cannot be named, the entity cannot be tracked — omit it."""


def extract_mentions(normalized_text: str) -> list:
    """
    Stage 1: Find all health-relevant mentions in the text.
    Uses tool_use for strict schema-enforced output.
    Does NOT structure or interpret — only finds and classifies.
    """
    # Inject KB registry so canonical labels are available at extraction time
    knowledge = _get_knowledge_context()
    system = SYSTEM_PROMPT
    if knowledge:
        system = SYSTEM_PROMPT + f"\n\n{knowledge}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        system=system,
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