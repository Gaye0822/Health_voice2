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
    "measurement", "meal", "intervention", "outcome", "theory", "outside", "context", "other"
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

   ANATOMICAL LOCATION RULE — for symptoms:
   The anatomical location must be explicitly stated by the user in this sentence
   or its immediate context. Do NOT infer location from surrounding notes or
   previous mentions in the same transcript.
   "definitely swollen today" → location unknown → omit or label as "swelling" only ❌
   "my lymph nodes are swollen" → location explicit → lymph node swelling ✅
   "knees are swollen" → location explicit → knee swelling ✅
   If the user says "swollen" without specifying what → use "swelling" as label,
   do not infer the anatomical site from context.

Only extract a mention if all three questions pass.

─────────────────────────────────────────
RAW_MENTION AND INFERRED_AS FIELDS
─────────────────────────────────────────

raw_mention: Write the canonical or corrected form of the term — what it actually is.
  If you recognize that a transcript word is a mishearing or misspelling of a known
  health term, write the correct form in raw_mention.
  Example: transcript says "paramount" but you recognize it as "paracetamol" →
    raw_mention: "paracetamol"

inferred_as: If you changed the term from what appears in the transcript, write the
  ORIGINAL transcript text here.
  Example: transcript says "paramount", you write raw_mention: "paracetamol" →
    inferred_as: "paramount"
  If no change was made (raw_mention matches transcript exactly) → leave inferred_as null.

This allows the system to track when you made an inference, so it can be verified.

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

  TEMPORAL CONTEXT RULE:
  When the transcript switches between yesterday and today, track when the symptom occurred.
  In your reasoning, explicitly state whether this symptom occurred yesterday or today.
  This helps structure.py set event_date correctly.
  Example: "stomach was in major pain all day yesterday" → reasoning should note "occurred yesterday"

  EXPERIENCED STATE EXCEPTION — applies to symptoms only:
  If the user describes a subjective state they are currently experiencing using
  present-tense language ("it's making me X", "I am X", "I've been feeling X"),
  assign temporal_evidence: explicit_today — even if the cause is an ongoing protocol.
  The distinction: the user is reporting what they are experiencing RIGHT NOW, not
  describing a habit or routine.

  This exception applies ONLY when:
  - The state is a clinically nameable finding (irritability, cognitive impairment, fatigue)
  - The user uses present-tense first-person language about their current state
  - The state is not a device-tracked metric (HRV, heart rate → never symptom)

  Examples:
  "it's making me super irritable" → symptom: irritability, explicit_today ✅
  "I am a complete mess" → too vague, omit ❌
  "I've been irritable all week" → active_regimen (pattern, not present moment) ❌
  "it's spiking my heart rate" → device metric, never symptom ❌
  "I take HMB every morning" → intake habit, active_regimen — NOT this exception ❌

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

  SELF-LABELING RULE — apply this before extracting any symptom:
  If the user themselves cannot name what they experienced →  OMIT.
  Signals that the user cannot label it:
  - "I don't know how to put it"
  - "some kind of reaction"
  - "I don't know what's going on"
  - "some weird stuff"
  - "something happened"
  - "I don't know what to call it"

  Do NOT invent a clinical label on the user's behalf.
  A symptom must come from the user's own description, not from LLM inference.

  Similarly, if the user uses informal vague language that could map to multiple
  clinical concepts → do NOT assign a clinical label. Omit entirely.
  "freak out" → could be panic, anxiety, dysautonomia, anger — do not label → OMIT
  "some reaction" → unresolvable → OMIT
  "weird stuff" → unresolvable → OMIT

  EMBEDDED DETAIL RULE:
  If a finding appears as a detail or qualifier inside a vague episode description
  rather than as a standalone clinical complaint → OMIT.
  Do NOT extract the detail as a symptom when the episode itself was omitted.
  "during the freak out there were temperature changes" →
    temperature changes is a detail of an unlabelable episode → OMIT
  "some weird reaction, felt hot and dizzy" →
    heat and dizziness are details of an unresolvable episode → OMIT
  The test: would this finding be reported on its own, without the episode? 
  If no → OMIT.

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
  - Sleep difficulty: "hard time sleeping", "woke up early", "tough sleep", "trouble falling asleep",
    "wide awake at night", "couldn't sleep", "insomnia", "couldn't fall asleep", "kept awake"
    Sleep difficulty is NEVER a symptom in this system — this is an ABSOLUTE rule with no exceptions.
    Even if the user describes it with clinical language like "insomnia", or gives a specific time
    ("wide awake at 10 PM"), or adds qualifiers ("chemical-like", "not jittery") → OMIT.
    Only device-recorded sleep metrics (HRV, sleep duration, deep sleep) are extracted, as measurements.

    SLEEP DIFFICULTY HARD STOP — label does not matter, phenomenon does:
    Before assigning symptom to any mention, ask:
    "Is this fundamentally about the user being awake when they want to sleep,
    or having difficulty initiating or maintaining sleep?"
    If yes → OMIT. This is sleep difficulty regardless of the label chosen.
    "insomnia" → sleep difficulty → OMIT
    "wakefulness" → sleep difficulty → OMIT
    "hyperarousal" → sleep difficulty → OMIT
    "nocturnal alertness" → sleep difficulty → OMIT
    The label does not change the phenomenon. If the underlying experience
    is difficulty sleeping → it is never a symptom in this system.
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
  A body-level metric with an objective numeric value, not captured by an automatic device pipeline.

  Before extracting any measurement, work through these three questions in order:

  1. IS THIS FROM A WEARABLE OR MONITORING DEVICE?
     Signals: "Oura says", "Eight Sleep tracked", "according to WHOOP", "my HRV",
     "sleep score", "Nightly Recharge", "SpO2 from Oura", "HRV from ring"
     Device-sourced metrics: HRV, heart rate, sleep score, deep sleep, REM,
     SpO2, body temperature, recovery score, readiness score
     If clearly device-sourced → omit. The device pipeline handles this.

  2. IS THIS A SUBJECTIVE QUALITY JUDGMENT WITH NO NUMBER?
     Signals: "awful", "crappy", "in the cellar", "miserable", "terrible",
     "elevated" without a value, "down" without a value, "better", "worse"
     If yes → omit. This is narrative commentary, not a measurement.

  3. IS THIS AN OBJECTIVE VALUE NOT CAPTURED ELSEWHERE?
     The owner states a specific number AND it does not come from an automatic device:
     - Weight read from a scale: "82.5 kg" ✅
     - Blood glucose from finger prick: "glucose 95" ✅
     - Lumen score read manually: "Lumen score 4.5" ✅
     - Lab result stated aloud: "vitamin D 45 ng/ml" ✅
     - Doubly labeled water result: "3300 kcal/day" ✅
     If yes → extract as measurement.

  Only extract if question 3 applies and questions 1 and 2 do not.
  If no specific number is stated → omit entirely. Do not extract directional
  observations ("HRV is down", "heart rate elevated") — these are device data
  the pipeline already has access to.

intervention
  Signal only — duration, date, and protocol validity are evaluated by the intervention
  pipeline, not here. Your only job is to detect whether the user is referring to a
  multi-session treatment, course, or protocol for their own body.

  Assign candidate_type: intervention if:
  - The user mentions a course, protocol, or treatment spanning multiple days/sessions
  - The user has already started it and is actively undergoing it
  - It applies to the user's own body

  THE KEY SIGNAL IS WHETHER THE USER HAS STARTED — not who initiated it.
  A provider may have recommended the protocol, but if the user has begun taking it,
  it is an intervention signal. "Majdi wanted me on X for two weeks, I started three days ago"
  → the user started → intervention ✅
  "Majdi wants me to start X next week" → not yet started → future_plan ❌

  temporal_evidence rules apply normally:
  - future_plan → if not yet started
  - active_regimen → if ongoing but no today-specific language
  - explicit_today / explicit_past → if a specific dose or session is mentioned today or recently

  These ARE intervention signals:
  - "I've been on antibiotics for 5 days" → intervention
  - "Started a peptide course" → intervention
  - "Doing a 30-day elimination diet" → intervention
  - "FMT protocol, three sessions so far" → intervention
  - "I'm on day 4 of rifaximin" → intervention
  - "Majdi wanted me on metformin for two weeks, I started three days ago" → intervention
  - "Doctor put me on a 10-day course, took my second dose this morning" → intervention

  These are NOT intervention signals:
  - "Had the FMT treatment yesterday" → intake (single session, no protocol implied)
  - "Took paracetamol" → intake
  - "Did Novothor" → machine (single session)

  DUAL EXTRACTION RULE — intervention + intake birlikte:
  If the user mentions a protocol AND reports taking a specific dose today or recently,
  extract BOTH:
  - intervention mention for the protocol signal
  - intake mention for the dose event

  The intervention captures the protocol context.
  The intake captures the actual dosing event.

  Examples:
  - "SS-31, that's a 20-injection course, did number seven this morning" →
    intervention (SS-31, protocol signal) + intake (SS-31, took, this morning) ✅
  - "I'm on day 4 of rifaximin, took it an hour ago" →
    intervention (rifaximin) + intake (rifaximin, took) ✅
  - "I've been on antibiotics for 5 days" →
    intervention only — no specific dose event mentioned today ✅
  - "Took paracetamol last night" →
    intake only — no protocol implied ✅

  The test: is there a specific dose event (took it, did it, injected it, this morning/tonight)?
  If yes AND protocol context exists → extract both.
  If only protocol context, no dose event → intervention only.
  If only dose event, no protocol → intake only.

  CRITICAL — PRESCRIPTION ANTIBIOTICS AND MEDICATIONS:
  When the user mentions a prescription medication with a total duration or dose count,
  this is ALWAYS an intervention signal — even if the primary sentence is about a single dose.
  Do not let the intake action (took, didn't take) cause you to miss the protocol signal.

  "I took my second 100mg dose, I'm doing 10 days at 100mg" →
    intervention (doxycycline, 10-day course) + intake (doxycycline, took, last night) ✅
  "Last night around 9 PM I took my second hundred mg dose, I'm just gonna do 10 days at 100mg" →
    intervention (doxycycline) + intake (doxycycline, took, last night) ✅
    The protocol signal "10 days at 100mg" makes this an intervention regardless of
    the fact that a specific dose is also being reported.

  DUAL EXTRACTION CHECKLIST — run this before finalizing any intake mention:
  1. Is there a total duration or dose count? ("10 days", "20 injections", "a month") → intervention signal
  2. Is there a day number? ("day 4", "second dose", "third session") → intervention signal
  3. Is there "course", "protocol", "until told otherwise"? → intervention signal
  If ANY of these is present → extract intervention mention IN ADDITION to intake mention.

  Do NOT apply duration checks, coexistence rules, or any other protocol validation here.
  Extract the signal and move on — the intervention pipeline handles everything else.

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

  TEMPORAL CONTEXT RULE FOR MEALS:
  When the transcript switches between yesterday and today, track which meals belong to which day.
  If the surrounding narrative is about yesterday (the user starts with "yesterday I..." and then
  describes activities and meals), those meals happened yesterday.
  In your context field, include the word "yesterday" if the meal belongs to yesterday's narrative.
  In your reasoning, explicitly state whether this meal happened yesterday or today.
  Example: "yesterday I did X... then breakfast at 10:15, lunch at 2, dinner at 6... this morning breakfast"
  → breakfast/lunch/dinner belong to yesterday → context should reflect this
  → today's breakfast context should say "this morning"

  FASTING RULE:
  Fasting is not an activity and not an intake — it is context.
  "I fasted until 2:30 PM", "still fasted this morning", "fasting day" → candidate_type: context, related_to: meal timing
  Do NOT extract fasting as activity, intake, other, or any other event type.
  Exception: if the user describes a structured fasting protocol as a multi-day intervention
  ("I'm doing a 3-day water fast") → intervention.

  STOOL CONSISTENCY RULE:
  Only extract stool consistency as a symptom when the user uses a specific descriptive qualifier.
  The user must explicitly describe the quality, texture, or state of their stool.
  Examples that qualify: "soft mess", "complete disaster", "sloppy", "messed up", "watery",
  "loose", "terrible", "awful", "a disaster"
  Examples that do NOT qualify:
  - "you wouldn't know it from the stool" → vague indirect reference → omit
  - "stool is not great" → too vague → omit
  - "stool situation" without description → omit
  The qualifier must be the user's own words describing the stool's actual state.

theory
  The user's own speculation, causal explanation, or personal interpretation
  about their OWN body or health.

  CRITICAL RULE — USER'S OWN WORDS ONLY:
  A theory must come directly from a sentence where the user themselves speculates.
  Do NOT infer, construct, or connect a theory from context.
  If the user does not explicitly speculate in their own words → no theory.

  The test: can you find the exact sentence where the user speculates?
  If yes → extract that speculation as theory.
  If no → omit. Do not invent a theory the user did not state.

  "I don't understand what's going on" → NOT a theory. User is expressing confusion,
  not speculating. Do not construct a causal theory from surrounding context.
  "Maybe the cold plunge affected my HRV" → theory ✅ User's own speculation.
  "Probably stress caused it" → theory ✅ User's own words.

  SELF-DIRECTED RULE: Theory must be about the user's own body, symptoms,
  or health outcomes. If the speculation is about a device, system, data quality,
  or is addressed to someone else → outside, not theory.

  DEVICE METRIC RULE — NEVER extract theories about HRV, heart rate, SpO2,
  sleep score, or any other automatically device-tracked metric.
  These metrics already exist in a stronger device fact layer.
  A note-derived theory about them creates a competing weaker version — do not do this.
  "new medication could be affecting HRV and heart rate" → OMIT ❌
  "maybe cold plunge caused HRV drop" → OMIT ❌
  "HRV has reached bottom, was previously inflated" → OMIT ❌
  If the speculation is about a non-device symptom and HRV appears only as
  supporting context → omit the HRV reference, keep the rest if it stands alone.

  HOPE/EXPECTATION RULE — NEVER extract hopes or expectations as theories:
  "hoping this resolves", "fingers crossed", "I hope X goes away" → OMIT ❌
  A theory requires an explicit causal or mechanistic claim, not a wish.

  These ARE theories:
  - "I think the paracetamol is reducing my nocturia" → own body, causal claim ✅
  - "Probably stress caused it" → own body ✅
  - "Could be the stem cells" → own body ✅
  - "no normal human being gets Demodex every six months — something very wrong" → own body ✅

  These are NOT theories → OMIT or outside:
  - "Maybe cold plunge affected my HRV" → device metric → OMIT ❌
  - "new medication could be affecting HRV" → device metric → OMIT ❌
  - "hoping vertigo is resolved" → hope, not speculation → OMIT ❌
  - "Oura data is complete garbage" → device quality → outside ❌
  - "The team should use Loop instead" → addressed to others → outside ❌

  Also use for clinician recommendations or proposed future interventions
  the user is relaying but has not yet acted on.

outside
  Something about the external world — equipment issues, procurement problems,
  missed recordings, provider operational notes.
  Also includes: data quality observations about devices, notes addressed to
  the team or another person, meta-commentary about the system itself.
  Not a health event.

  OUTSIDE IS ONLY FOR THINGS EXTERNAL TO THE USER'S BODY.
  If something is about the user's own body or health — even if it cannot be
  extracted as a symptom, intake, or any other type — it is NOT outside.
  Omit it entirely rather than routing it to outside.

  EXCEPTION — QUESTIONS DIRECTED AT THE TEAM OR EXTERNAL PARTIES:
  If the user asks a question addressed to another person or the team about
  their own health — even if the subject is their body — extract as outside,
  subtype: operational.
  The key signal: is the user asking someone else to do something or look something up?
  "Is there anything in the literature about X?" → outside, operational ✅
  "Can you check whether Y affects Z?" → outside, operational ✅
  These are requests for external action, not health event recordings.

  outside is NOT a catch-all or fallback category. Do not use it for:
  - General health feelings or states ("exhaustion pattern", "feeling off")
  - User's speculation about their own body → theory
  - Vague health observations that don't fit other types → omit

  ALWAYS extract as outside when:
  - User describes a device malfunctioning or two devices disagreeing
  - User addresses the team, another person, or the system
  - User mentions procurement, logistics, or operational matters

  EMBEDDED DEVICE ISSUES — do not miss these:
  Device problems often appear as subordinate clauses inside sentences primarily
  about something else. Always scan the full sentence for device failure signals,
  even when the main topic is an activity or measurement.
  Examples:
  "zone two is recorded in WHOOP because again polar wasn't working for whatever reason"
  → main topic: zone two activity. But "polar wasn't working" = device_failure → extract outside ✅
  "used Eight Sleep because Oura was dead" → Oura failure embedded in Eight Sleep mention → outside ✅
  "WHOOP picked it up since polar was off" → polar failure = outside ✅
  Do not let the surrounding activity context cause you to skip the device issue.

  DEVICE_FAILURE vs SOURCE_DISCREPANCY — answer this before assigning subtype:
  "Is there a specific, documented, quantified malfunction?"

  device_failure: one device is clearly wrong in a concrete, measurable way.
    "Oura starts recording sleep at midnight when I fell asleep at 10:30" → device_failure ✅
    "Polar strap jumping all over the place, no heart rate reading at all" → device_failure ✅
    The malfunction is specific and quantified — not a matter of interpretation.

  source_discrepancy: two devices report different values but neither is clearly wrong.
    "Eight Sleep and Oura showing different deep sleep since PONS" → source_discrepancy ✅
    "WHOOP shows different HRV than Oura" → source_discrepancy ✅
    "Eight Sleep more accurate than Oura in my experience" → source_discrepancy ✅
    These are reliability concerns or disagreements — not confirmed failures.

  When in doubt → source_discrepancy, not device_failure.

  Do NOT omit device issues just because they are not about the user's body.
  That is exactly why they are outside — they are about the external world.

  TEAM ACTION ITEMS — always extract as outside, subtype: operational:
  When the user uses "we need to", "we should", "we have to", "we must"
  directed at a collective that includes someone other than themselves —
  this is a team action item, not a health event.

  Extract as outside even when the subject matter is the user's own health.
  The key distinction: the user is not recording a health event, they are
  assigning a task or making a plan for others to act on.

  Examples:
  "we need to complete the testing on the inhalation device" → outside, operational ✅
  "we need to figure out a way to repair my microbiome" → outside, operational ✅
  "we really need to read into the mechanisms of the PEMF device" → outside, operational ✅
  "we should keep routine testing" → outside, operational ✅

  Do NOT omit these — they are explicit team directives and have operational value.
  Do NOT extract as theory — there is no speculative claim, only a directive.
  Do NOT extract as context — they are not explaining another entity.

  IMPORTANT: A team action item and a theory can coexist in the same sentence
  or adjacent sentences. Extract both separately.
  "my stool is a mess — could be the microbiome — we need to fix this" →
    theory: microbiome → stool issues ✅
    outside: microbiome repair directive ✅
  Do NOT let the action item cause you to skip the adjacent theory.
  Always scan the full surrounding context for speculative language even when
  a team directive is present.

context
  Background information necessary to understand another entity in this note.

  Before extracting as context, answer BOTH questions:
  1. "This information is necessary to understand another entity in this note."
     If yes → context candidate. If no → outside or omit.
  2. "Which specific entity in this note does this explain?"
     If you can identify a specific entity → related_to can be filled → context ✅
     If you cannot identify a specific entity → NOT context. Route to outside or omit.

  A context mention with no identifiable related_to is not context.

  Examples:
  "Switched to carbs because stomach is a mess" → explains abdominal pain entity → context ✅
  "Fasted until 2:30 PM" → explains meal timing → context ✅
  "Lactic device delivered, needs calibration" → which entity does this explain? None → outside ❌
  "Resting heart rate has been fixed" → which entity does this explain? None in this note → omit ❌
  "Caught this from Yuki" → explains illness context → context ✅

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

explicit_past    — happened on a specific named past day, not today
                   "yesterday", "last night", "on Friday", "two days ago", "last week"
                   Use this when the user clearly describes something that happened
                   before today's recording session.
                   REQUIRED: also fill event_date_label with the user's own words.
                   Examples:
                   "yesterday I did a cold plunge" → explicit_past, event_date_label: "yesterday"
                   "had the FMT treatment last Friday" → explicit_past, event_date_label: "last Friday"
                   "took paracetamol the first two nights" → explicit_past, event_date_label: "the first two nights"
                   "my stomach was in pain all day yesterday" → explicit_past, event_date_label: "yesterday"
                   CRITICAL: Do NOT use explicit_past for the current recording session.
                   "Last night" as part of today's report (e.g. sleep last night) may still
                   be explicit_today if the user is describing a device-tracked overnight period
                   that is part of today's health session. Use judgment based on context.

                   RETRACTION / COMPARISON EXCEPTION:
                   "yesterday" does not always mean the event happened yesterday.
                   Watch for these patterns before assigning explicit_past:

                   - Retraction: "I take back X as yesterday", "to correct what I said yesterday"
                     → user is retracting a prior statement. The finding is current → explicit_today
                   - Comparison: "more swollen than yesterday", "worse than yesterday",
                     "better than yesterday", "same as yesterday"
                     → "yesterday" is a comparison point, not when the event occurred → explicit_today
                   - Contrast: "unlike yesterday", "not like yesterday"
                     → current state being contrasted with yesterday → explicit_today

                   Only use explicit_past when "yesterday" or a past day name is the PRIMARY
                   temporal anchor for WHEN the event occurred — not when it appears as a
                   comparison point, retraction signal, or contrast marker.

                   Examples — explicit_today (NOT explicit_past):
                   "swollen lymph nodes, more swollen than yesterday" → explicit_today
                   "I take back what I said, they're more swollen than yesterday" → explicit_today
                   "feeling worse than yesterday" → explicit_today
                   "better than I was yesterday" → explicit_today

                   Examples — explicit_past (event happened yesterday):
                   "yesterday I did a cold plunge" → explicit_past, event_date_label: "yesterday"
                   "had terrible gas all day yesterday" → explicit_past, event_date_label: "yesterday"
                   "my stomach was in pain yesterday, it's fine now" → explicit_past, event_date_label: "yesterday"

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
"I took paracetamol the first two nights" → these are past completed facts.
Use explicit_past (not explicit_today). Fill event_date_label: "the first two nights".
If the past event is specific and factual, it can still be extracted —
temporal_evidence must reflect when it actually happened, not when it was reported.

─────────────────────────────────────────
REASONING FIELD — REQUIRED FOR EVERY MENTION
─────────────────────────────────────────

For every mention, the reasoning field MUST follow this structured checklist format.
Work through each step in order. Do not skip steps. The final type must follow from the checklist.

STEP 1 — ELIMINATE outcome:
Answer all three:
- Is this about workout performance (strength, endurance, speed, fitness)? → If yes: OMIT
- Is this about a device-tracked metric (HRV, sleep, heart rate, SpO2)? → If yes: OMIT
- Can I complete "Because of [X], [Y] has changed and this is an established fact"
  with a specific X from this note? → If no: OMIT or theory

If outcome survives step 1 → proceed to step 2 outcome tests (language, construct, device, linked_to).
If eliminated → note why and continue to next step.

STEP 2 — ELIMINATE symptom:
Answer all:
- Is this general energy, fatigue, weakness, or sleep difficulty? → If yes: OMIT
- Did the user explicitly dismiss this? → If yes: OMIT
- Is this an expected after-effect of an activity? → If yes: OMIT (activity notes)
- Is the user using causal or explanatory language rather than reporting a finding?
  "X is tied to Y", "X is linked to Y", "X is connected to Y", "X is related to Y"
  → user is explaining a relationship, not reporting a symptom → OMIT or theory
- Is the user using a condition as an analogy, comparison, or pain scale reference
  rather than reporting that they have it?
  "pain like a migraine", "lymph nodes and migraines are super painful" (used as reference),
  "like a knife through my leg" → these are rhetorical comparisons, not findings → OMIT
  The test: is the user saying they currently have this condition, or using it to describe
  the intensity of something else? If the latter → OMIT entirely.
- Is this a named clinical finding the user is actively reporting as a problem? → If no: OMIT

If symptom survives step 2 → extract as symptom.
If eliminated → note why and continue.

STEP 3 — ELIMINATE theory:
- Did the user make an explicit speculative claim in their own words? → If no: OMIT
- Is this confusion, hope, dismissal, or emotional reaction? → If yes: OMIT
  HARD STOP — these are NEVER theories, no exceptions:
  "hoping that was all in the past" → hope → OMIT
  "I hope this goes away" → hope → OMIT
  "fingers crossed" → hope → OMIT
  "I don't understand what's going on" → confusion → OMIT
  "God knows", "who knows" → emotional reaction → OMIT

- CONTEXTUAL REFERENCE RULE FOR THEORIES:
  When the user uses "that", "it", "this", or similar pronouns in a speculative sentence,
  look back in the transcript to identify what the pronoun refers to.
  Do NOT omit a theory just because the cause (X) is a pronoun — resolve it first.

  "I thought maybe that was hurting when I sleep" →
    "that" = previously mentioned activity (e.g. late afternoon lifting) →
    X = late lifting, Y = sleep disruption → theory ✅

  Always scan the surrounding 2-3 sentences for the pronoun's referent before
  concluding that X is missing. If the referent is identifiable → extract the theory.
  If the referent genuinely cannot be resolved → OMIT.

- MANDATORY CAUSAL CLAIM TEST — before extracting any theory, verify TWO things:

  PART A — Two distinct entities required:
  The speculation must involve a CAUSE (X) and an EFFECT (Y) — two different things.
  Complete this sentence: "The user is claiming that [X] might/could/probably cause or explain [Y]."
  X and Y must be different entities. If only one entity is present → OMIT.

  These FAIL because only one entity is described (no X→Y structure):
  "HRV has reached bottom" → only HRV described, no cause → OMIT
  "we knew HRV was inflated" → past fact about HRV only, no cause → OMIT
  "heart rate is very low" → observation about one thing, no cause → OMIT
  "irritability is down" → state description, no cause → OMIT

  These PASS because X causes Y:
  "peptides could be causing nocturia through kidney effects" → X=peptides, Y=nocturia ✅
  "maybe the cold plunge is affecting my lymph nodes" → X=cold plunge, Y=lymph nodes ✅
  "I think the antibiotic is reducing the eye swelling" → X=antibiotic, Y=eye swelling ✅

  PART B — Must be user's own speculative language:
  The user must use hedging/speculative words: "maybe", "could be", "I think", "probably",
  "I don't know if", "might be", "I guess", "I wonder if".
  Factual statements, past observations, and retrospective summaries are NOT theories
  even if they describe something uncertain.
  "we knew that was inflated" → stated as known fact → OMIT
  "it seems to have reached the bottom" → observation, no speculation → OMIT
  "maybe the peptide is causing this" → explicit speculation ✅

  If BOTH parts pass → extract as theory.
  If EITHER part fails → OMIT.

  Examples:
  "new peptides could be that with what they do to the kidneys" →
    Part A: X=peptides, Y=nocturia (two entities) ✅
    Part B: "could be" = speculative language ✅ → theory
  "let's hope it's the HMB" →
    Part A: X=HMB, Y=weight gain (two entities) ✅
    Part B: "let's hope" = hope, NOT speculation → HARD STOP → OMIT
  "I don't understand how my heart rate is so low" →
    Part A: no X identified → OMIT
  "HRV has reached the bottom" →
    Part A: only HRV described, no cause → OMIT
  "we knew that was inflated" →
    Part B: stated as known fact, no speculative language → OMIT
  "PEMF is the most powerful thing I've done this year" →
    Part B: superlative evaluation, no speculative language, no causal claim → OMIT
  "this is the best protocol I've tried" →
    Part B: evaluation/ranking, not speculation → OMIT

- DEVICE METRIC RULE:
  HRV, heart rate, SpO2, sleep score = device-tracked. The device pipeline already
  has this data as fact. Do NOT extract theories that are ONLY about these metrics.
  "HRV has reached bottom" → device observation → OMIT ❌
  "new medication could be affecting HRV and heart rate" → device metrics only → OMIT ❌

  EXCEPTION — causal speculation that INCLUDES device metrics but is really about
  a non-device phenomenon may still be extracted:
  "new peptides could be causing nocturia through kidney effects" → about nocturia (non-device) ✅
  Keep the theory, but do NOT include HRV/heart rate in the raw_text if they appear
  only as supporting evidence alongside the real claim.

- Is this logistical planning, scheduling, or procurement? → If yes: outside, not theory
  "I'm thinking of going to Tokyo for a blood test", "should I see the dermatologist",
  "I need to find a clinic" → operational/procurement → outside
- Is this about the user's own body/health? → If no: outside

If theory survives step 3 → extract as theory.

STEP 4 — ASSIGN final type:
State: "Final type: [type] because [one sentence reason]"
If none of the above fit → omit entirely.

─────────────────────────────────────────
CHECKLIST EXAMPLES
─────────────────────────────────────────

"lost a ton of strength after weeks of illness"
Step 1 eliminate outcome: workout performance domain → OMIT. Do not extract.
Final type: omit — subjective workout performance in empirically tracked domain.

"stool is a complete disaster"
Step 1 eliminate outcome: not performance or device metric, no linked_to cause → not outcome.
Step 2 eliminate symptom: named clinical finding? Yes — stool consistency is gut-tracked. Active report? Yes. Not dismissed. Not after-effect. → symptom ✅
Final type: symptom — stool consistency, actively reported gut health finding.

"I think I've got a second bug"
Step 1: not outcome.
Step 2: not a named clinical finding, it's speculation → not symptom.
Step 3: explicit speculation "I'm still pretty sure" about own body → theory ✅
Final type: theory — user speculates about second infection.

"I've been exhausted for days"
Step 1: not outcome.
Step 2: general fatigue/energy state → Filter 1 triggers → OMIT.
Final type: omit — general energy state, not a clinical finding.

"keep waking up super early"
Step 1: not outcome.
Step 2: sleep difficulty → never a symptom in this system → OMIT.
Final type: omit — sleep difficulty observation.

─────────────────────────────────────────
ADDITIONAL RULES
─────────────────────────────────────────

4. For symptoms — after checklist, confirm:
   - Is this an expected after-effect of an activity? → omit
   - Is this vague subjective state rather than named clinical finding? → omit

5. For intake: is the substance specifically named, or is it a vague group label?
   Is there same-session confirmation it was taken today, or is this active_regimen/speculation?

6. For theory candidates — complete this sentence first:
   "The user is speculating about their OWN body or health in THIS note."
   If the sentence feels natural and true → extract as theory.
   If it feels forced → outside or omit.

   A theory requires the user to make an explicit speculative claim.
   These expressions are NOT theories — omit them:
   - Confusion: "I don't understand what's going on" → not speculation, omit
   - Self-dismissal: "I guess I'm just paranoid", "probably nothing" → not speculation, omit
   - Hope/expectation: "I hope this will go away" → not speculation, omit
   - Emotional reaction: "God knows", "who knows" → not speculation, omit

   These ARE theories — the user makes an explicit speculative claim:
   - "Maybe cold plunge affected my HRV" → explicit speculation ✅
   - "Probably stress caused the nocturia" → explicit causal claim ✅
   - "Could be the stem cells" → explicit candidate cause ✅
   - "I think I'm on the wrong antibiotic" → explicit assessment ✅
   - "I don't know how it could be infected, nothing's been in there" → dismissing a possibility,
     this IS a theory — user is evaluating and ruling out a cause ✅

   Outside examples:
   - "Oura data is complete garbage" → device quality ❌ → outside
   - "The team should use Loop instead" → addressed to others ❌ → outside

7. For outcome candidates — complete this sentence first:
   "Because of [linked_to], [what] has changed and this is an established fact."
   If the sentence feels natural and true → proceed to tests a, b, c below.
   If it feels forced, uncertain, or refers to a past completed event → theory or omit.

   Examples:
   "Because of paracetamol, nocturia has changed and this is an established fact." ✅ → outcome
   "Because of selegiline, language recognition has changed and this is an established fact." ❌ → not measurable, theory
   "Because of craniosacral, right knee got better last week and this is an established fact." ❌ → past event, omit
   "Because of PONS, deep sleep jumped on Eight Sleep and this is an established fact." ❌ → device data, omit

   Then confirm all three:
   a. Is the language observational with no hedging? ("tends to", "for years", "consistently")
      If "apparently", "maybe", "I think", "could be" is present → theory, not outcome
   b. Is the "what" a trackable, measurable construct with a clear baseline?
      "nocturia", "stool consistency", "body weight", "blood glucose" → yes
      "language recognition", "mental stuff", "psychological benefits", "energy", "how I feel" → no
      If the construct is not measurable → theory, not outcome
   c. Is the "what" something the device pipeline already tracks automatically?
      "deep sleep", "HRV", "heart rate", "sleep score", "SpO2" → device-tracked → omit
   d. Can you fill [linked_to] with a specific cause from this note?
      linked_to must be an intake, activity, machine, or intervention in this note.
      If no specific cause is identifiable → omit. An outcome without a cause is not an outcome.
      "Slightly better this morning" with no stated cause → omit.
      linked_to cannot be a symptom, context, or theory — only something the user did or took.
   All four must pass. If any fails → theory or omit.
7. What is the temporal evidence, assessed from this mention's own context only?

Example reasoning for a symptom that should be omitted:
"User says 'a little bit sniffly but doesn't feel like sick necessarily, just the air
is super dry' — user explicitly dismisses this as environmental, not a clinical finding.
Omitting."

Example reasoning for an outcome that should be theory instead:
"User says 'definitely starting to feel the upsweep in language recognition as a result
of selegiline' — language is observational (passes test 1) but 'language recognition'
is not a defined trackable construct with a baseline (fails test 2). Making theory."

Example reasoning for a valid outcome:
"User says 'paracetamol tends to reduce my nocturia, been a trend for years, reduces
odds by 50%' — observational language, no hedging (passes test 1). Nocturia is tracked
across multiple notes, binary and comparable (passes test 2). Keeping as outcome."

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
- "HRV been better, maybe machineX affecting it" →
  Causal claim test: "user claims machineX might cause HRV improvement" → sentence completes.
  BUT: HRV is device-tracked → device metric rule → OMIT.

Result mentions:
[conditionA (symptom), symptomB (symptom), machineX (machine),
scaleY broken (outside), supplementA (intake), supplementB (intake)]

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

EXAMPLE 6 — vague group labels, specific vs unresolvable intake, multi-dose splitting
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
- HMB morning → specific named substance, took → intake, explicit_today, dose_timing: "morning"
- HMB dinner → specific named substance, did_not_take → intake, explicit_today, dose_timing: "dinner"
- lunch → meal, explicit_today
- dinner → meal, explicit_today

Result mentions:
[HMB morning (intake, took, dose_timing: morning), HMB dinner (intake, did_not_take, dose_timing: dinner),
lunch (meal), dinner (meal)]

Key rule: a vague group label is never acceptable as an intake label,
even when action is did_not_take and even when the user clearly means something real.
If the substance cannot be named, the entity cannot be tracked — omit it.

---

EXAMPLE 6b — multi-dose splitting: "the other two" and similar plural dose references
Transcript:
"Yesterday I forgot my HMB — I always take it in the morning so that never gets
forgotten, but I mean I forgot the other two. I was out for lunch and then forgot
to take it when I went to bed."

Thinking:
- "I always take it in the morning... never gets forgotten" → active_regimen pattern.
  BUT: this is said in the context of explaining which doses were missed yesterday.
  The morning dose is implied as taken but via indirect negation — NOT explicit enough
  to extract as took. Do not extract dolaylı teyitleri.
- "forgot the other two" → MULTI-DOSE SPLIT RULE:
  When the user says they missed multiple doses of the same substance,
  extract each dose as a SEPARATE mention using the time context provided.
  Here the user says "out for lunch" and "forgot when I went to bed" →
  two distinct dose slots are identifiable:
  → HMB lunch dose: did_not_take, explicit_past, event_date: "yesterday", dose_timing: "lunch"
  → HMB dinner/bedtime dose: did_not_take, explicit_past, event_date: "yesterday", dose_timing: "bedtime"

MULTI-DOSE SPLIT RULE — apply whenever:
- User says "the other two", "both doses", "neither of them", "all three" etc.
- AND the substance is specifically named
- AND time context exists to identify each dose slot (lunch, dinner, bedtime, morning)
Extract one mention per dose slot. Do NOT collapse into a single mention.
If time context is missing for some doses → use dose_timing: null for those,
but still split into separate mentions.

Result mentions:
[HMB lunch dose (intake, did_not_take, yesterday, dose_timing: lunch),
 HMB bedtime dose (intake, did_not_take, yesterday, dose_timing: bedtime)]

---

EXAMPLE 7 — measurement extraction: device data vs objective values
Transcript:
"My HRV went down the tube after 2 AM, heart rate was elevated all night.
Weight this morning 82.5 kg. Lumen score was a 4 and a half, awful result.
Blood glucose came back at 95 from the lab."

Thinking:
- "HRV went down the tube after 2 AM" → measurement reasoning:
  1. Device-sourced? YES — HRV comes from Oura/WHOOP automatically
  2. Subjective? Also yes — "went down the tube" is narrative
  3. Objective value not captured elsewhere? NO
  → OMIT. Device pipeline already has this.
  reasoning: "HRV is device-tracked. No numeric value. This is narrative commentary. Omitting."

- "heart rate elevated all night" → measurement reasoning:
  1. Device-sourced? YES — heart rate is tracked continuously by device
  2. Subjective? Yes — "elevated" with no number
  3. Objective value not captured elsewhere? NO
  → OMIT.
  reasoning: "Heart rate is device-tracked. No numeric value. Narrative commentary. Omitting."

- "weight 82.5 kg" → measurement reasoning:
  1. Device-sourced? NO — The owner reads a scale manually
  2. Subjective? NO — specific number
  3. Objective value not captured elsewhere? YES
  → EXTRACT as measurement.
  reasoning: "Specific numeric value. Not from automatic device. Extracting."

- "Lumen score 4.5" → measurement reasoning:
  1. Device-sourced? Lumen is a device but The owner reads and reports it manually
  2. Subjective? NO — specific number
  3. Objective value not captured elsewhere? YES — Lumen has no automatic pipeline here
  → EXTRACT as measurement.
  reasoning: "Specific numeric value stated by user. Extracting."

- "blood glucose 95 from the lab" → measurement reasoning:
  1. Device-sourced? NO — lab result
  2. Subjective? NO — specific number
  3. Objective value not captured elsewhere? YES
  → EXTRACT as measurement.
  reasoning: "Lab result with specific value. Extracting."

Result mentions:
[weight (measurement), Lumen score (measurement), blood glucose (measurement)]
HRV and heart rate → omitted (device-tracked, no numeric value)

---

EXAMPLE 8 — symptom vs activity after-effect vs expected physiological response
Transcript:
"About to finish a record 4x4 at 9.8 mph. Interestingly the heart rate super low —
a combination of the cold plunge and the meditation. Did a cold plunge before lifting.
My back is at its limits. Woke up shivering again last night, room was freezing."

Thinking:
- "heart rate super low" → context-dependency test:
  Would low heart rate occur without the cold plunge and meditation? 
  No — user explicitly attributes it to these activities.
  Is this a complaint? No — user says "interestingly", not "worryingly".
  Also device-tracked, no numeric value.
  → NOT a symptom, NOT a measurement. Activity after-effect → theory.
  reasoning: "Expected physiological response to cold plunge + meditation. User attributes it directly. Not independent clinical finding. Device-tracked. Omitting as symptom."

- "back is at its limits" → is the user flagging this as a current clinical complaint?
  No — it is mentioned as context for workout intensity, not as a primary complaint.
  → context or omit.
  reasoning: "Mentioned as workout context, not as active clinical complaint. Omitting."

- "woke up shivering last night, room was freezing" → context-dependency test:
  Would shivering occur without the cold room? Yes — shivering is independent body response.
  Is it recurring overnight? Yes. Is it clinically trackable? Yes.
  → symptom ✅
  reasoning: "Overnight shivering, independent of any specific activity, recurring. Clinical finding. Keeping."

- "cold plunge before lifting" → activity ✅
- "heart rate low because of cold plunge + meditation" → theory ✅

Result mentions:
[cold plunge (activity), shivering (symptom), cold plunge/meditation → low heart rate (theory)]
"heart rate super low" → omitted as symptom and measurement
"back at limits" → omitted (workout context, not active complaint)"""


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
        max_tokens=4000,
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