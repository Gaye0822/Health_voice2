import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Protected types — never sent to LLM validator
PROTECTED_TYPES = {"theory", "outside", "context"}


def _get_knowledge() -> str:
    try:
        from core.db import get_knowledge_for_prompt
        return get_knowledge_for_prompt()
    except Exception:
        return ""


def _resolve_time_references(entities: list) -> list:
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities


def _check_measurement_metrics(entities: list, transcript: str) -> tuple:
    """
    Deterministic guard: remove measurement entities whose metric name
    does not appear anywhere in the transcript.
    Prevents hallucinated metrics like HRV when only deep sleep was mentioned.
    """
    clean = []
    removed = []
    transcript_lower = transcript.lower()

    for entity in entities:
        if entity.get("type") != "measurement":
            clean.append(entity)
            continue

        metric = entity.get("metric", "")
        if not metric:
            clean.append(entity)
            continue

        # Check if any word of the metric appears in the transcript
        # Use multi-word check: all significant words must appear
        metric_words = [w for w in metric.lower().split() if len(w) > 2]
        if not metric_words:
            clean.append(entity)
            continue

        # At least one key word of the metric must appear in transcript
        found = any(word in transcript_lower for word in metric_words)
        if found:
            clean.append(entity)
        else:
            removed.append(f"Removed measurement [{metric}] — metric name not found in transcript")

    return clean, removed


def _check_outcome_linked_to(entities: list) -> tuple:
    """
    Deterministic guard: remove outcome entities whose linked_to
    does not match any other entity in this note, or is linked to
    a symptom/context/theory (invalid cause types).

    An outcome must be linked to something the user did or took:
    intake, activity, machine, or intervention.
    """
    # Valid cause types for outcome linked_to
    VALID_CAUSE_TYPES = {"intake", "activity", "machine", "intervention"}

    # Build map of label → type from non-outcome entities
    entity_label_types = {}
    for e in entities:
        if e.get("type") == "outcome":
            continue
        label = e.get("label", e.get("metric", e.get("raw_text", "")))
        if label:
            entity_label_types[label.lower()] = e.get("type", "")

    clean = []
    removed = []

    for entity in entities:
        if entity.get("type") != "outcome":
            clean.append(entity)
            continue

        linked_to = entity.get("linked_to", "")

        # No linked_to → omit
        if not linked_to:
            removed.append(
                f"Removed outcome with no linked_to — an outcome without a cause is not an outcome"
            )
            continue

        linked_lower = linked_to.lower()

        # Find matching entity
        matched_type = None
        for label, etype in entity_label_types.items():
            if linked_lower in label or label in linked_lower:
                matched_type = etype
                break

        if matched_type is None:
            removed.append(
                f"Removed outcome linked_to '{linked_to}' — no matching entity found in this note"
            )
        elif matched_type not in VALID_CAUSE_TYPES:
            removed.append(
                f"Removed outcome linked_to '{linked_to}' — linked to {matched_type}, "
                f"must be intake/activity/machine/intervention"
            )
        else:
            clean.append(entity)

    return clean, removed


def validate_entities(entities: list, normalized_text: str) -> tuple:
    """
    Stage 3: Validate extracted entities.
    Returns (corrected_entities, changes_list)

    Pipeline:
    1. schema_enforcer — deterministic mechanical fixes (no LLM)
    2. LLM validator — KB lookup + transcript contradiction check only
    """
    if not entities:
        return entities, []

    all_changes = []

    # ── Step 1b: Measurement metric presence check ────────────────────────
    entities, metric_violations = _check_measurement_metrics(entities, normalized_text)
    if metric_violations:
        print(f"⚙️  metric_guard: {len(metric_violations)} removal(s):")
        for v in metric_violations:
            print(f"   → {v}")
    all_changes.extend(metric_violations)

    # ── Step 1c: Outcome linked_to guard ─────────────────────────────────
    entities, outcome_violations = _check_outcome_linked_to(entities)
    if outcome_violations:
        print(f"⚙️  outcome_guard: {len(outcome_violations)} removal(s):")
        for v in outcome_violations:
            print(f"   → {v}")
    all_changes.extend(outcome_violations)

    # ── Step 2: Separate protected types — they skip the LLM ─────────────
    protected = [e for e in entities if e.get("type") in PROTECTED_TYPES]
    to_validate = [e for e in entities if e.get("type") not in PROTECTED_TYPES]

    if not to_validate:
        return _resolve_time_references(protected), all_changes

    # ── Step 3: LLM validator — KB lookup + contradiction check only ──────
    knowledge = _get_knowledge()
    kb_section = f"\n\n{knowledge}" if knowledge else ""

    system_prompt = """You are a health event validator.

ALWAYS respond in English.

You receive a list of extracted health entities and the original transcript.
Your job is narrow and specific — do NOT make broad ontological judgments.

You have exactly TWO jobs:

─────────────────────────────────────────
JOB 1: KNOWLEDGE BASE LOOKUP
─────────────────────────────────────────

If a knowledge base is provided at the end of this prompt:
- If an entity label matches a registry entry with a different entity_type → correct the type
- If an entity label matches a known alias → use the canonical form as the label
- If a correction entry shows this label was previously misclassified → apply the correction
- If an entity label appears in the removal list → remove that entity entirely
  The reason explains why — use it to confirm the match before removing.

For KB lookup: if you are unsure → leave unchanged.

─────────────────────────────────────────
JOB 2: TRANSCRIPT CONTRADICTION CHECK
─────────────────────────────────────────

For every entity, find the relevant passage in the original transcript and ask:
does the transcript explicitly state this did NOT happen, was skipped, was unavailable,
or is being negated?

Work through each entity one by one against the transcript before deciding.

IMPORTANT EXCEPTION — symptom entities with status: "absent":
These entities represent explicitly stated absences. The transcript negating them
is exactly why they exist. Do NOT remove a symptom entity that already has status: "absent".
That entity is correct — it captures the user's explicit statement that something did not occur.

IMPORTANT EXCEPTION — activity, machine, device entities:
NEVER remove an activity, machine, or device entity for any reason.
These entity types passed through mention_filter before reaching you.
That filter already eliminated future plans and unconfirmed events.
If an activity entity is in your input, it was confirmed as happened or explicitly noted as not done.
Your job is NOT to re-evaluate whether activities happened.

ACTIVITY STATUS CORRECTION — this is the ONLY change you may make to activity entities:
If an activity has status: "completed" but the transcript clearly states it was NOT done
("didn't lift", "didn't do PEMF", "couldn't get a cold plunge in") →
change status to "did_not_complete". Do NOT remove the entity.
If an activity has status: "did_not_complete", leave it as-is.
Even if the transcript contains ambiguous language like "didn't want" near an activity,
do NOT remove it — change status to "did_not_complete" only if negation is unambiguous.

MACHINE STATUS CORRECTION — this is the ONLY change you may make to machine entities:
If a machine has status: "used" but the transcript clearly states it was NOT used
("didn't do PEMF", "no PEMF today", "skipped the Novothor") →
change status to "did_not_use". Do NOT remove the entity.
If a machine has status: "did_not_use", leave it as-is.

IMPORTANT EXCEPTION — intake entities with action: "did_not_take":
These already capture the fact that something was NOT taken. Do NOT remove them.
They are correct by definition — the "did_not_take" action IS the negation.

For intake entities with action: "took":
If the transcript clearly states the substance was NOT taken → remove or change to did_not_take.

If the transcript negates it (and none of the above exceptions apply) → remove the entity,
or move to outside if it has operational value.
If you are unsure → leave unchanged. Default is preservation, not removal.

This rule applies only to clear, unambiguous negations. ASR errors frequently
distort words — "didn't want" may be an error for "did". When in doubt, preserve.

─────────────────────────────────────────
STRICT LIMITS — DO NOT:
─────────────────────────────────────────
- Change entity types based on your own judgment — this is an ABSOLUTE limit.
  The only exception is when the KB registry explicitly instructs a type correction.
  "Eel" being a meal, not an intake, is an ontological judgment — do NOT change it.
  If an entity came in as "meal", it must leave as "meal" unless KB says otherwise.
- Change status field values — this is an ABSOLUTE limit.
  "incomplete" must stay "incomplete". "completed" must stay "completed".
  If the transcript says something did not happen → remove the entity entirely, do not change status.
  NEVER change status to "absent" — this value does not exist in the schema.
- Remove entities for ontological reasons (EXCEPT when KB removal list says to remove)
- Add new entities not present in the input — this is an ABSOLUTE limit with no exceptions.
  Even if the transcript mentions something that was not extracted, you must NOT add it.
  Your input list is final. You can only remove or correct, never add.
  If you think something is missing → that is a job for the extraction stage, not validation.
- Change action fields (took / did_not_take)
- Make temporal judgments beyond what the transcript says
- Remove an entity based on ambiguous ASR language when specific timing exists.
  If an entity has a specific time (clock time, AM/PM, "this morning", "around X"),
  that timing is strong evidence the event occurred — do NOT remove it based on
  ambiguous phrasing like "didn't want", "wasn't sure", "maybe" nearby in the transcript.
  ASR errors frequently distort words — "didn't want" may be an error for "did do".

KB removal is an explicit instruction, not an ontological judgment.
If the label is in the removal list → remove it. This overrides the "do not remove" limit.

Output entity count must be ≤ input count (splitting one into two is the only exception).
If your output has MORE entities than your input → you have violated these rules. Fix it.""" + kb_section + """

Return ONLY this JSON:
{
  "entities": [...],
  "changes": ["brief description of each change made"]
}
If no changes needed, return original entities with empty changes list."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Validate these entities.\n\n"
                    f"Original transcript:\n{normalized_text}\n\n"
                    f"Entities to validate:\n{json.dumps(to_validate, indent=2)}"
                )
            }
        ]
    )

    raw = response.content[0].text.strip()
    print(f'⚠️ VALIDATOR RAW RESPONSE:\n{raw[:2000]}')

    # Extract JSON — handle cases where LLM writes analysis before JSON block
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    elif raw.startswith("{"):
        pass  # already pure JSON
    else:
        # Try to find JSON object in the response
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

    if not raw:
        print("⚠️ Validator empty response")
        return _resolve_time_references(to_validate + protected), all_changes

    try:
        parsed = json.loads(raw)
        llm_changes = parsed.get("changes", [])
        validated = parsed.get("entities", to_validate)

        # Deterministic guard: validator must never add entities
        # If output has more entities than input, it violated the rules — log and trim
        if len(validated) > len(to_validate):
            added_count = len(validated) - len(to_validate)
            print(f"⚠️ Validator added {added_count} entity(ies) — stripping extras")
            # Keep only entities whose labels exist in the input
            input_labels = {
                e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", ""))))
                for e in to_validate
            }
            validated = [
                e for e in validated
                if e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", "")))) in input_labels
            ]
            # If still more, just take first N
            if len(validated) > len(to_validate):
                validated = validated[:len(to_validate)]
            all_changes.append(f"Validator attempted to add {added_count} entity(ies) — stripped by guard")

        # Deterministic guard: validator must never change entity types
        # unless the KB explicitly instructs a type correction
        # Build set of labels that KB explicitly corrected in this round
        kb_corrected_labels = set()
        for change in llm_changes:
            change_lower = change.lower()
            if any(kw in change_lower for kw in ["kb", "registry", "knowledge base", "canonical", "correction"]):
                # Extract label from change message if possible
                for e in to_validate:
                    label = e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", ""))))
                    if label and label.lower() in change_lower:
                        kb_corrected_labels.add(label)

        # Set of (label, type) tuples — handles same label with multiple types (e.g. berberine intake + intervention)
        input_label_types = {
            (e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", "")))), e.get("type"))
            for e in to_validate
        }
        input_types = {
            e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", "")))): e.get("type")
            for e in to_validate
        }
        for e in validated:
            label = e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", ""))))
            current_type = e.get("type")
            # (label, type) pair exists in input → no change, skip guard
            if (label, current_type) in input_label_types:
                continue
            original_type = input_types.get(label)
            if original_type and current_type != original_type:
                # Exception 1: action field confirms intake
                if e.get("action") in ("took", "did_not_take"):
                    print(f"⚙️  Guard allowing intake correction for '{label}': action field confirms intake")
                    all_changes.append(f"Structure type error corrected: '{label}' was {original_type}, action field confirms intake")
                # Exception 3: intervention correction — structure LLM wrote intake but mention was intervention
                
                # Exception 2: KB explicitly instructed this type correction
                elif label in kb_corrected_labels:
                    print(f"⚙️  Guard allowing KB type correction for '{label}': {original_type} → {e['type']}")
                    all_changes.append(f"KB type correction applied: '{label}' {original_type} → {e['type']}")
                else:
                    print(f"⚠️ Validator changed type of '{label}' from {original_type} to {e['type']} — reverting")
                    all_changes.append(f"Validator attempted to change type of '{label}' from {original_type} to {e['type']} — reverted by guard")
                    e["type"] = original_type

        all_changes.extend(llm_changes)
        final_entities = _resolve_time_references(validated) + protected

        # Re-validate through Pydantic to ensure all fields have defaults
        try:
            from core.models import EntityOutput
        except ImportError:
            from models import EntityOutput
        try:
            reparsed = EntityOutput.model_validate({"entities": final_entities})
            final = [e.model_dump() for e in reparsed.entities]
        except Exception:
            final = final_entities

        return final, all_changes
    except json.JSONDecodeError as e:
        print(f"⚠️ Validator JSON parse error: {e}")
        return _resolve_time_references(to_validate + protected), all_changes