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

    # ── Step 1: Deterministic schema enforcement ──────────────────────────
    try:
        from core.schema_enforcer import enforce_schema, log_violations
    except ImportError:
        from schema_enforcer import enforce_schema, log_violations

    entities, enforcer_violations = enforce_schema(entities)
    log_violations(enforcer_violations)
    all_changes.extend(enforcer_violations)

    if not entities:
        return [], all_changes

    # ── Step 1b: Measurement metric presence check ────────────────────────
    entities, metric_violations = _check_measurement_metrics(entities, normalized_text)
    if metric_violations:
        print(f"⚙️  metric_guard: {len(metric_violations)} removal(s):")
        for v in metric_violations:
            print(f"   → {v}")
    all_changes.extend(metric_violations)

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

If the transcript negates it → remove the entity, or move to outside if it has operational value.
If you are unsure → remove. For contradiction check, the default is removal not preservation.

This rule has no exceptions. An entity that contradicts the transcript must not survive
validation regardless of how confidently it was extracted.

─────────────────────────────────────────
STRICT LIMITS — DO NOT:
─────────────────────────────────────────
- Change entity types based on your own judgment — this is an ABSOLUTE limit.
  The only exception is when the KB registry explicitly instructs a type correction.
  "Eel" being a meal, not an intake, is an ontological judgment — do NOT change it.
  If an entity came in as "meal", it must leave as "meal" unless KB says otherwise.
- Remove entities for ontological reasons (EXCEPT when KB removal list says to remove)
- Add new entities not present in the input — this is an ABSOLUTE limit with no exceptions.
  Even if the transcript mentions something that was not extracted, you must NOT add it.
  Your input list is final. You can only remove or correct, never add.
  If you think something is missing → that is a job for the extraction stage, not validation.
- Change action fields (took / did_not_take)
- Make temporal judgments beyond what the transcript says

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

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

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
        input_types = {
            e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", "")))): e.get("type")
            for e in to_validate
        }
        for e in validated:
            label = e.get("label", e.get("metric", e.get("linked_to", e.get("raw_text", ""))))
            original_type = input_types.get(label)
            if original_type and e.get("type") != original_type:
                print(f"⚠️ Validator changed type of '{label}' from {original_type} to {e['type']} — reverting")
                all_changes.append(f"Validator attempted to change type of '{label}' from {original_type} to {e['type']} — reverted by guard")
                e["type"] = original_type

        all_changes.extend(llm_changes)
        final = _resolve_time_references(validated) + protected
        return final, all_changes
    except json.JSONDecodeError as e:
        print(f"⚠️ Validator JSON parse error: {e}")
        return _resolve_time_references(to_validate + protected), all_changes