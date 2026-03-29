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
- Change entity types based on your own judgment
- Remove entities for ontological reasons (EXCEPT when KB removal list says to remove)
- Add new entities not present in the input
- Change action fields (took / did_not_take)
- Make temporal judgments beyond what the transcript says

KB removal is an explicit instruction, not an ontological judgment.
If the label is in the removal list → remove it. This overrides the "do not remove" limit.

Output entity count must be ≤ input count (splitting one into two is the only exception).""" + kb_section + """

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
        all_changes.extend(llm_changes)
        final = _resolve_time_references(validated) + protected
        return final, all_changes
    except json.JSONDecodeError as e:
        print(f"⚠️ Validator JSON parse error: {e}")
        return _resolve_time_references(to_validate + protected), all_changes