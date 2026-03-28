import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _get_knowledge() -> str:
    try:
        from core.db import get_knowledge_for_prompt
        return get_knowledge_for_prompt()
    except Exception:
        return ""


def _get_schemas() -> str:
    try:
        from core.structure import ENTITY_SCHEMAS
        return json.dumps(ENTITY_SCHEMAS, indent=2)
    except Exception:
        return "{}"


def _resolve_time_references(entities: list) -> list:
    from datetime import datetime
    now = datetime.now().isoformat()
    for entity in entities:
        for key, value in entity.items():
            if isinstance(value, str) and value.lower() in ["now", "just now", "right now"]:
                entity[key] = now
    return entities


def validate_entities(entities: list, normalized_text: str) -> tuple:
    """
    Stage 3: Validate extracted entities against schema rules and knowledge base.
    Returns (corrected_entities, changes_list)
    """
    if not entities:
        return entities, []

    # Separate protected entity types before passing to the LLM validator.
    # theory and outside entities are never modified by the validator —
    # they were placed there by the structurer and represent final decisions.
    PROTECTED_TYPES = {"theory", "outside", "context"}
    protected = [e for e in entities if e.get("type") in PROTECTED_TYPES]
    to_validate = [e for e in entities if e.get("type") not in PROTECTED_TYPES]

    if not to_validate:
        return entities, []

    entities_json = json.dumps(to_validate, indent=2)
    schemas_json = _get_schemas()
    knowledge = _get_knowledge()
    kb_section = f"\n\n{knowledge}" if knowledge else ""

    system_prompt = """You are a health event validator.

ALWAYS respond in English.

You receive extracted health entities and the original transcript.
Your job is to check each entity and fix violations using:
1. The schema rules below
2. The knowledge base at the end of this prompt (if present)

Entity schemas:
""" + schemas_json + """

─────────────────────────────────────────
HOW TO VALIDATE:
─────────────────────────────────────────

For each entity, ask:
- Does the type match what this thing actually is?
- Do the field values follow the schema rules?
- Is there anything in the knowledge base that clarifies what this entity should be?

Use the knowledge base to resolve named entities:
- If a term appears in the registry with a specific entity_type, use that type
- If a term has known aliases, use the canonical form
- If a correction entry shows this label was previously misclassified, apply the correction

─────────────────────────────────────────
SCHEMA VIOLATION RULES:
─────────────────────────────────────────

INTAKE:
- action must be "took" or "did_not_take" only
- "about to", "going to" in transcript → action: "did_not_take", notes: "planned"
- Two substances in one label → split into two separate intake entities
- Not obtained due to external reason → move to outside

SYMPTOM:
- Label must be a specific named condition — not a general feeling
- General expressions of feeling unwell → remove entity
- Sleep observations without a numeric value → remove entity
- Sleep metrics WITH a numeric value from a tracking device (duration, score, deep sleep, REM, etc.) → keep as measurement, do NOT remove
- Qualifier must describe the symptom itself, not personal assessments
- Positive observation (improvement, better) → change to outcome type

MEASUREMENT:
- value must be a specific number — if null or qualitative → remove entity

DEVICE / MACHINE:
- Broken or not-recorded → move to outside
- Only create if actually used today

INTERVENTION:
- Single procedure → remove, only reference in outcome linked_to
- status "completed" without explicit statement → change to "unknown"

GENERAL:
- Extra fields not in schema → remove
- String value "unknown" → change to null

─────────────────────────────────────────
CRITICAL RULES:
─────────────────────────────────────────
- Only fix real violations — do not change correct entities
- Never invent new entities not present in the input
- Output entity count must be ≤ input count (splitting one into two is the only exception)
- When the knowledge base contradicts the extracted type, trust the knowledge base""" + kb_section + """

Return ONLY this JSON:
{
  "entities": [...],
  "changes": ["brief description of each change made"]
}
If no changes needed, return original entities with empty changes list."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Validate these entities.\n\nOriginal transcript:\n{normalized_text}\n\nEntities to validate:\n{entities_json}"
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
        print("⚠️ Validator empty response")
        return to_validate + protected, []

    try:
        parsed = json.loads(raw)
        changes = parsed.get("changes", [])
        validated = parsed.get("entities", to_validate)
        # Re-attach protected entities (theory, outside, context) unchanged
        final = _resolve_time_references(validated) + protected
        return final, changes
    except json.JSONDecodeError as e:
        print(f"⚠️ Validator JSON parse error: {e}")
        return entities, []