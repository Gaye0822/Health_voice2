import json
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()


def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


# ─────────────────────────────────────────
# Corrections (Whisper normalization errors)
# ─────────────────────────────────────────

def get_corrections_from_db() -> list:
    """Returns corrections as list of {original, corrected, context_hint}"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT original_text, corrected_text, context_hint FROM corrections WHERE is_normalization = TRUE"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"original": row[0], "corrected": row[1], "context_hint": row[2]}
        for row in rows
    ]


def get_corrections_as_text() -> str:
    """Returns corrections formatted for prompt injection."""
    corrections = get_corrections_from_db()
    if not corrections:
        return ""
    lines = []
    for c in corrections:
        if c["context_hint"]:
            lines.append(f'- "{c["original"]}" → "{c["corrected"]}" (only when context matches: {c["context_hint"]})')
        else:
            lines.append(f'- "{c["original"]}" → "{c["corrected"]}"')
    return "\nKnown corrections (apply only when context matches):\n" + "\n".join(lines)


def save_correction(original: str, corrected: str, correction_type: str = "normalization", context_hint: str = None) -> bool:
    """
    Save a Whisper normalization correction.

    Also syncs the original term as an alias to KB registry IF the corrected
    term already exists there. If not in KB, does NOT create a new entry —
    the UI will ask the user whether to add it.

    Returns True if corrected term was found in KB (alias added),
    False if not found in KB (UI should offer to add).
    """
    if not original or not corrected:
        return False
    if original.strip() == corrected.strip():
        return False

    conn = get_db_connection()
    cur = conn.cursor()
    found_in_kb = False

    try:
        # ── Save to corrections table ──────────────────────────────────────────
        cur.execute(
            "SELECT id FROM corrections WHERE original_text = %s AND corrected_text = %s",
            (original, corrected)
        )
        if not cur.fetchone():
            cur.execute(
                """INSERT INTO corrections
                   (original_text, corrected_text, correction_type, is_normalization, context_hint)
                   VALUES (%s, %s, %s, TRUE, %s)""",
                (original, corrected, correction_type, context_hint)
            )
            conn.commit()

        # ── Sync alias to KB registry only if entry already exists ────────────
        if correction_type == "normalization":
            cur.execute(
                """SELECT id, example_before FROM knowledge_base
                   WHERE correction_type = 'registry'
                   AND LOWER(original_text) = LOWER(%s)""",
                (corrected,)
            )
            existing = cur.fetchone()

            if existing:
                found_in_kb = True
                entry_id = existing[0]
                before_json = existing[1] if existing[1] else {}
                aliases = before_json.get("aliases", [])
                if original not in aliases and original.lower() not in [a.lower() for a in aliases]:
                    aliases.append(original)
                    cur.execute(
                        """UPDATE knowledge_base
                           SET example_before = %s
                           WHERE id = %s""",
                        (json.dumps({"aliases": aliases}), entry_id)
                    )
                    conn.commit()
                    print(f"⚙️  KB registry: added alias '{original}' → '{corrected}'")
            # If not in KB → don't create, UI will ask

    except Exception as e:
        print(f"⚠️ save_correction error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return found_in_kb


# ─────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────

def save_knowledge(
    original_text: str,
    correction_type: str,
    corrected_value: str = None,
    context_hint: str = None,
    reason: str = None,
    example_before: dict = None,
    example_after: dict = None
):
    """
    Save a knowledge base entry.

    correction_type:
      - "normalization"  → Whisper mishearing (original_text = wrong term, corrected_value = correct term)
      - "classification" → Wrong entity type (original_text = label, corrected_value = correct type)
      - "entity_split"   → Two things merged as one (original_text = merged label)
      - "removal"        → Entity should not have been extracted (original_text = label)

    example_before / example_after: full entity dicts before and after correction
    reason: a general principle explaining why this correction was made
    """
    if not original_text or not correction_type:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, context_hint, reason, example_before, example_after)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                original_text,
                correction_type,
                corrected_value,
                context_hint,
                reason,
                json.dumps(example_before) if example_before else None,
                json.dumps(example_after) if example_after else None
            )
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ save_knowledge error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def get_knowledge_for_prompt() -> str:
    """
    Returns knowledge base entries formatted for prompt injection.
    Groups by correction_type for clarity.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT original_text, correction_type, corrected_value, context_hint, reason,
                  example_before, example_after
           FROM knowledge_base
           ORDER BY correction_type, created_at DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return ""

    normalizations = []
    classifications = []
    splits = []
    removals = []

    for row in rows:
        original, ctype, corrected, context, reason, before, after = row
        if ctype == "normalization":
            line = f'- "{original}" → "{corrected}"'
            if context:
                line += f' (when context: "{context}")'
            if reason:
                line += f' — {reason}'
            normalizations.append(line)
        elif ctype == "classification":
            line = f'- "{original}" should be type: {corrected}'
            if context:
                line += f' (when context: "{context}")'
            if reason:
                line += f' — {reason}'
            classifications.append(line)
        elif ctype == "entity_split":
            line = f'- "{original}" should be split into separate entities'
            if reason:
                line += f' — {reason}'
            splits.append(line)
        elif ctype == "removal":
            line = f'- "{original}" should NOT be extracted as an entity'
            if reason:
                line += f' — {reason}'
            removals.append(line)

    sections = []
    if normalizations:
        sections.append("Normalization corrections:\n" + "\n".join(normalizations))
    if classifications:
        sections.append("Classification corrections:\n" + "\n".join(classifications))
    if splits:
        sections.append("Entity split corrections:\n" + "\n".join(splits))
    if removals:
        sections.append("Removal corrections:\n" + "\n".join(removals))

    # Registry entries
    conn2 = get_db_connection()
    cur2 = conn2.cursor()
    cur2.execute(
        """SELECT original_text, corrected_value, example_before, example_after
           FROM knowledge_base WHERE correction_type = 'registry' ORDER BY created_at DESC"""
    )
    registry_rows = cur2.fetchall()
    cur2.close()
    conn2.close()

    registry_entries = []
    for row in registry_rows:
        canonical, entity_type, before_json, after_json = row
        before = before_json if before_json else {}
        after = after_json if after_json else {}
        aliases = before.get("aliases", [])
        subtype = after.get("subtype", "")
        description = after.get("description", "")
        line = f"- {canonical} → entity_type: {entity_type}"
        if aliases:
            line += f" | also written as: {', '.join(aliases)}"
        if subtype:
            line += f" | subtype: {subtype}"
        if description:
            line += f" | description: {description}"
        registry_entries.append(line)

    if registry_entries:
        sections.append("Canonical term registry:\n" + "\n".join(registry_entries))

    if not sections:
        return ""

    return "\n\nKnowledge base (learned from past corrections):\n" + "\n\n".join(sections)


# ─────────────────────────────────────────
# Transcripts
# ─────────────────────────────────────────

def save_transcript(raw_text: str, normalized_text: str, source: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO transcripts (raw_text, normalized_text, source) VALUES (%s, %s, %s) RETURNING id",
        (raw_text, normalized_text, source)
    )
    transcript_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return transcript_id


def get_transcript_by_id(transcript_id: int) -> dict:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, raw_text, normalized_text, source FROM transcripts WHERE id = %s",
        (transcript_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "raw_text": row[1], "normalized_text": row[2], "source": row[3]}


# ─────────────────────────────────────────
# Entities
# ─────────────────────────────────────────

def save_entities(entities: list, transcript_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    for entity in entities:
        entity = entity.copy()
        entity_type = entity.pop("type", "unknown")
        label = entity.pop("label", entity.pop("metric", entity.pop("linked_to", "")))
        cur.execute(
            "INSERT INTO entities (transcript_id, entity_type, label, attributes) VALUES (%s, %s, %s, %s)",
            (transcript_id, entity_type, label, json.dumps(entity))
        )
    conn.commit()
    cur.close()
    conn.close()


def get_entities_by_transcript(transcript_id: int) -> list:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT entity_type, label, attributes FROM entities WHERE transcript_id = %s",
        (transcript_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"type": row[0], "label": row[1], **row[2]} for row in rows]