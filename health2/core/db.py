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
    If corrected term exists in KB registry → adds original as alias automatically.
    If not in KB → returns False so UI can ask user to add it.
    Returns True if found in KB, False if not.
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
                mishearings = before_json.get("mishearings", [])
                if original not in mishearings and original.lower() not in [m.lower() for m in mishearings]:
                    mishearings.append(original)
                    cur.execute(
                        """UPDATE knowledge_base
                           SET example_before = %s
                           WHERE id = %s""",
                        (json.dumps({**before_json, "mishearings": mishearings}), entry_id)
                    )
                    conn.commit()
                    print(f"⚙️  KB registry: added mishearing '{original}' → '{corrected}'")
                else:
                    found_in_kb = True  # already in KB, no need to ask

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
        mishearings = before.get("mishearings", [])
        subtype = after.get("subtype", "")
        description = after.get("description", "")
        note = after.get("note", "")
        line = f"- {canonical} → entity_type: {entity_type}"
        if aliases:
            line += f" | also written as: {', '.join(aliases)}"
        if mishearings:
            line += f" | common ASR errors: {', '.join(mishearings)}"
        if subtype:
            line += f" | subtype: {subtype}"
        if description:
            line += f" | description: {description}"
        if note:
            line += f" | note: {note}"
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

def save_entities(entities: list, transcript_id: int, mentions: list = None):
    """
    Save entities to DB.
    If an entity is marked unverified (low confidence mention or label not in KB),
    saves with unverified=True and adds to unverified_entities queue.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Build set of KB canonical labels and mishearing map for quick lookup
    kb_labels, kb_mishearing_map = _get_kb_labels(cur)

    # Build mention map: raw_mention_lower → {confidence, reasoning, context}
    mention_map = {}
    if mentions:
        for m in mentions:
            raw = m.get("raw_mention", "").lower()
            mention_map[raw] = {
                "confidence": m.get("confidence", "high"),
                "reasoning": m.get("reasoning", ""),
                "context": m.get("context", "")
            }

    for entity in entities:
        entity = entity.copy()
        entity_type = entity.pop("type", "unknown")
        label = entity.pop("label", entity.pop("metric", entity.pop("linked_to", "")))

        # Determine if unverified
        unverified = False
        flag_reason = None
        reasoning = None
        context = None

        # Get raw_mention attached by structure.py
        raw_mention = entity.pop("_raw_mention", None)

        if entity_type == "intake":
            label_lower = label.lower()

            # Check 1: label not in KB at all
            if label_lower not in kb_labels:
                unverified = True
                flag_reason = f"Intake label '{label}' not found in KB registry"

            # Check 2: label IS in KB but raw_mention differs and is not a known mishearing
            elif raw_mention and raw_mention.lower() != label_lower:
                raw_lower = raw_mention.lower()
                # Is raw_mention a registered mishearing of this label?
                mapped_canonical = kb_mishearing_map.get(raw_lower)
                if mapped_canonical != label_lower:
                    unverified = True
                    flag_reason = f"Label '{label}' inferred from '{raw_mention}' — not a registered mishearing"
                    # Find reasoning and context from mentions
                    for m in (mentions or []):
                        if m.get("raw_mention", "").lower() == raw_lower:
                            reasoning = m.get("reasoning", "")
                            context = m.get("context", "")
                            break
                    print(f"⚙️  unverified: '{raw_mention}' → '{label}' (inferred, not in mishearings)")

        # Check mention confidence
        mention_info = mention_map.get(label.lower())

        if mention_info and mention_info["confidence"] == "low":
            unverified = True
            flag_reason = (flag_reason + " | " if flag_reason else "") + "Low confidence mention"
            if not reasoning:
                reasoning = mention_info.get("reasoning")
            if not context:
                context = mention_info.get("context")

        cur.execute(
            """INSERT INTO entities (transcript_id, entity_type, label, attributes, unverified, flag_reason)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (transcript_id, entity_type, label, json.dumps(entity), unverified, flag_reason)
        )
        entity_id = cur.fetchone()[0]

        # Add to unverified queue
        if unverified:
            entity_for_queue = {"type": entity_type, "label": label, **entity}
            cur.execute(
                """INSERT INTO unverified_entities
                   (transcript_id, entity_id, entity_json, flag_reason, context, reasoning)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    transcript_id,
                    entity_id,
                    json.dumps(entity_for_queue),
                    flag_reason,
                    context,
                    reasoning
                )
            )

    conn.commit()
    cur.close()
    conn.close()


def _get_kb_labels(cur) -> tuple:
    """
    Returns:
      - set of canonical labels + aliases (lowercase) — for KB lookup
      - dict of mishearing → canonical — for inferred label detection
    """
    cur.execute(
        """SELECT original_text, example_before FROM knowledge_base
           WHERE correction_type = 'registry'"""
    )
    rows = cur.fetchall()
    labels = set()
    mishearing_map = {}  # mishearing_lower → canonical_lower

    for row in rows:
        canonical = row[0]
        canonical_lower = canonical.lower()
        labels.add(canonical_lower)

        before = row[1] if row[1] else {}
        for alias in before.get("aliases", []):
            labels.add(alias.lower())
        for mishearing in before.get("mishearings", []):
            mishearing_map[mishearing.lower()] = canonical_lower

    return labels, mishearing_map


def _get_kb_labels_simple() -> set:
    """Public version — returns just the set of KB labels for UI use."""
    conn = get_db_connection()
    cur = conn.cursor()
    labels, _ = _get_kb_labels(cur)
    cur.close()
    conn.close()
    return labels


def get_unverified_entities(status: str = "pending") -> list:
    """Returns unverified entities for review."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT u.id, u.transcript_id, u.entity_id, u.entity_json,
                  u.flag_reason, u.context, u.reasoning, u.created_at,
                  t.normalized_text
           FROM unverified_entities u
           JOIN transcripts t ON u.transcript_id = t.id
           WHERE u.status = %s
           ORDER BY u.created_at DESC""",
        (status,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": row[0],
            "transcript_id": row[1],
            "entity_id": row[2],
            "entity": row[3],
            "flag_reason": row[4],
            "context": row[5],
            "reasoning": row[6],
            "created_at": row[7],
            "transcript_text": row[8]
        }
        for row in rows
    ]


def approve_unverified_entity(unverified_id: int, canonical_label: str = None):
    """
    Approve an unverified entity.
    If canonical_label provided, updates entity label.
    Marks entity as verified in entities table.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT entity_id FROM unverified_entities WHERE id = %s",
            (unverified_id,)
        )
        row = cur.fetchone()
        if not row:
            return
        entity_id = row[0]

        if canonical_label:
            cur.execute(
                "UPDATE entities SET label = %s, unverified = FALSE, flag_reason = NULL WHERE id = %s",
                (canonical_label, entity_id)
            )
        else:
            cur.execute(
                "UPDATE entities SET unverified = FALSE, flag_reason = NULL WHERE id = %s",
                (entity_id,)
            )

        cur.execute(
            "UPDATE unverified_entities SET status = 'approved', reviewed_at = NOW() WHERE id = %s",
            (unverified_id,)
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ approve_unverified_entity error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def reject_unverified_entity(unverified_id: int):
    """Remove entity from entities table and mark as rejected."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT entity_id FROM unverified_entities WHERE id = %s",
            (unverified_id,)
        )
        row = cur.fetchone()
        if not row:
            return
        entity_id = row[0]

        cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
        cur.execute(
            "UPDATE unverified_entities SET status = 'rejected', reviewed_at = NOW() WHERE id = %s",
            (unverified_id,)
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ reject_unverified_entity error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def get_entities_by_transcript(transcript_id: int) -> list:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT entity_type, label, attributes, unverified, flag_reason FROM entities WHERE transcript_id = %s",
        (transcript_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"type": row[0], "label": row[1], **row[2],
         "unverified": row[3], "flag_reason": row[4]}
        for row in rows
    ]