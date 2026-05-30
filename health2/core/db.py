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
            # First try exact match
            cur.execute(
                """SELECT id, example_before, original_text FROM knowledge_base
                   WHERE correction_type = 'registry'
                   AND LOWER(original_text) = LOWER(%s)""",
                (corrected,)
            )
            existing = cur.fetchone()

            # If no exact match, try fuzzy: KB label starts with corrected term
            if not existing:
                cur.execute(
                    """SELECT id, example_before, original_text FROM knowledge_base
                       WHERE correction_type = 'registry'
                       AND LOWER(original_text) LIKE LOWER(%s)""",
                    (corrected + "%",)
                )
                existing = cur.fetchone()
                if existing:
                    print(f"⚙️  KB fuzzy match: '{corrected}' → '{existing[2]}'")

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
                    print(f"⚙️  KB registry: added mishearing '{original}' → '{existing[2]}'")
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

def save_transcript(raw_text: str, normalized_text: str, source: str, note_date=None) -> tuple:
    """
    Save transcript to DB. Returns (transcript_id, source_note_id).
    source_note_id is a stable UUID for this transcript — used as envelope idempotency key.

    Upsert: if a transcript with the same source (filename) already exists,
    update it in place and return the existing transcript_id.
    Entities for that transcript are deleted before re-save so pipeline
    can write fresh entities on top.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if this source already exists
    cur.execute(
        "SELECT id, source_note_id FROM transcripts WHERE source = %s",
        (source,)
    )
    existing = cur.fetchone()

    if existing:
        transcript_id = existing[0]
        source_note_id = str(existing[1])
        # Update transcript text
        cur.execute(
            """UPDATE transcripts
               SET raw_text = %s, normalized_text = %s, note_date = %s
               WHERE id = %s""",
            (raw_text, normalized_text, note_date, transcript_id)
        )
        # Delete existing unverified_entities first (FK references entities)
        cur.execute(
            "DELETE FROM unverified_entities WHERE transcript_id = %s",
            (transcript_id,)
        )
        # Then delete entities
        cur.execute(
            "DELETE FROM entities WHERE transcript_id = %s",
            (transcript_id,)
        )
        print(f"⚙️  upsert: existing transcript updated (id={transcript_id}, source={source})")
    else:
        cur.execute(
            """INSERT INTO transcripts (raw_text, normalized_text, source, note_date)
               VALUES (%s, %s, %s, %s) RETURNING id, source_note_id""",
            (raw_text, normalized_text, source, note_date)
        )
        row = cur.fetchone()
        transcript_id = row[0]
        source_note_id = str(row[1])
        print(f"⚙️  upsert: new transcript created (id={transcript_id}, source={source})")

    conn.commit()
    cur.close()
    conn.close()
    return transcript_id, source_note_id


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

def save_entities(entities: list, transcript_id: int, mentions: list = None, note_date=None):
    """
    Save entities to DB.
    If an entity is marked unverified (low confidence mention or label not in KB),
    saves with unverified=True and adds to unverified_entities queue.
    note_date: the actual date of the voice note — written to each entity row for
    time-based querying (e.g. recurring symptom detection).
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

    # Build symptom mention list in order — parallel to symptom entities
    _symptom_mentions = [
        m for m in (mentions or [])
        if m.get("candidate_type") == "symptom"
    ]
    _symptom_entity_idx = 0  # tracks which symptom entity we're on



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

        # For intake and machine only: if raw_mention differs from label → inferred, flag as unverified
        # Symptom labels are clinical interpretations by LLM — not flagged
        if not unverified and entity_type in ("intake", "machine") and raw_mention and raw_mention.lower() != label.lower():
            raw_lower = raw_mention.lower()
            mapped_canonical = kb_mishearing_map.get(raw_lower)
            if mapped_canonical != label.lower():
                unverified = True
                flag_reason = f"Label '{label}' inferred from '{raw_mention}' — not a registered mishearing"
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

        # entity_date: event_date string'ini note_date'e göre gerçek tarihe çevir
        # Örn: event_date="yesterday" + note_date=2026-04-06 → entity_date=2026-04-05
        entity_date = note_date  # default: note_date
        event_date_raw = entity.get("event_date")
        if event_date_raw and note_date:
            try:
                from core.schema_enforcer import _resolve_start_date_to_iso
                resolved = _resolve_start_date_to_iso(event_date_raw, note_date)
                if resolved:
                    entity_date = resolved
            except Exception:
                pass

        cur.execute(
            """INSERT INTO entities (transcript_id, entity_type, label, attributes, unverified, flag_reason, note_date, entity_date)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (transcript_id, entity_type, label, json.dumps(entity), unverified, flag_reason, note_date, entity_date)
        )
        entity_id = cur.fetchone()[0]

        # Low-confidence symptom → symptom_label_reviews
        # Nth symptom entity → Nth symptom mention (order-preserving across all symptoms)
        if entity_type == "symptom":
            if _symptom_entity_idx < len(_symptom_mentions):
                _sm = _symptom_mentions[_symptom_entity_idx]
                if _sm.get("confidence") == "low":
                    try:
                        import sys as _sys, os as _os
                        _parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                        if _parent not in _sys.path:
                            _sys.path.insert(0, _parent)
                        from symptom_label_review import save_symptom_for_review
                        save_symptom_for_review(
                            proposed_label=label,
                            transcript_context=_sm.get("context", ""),
                            llm_reasoning=_sm.get("reasoning", ""),
                            entity_json={"type": entity_type, "label": label, **{k: v for k, v in entity.items() if not k.startswith("_")}},
                            transcript_id=transcript_id,
                            entity_id=entity_id,
                            note_date=note_date,
                        )
                    except Exception as _e:
                        print(f"⚠️  symptom_label_review queue error: {_e}")
            _symptom_entity_idx += 1

        # Recurring detection + fuzzy matching — terminal only, no schema change
        if entity_type == "symptom" and note_date is not None:
            # 1. Exact recurring check
            cur.execute(
                """SELECT COUNT(*), MIN(entity_date), MAX(entity_date)
                   FROM entities
                   WHERE entity_type = 'symptom'
                   AND LOWER(label) = LOWER(%s)
                   AND entity_date >= %s - INTERVAL '7 days'
                   AND entity_date < %s""",
                (label, note_date, note_date)
            )
            row = cur.fetchone()
            freq = row[0] if row else 0
            first_seen = row[1] if row else None
            last_seen = row[2] if row else None
            if freq > 0:
                print(f"🔁 RECURRING [{freq}x in past 7 days]: symptom '{label}' "
                      f"| first: {first_seen} | last: {last_seen} | today: {note_date}")
            else:
                print(f"🆕 NEW symptom '{label}' | note_date: {note_date}")

            # 2. Fuzzy match against all distinct symptom labels in DB (past 30 days)
            cur.execute(
                """SELECT DISTINCT label FROM entities
                   WHERE entity_type = 'symptom'
                   AND LOWER(label) != LOWER(%s)
                    AND entity_date >= %s - INTERVAL '30 days'""",
                (label, note_date)
            )
            existing_labels = [r[0] for r in cur.fetchall()]
            if existing_labels:
                try:
                    from rapidfuzz import fuzz

                    # Anatomical body part tokens — if labels share no body part token,
                    # they cannot be the same symptom regardless of string similarity.
                    # Only true anatomical locations — NOT generic symptom words like pain/swelling
                    # If two labels share no anatomical token, they cannot be the same symptom
                    BODY_PARTS = {
                        "eye", "knee", "joint", "lymph", "node", "groin", "elbow",
                        "back", "head", "ear", "hand", "finger", "shoulder", "hip",
                        "ankle", "foot", "toe", "neck", "chest", "stomach", "abdomen",
                        "face", "jaw", "skin", "eyelid", "gland", "vision",
                        "nocturia", "stool", "bowel", "sinusitis", "blepharitis",
                        "vertigo", "headache", "tremor", "tingling",
                    }

                    def _body_tokens(s):
                        return {w for w in s.lower().split() if w in BODY_PARTS}

                    def _similarity(a, b):
                        # Require at least one shared body part token
                        if not (_body_tokens(a) & _body_tokens(b)):
                            return 0
                        # Use weighted combo: partial_ratio catches substrings,
                        # token_sort_ratio handles reordering
                        pr = fuzz.partial_ratio(a.lower(), b.lower())
                        tsr = fuzz.token_sort_ratio(a.lower(), b.lower())
                        return round((pr * 0.4 + tsr * 0.6), 1)

                    matches = []
                    for existing in existing_labels:
                        score = _similarity(label, existing)
                        if score >= 70:
                            matches.append((existing, score))
                    matches.sort(key=lambda x: x[1], reverse=True)
                    if matches:
                        match_str = " | ".join(f"'{m}' ({s}%)" for m, s in matches[:5])
                        print(f"   🔍 FUZZY MATCHES for '{label}': {match_str}")
                    else:
                        print(f"   🔍 FUZZY: no similar labels found (threshold: 70%)")
                except ImportError:
                    print("   ⚠️  rapidfuzz not installed")

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


def save_normalize_flags(transcript_id: int, applied_corrections: list, entities: list):
    """
    Pre_normalize'in uyguladığı correction'lardan KB'deki terimlere yapılanları tespit eder.
    Eğer o terim pipeline'dan tracked entity (intake/machine/activity) olarak çıkmadıysa
    unverified_entities tablosuna yazar — entity_type="normalize_flag" olarak.

    applied_corrections: pre_result["applied"] listesi
      format: '"original" -> "corrected" (source)'
    entities: pipeline'dan çıkan entity listesi
    """
    if not applied_corrections:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        kb_labels, _ = _get_kb_labels(cur)

        # Pipeline'dan çıkan tracked entity label'larını topla
        tracked_labels = set()
        for e in entities:
            etype = e.get("type", "")
            if etype in ("intake", "machine", "activity"):
                label = e.get("label", "").lower()
                if label:
                    tracked_labels.add(label)

        # applied_corrections'ı parse et ve KB'deki terimleri bul
        import re
        for correction in applied_corrections:
            # Format: '"original" -> "corrected" (source)' veya
            # '"original" → "corrected" (source)'
            match = re.search(r'"([^"]+)"\s*(?:→|->)\s*"([^"]+)"', correction)
            if not match:
                continue

            # Mishearing ile yapılan düzeltmeler zaten onaylı — unverified'a yazma
            if "registry mishearing" in correction:
                continue
            corrected_full = match.group(2).strip().lower()

            # Corrected terim KB'de var mı? — tam eşleşme veya token bazlı SQL
            corrected_match = None
            if corrected_full in kb_labels:
                corrected_match = corrected_full
            else:
                # "Dexcom thing" gibi durumlarda her token için SQL'de ara
                for token in corrected_full.split():
                    if len(token) <= 3:
                        continue
                    cur.execute(
                        """SELECT original_text FROM knowledge_base
                           WHERE correction_type = 'registry'
                           AND LOWER(original_text) LIKE LOWER(%s)
                           ORDER BY LENGTH(original_text) ASC
                           LIMIT 1""",
                        (f"%{token}%",)
                    )
                    kb_row = cur.fetchone()
                    if kb_row:
                        corrected_match = token
                        break

            if not corrected_match:
                continue
            corrected = corrected_match

            # Pipeline'dan tracked entity olarak çıktı mı?
            if corrected in tracked_labels:
                continue

            # Çıkmadı → unverified_entities'e yaz
            # corrected = KB'deki canonical label (token bazlı bulunmuş)
            canonical_label = corrected  # KB'de bulunan temiz label
            flag_reason = f"normalize_flag: '{match.group(1)}' → '{canonical_label}' detected by normalize but not tracked as entity"
            entity_json = {
                "type": "normalize_flag",
                "label": canonical_label,
                "original": match.group(1),
                "corrected": canonical_label,
                "source": correction
            }

            cur.execute(
                """INSERT INTO entities (transcript_id, entity_type, label, attributes, unverified, flag_reason)
                   VALUES (%s, %s, %s, %s, TRUE, %s) RETURNING id""",
                (transcript_id, "normalize_flag", canonical_label, json.dumps(entity_json), flag_reason)
            )
            entity_id = cur.fetchone()[0]

            cur.execute(
                """INSERT INTO unverified_entities
                   (transcript_id, entity_id, entity_json, flag_reason, context, reasoning)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    transcript_id,
                    entity_id,
                    json.dumps(entity_json),
                    flag_reason,
                    None,
                    f"pre_normalize corrected '{match.group(1)}' to '{match.group(2)}' which is in KB but did not appear as tracked entity in pipeline output"
                )
            )
            print(f"⚙️  normalize_flag: '{match.group(1)}' → '{match.group(2)}' saved to unverified")

        conn.commit()
    except Exception as e:
        print(f"⚠️ save_normalize_flags error: {e}")
    finally:
        cur.close()
        conn.close()

def get_unverified_entities(status: str = "pending") -> list:
    """Returns unverified entities for review."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT u.id, u.transcript_id, u.entity_id, u.entity_json,
                  u.flag_reason, u.context, u.reasoning, u.created_at,
                  t.normalized_text, t.raw_text
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
            "transcript_text": row[8],
            "raw_transcript_text": row[9],
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
                """UPDATE entities
                   SET label = %s,
                       unverified = FALSE,
                       flag_reason = NULL,
                       attributes = attributes || jsonb_build_object('label', %s::text)
                   WHERE id = %s""",
                (canonical_label, canonical_label, entity_id)
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

        # normalize_flag tipi için: original'ı canonical'ın mishearing'ine ekle
        cur.execute(
            "SELECT entity_type, attributes FROM entities WHERE id = %s",
            (entity_id,)
        )
        entity_row = cur.fetchone()
        if entity_row and entity_row[0] == "normalize_flag":
            attrs = entity_row[1] if isinstance(entity_row[1], dict) else {}
            original = attrs.get("original", "")
            corrected = canonical_label or attrs.get("corrected", "")
            if original and corrected:
                # KB'de canonical entry'yi bul ve mishearing ekle
                cur.execute(
                    """SELECT id, example_before FROM knowledge_base
                       WHERE correction_type = 'registry'
                       AND (LOWER(original_text) = LOWER(%s)
                            OR LOWER(original_text) LIKE LOWER(%s))
                       ORDER BY LENGTH(original_text) ASC
                       LIMIT 1""",
                    (corrected, f"%{corrected}%")
                )
                kb_row = cur.fetchone()
                if kb_row:
                    kb_id = kb_row[0]
                    before = kb_row[1] if isinstance(kb_row[1], dict) else {}
                    mishearings = before.get("mishearings", [])
                    if original.lower() not in [m.lower() for m in mishearings]:
                        mishearings.append(original)
                        before["mishearings"] = mishearings
                        cur.execute(
                            "UPDATE knowledge_base SET example_before = %s WHERE id = %s",
                            (json.dumps(before), kb_id)
                        )
                        print(f"⚙️  normalize_flag approved: '{original}' added to mishearings of '{corrected}'")

        conn.commit()

        # ── Envelope güncelle ──────────────────────────────────────────────────
        # Entity onaylandıktan sonra ilgili transcript'in envelopunu yeniden üret
        try:
            cur.execute(
                """SELECT t.id, t.normalized_text, t.note_date, e2.source_note_id
                   FROM entities e
                   JOIN transcripts t ON t.id = e.transcript_id
                   JOIN envelopes e2 ON e2.transcript_id = t.id
                   WHERE e.id = %s""",
                (entity_id,)
            )
            tr = cur.fetchone()
            if tr:
                transcript_id, normalized_text, note_date, source_note_id = tr
                all_entities = get_entities_by_transcript(transcript_id)
                from core.envelope import emit_envelope
                new_envelope = emit_envelope(
                    entities=all_entities,
                    validation_changes=[],
                    enforcer_violations=[],
                    normalized_transcript=normalized_text or "",
                    note_date=note_date,
                    source_note_id=str(source_note_id),
                )
                save_envelope(new_envelope, transcript_id=transcript_id)
                print(f"⚙️  envelope updated after approval: source_note_id={source_note_id}")
        except Exception as env_err:
            print(f"⚠️  envelope update failed: {env_err}")

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
            """SELECT u.entity_id, t.id, t.normalized_text, t.note_date, e2.source_note_id
               FROM unverified_entities u
               JOIN entities en ON en.id = u.entity_id
               JOIN transcripts t ON t.id = en.transcript_id
               LEFT JOIN envelopes e2 ON e2.transcript_id = t.id
               WHERE u.id = %s""",
            (unverified_id,)
        )
        row = cur.fetchone()
        if not row:
            return
        entity_id, transcript_id, normalized_text, note_date, source_note_id = row

        cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
        cur.execute(
            "UPDATE unverified_entities SET status = 'rejected', reviewed_at = NOW() WHERE id = %s",
            (unverified_id,)
        )
        conn.commit()

        # ── Envelope güncelle ──────────────────────────────────────────────────
        if source_note_id:
            try:
                all_entities = get_entities_by_transcript(transcript_id)
                from core.envelope import emit_envelope
                new_envelope = emit_envelope(
                    entities=all_entities,
                    validation_changes=[],
                    enforcer_violations=[],
                    normalized_transcript=normalized_text or "",
                    note_date=note_date,
                    source_note_id=str(source_note_id),
                )
                save_envelope(new_envelope, transcript_id=transcript_id)
                print(f"⚙️  envelope updated after rejection: source_note_id={source_note_id}")
            except Exception as env_err:
                print(f"⚠️  envelope update failed: {env_err}")

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

def get_recurring_info(symptom_labels: list, note_date=None, days_back: int = 7) -> dict:
    """
    For a list of symptom labels, check if they have appeared in the past N days.
    Returns dict: label → {frequency_7d, first_seen, last_seen, dates}
    Read-only — used for UI display and envelope enrichment.
    """
    if not symptom_labels or note_date is None:
        return {}

    from datetime import timedelta, date

    today = note_date if isinstance(note_date, date) else date.today()
    cutoff = today - timedelta(days=days_back)

    conn = get_db_connection()
    cur = conn.cursor()

    # symptom_labels can be a list or a dict {label: status}
    if isinstance(symptom_labels, dict):
        label_status_map = symptom_labels
    else:
        label_status_map = {label: "present" for label in symptom_labels}

    results = {}
    for label, current_status in label_status_map.items():
        target = 'absent' if current_status == 'absent' else 'present'
        cur.execute(
            """SELECT entity_date, COALESCE(attributes->>'status', 'present') AS status
               FROM entities
               WHERE entity_type = 'symptom'
               AND LOWER(label) = LOWER(%s)
               AND entity_date >= %s
               AND entity_date < %s
               ORDER BY entity_date ASC""",
            (label, cutoff, today)
        )
        rows = cur.fetchall()
        matched_dates = [
            str(r[0]) for r in rows if r[0] and
            (r[1] == 'absent' if target == 'absent' else r[1] != 'absent')
        ]
        if matched_dates:
            results[label] = {
                "frequency_7d": len(matched_dates),
                "first_seen": matched_dates[0],
                "last_seen": matched_dates[-1],
                "dates": matched_dates,
            }

    cur.close()
    conn.close()
    return results


def get_symptom_fuzzy_matches(symptom_labels: list, note_date=None, days_back: int = 30) -> dict:
    """
    For a list of symptom labels, find fuzzy matches from DB (past N days).
    Returns dict: label → list of (matched_label, score) tuples.
    Read-only — used for review UI display, no writes.
    """
    if not symptom_labels:
        return {}

    from rapidfuzz import fuzz
    from datetime import date

    # Anatomical body part tokens — both labels must share at least one
    # to be considered a valid match. Prevents cross-region false positives
    # like "left elbow pain" matching "left eye pain" via shared "left".
    BODY_PARTS = {
        # Eye and face
        "eye", "eyelid", "gland", "vision", "jaw", "face",
        # Musculoskeletal
        "knee", "elbow", "shoulder", "hip", "ankle", "foot", "toe",
        "finger", "hand", "wrist", "back", "neck", "spine", "joint",
        # Lymph / vascular
        "lymph", "node", "groin",
        # Internal / digestive
        "chest", "stomach", "abdomen", "bowel", "stool",
        # Neurological / ENT
        "head", "ear", "sinus",
        # Tracked phenomena with stable canonical labels
        "nocturia", "blepharitis", "sinusitis", "vertigo",
        "headache", "tremor", "tingling",
    }

    def _body_tokens(s):
        return {w for w in s.lower().split() if w in BODY_PARTS}

    def _similarity(a, b):
        # Require at least one shared anatomical token
        shared = _body_tokens(a) & _body_tokens(b)
        if not shared:
            return 0
        pr = fuzz.partial_ratio(a.lower(), b.lower())
        tsr = fuzz.token_sort_ratio(a.lower(), b.lower())
        return round((pr * 0.4 + tsr * 0.6), 1)

    today = note_date or date.today()

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all distinct symptom labels from past N days in one query
    from datetime import timedelta
    cutoff = today - timedelta(days=days_back)
    cur.execute(
        """SELECT DISTINCT label FROM entities
           WHERE entity_type = 'symptom'
           AND (entity_date IS NULL OR entity_date >= %s)""",
        (cutoff,)
    )
    existing_labels = [r[0] for r in cur.fetchall()]
    print(f"DEBUG fuzzy: symptom_labels={symptom_labels}")
    print(f"DEBUG fuzzy: existing_labels={existing_labels}")
    cur.close()
    conn.close()

    results = {}
    for label in symptom_labels:
        matched = []    # guard passed — valid matches
        raw_scores = [] # guard failed — shown diagnostically only

        for existing in existing_labels:
            if existing.lower() == label.lower():
                continue  # skip exact self-match
            score = _similarity(label, existing)
            if score > 0:
                matched.append((existing, score))
            else:
                # Guard failed but show raw score for diagnostics
                from rapidfuzz import fuzz as _fuzz
                raw = round((_fuzz.partial_ratio(label.lower(), existing.lower()) * 0.4 +
                             _fuzz.token_sort_ratio(label.lower(), existing.lower()) * 0.6), 1)
                if raw > 0:
                    raw_scores.append((existing, raw, "⚠️ no shared body token"))

        matched.sort(key=lambda x: x[1], reverse=True)
        raw_scores.sort(key=lambda x: x[1], reverse=True)

        # Format: matched entries + diagnostic entries (marked)
        results[label] = {
            "matches": matched[:5],
            "diagnostic": raw_scores[:3]
        }

    return results


# ─────────────────────────────────────────
# Envelopes
# ─────────────────────────────────────────

def save_envelope(envelope: dict, transcript_id: int = None) -> str:
    """
    Save a full HDS envelope to the envelopes table.
    Returns the source_note_id.

    If an envelope with the same source_note_id already exists,
    it is replaced (full re-extraction policy — Q4).
    """
    env_block = envelope.get("envelope", {})
    source_note_id = env_block.get("source_note_id")
    note_date = env_block.get("note_date")
    pipeline_version = env_block.get("pipeline_version")

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO envelopes
                (source_note_id, transcript_id, note_date, pipeline_version, payload)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source_note_id)
            DO UPDATE SET
                transcript_id    = EXCLUDED.transcript_id,
                note_date        = EXCLUDED.note_date,
                pipeline_version = EXCLUDED.pipeline_version,
                payload          = EXCLUDED.payload,
                created_at       = NOW(),
                sent_to_hds      = FALSE,
                sent_at          = NULL
            """,
            (
                source_note_id,
                transcript_id,
                note_date,
                pipeline_version,
                json.dumps(envelope),
            )
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    return source_note_id


def get_unsent_envelopes() -> list:
    """
    Return all envelopes not yet sent to HDS (sent_to_hds = FALSE).
    Used for future AWS delivery.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT source_note_id, payload, created_at
            FROM envelopes
            WHERE sent_to_hds = FALSE
            ORDER BY created_at ASC
            """
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return [
        {
            "source_note_id": str(row[0]),
            "payload": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
        }
        for row in rows
    ]


def mark_envelope_sent(source_note_id: str):
    """
    Mark an envelope as sent to HDS.
    Called after successful AWS delivery.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE envelopes
            SET sent_to_hds = TRUE, sent_at = NOW()
            WHERE source_note_id = %s
            """,
            (source_note_id,)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()