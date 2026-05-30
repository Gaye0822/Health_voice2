"""
symptom_label_review.py
─────────────────────────────────────────────────────────────────────────────
Mevcut sisteme DOKUNMADAN çalışan bağımsız modül.

Görevleri:
  1. create_symptom_label_review_table()  — tabloyu oluşturur (bir kez çalıştır)
  2. save_symptom_for_review()            — pipeline'dan low-confidence symptom gelince çağrılır
  3. get_pending_symptom_reviews()        — UI için bekleyen kayıtları döner
  4. approve_symptom_label()              — Gabriel onaylarsa KB'ye yazar
  5. correct_symptom_label()             — Gabriel değiştirirse yeni label ile KB'ye yazar
  6. reject_symptom_label()              — Gabriel reddederse kaydı reddedildi olarak işaretler

Tablo: symptom_label_reviews
  - mevcut entities / unverified_entities tablolarına dokunmaz
  - mevcut knowledge_base tablosuna sadece Gabriel onayladıktan SONRA yazar

Kullanım:
  from symptom_label_review import (
      save_symptom_for_review,
      get_pending_symptom_reviews,
      approve_symptom_label,
      correct_symptom_label,
      reject_symptom_label,
  )
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.db import get_db_connection as _get_conn


# ─────────────────────────────────────────
# 1. Tablo oluşturma — bir kez çalıştır
# ─────────────────────────────────────────

def create_symptom_label_review_table():
    """
    symptom_label_reviews tablosunu oluşturur.
    Güvenlidir — tablo zaten varsa hiçbir şey yapmaz.
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS symptom_label_reviews (
            id                  SERIAL PRIMARY KEY,
            transcript_id       INTEGER,
            entity_id           INTEGER,
            proposed_label      TEXT NOT NULL,
            transcript_context  TEXT,
            llm_reasoning       TEXT,
            entity_json         JSONB,
            status              TEXT NOT NULL DEFAULT 'pending',
            approved_label      TEXT,
            kb_entry_id         INTEGER,
            note_date           DATE,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            reviewed_at         TIMESTAMPTZ
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ symptom_label_reviews table ready")


# ─────────────────────────────────────────
# 2. Low-confidence symptom kaydetme
# ─────────────────────────────────────────

def save_symptom_for_review(
    proposed_label: str,
    transcript_context: str,
    llm_reasoning: str,
    entity_json: dict,
    transcript_id: int = None,
    entity_id: int = None,
    note_date=None,
) -> int:
    """
    Pipeline'dan gelen low-confidence symptom'u review kuyruğuna yazar.
    Mevcut entities tablosuna dokunmaz.
    Returns: yeni kaydın id'si
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        # Duplicate check — same transcript + same label already queued → skip
        cur.execute(
            """SELECT id FROM symptom_label_reviews
               WHERE transcript_id = %s AND proposed_label = %s AND status = 'pending'""",
            (transcript_id, proposed_label)
        )
        existing = cur.fetchone()
        if existing:
            print(f"⚙️  symptom_label_review: duplicate skipped (transcript_id={transcript_id}, label='{proposed_label}')")
            cur.close()
            conn.close()
            return existing[0]

        cur.execute(
            """INSERT INTO symptom_label_reviews
               (transcript_id, entity_id, proposed_label, transcript_context,
                llm_reasoning, entity_json, note_date)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                transcript_id,
                entity_id,
                proposed_label,
                transcript_context,
                llm_reasoning,
                json.dumps(entity_json),
                note_date,
            )
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        print(f"⚙️  symptom_label_review: queued '{proposed_label}' (id={new_id})")
        return new_id
    except Exception as e:
        conn.rollback()
        print(f"⚠️  save_symptom_for_review error: {e}")
        return -1
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────
# 3. UI için bekleyen kayıtları getir
# ─────────────────────────────────────────

def get_pending_symptom_reviews(status: str = "pending") -> list:
    """
    Belirtilen status'taki symptom review kayıtlarını döner.
    status: "pending" | "approved" | "corrected" | "rejected"
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, transcript_id, entity_id, proposed_label,
                  transcript_context, llm_reasoning, entity_json,
                  status, approved_label, note_date, created_at
           FROM symptom_label_reviews
           WHERE status = %s
           ORDER BY created_at DESC""",
        (status,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id":                 row[0],
            "transcript_id":      row[1],
            "entity_id":          row[2],
            "proposed_label":     row[3],
            "transcript_context": row[4],
            "llm_reasoning":      row[5],
            "entity_json":        row[6],
            "status":             row[7],
            "approved_label":     row[8],
            "note_date":          row[9],
            "created_at":         row[10],
        })
    return results


# ─────────────────────────────────────────
# 4. Onaylama — KB'ye yaz
# ─────────────────────────────────────────

def approve_symptom_label(review_id: int, description: str = None) -> bool:
    """
    Gabriel proposed_label'ı olduğu gibi onayladı.
    KB'ye canonical symptom olarak ekler.
    description: Gabriel'in bu label için verdiği tanım (opsiyonel ama önerilir)
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        # Review kaydını getir
        cur.execute(
            "SELECT proposed_label, entity_json, transcript_id FROM symptom_label_reviews WHERE id = %s",
            (review_id,)
        )
        row = cur.fetchone()
        if not row:
            print(f"⚠️  approve_symptom_label: id={review_id} not found")
            return False

        label = row[0]
        entity_json = row[1] or {}
        transcript_id_val = row[2]

        # KB'ye yaz — normalization olarak kaydet, pipeline bu tipi tanıyor
        # label → label (aynı, ama KB'de kayıtlı olduğunu belirtir)
        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, reason, example_before, example_after)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT DO NOTHING
               RETURNING id""",
            (
                label,
                "registry",
                "symptom",
                description or f"Approved by Gabriel from symptom label review (id={review_id})",
                json.dumps({"description": description or "", "aliases": [], "entity_type": "symptom"}),
                json.dumps({"canonical_label": label})
            )
        )
        kb_id = cur.fetchone()[0]

        # Entity'yi güncelle (varsa)
        entity_id = None
        cur.execute(
            "SELECT entity_id FROM symptom_label_reviews WHERE id = %s",
            (review_id,)
        )
        eid_row = cur.fetchone()
        if eid_row:
            entity_id = eid_row[0]

        if entity_id:
            cur.execute(
                "UPDATE entities SET unverified = FALSE WHERE id = %s",
                (entity_id,)
            )

        # Review kaydını güncelle
        cur.execute(
            """UPDATE symptom_label_reviews
               SET status = 'approved', approved_label = %s,
                   kb_entry_id = %s, reviewed_at = NOW()
               WHERE id = %s""",
            (label, kb_id, review_id)
        )
        conn.commit()

        # Envelope patch — label zaten aynı, kb_matched güncellenir
        _patch_envelope_label(transcript_id_val, label, label)

        print(f"✅ symptom_label approved: '{label}' → KB id={kb_id}")
        return True

    except Exception as e:
        conn.rollback()
        print(f"⚠️  approve_symptom_label error: {e}")
        return False
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────
# 5. Düzeltme — yeni label ile KB'ye yaz
# ─────────────────────────────────────────

def correct_symptom_label(review_id: int, new_label: str, description: str = None) -> bool:
    """
    Gabriel proposed_label'ı değiştirdi.
    Yeni label'ı KB'ye ekler, proposed_label'ı alias olarak kaydeder.
    Entity varsa label'ı günceller.
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT proposed_label, entity_id, transcript_id FROM symptom_label_reviews WHERE id = %s",
            (review_id,)
        )
        row = cur.fetchone()
        if not row:
            return False

        original_proposed = row[0]
        entity_id = row[1]
        transcript_id_val = row[2]

        # KB'ye registry olarak yaz — new_label canonical symptom
        # description = transcript context (tanımı) + Gabriel'in açıklaması
        # Pipeline mention stage'de bunu okuyup eşleştirir
        cur.execute(
            """SELECT transcript_context FROM symptom_label_reviews WHERE id = %s""",
            (review_id,)
        )
        ctx_row = cur.fetchone()
        transcript_context = ctx_row[0] if ctx_row else ""

        kb_description = description or transcript_context or f"Symptom labeled by Gabriel (review id={review_id})"

        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, reason, example_before, example_after)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                new_label,
                "registry",
                "symptom",
                f"Corrected by Gabriel: '{original_proposed}' → '{new_label}' (review id={review_id})",
                json.dumps({
                    "aliases": [],
                    "mishearings": [],
                    "description": kb_description
                }),
                json.dumps({"canonical_label": new_label, "subtype": "", "description": kb_description})
            )
        )
        kb_id = cur.fetchone()[0]

        # Entity label'ını güncelle (varsa)
        if entity_id:
            cur.execute(
                """UPDATE entities
                   SET label = %s, unverified = FALSE
                   WHERE id = %s""",
                (new_label, entity_id)
            )

        # Review kaydını güncelle
        cur.execute(
            """UPDATE symptom_label_reviews
               SET status = 'corrected', approved_label = %s,
                   kb_entry_id = %s, reviewed_at = NOW()
               WHERE id = %s""",
            (new_label, kb_id, review_id)
        )
        conn.commit()

        # Envelope patch — label değişti
        _patch_envelope_label(transcript_id_val, original_proposed, new_label)

        print(f"✅ symptom_label corrected: '{original_proposed}' → '{new_label}' (KB id={kb_id})")
        return True

    except Exception as e:
        conn.rollback()
        print(f"⚠️  correct_symptom_label error: {e}")
        return False
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────
# 6. Reddetme
# ─────────────────────────────────────────

def reject_symptom_label(review_id: int) -> bool:
    """
    Gabriel bu symptom'u reddetti — KB'ye eklenmez.
    Entity varsa unverified=FALSE olarak işaretlenir (temizlendi sayılır).
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT entity_id FROM symptom_label_reviews WHERE id = %s",
            (review_id,)
        )
        row = cur.fetchone()
        entity_id = row[0] if row else None

        if entity_id:
            cur.execute(
                "UPDATE entities SET unverified = FALSE WHERE id = %s",
                (entity_id,)
            )

        cur.execute(
            """UPDATE symptom_label_reviews
               SET status = 'rejected', reviewed_at = NOW()
               WHERE id = %s""",
            (review_id,)
        )
        conn.commit()
        print(f"✅ symptom_label rejected (review id={review_id})")
        return True

    except Exception as e:
        conn.rollback()
        print(f"⚠️  reject_symptom_label error: {e}")
        return False
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────
# Yardımcı — Envelope patch
# ─────────────────────────────────────────

def _patch_envelope_label(transcript_id: int, old_label: str, new_label: str):
    """
    Envelope payload'ındaki symptom label'ını patch eder.
    event_layer içindeki ilk eşleşen 'label' alanını günceller.
    """
    if not transcript_id:
        return
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, payload FROM envelopes WHERE transcript_id = %s",
            (transcript_id,)
        )
        row = cur.fetchone()
        if not row:
            return

        envelope_id = row[0]
        payload = row[1]
        if isinstance(payload, str):
            import json as _json
            payload = _json.loads(payload)

        # event_layer içinde label'ı güncelle
        changed = False
        event_layer = payload.get("layers", {}).get("event_layer", [])
        for entity in event_layer:
            if (entity.get("type") == "symptom" and
                    entity.get("label", "").lower() == old_label.lower()):
                entity["label"] = new_label
                changed = True
                break

        if changed:
            import json as _json
            cur.execute(
                "UPDATE envelopes SET payload = %s WHERE id = %s",
                (_json.dumps(payload), envelope_id)
            )
            conn.commit()
            print(f"⚙️  envelope patched: '{old_label}' → '{new_label}' (transcript_id={transcript_id})")
    except Exception as e:
        conn.rollback()
        print(f"⚠️  envelope patch error: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────
# Tablo oluşturma — doğrudan çalıştırılırsa
# ─────────────────────────────────────────

if __name__ == "__main__":
    create_symptom_label_review_table()