"""
migrate_mishearings_to_kb.py — Corrections tablosundaki kayıtları kontrol eder.
Eğer corrected_text KB registry'de varsa → original_text'i mishearing olarak ekler
ve corrections tablosundan siler.
KB'de yoksa → corrections'ta bırakır.

Kullanım:
  cd ~/Desktop/Health_voice2/health2
  source venv/bin/activate
  python migrate_mishearings_to_kb.py --dry-run
  python migrate_mishearings_to_kb.py
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def migrate(dry_run=False):
    from core.db import get_db_connection
    conn = get_db_connection()
    cur = conn.cursor()

    # Tüm corrections'ı çek
    cur.execute(
        """SELECT id, original_text, corrected_text FROM corrections
           WHERE is_normalization = TRUE AND correction_type = 'normalization'
           ORDER BY id"""
    )
    corrections = cur.fetchall()
    print(f"Found {len(corrections)} corrections to check.\n")

    moved = 0
    kept = 0

    for corr_id, original, corrected in corrections:
        if not original or not corrected:
            kept += 1
            continue

        # KB'de corrected term var mı?
        cur.execute(
            """SELECT id, example_before FROM knowledge_base
               WHERE correction_type = 'registry'
               AND LOWER(original_text) = LOWER(%s)""",
            (corrected,)
        )
        existing = cur.fetchone()

        if existing:
            entry_id = existing[0]
            before_json = existing[1] if existing[1] else {}
            mishearings = before_json.get("mishearings", [])
            aliases = before_json.get("aliases", [])

            # Zaten mishearing listesinde var mı?
            already_there = (
                original in mishearings or
                original.lower() in [m.lower() for m in mishearings]
            )

            if already_there:
                print(f"  ⏭️  SKIP (already in KB): '{original}' → '{corrected}'")
                if not dry_run:
                    cur.execute("DELETE FROM corrections WHERE id = %s", (corr_id,))
                moved += 1
            else:
                mishearings.append(original)
                updated_before = {**before_json, "mishearings": mishearings}
                print(f"  ✅ MOVE to KB mishearing: '{original}' → '{corrected}'")
                if not dry_run:
                    cur.execute(
                        "UPDATE knowledge_base SET example_before = %s WHERE id = %s",
                        (json.dumps(updated_before), entry_id)
                    )
                    cur.execute("DELETE FROM corrections WHERE id = %s", (corr_id,))
                moved += 1
        else:
            print(f"  📌 KEEP in corrections: '{original}' → '{corrected}' (not in KB)")
            kept += 1

    if not dry_run:
        conn.commit()

    cur.close()
    conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migration complete:")
    print(f"  {moved} moved to KB (or already there, deleted from corrections)")
    print(f"  {kept} kept in corrections (corrected term not in KB registry)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)