"""
reset_and_import_clean_kb.py — Mevcut KB ve corrections tablolarını temizleyip
temiz CSV'leri import eder.

UYARI: Bu script mevcut tüm KB ve corrections verilerini siler!
Önce backup alındığından emin ol.

Kullanım:
  cd ~/Desktop/Health_voice2/health2
  source venv/bin/activate
  python reset_and_import_clean_kb.py --dry-run   # önce kontrol et
  python reset_and_import_clean_kb.py             # sonra çalıştır
"""

import sys
import os
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_csv_path(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)
    return path


def reset_and_import(dry_run=False):
    from core.db import get_db_connection
    conn = get_db_connection()
    cur = conn.cursor()

    if not dry_run:
        print("🗑️  Clearing knowledge_base table...")
        cur.execute("DELETE FROM knowledge_base")
        print("🗑️  Clearing corrections table...")
        cur.execute("DELETE FROM corrections")
        conn.commit()
        print("✅ Tables cleared\n")

    # ── 1. Registry ────────────────────────────────────────────────────────
    registry_path = get_csv_path('kb_registry_clean.csv')
    registry_count = 0

    with open(registry_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row['canonical'].strip()
            entity_type = row['entity_type'].strip() or None
            mishearings = json.loads(row['mishearings']) if row['mishearings'] else []
            aliases = json.loads(row['aliases']) if row['aliases'] else []
            description = row['description'].strip()
            note = row['note'].strip()

            example_before = {}
            if aliases:
                example_before['aliases'] = aliases
            if mishearings:
                example_before['mishearings'] = mishearings

            example_after = {
                'subtype': '',
                'description': description,
                'note': note
            }

            if dry_run:
                print(f"  [REGISTRY] {canonical} ({entity_type})"
                      + (f" | mishearings: {mishearings}" if mishearings else "")
                      + (f" | aliases: {aliases}" if aliases else ""))
            else:
                cur.execute(
                    """INSERT INTO knowledge_base
                       (original_text, correction_type, corrected_value, reason,
                        example_before, example_after)
                       VALUES (%s, 'registry', %s, %s, %s, %s)""",
                    (
                        canonical,
                        entity_type,
                        'Imported from clean registry',
                        json.dumps(example_before),
                        json.dumps(example_after)
                    )
                )
            registry_count += 1

    if not dry_run:
        conn.commit()
    print(f"\n{'[DRY RUN] ' if dry_run else ''}✅ Registry: {registry_count} entries")

    # ── 2. Removals ────────────────────────────────────────────────────────
    removals_path = get_csv_path('kb_removals_clean.csv')
    removal_count = 0

    with open(removals_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip()
            reason = row['reason'].strip()

            if dry_run:
                print(f"  [REMOVAL] {label}")
            else:
                cur.execute(
                    """INSERT INTO knowledge_base
                       (original_text, correction_type, reason)
                       VALUES (%s, 'removal', %s)""",
                    (label, reason)
                )
            removal_count += 1

    if not dry_run:
        conn.commit()
    print(f"{'[DRY RUN] ' if dry_run else ''}✅ Removals: {removal_count} entries")

    # ── 3. Classifications ─────────────────────────────────────────────────
    classifications_path = get_csv_path('kb_classifications_clean.csv')
    classification_count = 0

    with open(classifications_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip()
            correct_type = row['correct_type'].strip()
            reason = row['reason'].strip()
            context = row['context'].strip()

            if dry_run:
                print(f"  [CLASSIFICATION] {label} → {correct_type}")
            else:
                cur.execute(
                    """INSERT INTO knowledge_base
                       (original_text, correction_type, corrected_value, context_hint, reason)
                       VALUES (%s, 'classification', %s, %s, %s)""",
                    (label, correct_type, context or None, reason)
                )
            classification_count += 1

    if not dry_run:
        conn.commit()
    print(f"{'[DRY RUN] ' if dry_run else ''}✅ Classifications: {classification_count} entries")

    # ── 4. Corrections ─────────────────────────────────────────────────────
    corrections_path = get_csv_path('corrections_clean.csv')
    correction_count = 0

    with open(corrections_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row['original_text'].strip()
            corrected = row['corrected_text'].strip()
            context = row['context_hint'].strip()

            if dry_run:
                print(f"  [CORRECTION] '{original}' → '{corrected}'")
            else:
                cur.execute(
                    """INSERT INTO corrections
                       (original_text, corrected_text, correction_type, is_normalization, context_hint)
                       VALUES (%s, %s, 'normalization', TRUE, %s)""",
                    (original, corrected, context or None)
                )
            correction_count += 1

    if not dry_run:
        conn.commit()
    print(f"{'[DRY RUN] ' if dry_run else ''}✅ Corrections: {correction_count} entries")

    cur.close()
    conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Import complete:")
    print(f"  {registry_count} registry entries")
    print(f"  {removal_count} removal rules")
    print(f"  {classification_count} classification corrections")
    print(f"  {correction_count} normalization corrections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not args.dry_run:
        confirm = input("⚠️  Bu işlem mevcut tüm KB ve corrections verilerini siler. Devam? (yes/no): ")
        if confirm.lower() != 'yes':
            print("İptal edildi.")
            sys.exit(0)

    reset_and_import(dry_run=args.dry_run)