"""
import_gabriel_registry.py — Gabriel'in Registry_of_items.xlsx dosyasını KB'ye import eder.

Kurallar:
- Canonical term → original_text
- Class → corrected_value (entity_type)
- Mishearings → example_before["mishearings"] (fuzzy match için)
- Aliases → example_after["aliases"] (sadece LLM context için)
- Note → example_after["note"]
- Description → example_after["description"] (subtype'dan türetilir)

Zaten var olan kayıtları skip eder (canonical term eşleşmesi ile kontrol).

Kullanım:
  cd ~/Desktop/health2
  source venv/bin/activate
  python import_gabriel_registry.py [--sheet All] [--dry-run]
"""

import sys
import os
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Class → entity_type mapping
CLASS_TO_ENTITY_TYPE = {
    "prescipriton medication": "intake",
    "prescription medication": "intake",
    "medication": "intake",
    "supplement": "intake",
    "supplement/peptide": "intake",
    "device/test platform": "machine",
    "device": "device",
    "monitoring &wearable device": "device",
    "monitoring & wearable device": "device",
    "sleep &circadian device": "device",
    "sleep & circadian device": "device",
    "exercise&performance device": "device",
    "exercise & performance device": "device",
    "recovery&theraputic device": "device",
    "recovery & therapeutic device": "device",
    "light therapy device": "machine",
    "neurological&neurostimualtion device": "machine",
    "neurological & neurostimulation device": "machine",
    "metabolic& breath testing device": "device",
    "metabolic & breath testing device": "device",
    "test": "test",
    "activity": "activity",
    "activity/intervention": "activity",
    "exercise activity": "activity",
    "exercise activity / device": "activity",
    "intervention": "intervention",
    "other": "other",
}


def normalize_entity_type(class_str: str) -> str:
    if not class_str or pd.isna(class_str):
        return None
    key = class_str.strip().lower()
    return CLASS_TO_ENTITY_TYPE.get(key, None)


def parse_list(cell) -> list:
    """Parse a comma/semicolon separated cell into a list."""
    if not cell or pd.isna(cell):
        return []
    parts = str(cell).replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def import_registry(xlsx_path: str, sheet_name: str = "All", dry_run: bool = False):
    from core.db import get_db_connection

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    # Column mapping
    col_canonical = "Canonical term"
    col_aliases = "Aliases"
    col_mishearings = "Mishearings"
    col_class = "Class (medication, supplement, device, test, brand, modality, etc.)"
    col_note = "Note (optional normalization hint if there's an obvious common confusion)"

    conn = get_db_connection()
    cur = conn.cursor()

    added = 0
    skipped = 0
    updated = 0

    for _, row in df.iterrows():
        canonical = str(row.get(col_canonical, "")).strip()
        if not canonical or canonical == "nan":
            continue

        aliases = parse_list(row.get(col_aliases, ""))
        mishearings = parse_list(row.get(col_mishearings, ""))
        entity_type = normalize_entity_type(row.get(col_class, ""))
        note = str(row.get(col_note, "")).strip() if not pd.isna(row.get(col_note, "")) else ""
        if note == "nan":
            note = ""

        example_before = {}
        if aliases:
            example_before["aliases"] = aliases
        if mishearings:
            example_before["mishearings"] = mishearings

        example_after = {
            "subtype": "",
            "description": "",
            "note": note
        }

        # Check if already exists
        cur.execute(
            """SELECT id, example_before, example_after FROM knowledge_base
               WHERE correction_type = 'registry'
               AND LOWER(original_text) = LOWER(%s)""",
            (canonical,)
        )
        existing = cur.fetchone()

        if existing:
            entry_id = existing[0]
            existing_before = existing[1] if existing[1] else {}
            existing_after = existing[2] if existing[2] else {}

            # Merge mishearings and aliases into existing entry
            existing_aliases = existing_before.get("aliases", [])
            existing_mishearings = existing_before.get("mishearings", [])

            new_aliases = list(set(existing_aliases + aliases))
            new_mishearings = list(set(existing_mishearings + mishearings))

            merged_before = dict(existing_before)
            merged_before["aliases"] = new_aliases
            merged_before["mishearings"] = new_mishearings

            merged_after = dict(existing_after)
            if note and not merged_after.get("note"):
                merged_after["note"] = note

            if not dry_run:
                cur.execute(
                    """UPDATE knowledge_base
                       SET example_before = %s, example_after = %s
                       WHERE id = %s""",
                    (json.dumps(merged_before), json.dumps(merged_after), entry_id)
                )
                conn.commit()
            print(f"  🔄 Updated: {canonical} (merged aliases/mishearings)")
            updated += 1
        else:
            if not dry_run:
                cur.execute(
                    """INSERT INTO knowledge_base
                       (original_text, correction_type, corrected_value, context_hint, reason,
                        example_before, example_after)
                       VALUES (%s, 'registry', %s, %s, %s, %s, %s)""",
                    (
                        canonical,
                        entity_type,
                        None,
                        "Imported from Gabriel's registry list",
                        json.dumps(example_before),
                        json.dumps(example_after)
                    )
                )
                conn.commit()
            print(f"  ✅ Added: {canonical} ({entity_type or 'unknown type'})"
                  + (f" | mishearings: {mishearings}" if mishearings else "")
                  + (f" | aliases: {aliases}" if aliases else ""))
            added += 1

    cur.close()
    conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Import complete:")
    print(f"  {added} new entries added")
    print(f"  {updated} existing entries updated (aliases/mishearings merged)")
    print(f"  {skipped} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default="All", help="Sheet name to import (default: All)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Registry_of_items.xlsx")
    if not os.path.exists(xlsx_path):
        print(f"❌ File not found: {xlsx_path}")
        print("Place Registry_of_items.xlsx in the project root and try again.")
        sys.exit(1)

    print(f"Importing from sheet: {args.sheet}")
    print(f"Dry run: {args.dry_run}")
    print()
    import_registry(xlsx_path, sheet_name=args.sheet, dry_run=args.dry_run)