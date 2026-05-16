"""
export_envelopes.py — Export envelopes from DB to individual JSON files.

Usage:
    python export_envolops.py                        # exports all envelopes
    python export_envolops.py --limit 5              # exports last 5
    python export_envolops.py --start 57 --end 85    # exports rows 57 to 85
    python export_envolops.py --out ./output         # custom output directory

Each envelope is saved as a separate JSON file named by source_note_id and note_date:
    exports/2025-12-03_a1b2c3d4.json
"""

import json
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.db import get_db_connection


def export_envelopes(output_dir: str = "./exports", limit: int = None, start: int = None, end: int = None):
    """
    Read envelopes from DB and write each as a separate JSON file.
    Files are named: <note_date>_<source_note_id[:8]>.json
    """
    os.makedirs(output_dir, exist_ok=True)

    conn = get_db_connection()
    cur = conn.cursor()

    if start is not None and end is not None:
        query = f"""
            SELECT source_note_id, note_date, payload, created_at
            FROM (
                SELECT source_note_id, note_date, payload, created_at,
                       ROW_NUMBER() OVER (ORDER BY note_date ASC, created_at ASC) AS rn
                FROM envelopes
            ) ranked
            WHERE rn BETWEEN {start} AND {end}
        """
    else:
        query = """
            SELECT source_note_id, note_date, payload, created_at
            FROM envelopes
            ORDER BY note_date ASC, created_at ASC
        """
        if limit:
            query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("⚠️  No envelopes found in DB.")
        return

    print(f"📦 Exporting {len(rows)} envelope(s) to {output_dir}/\n")

    exported = []
    for row in rows:
        source_note_id, note_date, payload, created_at = row

        date_str = str(note_date) if note_date else "unknown_date"
        id_str = str(source_note_id)[:8]
        filename = f"{date_str}_{id_str}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        print(f"  ✅ {filename}")
        exported.append(filepath)

    print(f"\n✅ Done — {len(exported)} file(s) written to {output_dir}/")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export envelopes from DB to JSON files")
    parser.add_argument("--out", default="./exports", help="Output directory (default: ./exports)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of envelopes to export")
    parser.add_argument("--start", type=int, default=None, help="Start row number (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End row number (1-indexed)")
    args = parser.parse_args()

    export_envelopes(output_dir=args.out, limit=args.limit, start=args.start, end=args.end)