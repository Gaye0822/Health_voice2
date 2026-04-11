"""
create_unverified_entities_table.py — unverified_entities tablosunu oluşturur.

Kullanım:
  cd ~/Desktop/Health_voice2/health2
  source venv/bin/activate
  python create_unverified_entities_table.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.db import get_db_connection


def create_table():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS unverified_entities (
            id SERIAL PRIMARY KEY,
            transcript_id INTEGER REFERENCES transcripts(id),
            entity_id INTEGER REFERENCES entities(id),
            entity_json JSONB NOT NULL,
            flag_reason TEXT NOT NULL,
            context TEXT,
            reasoning TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            corrected_label TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            reviewed_at TIMESTAMPTZ
        )
    """)

    # entities tablosuna unverified field ekle (yoksa)
    cur.execute("""
        ALTER TABLE entities
        ADD COLUMN IF NOT EXISTS unverified BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS flag_reason TEXT
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ unverified_entities table created")
    print("✅ entities table updated with unverified and flag_reason columns")


if __name__ == "__main__":
    create_table()