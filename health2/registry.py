import streamlit as st
import json
from dotenv import load_dotenv

from core.db import get_db_connection

load_dotenv()

st.set_page_config(page_title="Term Registry — Health Voice System", layout="wide")
st.title("📚 Canonical Term Registry")
st.caption("Register recurring named things so the system knows what they are.")

ENTITY_TYPES = [
    "intake", "symptom", "activity", "machine", "device",
    "measurement", "meal", "intervention", "outcome", "test",
    "context", "theory", "outside"
]

# ─────────────────────────────────────────
# DB functions
# ─────────────────────────────────────────

def get_registry() -> list:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, original_text, corrected_value, context_hint, reason,
                  example_before, example_after, created_at
           FROM knowledge_base
           WHERE correction_type = 'registry'
           ORDER BY created_at DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    results = []
    for row in rows:
        before = row[5] if row[5] else {}
        after = row[6] if row[6] else {}
        results.append({
            "id": row[0],
            "canonical": row[1],
            "entity_type": row[2],
            "aliases": before.get("aliases", []),
            "subtype": after.get("subtype", ""),
            "description": after.get("description", ""),
            "created_at": row[7]
        })
    return results


def save_registry_entry(canonical, entity_type, aliases, subtype, description):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, context_hint, reason,
                example_before, example_after)
               VALUES (%s, 'registry', %s, %s, %s, %s, %s)""",
            (
                canonical,
                entity_type,
                None,
                None,
                json.dumps({"aliases": aliases}),
                json.dumps({"subtype": subtype, "description": description})
            )
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def delete_registry_entry(entry_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM knowledge_base WHERE id = %s AND correction_type = 'registry'",
        (entry_id,)
    )
    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────
# Add new entry
# ─────────────────────────────────────────

st.header("Add New Term")

with st.form("add_registry"):
    col1, col2 = st.columns(2)

    with col1:
        canonical = st.text_input("Canonical name *", placeholder="QIAstat")
        entity_type = st.selectbox("Entity type *", ENTITY_TYPES)
        subtype = st.text_input("Subtype", placeholder="syndromic diagnostic platform")

    with col2:
        aliases_raw = st.text_input(
            "Aliases / ASR errors (comma separated)",
            placeholder="QI stat, QI Stat, kaistat"
        )
        description = st.text_area(
            "Description",
            placeholder="Rapid molecular test system for respiratory, GI and meningitis panels.",
            height=100
        )

    submitted = st.form_submit_button("➕ Add", type="primary")

    if submitted:
        if not canonical or not entity_type:
            st.error("Canonical name and entity type are required.")
        else:
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []
            success = save_registry_entry(
                canonical=canonical,
                entity_type=entity_type,
                aliases=aliases,
                subtype=subtype,
                description=description
            )
            if success:
                st.success(f"✅ '{canonical}' added.")
                st.rerun()

st.divider()

# ─────────────────────────────────────────
# View entries
# ─────────────────────────────────────────

st.header("Registry")

entries = get_registry()

if not entries:
    st.info("No entries yet.")
else:
    st.write(f"**{len(entries)} registered terms**")

    search = st.text_input("🔍 Search", placeholder="QIAstat, Novothor...")
    if search:
        entries = [e for e in entries if
                   search.lower() in e["canonical"].lower() or
                   any(search.lower() in a.lower() for a in e["aliases"])]

    for entry in entries:
        aliases_str = ", ".join(entry["aliases"]) if entry["aliases"] else "—"
        with st.expander(f"**{entry['canonical']}** — {entry['entity_type']}"):
            col1, col2 = st.columns([4, 1])
            with col1:
                if entry["subtype"]:
                    st.write(f"**Subtype:** {entry['subtype']}")
                if entry["description"]:
                    st.write(f"**Description:** {entry['description']}")
                st.write(f"**Aliases:** {aliases_str}")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{entry['id']}"):
                    delete_registry_entry(entry["id"])
                    st.toast(f"Deleted: {entry['canonical']}")
                    st.rerun()