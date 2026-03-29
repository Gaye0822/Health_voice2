import streamlit as st
import json
from dotenv import load_dotenv

from core.db import get_db_connection

load_dotenv()

st.set_page_config(page_title="Term Registry — Health Voice System", layout="wide")
st.title("📚 Knowledge Base Manager")
st.caption("Registry, removal rules, and classification corrections.")

ENTITY_TYPES = [
    "intake", "symptom", "activity", "machine", "device",
    "measurement", "meal", "intervention", "outcome", "test",
    "context", "theory", "outside"
]

# ─────────────────────────────────────────
# DB functions — Registry
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


# ─────────────────────────────────────────
# DB functions — Removal
# ─────────────────────────────────────────

def get_removals() -> list:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, original_text, context_hint, reason, created_at
           FROM knowledge_base
           WHERE correction_type = 'removal'
           ORDER BY created_at DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": row[0],
            "label": row[1],
            "context_hint": row[2],
            "reason": row[3],
            "created_at": row[4]
        }
        for row in rows
    ]


def save_removal_entry(label, reason, context_hint):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, context_hint, reason)
               VALUES (%s, 'removal', NULL, %s, %s)""",
            (label, context_hint or None, reason)
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


# ─────────────────────────────────────────
# DB functions — Classification
# ─────────────────────────────────────────

def get_classifications() -> list:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, original_text, corrected_value, context_hint, reason, created_at
           FROM knowledge_base
           WHERE correction_type = 'classification'
           ORDER BY created_at DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": row[0],
            "label": row[1],
            "correct_type": row[2],
            "context_hint": row[3],
            "reason": row[4],
            "created_at": row[5]
        }
        for row in rows
    ]


def save_classification_entry(label, correct_type, reason, context_hint):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO knowledge_base
               (original_text, correction_type, corrected_value, context_hint, reason)
               VALUES (%s, 'classification', %s, %s, %s)""",
            (label, correct_type, context_hint or None, reason)
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


# ─────────────────────────────────────────
# Shared delete
# ─────────────────────────────────────────

def delete_entry(entry_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM knowledge_base WHERE id = %s", (entry_id,))
    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────
# UI — Tabs
# ─────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📋 Registry", "🗑️ Removals", "🔀 Classifications"])


# ══════════════════════════════════════════
# TAB 1 — Registry
# ══════════════════════════════════════════

with tab1:
    st.header("Add New Term")
    st.caption("Register a canonical term so the system knows what it is and how it may be written.")

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
    st.header("Registry")

    entries = get_registry()

    if not entries:
        st.info("No entries yet.")
    else:
        st.write(f"**{len(entries)} registered terms**")

        search = st.text_input("🔍 Search", placeholder="QIAstat, Novothor...", key="reg_search")
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
                    if st.button("🗑️ Delete", key=f"del_reg_{entry['id']}"):
                        delete_entry(entry["id"])
                        st.toast(f"Deleted: {entry['canonical']}")
                        st.rerun()


# ══════════════════════════════════════════
# TAB 2 — Removals
# ══════════════════════════════════════════

with tab2:
    st.header("Add Removal Rule")
    st.caption("Tell the validator that a specific label should never become a structured entity.")

    with st.form("add_removal"):
        label = st.text_input(
            "Label to remove *",
            placeholder="wake up supplements"
        )
        reason = st.text_area(
            "Reason * — this is injected into the validator's chain of thought",
            placeholder="Vague group label, not resolvable to specific substances. Cannot be tracked over time.",
            height=100
        )
        context_hint = st.text_input(
            "Context hint (optional)",
            placeholder="only when mentioned as a group without individual names"
        )

        submitted2 = st.form_submit_button("➕ Add Removal Rule", type="primary")

        if submitted2:
            if not label or not reason:
                st.error("Label and reason are required.")
            else:
                success = save_removal_entry(
                    label=label,
                    reason=reason,
                    context_hint=context_hint
                )
                if success:
                    st.success(f"✅ Removal rule added for '{label}'.")
                    st.rerun()

    st.divider()
    st.header("Removal Rules")

    removals = get_removals()

    if not removals:
        st.info("No removal rules yet.")
    else:
        st.write(f"**{len(removals)} removal rules**")

        for entry in removals:
            with st.expander(f"**{entry['label']}**"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Reason:** {entry['reason']}")
                    if entry["context_hint"]:
                        st.write(f"**Context:** {entry['context_hint']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"del_rem_{entry['id']}"):
                        delete_entry(entry["id"])
                        st.toast(f"Deleted: {entry['label']}")
                        st.rerun()


# ══════════════════════════════════════════
# TAB 3 — Classifications
# ══════════════════════════════════════════

with tab3:
    st.header("Add Classification Correction")
    st.caption("Tell the validator that a label was assigned the wrong entity type.")

    with st.form("add_classification"):
        col1, col2 = st.columns(2)

        with col1:
            label_c = st.text_input(
                "Label *",
                placeholder="intestinal improvement"
            )
            correct_type = st.selectbox("Correct entity type *", ENTITY_TYPES)

        with col2:
            reason_c = st.text_area(
                "Reason * — injected into validator chain of thought",
                placeholder="Observed directional change linked to FMT intervention. Should be outcome, not symptom.",
                height=100
            )
            context_hint_c = st.text_input(
                "Context hint (optional)",
                placeholder="when mentioned in relation to FMT or gut health intervention"
            )

        submitted3 = st.form_submit_button("➕ Add Classification Correction", type="primary")

        if submitted3:
            if not label_c or not correct_type or not reason_c:
                st.error("Label, correct type, and reason are required.")
            else:
                success = save_classification_entry(
                    label=label_c,
                    correct_type=correct_type,
                    reason=reason_c,
                    context_hint=context_hint_c
                )
                if success:
                    st.success(f"✅ Classification correction added for '{label_c}' → {correct_type}.")
                    st.rerun()

    st.divider()
    st.header("Classification Corrections")

    classifications = get_classifications()

    if not classifications:
        st.info("No classification corrections yet.")
    else:
        st.write(f"**{len(classifications)} classification corrections**")

        for entry in classifications:
            with st.expander(f"**{entry['label']}** → {entry['correct_type']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Reason:** {entry['reason']}")
                    if entry["context_hint"]:
                        st.write(f"**Context:** {entry['context_hint']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"del_cls_{entry['id']}"):
                        delete_entry(entry["id"])
                        st.toast(f"Deleted: {entry['label']}")
                        st.rerun()