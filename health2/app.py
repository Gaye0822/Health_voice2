import streamlit as st
import json
import os
from dotenv import load_dotenv

from core.transcribe import transcribe_audio
from core.normalize import normalize_transcript
from core.mention import extract_mentions
from core.structure import structure_mentions
from core.validate import validate_entities
from core.db import (
    save_transcript,
    save_entities,
    save_correction,
    save_knowledge,
    get_corrections_from_db
)

load_dotenv()

# --- Config ---
st.set_page_config(page_title="Health Voice System", layout="wide")
st.title("🎙️ Health Voice System")

# --- Session State ---
defaults = {
    "transcript": None,
    "normalized": None,
    "edited": None,
    "low_confidence_segments": [],
    "mentions": None,
    "entities": None,
    "validation_changes": [],
    "transcript_id": None,
    "step": "upload"
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Type icons ---
TYPE_ICONS = {
    "symptom": "🔴",
    "intake": "💊",
    "activity": "🏃",
    "machine": "🔧",
    "device": "📱",
    "measurement": "📊",
    "meal": "🍽️",
    "intervention": "📋",
    "outcome": "📈",
    "test": "🧪",
    "context": "💬",
    "theory": "💭",
    "outside": "📤",
    "other": "📌",
}

CANDIDATE_TYPES = [
    "intake",
    "symptom",
    "activity",
    "machine",
    "device",
    "measurement",
    "meal",
    "intervention",
    "outcome",
    "test",
    "context",
    "theory",
    "outside",
    "other"
]

# ─────────────────────────────────────────
# STEP 1 — Upload Audio or Paste Transcript
# ─────────────────────────────────────────
if st.session_state.step == "upload":
    st.header("Step 1 — Input")

    input_mode = st.radio(
        "Input method",
        ["🎙️ Audio file", "📋 JSON transcript"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── Audio mode ────────────────────────────────────────────────────────
    if input_mode == "🎙️ Audio file":
        uploaded_file = st.file_uploader("Upload voice note", type=["m4a", "mp3", "wav", "ogg"])

        if uploaded_file and st.button("Transcribe", type="primary"):
            with st.spinner("Transcribing with Whisper..."):
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                raw = transcribe_audio(temp_path)

            with st.spinner("Normalizing..."):
                result = normalize_transcript(raw)

            st.session_state.transcript = raw
            st.session_state.normalized = result["normalized_text"]
            st.session_state.edited = result["normalized_text"]
            st.session_state.low_confidence_segments = result["low_confidence_segments"]
            st.session_state.step = "review_transcript"
            st.rerun()

    # ── JSON transcript mode ──────────────────────────────────────────────
    else:
        st.caption('Upload a `.json` file containing a single `{"Content": "...", "Created_at": "..."}` object or an array of them.')

        json_file = st.file_uploader("Upload JSON transcript", type=["json"], label_visibility="collapsed")

        if json_file and st.button("Load Transcript", type="primary"):
            try:
                parsed = json.loads(json_file.read().decode("utf-8"))

                # Normalise: accept single object or array
                if isinstance(parsed, dict):
                    entries = [parsed]
                elif isinstance(parsed, list):
                    entries = parsed
                else:
                    st.error("Expected a JSON object or array.")
                    st.stop()

                # Validate each entry has Content
                if not all("Content" in e for e in entries):
                    st.error('Each entry must have a "Content" field.')
                    st.stop()

                if len(entries) == 1:
                    # Single transcript — go straight to normalize
                    raw = entries[0]["Content"]
                    created_at = entries[0].get("Created_at", "")

                    with st.spinner("Normalizing..."):
                        result = normalize_transcript(raw)

                    st.session_state.transcript = raw
                    st.session_state.normalized = result["normalized_text"]
                    st.session_state.edited = result["normalized_text"]
                    st.session_state.low_confidence_segments = result["low_confidence_segments"]
                    st.session_state.json_created_at = created_at
                    st.session_state.step = "review_transcript"
                    st.rerun()

                else:
                    # Multiple transcripts — let user pick which one to process
                    st.session_state._json_entries = entries
                    st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {e}")

        # Multi-entry picker (shown after a multi-entry paste)
        if hasattr(st.session_state, "_json_entries") and st.session_state._json_entries:
            entries = st.session_state._json_entries
            st.divider()
            st.write(f"**{len(entries)} transcripts found — select one to process:**")

            for i, entry in enumerate(entries):
                preview = entry["Content"][:120].replace("\n", " ")
                created = entry.get("Created_at", "")
                label = f"**[{i+1}]** {created[:10]}  —  {preview}…"
                if st.button(label, key=f"pick_entry_{i}"):
                    raw = entry["Content"]
                    with st.spinner("Normalizing..."):
                        result = normalize_transcript(raw)
                    st.session_state.transcript = raw
                    st.session_state.normalized = result["normalized_text"]
                    st.session_state.edited = result["normalized_text"]
                    st.session_state.low_confidence_segments = result["low_confidence_segments"]
                    st.session_state.json_created_at = entry.get("Created_at", "")
                    st.session_state._json_entries = []
                    st.session_state.step = "review_transcript"
                    st.rerun()

# ─────────────────────────────────────────
# STEP 2 — Review Transcript
# ─────────────────────────────────────────
elif st.session_state.step == "review_transcript":
    st.header("Step 2 — Review Transcript")

    low_conf = st.session_state.low_confidence_segments or []
    # Filter already resolved segments
    unresolved = [s for s in low_conf if not s.get("_resolved")]

    if unresolved:
        st.warning(f"⚠️ {len(unresolved)} uncertain term(s) — please review:")
        st.divider()

        for i, seg in enumerate(low_conf):
            if seg.get("_resolved"):
                continue

            original = seg.get("original", "")
            suggested = seg.get("suggested")
            context = seg.get("context", "")
            reason = seg.get("reason", "")

            with st.expander(f"❓ **{original}** — context: *\"{context}\"*"):
                st.caption(f"Reason: {reason}")
                col1, col2, col3 = st.columns([2, 2, 3])

                with col1:
                    if suggested and st.button(f"✅ Use **{suggested}**", key=f"accept_{i}", use_container_width=True):
                        # Update both normalized and edited so text_area reflects change
                        st.session_state.normalized = st.session_state.normalized.replace(original, suggested, 1)
                        st.session_state.edited = st.session_state.normalized
                        save_correction(original=original, corrected=suggested, correction_type="normalization", context_hint=context)
                        low_conf[i]["_resolved"] = True
                        st.toast(f"✅ Saved: '{original}' → '{suggested}'")
                        st.rerun()

                with col2:
                    if st.button(f"❌ Keep **{original}**", key=f"keep_{i}", use_container_width=True):
                        low_conf[i]["_resolved"] = True
                        st.toast(f"Kept original: '{original}'")
                        st.rerun()

                with col3:
                    custom_col, apply_col = st.columns([3, 1])
                    with custom_col:
                        custom = st.text_input("Custom:", key=f"custom_{i}", placeholder="type correct term...", label_visibility="collapsed")
                    with apply_col:
                        if st.button("Apply", key=f"apply_{i}", use_container_width=True):
                            if custom:
                                st.session_state.normalized = st.session_state.normalized.replace(original, custom, 1)
                                st.session_state.edited = st.session_state.normalized
                                save_correction(original=original, corrected=custom, correction_type="normalization", context_hint=context)
                                low_conf[i]["_resolved"] = True
                                st.toast(f"✅ Saved: '{original}' → '{custom}'")
                                st.rerun()

        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original (Whisper)")
        st.text_area(
            "whisper_raw",
            st.session_state.transcript,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
    with col2:
        st.subheader("Normalized (editable)")
        # Use session_state.edited as the source of truth for the text area
        new_edited = st.text_area(
            "normalized_editable",
            value=st.session_state.edited,
            height=200,
            key="normalized_editable"
        )
        # Sync back any manual edits
        if new_edited != st.session_state.edited:
            st.session_state.edited = new_edited

    if st.session_state.edited != st.session_state.normalized:
        st.info("✏️ You've made manual changes — these will be saved as corrections.")

    if st.button("Confirm & Find Mentions", type="primary"):
        st.session_state.normalized = st.session_state.edited

        with st.spinner("Finding mentions (Stage 1)..."):
            mentions = extract_mentions(st.session_state.normalized)
            st.session_state.mentions = mentions
            st.session_state.step = "review_mentions"
            st.rerun()

# ─────────────────────────────────────────
# STEP 3 — Review Mentions
# ─────────────────────────────────────────
elif st.session_state.step == "review_mentions":
    st.header("Step 3 — Review Mentions")
    st.caption("Check what was found before structuring. Edit types, remove noise.")

    mentions = st.session_state.mentions
    st.write(f"**{len(mentions)} mentions found**")

    updated_mentions = []

    for i, mention in enumerate(mentions):
        candidate_type = mention.get("candidate_type", "other")
        if candidate_type not in CANDIDATE_TYPES:
            candidate_type = "other"

        icon = TYPE_ICONS.get(candidate_type, "📌")
        conf = mention.get("confidence", "high")
        conf_badge = "⚠️" if conf == "low" else ""

        temp_ev = mention.get("temporal_evidence", "")
        temp_badge = {
            "explicit_today": "✅",
            "active_regimen": "🔄",
            "future_plan": "📅",
            "consultation_relay": "👨‍⚕️",
            "unclear": "❓"
        }.get(temp_ev, "")

        with st.expander(f"{icon} {candidate_type.upper()} — {mention.get('raw_mention', '')} {conf_badge} {temp_badge}"):
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.text_input("Raw mention", mention.get("raw_mention", ""), key=f"raw_{i}", disabled=True)
                st.text_input("Context", mention.get("context", ""), key=f"ctx_{i}", disabled=True)
                if temp_ev:
                    st.caption(f"⏱ temporal: {temp_ev}")

            with col2:
                type_index = CANDIDATE_TYPES.index(candidate_type) if candidate_type in CANDIDATE_TYPES else CANDIDATE_TYPES.index("other")
                new_type = st.selectbox("Type", CANDIDATE_TYPES, index=type_index, key=f"type_{i}")

            with col3:
                delete = st.checkbox("🗑️ Remove", key=f"del_{i}")

            updated_mention = mention.copy()
            updated_mention["candidate_type"] = new_type
            updated_mention["user_reviewed"] = True  # mark as user-reviewed

            if not delete:
                updated_mentions.append(updated_mention)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Structure These Mentions", type="primary"):
            # Compare updated_mentions with original to find changes
            original_mentions = {m.get("raw_mention", ""): m for m in st.session_state.mentions}
            kept_raw = {m.get("raw_mention", "") for m in updated_mentions}

            # Save removals
            for mention in st.session_state.mentions:
                raw = mention.get("raw_mention", "")
                original_type = mention.get("candidate_type", "other")
                if original_type not in CANDIDATE_TYPES:
                    original_type = "other"
                if raw not in kept_raw:
                    save_knowledge(
                        original_text=raw,
                        correction_type="removal",
                        context_hint=mention.get("context", ""),
                        reason=f"user removed during review (was {original_type})",
                        example_before={"candidate_type": original_type, "raw_mention": raw}
                    )

            # Save type changes
            for updated in updated_mentions:
                raw = updated.get("raw_mention", "")
                new_type = updated.get("candidate_type", "other")
                original = original_mentions.get(raw, {})
                original_type = original.get("candidate_type", "other")
                if original_type not in CANDIDATE_TYPES:
                    original_type = "other"
                if new_type != original_type:
                    save_knowledge(
                        original_text=raw,
                        correction_type="classification",
                        corrected_value=new_type,
                        context_hint=updated.get("context", ""),
                        reason=f"user changed type from {original_type} to {new_type}",
                        example_before={"candidate_type": original_type, "raw_mention": raw},
                        example_after={"candidate_type": new_type, "raw_mention": raw}
                    )
                    st.toast(f"✅ Saved: {raw} → {new_type}")

            with st.spinner("Structuring entities (Stage 2)..."):
                entities = structure_mentions(updated_mentions, st.session_state.normalized)
            with st.spinner("Validating entities (Stage 3)..."):
                entities, validation_changes = validate_entities(entities, st.session_state.normalized)
            st.session_state.mentions = updated_mentions
            st.session_state.entities = entities
            st.session_state.validation_changes = validation_changes
            st.session_state.step = "review_entities"
            st.rerun()
    with col2:
        if st.button("↩️ Back to Transcript"):
            st.session_state.step = "review_transcript"
            st.rerun()

# ─────────────────────────────────────────
# STEP 4 — Review Entities
# ─────────────────────────────────────────
elif st.session_state.step == "review_entities":
    st.header("Step 4 — Review Structured Entities")

    entities = st.session_state.entities

    event_entities = [e for e in entities if e.get("type") not in ("theory", "outside", "meal", "context")]
    meal_entities = [e for e in entities if e.get("type") == "meal"]
    context_entities = [e for e in entities if e.get("type") == "context"]
    theory_entities = [e for e in entities if e.get("type") == "theory"]
    outside_entities = [e for e in entities if e.get("type") == "outside"]

    st.write(f"**{len(event_entities)} events · {len(meal_entities)} meals · {len(theory_entities)} theories · {len(outside_entities)} outside**")

    # Show validation changes if any
    validation_changes = st.session_state.get("validation_changes", [])
    if validation_changes:
        with st.expander(f"🔍 Validator made {len(validation_changes)} correction(s)", expanded=True):
            for change in validation_changes:
                st.write(f"  → {change}")

    if event_entities:
        st.subheader("Event Layer")
        for entity in event_entities:
            entity_type = entity.get("type", "unknown")
            label = entity.get("label", entity.get("metric", entity.get("linked_to", "unknown")))
            icon = TYPE_ICONS.get(entity_type, "📌")
            with st.expander(f"{icon} {entity_type.upper()} — {label}"):
                st.json(entity)

    if meal_entities:
        st.subheader("Nutrition Layer")
        for entity in meal_entities:
            with st.expander(f"🍽️ MEAL — {entity.get('label', '?')}"):
                st.json(entity)

    if context_entities:
        st.subheader("Context")
        for entity in context_entities:
            with st.expander(f"💬 CONTEXT — {entity.get('raw_text', '')[:60]}"):
                st.json(entity)

    if theory_entities:
        st.subheader("Theory Layer")
        for entity in theory_entities:
            with st.expander(f"💭 THEORY — {entity.get('linked_to_label', '?')}"):
                st.json(entity)

    if outside_entities:
        st.subheader("Outside Bucket")
        for entity in outside_entities:
            with st.expander(f"📤 OUTSIDE — {entity.get('raw_text', '')[:60]}"):
                st.json(entity)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save to Database", type="primary"):
            transcript_id = save_transcript(
                st.session_state.transcript,
                st.session_state.normalized,
                "voice_note"
            )
            save_entities(st.session_state.entities, transcript_id)
            st.session_state.transcript_id = transcript_id
            st.session_state.step = "done"
            st.rerun()
    with col2:
        if st.button("↩️ Back to Mentions"):
            st.session_state.step = "review_mentions"
            st.rerun()

# ─────────────────────────────────────────
# STEP 5 — Done
# ─────────────────────────────────────────
elif st.session_state.step == "done":
    st.header("✅ Done!")
    st.success(f"Saved. Transcript ID: {st.session_state.transcript_id}")

    if st.button("Process Another Note", type="primary"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()