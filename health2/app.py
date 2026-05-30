import streamlit as st
import json
import os
from dotenv import load_dotenv

from core.transcribe import transcribe_audio
from core.normalize import normalize_transcript
from core.mention import extract_mentions
from core.pipeline import run_pipeline
from core.envelope import emit_envelope
from core.db import (
    save_transcript,
    save_entities,
    save_normalize_flags,
    save_correction,
    save_knowledge,
    get_corrections_from_db,
    get_symptom_fuzzy_matches,
    get_recurring_info,
    save_envelope,
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
    "step": "upload",
    "edit_mode": False,
    "pending_flags": [],
    "note_date": None,           # date of the voice note from Created_at field
    "source_filename": None,     # original filename — used as unique key for upsert
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
# STEP 1 — Upload & Transcribe
# ─────────────────────────────────────────
if st.session_state.step == "upload":
    st.header("Step 1 — Upload")

    input_mode = st.radio(
        "Input type",
        ["🎙️ Audio file", "📄 JSON transcript"],
        horizontal=True,
        label_visibility="collapsed"
    )

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
            st.session_state.applied_corrections = result.get("applied_corrections", [])
            # Try to parse note_date from filename (e.g. 20260506T052216Z-xxx.m4a)
            import re as _re
            _fname = uploaded_file.name
            _m = _re.match(r"(\d{4})(\d{2})(\d{2})T", _fname)
            if _m:
                from datetime import date as _date
                _note_date = _date(int(_m.group(1)), int(_m.group(2)), int(_m.group(3)))
                print(f"⚙️  Note date from filename: {_note_date}")
            else:
                _note_date = datetime.now().date()
                print(f"⚠️  No date in filename, using today: {_note_date}")
            st.session_state.note_date = _note_date
            st.session_state.source_filename = uploaded_file.name
            st.session_state.edit_mode = False
            st.session_state.pending_flags = []
            st.session_state.step = "review_transcript"
            st.rerun()

    else:
        # ── JSON transcript upload ────────────────────────────────────
        st.caption('Expected format: `{"Content": "transcript text", "Created_at": "2025-11-14T14:10:25+09:00"}`')
        json_file = st.file_uploader("Upload JSON transcript", type=["json"])

        if json_file and st.button("Load & Normalize", type="primary"):
            try:
                data = json.load(json_file)
                raw = data.get("Content", "").strip()
                if not raw:
                    st.error("JSON file does not contain a 'Content' field.")
                else:
                    # Extract note date from Created_at field
                    from datetime import datetime, timezone
                    created_at_raw = data.get("Created_at") or data.get("created_at")
                    note_date = None
                    if created_at_raw:
                        try:
                            dt = datetime.fromisoformat(created_at_raw)
                            note_date = dt.date()
                        except Exception:
                            note_date = None
                    if note_date is None:
                        note_date = datetime.now().date()
                        print(f"⚠️  No Created_at found, using today: {note_date}")
                    else:
                        print(f"⚙️  Note date from Created_at: {note_date}")

                    with st.spinner("Normalizing..."):
                        result = normalize_transcript(raw)

                    st.session_state.transcript = raw
                    st.session_state.normalized = result["normalized_text"]
                    st.session_state.edited = result["normalized_text"]
                    st.session_state.low_confidence_segments = result["low_confidence_segments"]
                    st.session_state.applied_corrections = result.get("applied_corrections", [])
                    st.session_state.note_date = note_date
                    st.session_state.source_filename = json_file.name
                    st.session_state.edit_mode = False
                    st.session_state.pending_flags = []
                    st.session_state.step = "review_transcript"
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Could not parse JSON file. Please check the format.")

# ─────────────────────────────────────────
# STEP 2 — Review Transcript
# ─────────────────────────────────────────
elif st.session_state.step == "review_transcript":
    st.header("Step 2 — Review Transcript")

    low_conf = st.session_state.low_confidence_segments or []
    unresolved = [s for s in low_conf if not s.get("_resolved")]

    # ── Low confidence segments ───────────────────────────────────────
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
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

                with col1:
                    if suggested and st.button(f"✅ Use **{suggested}**", key=f"accept_{i}", use_container_width=True):
                        st.session_state.edited = st.session_state.edited.replace(original, suggested, 1)
                        st.session_state.normalized = st.session_state.edited
                        found_in_kb = save_correction(
                            original=original,
                            corrected=suggested,
                            correction_type="normalization",
                            context_hint=context
                        )
                        low_conf[i]["_resolved"] = True
                        if not found_in_kb:
                            st.session_state["kb_prompt"] = {
                                "original": original,
                                "corrected": suggested
                            }
                        st.toast(f"✅ Applied: '{original}' → '{suggested}'")
                        st.rerun()

                with col2:
                    if st.button(f"❌ Keep **{original}**", key=f"keep_{i}", use_container_width=True):
                        low_conf[i]["_resolved"] = True
                        st.toast(f"Kept original: '{original}'")
                        st.rerun()

                with col3:
                    custom_col, apply_col = st.columns([3, 1])
                    with custom_col:
                        custom = st.text_input(
                            "Custom:",
                            key=f"custom_{i}",
                            placeholder="type correct term...",
                            label_visibility="collapsed"
                        )
                    with apply_col:
                        if st.button("Apply", key=f"apply_{i}", use_container_width=True):
                            if custom:
                                st.session_state.edited = st.session_state.edited.replace(original, custom, 1)
                                st.session_state.normalized = st.session_state.edited
                                found_in_kb = save_correction(
                                    original=original,
                                    corrected=custom,
                                    correction_type="normalization",
                                    context_hint=context
                                )
                                low_conf[i]["_resolved"] = True
                                if not found_in_kb:
                                    st.session_state["kb_prompt"] = {
                                        "original": original,
                                        "corrected": custom
                                    }
                                st.toast(f"✅ Applied: '{original}' → '{custom}'")
                                st.rerun()

                # Flag for the owner audio review
                with col4:
                    already_flagged = any(
                        f["original"] == original for f in st.session_state.pending_flags
                    )
                    if already_flagged:
                        st.button("🎧 Flagged", key=f"flag_{i}", disabled=True, use_container_width=True)
                    else:
                        if st.button("🎧 Ask", key=f"flag_{i}", use_container_width=True):
                            st.session_state.pending_flags.append({
                                "original": original,
                                "suggested": suggested,
                                "context": context,
                                "reason": reason
                            })
                            low_conf[i]["_resolved"] = True
                            st.toast(f"🎧 Flagged for audio review: '{original}'")
                            st.rerun()

        st.divider()

    # ── KB Registry prompt ────────────────────────────────────────────
    if st.session_state.get("kb_prompt"):
        kb_data = st.session_state["kb_prompt"]
        st.warning(f"**'{kb_data['corrected']}'** is not in the KB registry yet. Add it?")
        kb_col1, kb_col2, kb_col3 = st.columns([2, 2, 1])
        with kb_col1:
            kb_entity_type = st.selectbox(
                "Entity type",
                CANDIDATE_TYPES,
                key="kb_entity_type_global",
                label_visibility="collapsed"
            )
        with kb_col2:
            if st.button("➕ Add to KB", key="kb_add_global", use_container_width=True):
                from core.db import save_knowledge
                save_knowledge(
                    original_text=kb_data["corrected"],
                    correction_type="registry",
                    corrected_value=kb_entity_type,
                    reason="Added from normalization correction",
                    example_before={"aliases": [], "mishearings": [kb_data["original"]]},
                    example_after={"subtype": "", "description": "", "note": ""}
                )
                del st.session_state["kb_prompt"]
                st.toast(f"✅ '{kb_data['corrected']}' added to KB as {kb_entity_type}")
                st.rerun()
        with kb_col3:
            if st.button("Skip", key="kb_skip_global", use_container_width=True):
                del st.session_state["kb_prompt"]
                st.rerun()

    # ── Pending flags summary ─────────────────────────────────────────
    if st.session_state.pending_flags:
        with st.expander(f"🎧 {len(st.session_state.pending_flags)} term(s) flagged for The owners audio review"):
            for flag in st.session_state.pending_flags:
                st.write(f"• **{flag['original']}** — context: *\"{flag['context']}\"*")
                if flag.get("suggested"):
                    st.caption(f"  Suggested: {flag['suggested']}")

    # ── Transcript display ────────────────────────────────────────────
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
        st.subheader("Normalized")

        if not st.session_state.edit_mode:
            # READ-ONLY mode — show current edited text
            st.text_area(
                "normalized_readonly",
                value=st.session_state.edited,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
            if st.button("✏️ Edit Transcript"):
                st.session_state.edit_mode = True
                st.rerun()
        else:
            # EDIT mode — editable text area
            new_edited = st.text_area(
                "normalized_editable",
                value=st.session_state.edited,
                height=200,
                label_visibility="collapsed",
                key="normalized_editable_input"
            )
            col_apply, col_cancel = st.columns(2)
            with col_apply:
                if st.button("✅ Apply Changes", type="primary", use_container_width=True):
                    st.session_state.edited = new_edited
                    st.session_state.normalized = new_edited
                    st.session_state.edit_mode = False
                    st.toast("✅ Transcript updated.")
                    st.rerun()
            with col_cancel:
                if st.button("✖️ Cancel", use_container_width=True):
                    st.session_state.edit_mode = False
                    st.rerun()

    # ── Proceed ───────────────────────────────────────────────────────
    st.divider()

    if st.button("Confirm & Find Mentions", type="primary"):
        st.session_state.normalized = st.session_state.edited

        # Save pending flags to DB as pending_audio_review
        for flag in st.session_state.pending_flags:
            save_correction(
                original=flag["original"],
                corrected=flag.get("suggested") or flag["original"],
                correction_type="pending_audio_review",
                context_hint=flag["context"]
            )

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

                # Show transcript snippet anchored to inferred_as (original Whisper word)
                # if present, otherwise fall back to raw_mention.
                # This ensures "Korea" → "nocturia" shows the correct transcript location.
                transcript_text = st.session_state.get("normalized", "") or ""
                search_term = mention.get("inferred_as") or mention.get("raw_mention", "")
                snippet = ""
                if search_term and transcript_text:
                    idx = transcript_text.lower().find(search_term.lower())
                    if idx != -1:
                        start = max(0, idx - 60)
                        end = min(len(transcript_text), idx + len(search_term) + 60)
                        snippet = "..." + transcript_text[start:end].strip() + "..."
                if snippet:
                    st.caption(f"📍 Transcript: *\"{snippet}\"*")
                else:
                    st.text_input("Context", mention.get("context", ""), key=f"ctx_{i}", disabled=True)

                if temp_ev:
                    st.caption(f"⏱ temporal: {temp_ev}")
                reasoning = mention.get("reasoning", "")
                if reasoning:
                    st.caption(f"💭 {reasoning}")

            with col2:
                type_index = CANDIDATE_TYPES.index(candidate_type) if candidate_type in CANDIDATE_TYPES else CANDIDATE_TYPES.index("other")
                new_type = st.selectbox("Type", CANDIDATE_TYPES, index=type_index, key=f"type_{i}")

            with col3:
                delete = st.checkbox("🗑️ Remove", key=f"del_{i}")

            updated_mention = mention.copy()
            updated_mention["_original_type"] = candidate_type  # UI'da gösterilen type
            updated_mention["candidate_type"] = new_type
            updated_mention["user_reviewed"] = True

            if not delete:
                updated_mentions.append(updated_mention)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Structure These Mentions", type="primary"):
            original_mentions = {m.get("raw_mention", ""): m for m in st.session_state.mentions}
            kept_raw = {m.get("raw_mention", "") for m in updated_mentions}

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

            for updated in updated_mentions:
                raw = updated.get("raw_mention", "")
                new_type = updated.get("candidate_type", "other")
                original_type = updated.pop("_original_type", new_type)
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

            # ── Run pipeline ──────────────────────────────────────────────
            note_date = st.session_state.get("note_date")
            with st.spinner("Structuring & validating entities..."):
                pipeline_result = run_pipeline(
                    updated_mentions,
                    st.session_state.normalized,
                    note_date=note_date
                )

            entities = pipeline_result["entities"]
            validation_changes = pipeline_result["validation_changes"]
            enforcer_violations = pipeline_result["enforcer_violations"]

            if enforcer_violations:
                for v in enforcer_violations:
                    print(f"⚙️  schema_enforcer: {v}")

            # ── KB badge check (UI only — not saved to DB) ────────────────
            from core.db import _get_kb_labels_simple
            kb_labels = _get_kb_labels_simple()

            for entity in entities:
                label = entity.get("label", entity.get("metric", entity.get("linked_to", "")))
                entity_type = entity.get("type", "")

                # Check inferred label (paramount → paracetamol)
                for m in updated_mentions:
                    m_raw = m.get("raw_mention", "")
                    m_inferred = m.get("inferred_as")
                    if m_raw.lower() == label.lower() and m_inferred:
                        entity["_raw_mention"] = m_inferred
                        break

                # Check KB — intake label not in KB
                if entity_type == "intake" and not entity.get("_raw_mention"):
                    if label.lower() not in kb_labels:
                        entity["_not_in_kb"] = True

                # Check KB — symptom label not in KB
                if entity_type == "symptom":
                    if label.lower() not in kb_labels:
                        entity["_not_in_kb"] = True

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

    # Fuzzy match lookup for symptoms — single DB call, read-only
    symptom_labels = [
        e.get("label", "") for e in event_entities
        if e.get("type") == "symptom" and e.get("label")
    ]
    fuzzy_matches = {}
    recurring_info = {}
    if symptom_labels:
        try:
            fuzzy_matches = get_symptom_fuzzy_matches(
                symptom_labels,
                note_date=st.session_state.get("note_date")
            )
            symptom_label_status = {
                e.get("label", ""): e.get("status", "present")
                for e in event_entities
                if e.get("type") == "symptom" and e.get("label")
            }
            recurring_info = get_recurring_info(
                symptom_label_status,
                note_date=st.session_state.get("note_date")
            )
            # Write _recurring into entities for envelope enrichment
            for entity in event_entities:
                if entity.get("type") == "symptom":
                    lbl = entity.get("label", "")
                    if lbl in recurring_info:
                        entity["_recurring"] = recurring_info[lbl]
        except Exception as e:
            print(f"⚠️  fuzzy/recurring lookup error: {e}")

    st.write(f"**{len(event_entities)} events · {len(meal_entities)} meals · {len(theory_entities)} theories · {len(outside_entities)} outside**")

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
            unverified_badge = " ⚠️" if (entity.get("_raw_mention") or entity.get("_not_in_kb")) else ""
            with st.expander(f"{icon} {entity_type.upper()} — {label}{unverified_badge}"):
                if entity.get("_raw_mention"):
                    st.caption(f"⚠️ Inferred from transcript: '{entity['_raw_mention']}'")
                if entity.get("_not_in_kb"):
                    st.caption(f"⚠️ Label '{label}' not found in KB registry")
                # Recurring badge
                if entity_type == "symptom" and entity.get("_recurring"):
                    rec = entity["_recurring"]
                    freq = rec.get("frequency_7d", 0)
                    first = rec.get("first_seen", "?")
                    last = rec.get("last_seen", "?")
                    st.caption(f"🔁 Recurring: {freq}x in past 7 days · first: {first} · last: {last}")

                # Fuzzy match results for symptoms
                if entity_type == "symptom" and label in fuzzy_matches:
                    result = fuzzy_matches[label]
                    matches = result.get("matches", [])
                    diagnostic = result.get("diagnostic", [])
                    if matches:
                        match_str = " · ".join(f"**{m}** ({s}%)" for m, s in matches)
                        st.caption(f"🔍 Similar in DB: {match_str}")
                    if diagnostic:
                        diag_str = " · ".join(f"{m} ({s}% ⚠️)" for m, s, _ in diagnostic)
                        st.caption(f"🔍 No body token match: {diag_str}")
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
            transcript_id, source_note_id = save_transcript(
                st.session_state.transcript,
                st.session_state.normalized,
                st.session_state.get("source_filename") or "voice_note",
                note_date=st.session_state.get("note_date")
            )
            save_entities(st.session_state.entities, transcript_id, st.session_state.mentions,
                         note_date=st.session_state.get("note_date"))
            print(f"⚙️  applied_corrections at save: {st.session_state.get('applied_corrections', [])}")
            applied_corrections = st.session_state.get("applied_corrections", [])
            if applied_corrections:
                save_normalize_flags(transcript_id, applied_corrections, st.session_state.entities)
            st.session_state.transcript_id = transcript_id

            # ── Envelope emit ──────────────────────────────────────────────────
            envelope = emit_envelope(
                entities=st.session_state.entities,
                validation_changes=st.session_state.get("validation_changes", []),
                enforcer_violations=st.session_state.get("enforcer_violations", []),
                normalized_transcript=st.session_state.normalized or "",
                note_date=st.session_state.get("note_date"),
                source_note_id=source_note_id,
            )
            save_envelope(envelope, transcript_id=transcript_id)
            st.session_state["last_envelope"] = envelope
            print(f"⚙️  envelope saved: source_note_id={source_note_id}")

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

    if st.session_state.get("last_envelope"):
        with st.expander("📦 Envelope (HDS format)", expanded=False):
            st.json(st.session_state["last_envelope"])

    if st.button("Process Another Note", type="primary"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()