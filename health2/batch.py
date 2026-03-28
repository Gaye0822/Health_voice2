import streamlit as st
import json
import os
import glob
from dotenv import load_dotenv

from core.transcribe import transcribe_audio
from core.normalize import normalize_transcript
from core.mention import extract_mentions
from core.structure import structure_mentions, validate_entities
from core.db import (
    save_transcript,
    save_entities,
    save_correction
)

load_dotenv()

st.set_page_config(page_title="Batch Testing — Health Voice System", layout="wide")
st.title("🧪 Batch Testing")
st.caption("Process multiple audio files at once. Review and save or skip each one.")

# --- Session State ---
if "batch_files" not in st.session_state:
    st.session_state.batch_files = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}  # filename → {transcript, normalized, entities, status}
if "batch_processing" not in st.session_state:
    st.session_state.batch_processing = False

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

# ─────────────────────────────────────────
# FOLDER INPUT
# ─────────────────────────────────────────
st.header("Step 1 — Select Folder")

folder_path = st.text_input(
    "Folder path",
    placeholder="/Users/gayecetindere/Desktop/health2/batch_notes",
    help="Full path to the folder containing your audio files"
)

SUPPORTED_FORMATS = ["*.m4a", "*.mp3", "*.wav", "*.ogg"]

if folder_path and os.path.isdir(folder_path):
    files = []
    for fmt in SUPPORTED_FORMATS:
        files.extend(glob.glob(os.path.join(folder_path, fmt)))
    files = sorted(files)

    if files:
        st.success(f"✅ Found {len(files)} audio file(s):")
        for f in files:
            st.write(f"  • {os.path.basename(f)}")

        if st.button("🚀 Process All Files", type="primary"):
            st.session_state.batch_files = files
            st.session_state.batch_results = {}
            st.session_state.batch_processing = True
            st.rerun()
    else:
        st.warning("No supported audio files found in this folder (m4a, mp3, wav, ogg).")

elif folder_path:
    st.error("Folder not found. Please check the path.")

# ─────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────
if st.session_state.batch_processing and st.session_state.batch_files:
    files = st.session_state.batch_files
    results = st.session_state.batch_results

    # Process any files not yet done
    unprocessed = [f for f in files if os.path.basename(f) not in results]

    if unprocessed:
        st.header("Step 2 — Processing...")
        progress = st.progress(0)
        status_text = st.empty()

        for i, filepath in enumerate(unprocessed):
            filename = os.path.basename(filepath)
            status_text.text(f"Processing {filename}... ({i+1}/{len(unprocessed)})")

            try:
                # Transcribe
                raw = transcribe_audio(filepath)

                # Normalize
                norm_result = normalize_transcript(raw)
                normalized = norm_result["normalized_text"]
                low_conf = norm_result["low_confidence_segments"]

                # Mentions
                mentions = extract_mentions(normalized)

                # Structure
                entities = structure_mentions(mentions, normalized)

                # Validate
                entities = validate_entities(entities, normalized)

                results[filename] = {
                    "filepath": filepath,
                    "transcript": raw,
                    "normalized": normalized,
                    "low_confidence_segments": low_conf,
                    "mentions": mentions,
                    "entities": entities,
                    "status": "pending"  # pending / saved / skipped
                }

            except Exception as e:
                results[filename] = {
                    "filepath": filepath,
                    "transcript": None,
                    "normalized": None,
                    "low_confidence_segments": [],
                    "mentions": [],
                    "entities": [],
                    "status": "error",
                    "error": str(e)
                }

            progress.progress((i + 1) / len(unprocessed))
            st.session_state.batch_results = results

        status_text.text("✅ All files processed!")
        st.session_state.batch_processing = False
        st.rerun()

# ─────────────────────────────────────────
# REVIEW
# ─────────────────────────────────────────
if st.session_state.batch_results:
    results = st.session_state.batch_results
    files = st.session_state.batch_files

    # Summary bar
    total = len(results)
    saved = sum(1 for r in results.values() if r["status"] == "saved")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    pending = sum(1 for r in results.values() if r["status"] == "pending")
    errors = sum(1 for r in results.values() if r["status"] == "error")

    st.header("Step 3 — Review Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("✅ Saved", saved)
    col3.metric("⏭️ Skipped", skipped)
    col4.metric("⏳ Pending", pending)

    if errors:
        st.error(f"⚠️ {errors} file(s) failed to process.")

    st.divider()

    # Save all pending button
    pending_files = [fn for fn, r in results.items() if r["status"] == "pending"]
    if pending_files:
        if st.button(f"✅ Save All Pending ({len(pending_files)})", type="primary"):
            for filename in pending_files:
                r = results[filename]
                if r["entities"]:
                    transcript_id = save_transcript(r["transcript"], r["normalized"], "batch")
                    save_entities(r["entities"], transcript_id)
                    results[filename]["status"] = "saved"
            st.session_state.batch_results = results
            st.toast(f"Saved {len(pending_files)} transcripts!")
            st.rerun()

    st.divider()

    # Individual file review
    for filename in sorted(results.keys()):
        r = results[filename]
        status = r["status"]

        status_icon = {"saved": "✅", "skipped": "⏭️", "pending": "⏳", "error": "❌"}.get(status, "?")
        entity_count = len(r["entities"]) if r["entities"] else 0

        with st.expander(f"{status_icon} **{filename}** — {entity_count} entities — {status.upper()}"):

            if status == "error":
                st.error(f"Error: {r.get('error', 'Unknown error')}")
                continue

            # Transcript columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Transcript (Whisper)")
                st.text_area(
                    f"raw_{filename}",
                    r["transcript"] or "",
                    height=120,
                    disabled=True,
                    label_visibility="collapsed"
                )
            with col2:
                st.subheader("Normalized")
                st.text_area(
                    f"norm_{filename}",
                    r["normalized"] or "",
                    height=120,
                    disabled=True,
                    label_visibility="collapsed"
                )

            # Low confidence segments
            low_conf = r.get("low_confidence_segments", [])
            if low_conf:
                st.warning(f"⚠️ {len(low_conf)} uncertain term(s) flagged during normalization:")
                for seg in low_conf:
                    st.write(f"  • **{seg.get('original')}** → suggested: *{seg.get('suggested')}* — {seg.get('reason', '')}")

            # Entities
            if r["entities"]:
                st.subheader("Entities")

                event_entities = [e for e in r["entities"] if e.get("type") not in ("theory", "outside")]
                theory_entities = [e for e in r["entities"] if e.get("type") == "theory"]
                outside_entities = [e for e in r["entities"] if e.get("type") == "outside"]

                if event_entities:
                    st.write("**Event Layer**")
                    for entity in event_entities:
                        entity_type = entity.get("type", "unknown")
                        label = entity.get("label", entity.get("metric", "unknown"))
                        icon = TYPE_ICONS.get(entity_type, "📌")
                        st.write(f"{icon} `{entity_type.upper()}` — **{label}**")
                        # Show key fields inline
                        preview_fields = {k: v for k, v in entity.items()
                                         if k not in ("type", "label") and v is not None}
                        if preview_fields:
                            st.caption("  " + "  ·  ".join([f"{k}: {v}" for k, v in list(preview_fields.items())[:4]]))

                if theory_entities:
                    st.write("**Theories**")
                    for entity in theory_entities:
                        st.write(f"💭 {entity.get('raw_text', '')[:80]}...")

                if outside_entities:
                    st.write("**Outside Bucket**")
                    for entity in outside_entities:
                        st.write(f"📤 {entity.get('raw_text', '')[:80]}")

            else:
                st.info("No entities extracted.")

            # Action buttons
            if status == "pending":
                col_save, col_skip = st.columns(2)
                with col_save:
                    if st.button("✅ Save", key=f"save_{filename}", use_container_width=True):
                        transcript_id = save_transcript(r["transcript"], r["normalized"], "batch")
                        save_entities(r["entities"], transcript_id)
                        results[filename]["status"] = "saved"
                        st.session_state.batch_results = results
                        st.toast(f"Saved: {filename}")
                        st.rerun()
                with col_skip:
                    if st.button("⏭️ Skip", key=f"skip_{filename}", use_container_width=True):
                        results[filename]["status"] = "skipped"
                        st.session_state.batch_results = results
                        st.rerun()

            elif status == "saved":
                st.success("Saved to database.")

            elif status == "skipped":
                st.info("Skipped.")
                if st.button("↩️ Undo skip", key=f"undo_{filename}"):
                    results[filename]["status"] = "pending"
                    st.session_state.batch_results = results
                    st.rerun()

    # Reset button
    st.divider()
    if st.button("🔄 Start New Batch"):
        st.session_state.batch_files = []
        st.session_state.batch_results = {}
        st.session_state.batch_processing = False
        st.rerun()