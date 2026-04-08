"""
watcher.py — Audio file watcher for the health voice pipeline.

Monitors ~/AudioProcessing for new audio files dropped by icloud_watcher.py.
When a new file is detected, runs the full pipeline:
  transcribe → normalize → mention → structure → validate → save to DB

Usage:
  python watcher.py

Run this alongside icloud_watcher.py. Both can be managed via launchd.
"""

import time
import os
import sys
import shutil
from datetime import datetime
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
WATCH_DIR     = os.path.expanduser("~/AudioProcessing")
PROCESSED_DIR = os.path.expanduser("~/AudioProcessing/processed")
LOG_FILE      = os.path.expanduser("~/audio-pipeline/watcher.log")

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(WATCH_DIR, exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Pipeline imports ───────────────────────────────────────────────────────────
# Add project root to path so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcribe import transcribe_audio
from core.normalize import normalize_transcript
from core.mention import extract_mentions
from core.structure import structure_mentions
from core.validate import validate_entities
from core.db import save_transcript, save_entities


# ── Core pipeline ──────────────────────────────────────────────────────────────

def process_audio_file(filepath: str):
    """
    Run the full health voice pipeline on a single audio file.
    Saves transcript and entities to the database.
    Archives the processed file to ~/AudioProcessing/processed/
    """
    fname = os.path.basename(filepath)
    log(f"[START] Processing: {fname}")

    try:
        # Stage 1 — Transcribe
        log(f"[1/6] Transcribing {fname}...")
        raw = transcribe_audio(filepath)
        log(f"[1/6] Transcript: {raw[:80]}...")

        # Stage 2 — Normalize
        log(f"[2/6] Normalizing...")
        result = normalize_transcript(raw)
        normalized = result["normalized_text"]
        low_conf = result.get("low_confidence_segments", [])
        if low_conf:
            log(f"[2/6] {len(low_conf)} low-confidence segment(s) flagged")

        # Stage 3 — Save transcript to DB
        log(f"[3/6] Saving transcript to DB...")
        transcript_id = save_transcript(raw, normalized, "voice_note_auto")
        log(f"[3/6] Transcript ID: {transcript_id}")

        # Stage 4 — Extract mentions
        log(f"[4/6] Extracting mentions...")
        mentions = extract_mentions(normalized)
        log(f"[4/6] {len(mentions)} mention(s) found")

        # Stage 5 — Structure
        log(f"[5/6] Structuring entities...")
        entities = structure_mentions(mentions, normalized)
        log(f"[5/6] {len(entities)} entity(ies) structured")

        # Stage 6 — Validate + Save
        log(f"[6/6] Validating and saving...")
        entities, changes = validate_entities(entities, normalized)
        save_entities(entities, transcript_id)

        log(f"[DONE] {fname} → transcript_id={transcript_id}, {len(entities)} entities saved")
        if changes:
            for change in changes:
                log(f"       validator: {change}")

        # Archive processed file
        dst = os.path.join(PROCESSED_DIR, fname)
        shutil.move(filepath, dst)
        log(f"[ARCHIVED] {fname} → processed/")

    except Exception as e:
        log(f"[ERROR] {fname}: {e}")
        import traceback
        log(traceback.format_exc())


# ── File readiness check ───────────────────────────────────────────────────────

def wait_for_file(filepath: str, timeout: int = 120) -> bool:
    """Wait until file is fully written and stable."""
    start = time.time()
    last_size = -1
    while time.time() - start < timeout:
        try:
            if not os.path.exists(filepath):
                time.sleep(5)
                continue
            current_size = os.path.getsize(filepath)
            if current_size == last_size and current_size > 0:
                with open(filepath, "rb") as f:
                    f.read(1024)
                return True
            last_size = current_size
        except (OSError, IOError):
            pass
        time.sleep(5)
    return False


# ── Watchdog handler ───────────────────────────────────────────────────────────

PROCESSED = set()


class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        fname = os.path.basename(filepath)

        # Skip processed subfolder events
        if "processed" in filepath:
            return

        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return

        if fname in PROCESSED:
            log(f"[SKIPPED] Already processed: {fname}")
            return

        if not os.path.exists(filepath):
            log(f"[SKIPPED] File no longer exists: {fname}")
            return

        log(f"[DETECTED] New audio file: {fname}")
        PROCESSED.add(fname)

        if wait_for_file(filepath):
            process_audio_file(filepath)
        else:
            log(f"[SKIPPED] File never became ready: {fname}")
            PROCESSED.discard(fname)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log(f"[WATCHER] Starting. Monitoring: {WATCH_DIR}")

    # Process any files already present at startup
    for fname in os.listdir(WATCH_DIR):
        if fname == "processed":
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        filepath = os.path.join(WATCH_DIR, fname)
        if not os.path.isfile(filepath):
            continue
        log(f"[STARTUP] Found existing file: {fname}")
        PROCESSED.add(fname)
        if wait_for_file(filepath):
            process_audio_file(filepath)
        else:
            PROCESSED.discard(fname)

    # Start watchdog observer
    event_handler = AudioHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    log("[WATCHER] Observer started. Waiting for files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("[WATCHER] Stopping...")
        observer.stop()
    observer.join()
    log("[WATCHER] Stopped.")