"""
icloud_watcher.py — Monitors iCloud AudioInbox and copies new audio files to ~/AudioProcessing.

Gabriel's requirements:
  - Copy raw audio into local storage first (~/AudioProcessing)
  - Preserve the original file in iCloud — do NOT delete it
  - Preserve original file timestamp (shutil.copy2)
  - Queue processing from there (watcher.py handles the rest)
  - Simple logging: ingested vs failed clearly visible

Run alongside watcher.py.
"""

import time
import os
import shutil
import subprocess
from datetime import datetime

ICLOUD_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/AudioInbox"
)
LOCAL_DIR     = os.path.expanduser("~/AudioProcessing")
LOG_FILE      = os.path.expanduser("~/audio-pipeline/icloud_watcher.log")
PROCESSED_LOG = os.path.expanduser("~/audio-pipeline/processed_files.log")
SUPPORTED     = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}

os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(ICLOUD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Processed file tracking ───────────────────────────────────────────────────

def load_processed():
    """Load the list of already copied files from disk."""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(fname):
    """Save a filename to the processed log so it survives restarts."""
    with open(PROCESSED_LOG, "a") as f:
        f.write(fname + "\n")


# ── File readiness ────────────────────────────────────────────────────────────

def is_fully_downloaded(path):
    """Check if iCloud has fully downloaded the file (stable size, readable)."""
    try:
        size1 = os.path.getsize(path)
        if size1 == 0:
            return False
        time.sleep(3)
        size2 = os.path.getsize(path)
        if size1 != size2:
            return False
        with open(path, "rb") as f:
            f.read(1024)
        return True
    except (OSError, IOError):
        return False


def already_handled(fname, processed):
    """Skip if already copied or logged."""
    if fname in processed:
        return True
    if os.path.exists(os.path.join(LOCAL_DIR, fname)):
        return True
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────

processed = load_processed()
log(f"[ICLOUD WATCHER] Started. {len(processed)} files already processed.")
log(f"[ICLOUD WATCHER] Monitoring: {ICLOUD_DIR}")

while True:
    try:
        files = os.listdir(ICLOUD_DIR)
        for fname in files:
            # Skip hidden files and non-audio files
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED:
                continue

            # Skip files already handled
            if already_handled(fname, processed):
                continue

            src = os.path.join(ICLOUD_DIR, fname)
            log(f"[FOUND] {fname} — forcing iCloud download...")

            # Force iCloud to download the file
            subprocess.run(["brctl", "download", src], capture_output=True)

            # Wait up to ~3 minutes for file to be fully available
            ready = False
            for attempt in range(24):
                if is_fully_downloaded(src):
                    ready = True
                    break
                log(f"[WAITING] {fname} — attempt {attempt + 1}/24")
                time.sleep(5)

            if ready:
                dst = os.path.join(LOCAL_DIR, fname)
                # copy2 preserves original timestamps
                shutil.copy2(src, dst)
                # Original file in iCloud is intentionally NOT deleted
                processed.add(fname)
                mark_processed(fname)
                log(f"[INGESTED] {fname} → ~/AudioProcessing (original preserved in iCloud)")
            else:
                log(f"[FAILED] {fname} — timed out waiting for iCloud download, will retry next cycle")

    except Exception as e:
        log(f"[ERROR] {e}")

    time.sleep(15)