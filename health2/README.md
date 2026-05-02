# Health Voice System

A voice-note processing pipeline that transcribes audio recordings, extracts structured health entities, and emits envelope-format JSON for downstream ingestion.

---

## What It Does

1. Watches a folder for new audio files (`.m4a`, `.mp3`, `.wav`)
2. Transcribes audio using OpenAI Whisper
3. Normalizes and classifies the transcript
4. Extracts structured health entities across five layers: `event`, `nutrition`, `theory`, `context`, `outside`
5. Validates and enforces schema rules
6. Saves entities to Postgres and emits an HDS-compatible envelope JSON

---

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.11+
- PostgreSQL 14+
- OpenAI API key (for extraction)
- `ffmpeg` (for audio processing)

---

## Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd health2
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg (if not already installed)

```bash
brew install ffmpeg
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```
OPENAI_API_KEY=your_openai_api_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=health_voice
DB_USER=health_user
DB_PASSWORD=your_db_password
```

---

## Database Setup

### 1. Create database and user

```bash
psql -U <your_superuser> -c "CREATE USER health_user WITH PASSWORD 'your_password';"
psql -U <your_superuser> -c "CREATE DATABASE health_voice OWNER health_user;"
psql -U <your_superuser> -c "GRANT ALL PRIVILEGES ON DATABASE health_voice TO health_user;"
```

### 2. Run schema

```bash
psql -U <your_superuser> -d health_voice -f schema.sql
psql -U <your_superuser> -d health_voice -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO health_user;"
psql -U <your_superuser> -d health_voice -c "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO health_user;"
```

### 3. Seed knowledge base

```bash
psql -U <your_superuser> -d health_voice -f seed_data.sql
```

---

## Running the System

### Option A — Streamlit UI (manual review)

```bash
source venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser. You can upload an audio file or paste a transcript directly.

### Option B — Watcher (automated)

The watcher monitors a local folder and processes new audio files automatically.

#### One-time iCloud setup

The system expects a folder called `AudioInbox` in your iCloud Drive. Create it once:

```bash
mkdir -p ~/Library/Mobile\ Documents/com~apple~CloudDocs/AudioInbox
```

Drop any audio file (`.m4a`, `.mp3`, `.wav`) into this folder — the watcher picks it up automatically.

#### Start the watchers

```bash
# Terminal 1 — iCloud sync watcher (monitors ~/iCloud Drive/AudioInbox)
source venv/bin/activate
python icloud_watcher.py

# Terminal 2 — pipeline watcher (processes files from ~/AudioProcessing)
source venv/bin/activate
python watcher.py
```

Audio files are picked up automatically, processed end-to-end, and results saved to Postgres.

#### How it works

```
iCloud Drive/AudioInbox/   ← drop audio files here
        ↓ icloud_watcher.py copies to local
~/AudioProcessing/
        ↓ watcher.py picks up and processes
Pipeline → Postgres + Envelope JSON
```

---

## Exporting Envelopes

To export processed envelopes as individual JSON files:

```bash
python export_envelopes.py
```

Files are written to `./exports/` named `<note_date>_<id>.json`.

Options:

```bash
python export_envelopes.py --out ./my_output    # custom output directory
python export_envelopes.py --limit 5            # export last 5 only
```

---

## Reviewing Entities

Open the Streamlit UI and navigate to the **Registry** tab to:

- Review and approve/reject unverified entities
- Manage the knowledge base (canonical labels, removal rules)
- View classification corrections

---

## Project Structure

```
health2/
├── app.py                      # Streamlit UI
├── watcher.py                  # Automated pipeline watcher
├── icloud_watcher.py           # iCloud folder sync watcher
├── export_envelopes.py         # Envelope JSON exporter
├── schema.sql                  # Database schema
├── seed_data.sql               # Knowledge base seed data
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── core/
    ├── db.py                   # Database operations
    ├── envelope.py             # Envelope builder (HDS format)
    ├── mention.py              # Entity mention extraction
    ├── mention_filter.py       # Mention filtering rules
    ├── models.py               # Pydantic entity models
    ├── normalize.py            # Transcript normalization
    ├── pre_normalize.py        # Pre-normalization cleaning
    ├── pipeline.py             # Main extraction pipeline
    ├── schema_enforcer.py      # Schema validation and enforcement
    ├── structure.py            # Entity structuring (LLM)
    ├── transcribe.py           # Audio transcription (Whisper)
    ├── validate.py             # Entity validation rules
    ├── intervention_pipeline.py    # Intervention extraction
    ├── intervention_enricher.py    # Intervention enrichment
    └── intervention_merger.py      # Intervention entity merging
```

---

## Envelope Format

The system emits HDS-compatible envelopes in the following shape:

```json
{
  "envelope": {
    "source_note_id": "<uuid>",
    "note_date": "2025-12-03",
    "created_at": "2025-12-03T08:15:00Z",
    "pipeline_version": "1.0.0",
    "extraction_model": "claude-sonnet-4-20250514",
    "transcription_model": "whisper-1",
    "audio_file": null,
    "normalized_transcript": "..."
  },
  "layers": {
    "event_layer": [...],
    "nutrition_layer": [...],
    "theory_layer": [...],
    "context_layer": [...],
    "outside_bucket": [...]
  },
  "meta": {
    "warnings": [],
    "validation_changes": [],
    "enforcer_violations": []
  }
}
```

Each record within a layer carries a `warnings` array using a closed vocabulary:

| Warning | Meaning |
|---|---|
| `label not found in KB registry` | No canonical match in knowledge base |
| `inferred from transcript` | Label inferred by LLM, not a direct match |
| `no body token match` | Raw mention could not be matched to a known term |
| `raw mention preserved` | Entity label is the raw transcript text |
| `low confidence mention` | Mention extraction flagged as low confidence |

---

## Notes

- The pipeline is calibrated to the specific language patterns in Gabriel's voice notes. If note style changes significantly, some prompt rules may need recalibration.
- Theory extraction handles speculative and indirect language — edge cases are actively being improved.
- The knowledge base (KB) is the source of truth for canonical labels and mishearing corrections. It lives in Postgres and is managed through the Registry UI.
- All corrections made through the review UI feed back into the KB and improve future extractions.