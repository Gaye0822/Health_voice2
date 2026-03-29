# Health Voice System — Technical Documentation

**Prepared for:** Gabriel  
**System Version:** Current (as of project files)  
**Author:** Gaye Çetindere

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Project Structure](#2-project-structure)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [The Pipeline — Stage by Stage](#4-the-pipeline--stage-by-stage)
5. [The Knowledge Base](#5-the-knowledge-base)
6. [User Interfaces](#6-user-interfaces)
7. [Entity Types Reference](#7-entity-types-reference)
8. [Running the System](#8-running-the-system)
9. [Regression Testing](#9-regression-testing)
10. [Design Principles & Key Decisions](#10-design-principles--key-decisions)

---

## 1. What This System Does

The Health Voice System converts spoken health logs into structured, queryable data records stored in a PostgreSQL database. You record a voice note describing what happened with your health today — supplements taken, symptoms, measurements, activities — and the system:

1. Transcribes the audio (Whisper)
2. Normalizes transcription errors
3. Finds all health-relevant mentions in the text
4. Filters out noise (future plans, consultation relays, vague statements)
5. Structures each mention into a typed entity with defined fields
6. Validates entities against a knowledge base and the original transcript
7. Stores everything in the database

The system is designed to generalize across a wide range of health contexts. It is intentionally conservative: it is better to miss an entity than to create a wrong one.

---

## 2. Project Structure

```
health2/
├── app.py                  # Main Streamlit UI — single voice note workflow
├── batch.py                # Batch UI — process a folder of audio files
├── registry.py             # Knowledge Base Manager UI
├── regression_test.py      # CLI regression test suite
│
├── core/
│   ├── transcribe.py       # Stage 0 — Whisper transcription
│   ├── normalize.py        # Stage 0b — Transcription normalization
│   ├── mention.py          # Stage 1 — Mention extraction (LLM)
│   ├── mention_filter.py   # Stage 1b — Deterministic mention filter
│   ├── structure.py        # Stage 2 — Entity structuring (LLM)
│   ├── validate.py         # Stage 3 — Validation (LLM + schema enforcer)
│   ├── schema_enforcer.py  # Stage 3a — Deterministic post-processing
│   ├── models.py           # Pydantic entity models — single source of truth
│   └── db.py               # All database operations
│
├── tests/
│   └── snapshots/          # Baseline JSON files for regression testing
│
└── requirements.txt
```

---

## 3. Prerequisites & Setup

### Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
DB_NAME=health_voice
DB_USER=health_user
DB_PASSWORD=...
DB_HOST=localhost
DB_PORT=5432
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

```
anthropic>=0.40.0
openai>=1.50.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
streamlit>=1.40.0
```

### Database

PostgreSQL database named `health_voice`. The system expects these tables:

- `transcripts` — raw and normalized text, source type
- `entities` — structured entity records with type, label, and JSON attributes
- `corrections` — Whisper normalization corrections learned from user review
- `knowledge_base` — canonical registry, removal rules, and classification corrections

### External Tools (for regression testing only)

```bash
brew install pandoc   # macOS
```

---

## 4. The Pipeline — Stage by Stage

```
Audio File
    │
    ▼
[transcribe.py]        Stage 0   — Whisper ASR → raw text
    │
    ▼
[normalize.py]         Stage 0b  — Fix transcription errors, flag uncertain terms
    │
    ▼
[mention.py]           Stage 1   — Find all health-relevant mentions (LLM)
    │
    ▼
[mention_filter.py]    Stage 1b  — Drop future plans, consultation relays (no LLM)
    │
    ▼
[structure.py]         Stage 2   — Convert mentions to typed entity dicts (LLM)
    │
    ▼
[schema_enforcer.py]   Stage 3a  — Deterministic field-level fixes (no LLM)
    │
    ▼
[validate.py]          Stage 3   — KB lookup + transcript contradiction check (LLM)
    │
    ▼
PostgreSQL Database
```

---

### Stage 0 — Transcription (`transcribe.py`)

**Model:** OpenAI Whisper (`whisper-1`)  
**Input:** Audio file path (`.m4a`, `.mp3`, `.wav`, `.ogg`)  
**Output:** Raw transcript string

Whisper is given a domain hint prompt mentioning supplements, biomarkers, devices, symptoms, and measurements to improve recognition of health-specific vocabulary. No further processing happens here — the raw text is passed directly to normalize.

---

### Stage 0b — Normalization (`normalize.py`)

**Model:** Claude Sonnet  
**Input:** Raw transcript string  
**Output:** `{ normalized_text, low_confidence_segments }`

The normalizer corrects ASR errors while being deliberately conservative. It distinguishes between two confidence levels:

**High confidence (auto-fix):** The correction is unambiguous given context — e.g., a capitalization error for a known device name, or an obvious mishearing where the correct form is clear from surrounding words.

**Low confidence (flagged for user review):** The term looks wrong but the correct form is uncertain — brand names, supplement names, device names, or proper nouns where Whisper may have produced a phonetically similar but incorrect word. These are returned as `low_confidence_segments` with the original, a suggested correction, surrounding context, and a reason.

The normalizer also injects any previously learned corrections from the `corrections` database table, so recurring Whisper errors on known terms are fixed automatically over time.

In the **single note UI** (`app.py`), low-confidence segments are shown to you for review before extraction proceeds. In **batch mode** (`batch.py`), they are logged but not reviewed interactively.

---

### Stage 1 — Mention Extraction (`mention.py`)

**Model:** Claude Sonnet (tool_use)  
**Input:** Normalized transcript  
**Output:** List of mention dicts, each with:

| Field | Description |
|---|---|
| `raw_mention` | Exact text of the mention |
| `candidate_type` | Proposed entity type (intake, symptom, activity, etc.) |
| `confidence` | `high` or `low` |
| `context` | Surrounding words for traceability |
| `temporal_evidence` | When this happened (see below) |

The LLM uses a **chain-of-thought prompting approach** — before assigning a type to any mention, it works through three questions in order:

1. **Did this actually happen?** (not a future plan, not a provider recommendation)
2. **Is this about the user's own body or actions?** (not about another person or equipment)
3. **Is the label specific and resolvable?** (can it be tracked and compared over time?)

A mention is only extracted if all three pass.

**Temporal evidence values:**

| Value | Meaning |
|---|---|
| `explicit_today` | Clearly happened today — specific time, "this morning", "just took" |
| `active_regimen` | Ongoing habit, no today confirmation — "I take", "I've been on" |
| `future_plan` | Not yet done — "going to", "will", "planning to" |
| `consultation_relay` | What a provider said — "he said", "she wants me to" |
| `unclear` | Not enough context |

**What gets excluded at this stage:**
- General feelings ("feel awful", "feel terrible", "not feeling it")
- Energy/motivation states ("exhausted", "groggy", "low energy")
- Sleep difficulty ("hard time sleeping", "woke up early") — sleep is only captured as a device measurement
- Vague discomfort ("under the weather", "a bit off")
- Pathogen names used as cause/source context ("I caught rhinovirus from X") — the actual symptoms are extracted, not the pathogen
- Mentions that don't pass all three questions above

Output is validated against Pydantic models (`MentionOutput`, `Mention`) before passing downstream.

---

### Stage 1b — Mention Filter (`mention_filter.py`)

**No LLM — pure Python**  
**Input:** Mention list + normalized transcript  
**Output:** `(kept_mentions, dropped_mentions)`

This deterministic layer enforces temporal rules that the LLM assigned but only Python can reliably act on:

| Temporal Evidence | Rule |
|---|---|
| `future_plan` | Always dropped |
| `consultation_relay` | Always dropped — **except** `measurement` type (blood test results reported by a provider are real data) |
| `active_regimen` | Kept only if a same-session confirmation signal is found near the mention in the transcript |
| `unclear` | Same as `active_regimen` — needs confirmation |
| `explicit_today` | Always kept |
| `theory`, `outside`, `context`, `other` | Always passed through — handled by structure.py |

**Same-session confirmation signals** are words and phrases like: "today", "this morning", "just took", "last night", "past few days", "yesterday", and similar temporal anchors. The filter checks the mention's `raw_mention`, its `context` field, and a 150-character window around the mention in the full transcript.

This layer is deterministic and has no prompt — it cannot hallucinate. It is the right place for this logic precisely because it needs to be reliable and testable.

---

### Stage 2 — Entity Structuring (`structure.py`)

**Model:** Claude Sonnet (tool_use)  
**Input:** Filtered mention list + normalized transcript  
**Output:** List of typed entity dicts

The structurer converts each mention into a fully specified entity conforming to the schema for its type. It uses the **`extract_entities` tool** — the Pydantic `EntityOutput` model is converted to a JSON schema and passed to the Anthropic API as a tool definition, forcing the model to return schema-valid output.

The structurer works through four questions for each mention:

1. **What type is this?** — use the candidate_type from the mention unless it clearly violates schema
2. **What actually happened?** — use temporal_evidence to decide action (took / did_not_take / omit)
3. **Is this a fact or an interpretation?** — facts go to event layer; speculation goes to theory; provider recommendations not yet acted on go to theory or are omitted
4. **Which fields are actually known?** — only fill fields explicitly present in the transcript; everything else is `null`

**Key structuring rules:**

- `active_regimen` mentions are structured only if same-session confirmation exists in the transcript. Provider advice to continue something is **not** confirmation.
- `future_plan` and `consultation_relay` are omitted entirely (the filter already dropped them, but structure.py has the same rule as a safety net).
- If a movement or exercise is described as part of a machine session, it goes in the machine's `notes` field — not as a separate activity entity.
- Dose field: only filled if a single unambiguous numeric value is stated. Ranges, approximations, or vague quantities → `null`.
- Pathogen names ("rhinovirus", "strep") are context, not symptoms. The actual symptoms are what get extracted.
- Sleep difficulty is never a symptom — not even if it sounds clinical. Only device-recorded sleep metrics (HRV, sleep duration) are extracted, as measurements.
- Same entity mentioned twice → one record (merge rule).

Output is validated against `EntityOutput` Pydantic model.

---

### Stage 3a — Schema Enforcer (`schema_enforcer.py`)

**No LLM — pure Python**  
**Input:** Raw entity list from structure.py  
**Output:** `(clean_entities, violations_log)`

This layer runs before the LLM validator and fixes mechanical issues that LLMs consistently get wrong. Rules applied in order:

1. **Remove extra fields** — any field not in the schema for that entity type is stripped
2. **Fix "unknown" strings** — the string `"unknown"` in any field is replaced with `null`
3. **Fix invalid time fields** — relative phrases like "recently", "last week", "three weeks ago" in time fields are set to `null`
4. **Fix dose field** — vague quantity strings ("two or three", "a few", "some") → `null` and unit cleared
5. **Duration conversion** — string durations like "7 hours 20 minutes" are reformatted as `"7h 20m"` and `unit` is set to `"duration"` (not converted to decimal, which loses precision)
6. **Measurement value coercion** — string numeric values are coerced to float; non-numeric values cause the entity to be flagged for removal

Entities flagged for removal are dropped with a reason logged. The violations log is surfaced in the Streamlit UI.

---

### Stage 3 — Validation (`validate.py`)

**Model:** Claude Sonnet  
**Input:** Schema-enforced entity list + normalized transcript + knowledge base  
**Output:** `(validated_entities, changes_list)`

The validator has exactly two jobs — it is explicitly prohibited from making broader ontological decisions:

**Job 1: Knowledge Base Lookup**
- If an entity label matches a **registry entry** with a different type → correct the type
- If a label matches a known **alias** → replace with canonical form
- If a label is in the **removal list** → remove the entity (the reason field is injected into the prompt to help the model confirm the match before removing)
- If a label has a **classification correction** → apply it

**Job 2: Transcript Contradiction Check**
- For every entity, the model finds the relevant passage in the transcript and asks: does the transcript explicitly state this did NOT happen, was skipped, was unavailable, or is being negated?
- If the transcript negates it → remove or move to outside
- **Default is removal, not preservation.** If uncertain → remove.

**Protected types** (`theory`, `outside`, `context`) bypass the LLM validator entirely — they are passed through unchanged. Only event-layer entities are validated.

The validator cannot:
- Change entity types on its own judgment
- Add new entities
- Change `action` fields (took / did_not_take)
- Make temporal judgments beyond what the transcript says

---

## 5. The Knowledge Base

Managed via `registry.py` (the Knowledge Base Manager UI). Three entry types:

### Registry Entries

Canonical term definitions. Used by the validator to correct entity types and normalize aliases.

**Fields:**
- **Canonical name** — the official label (e.g., `QIAstat`)
- **Entity type** — what type this term is (e.g., `test`)
- **Aliases** — alternate spellings, ASR variants (e.g., `QI stat, QI Stat, kaistat`)
- **Subtype** — optional descriptor
- **Description** — optional free text

**Effect:** When the validator sees any alias, it uses the canonical name. If the entity type in the extracted record doesn't match the registry type, the validator corrects it.

### Removal Rules

Labels that should never become structured entities.

**Fields:**
- **Label** — the term to never extract (e.g., `wake up supplements`)
- **Reason** — *this is injected verbatim into the validator's chain of thought.* Write it as an explanation, not just a category. Example: "Vague group label, not resolvable to specific substances. Cannot be tracked over time."
- **Context hint** — optional qualifier for when the rule applies

**Effect:** When the validator sees this label in the entity list, it removes it. The reason is what the model reads to confirm the match — write it to be unambiguous.

### Classification Corrections

Labels that were previously assigned the wrong entity type.

**Fields:**
- **Label** — the term that was misclassified
- **Correct entity type** — what it should be
- **Reason** — injected into validator chain of thought
- **Context hint** — optional

**Effect:** When the validator sees this label, it corrects the entity type.

---

### How KB Entries Flow Through the Pipeline

The knowledge base is fetched and injected at two points:

1. **`normalize.py`** — corrections table is injected so recurring Whisper errors on known terms are auto-fixed before extraction even starts
2. **`validate.py`** — full KB (registry, removals, classifications) is injected into the validator's system prompt

---

## 6. User Interfaces

Three Streamlit pages, run from the `health2/` directory:

### Single Note UI — `app.py`

```bash
streamlit run app.py
```

Five-step interactive workflow:

| Step | What happens |
|---|---|
| **1 — Upload** | Upload audio file, Whisper transcribes it |
| **2 — Review Transcript** | Review low-confidence segments one by one; manually edit normalized text if needed |
| **3 — Review Mentions** | See all extracted mentions; change types or remove noise before structuring |
| **4 — Review Entities** | See structured entities grouped by layer; validator changes are shown |
| **5 — Done** | Save to database; transcript ID returned |

User corrections at Step 3 (type changes, removals) are saved to the knowledge base automatically so they influence future runs.

**Note:** If the UI appears to show stale output after a code change, do a hard refresh (Cmd+Shift+R / Ctrl+Shift+F5) to clear Streamlit's cache.

---

### Batch UI — `batch.py`

```bash
streamlit run batch.py
```

Process a folder of audio files at once:

1. Enter folder path
2. Click "Process All Files" — pipeline runs on every file
3. Review results file by file, or click "Save All Pending" to bulk-save
4. Individual files can be saved or skipped; skips can be undone

Low-confidence normalization segments are logged per file but not interactively reviewed.

---

### Knowledge Base Manager — `registry.py`

```bash
streamlit run registry.py
```

Three tabs:

- **Registry** — Add/view/delete canonical term entries
- **Removals** — Add/view/delete removal rules
- **Classifications** — Add/view/delete classification corrections

All entries can be deleted individually. Search is available in the Registry tab.

---

## 7. Entity Types Reference

All entity types are defined in `models.py` (Pydantic models). This is the single source of truth — the tool schema sent to the Anthropic API is generated from these models automatically.

### Event Layer

| Type | Label field | Key fields | Notes |
|---|---|---|---|
| `intake` | `label` | `action` (took/did_not_take), `dose`, `unit`, `time`, `category` | category: supplement / prescription / OTC / food |
| `symptom` | `label` | `onset_time`, `severity`, `qualifier`, `duration` | Clinical findings only — not general feelings |
| `activity` | `label` | `start_time`, `duration`, `status` | status: completed / planned / incomplete |
| `machine` | `label` | `start_time`, `duration`, `status` | Therapy equipment, PBM devices, treadmills |
| `device` | `label` | `start_time`, `status` | Passive monitoring devices worn/carried |
| `measurement` | `metric` | `value` (required float), `unit`, `time`, `source` | Objective metrics with numeric values only |
| `meal` | `label` | `time`, `eaten_out`, `items` | Items: high-level only, not ingredient lists |
| `intervention` | `label` | `start_date`, `end_date`, `status` | Ongoing or completed protocols |
| `outcome` | `linked_to` | `direction`, `qualifier`, `onset_time` | Observed change over time linked to something |
| `test` | `label` | `time`, `status`, `result` | Diagnostic tests; status: planned / done |

### Non-Event Layer

| Type | Key field | Purpose |
|---|---|---|
| `context` | `raw_text` | Background information (e.g., infection source, environment) |
| `theory` | `raw_text` | User speculation, causal explanations, provider recommendations not acted on |
| `outside` | `raw_text` | Equipment issues, procurement notes, missed recordings |

---

## 8. Running the System

### Normal Use (Single Note)

```bash
cd /Users/gayecetindere/Desktop/health2
streamlit run app.py
```

### Batch Processing

```bash
streamlit run batch.py
# Enter folder path, e.g.: /Users/gayecetindere/Desktop/health2/batch_notes
```

### Managing the Knowledge Base

```bash
streamlit run registry.py
```

### All three UIs can run simultaneously on different ports:

```bash
streamlit run app.py --server.port 8501
streamlit run batch.py --server.port 8502
streamlit run registry.py --server.port 8503
```

---

## 9. Regression Testing

`regression_test.py` is a CLI tool for detecting pipeline regressions against known-good outputs.

### Workflow

**Step 1: Save a baseline** from a batch `.docx` review file (the old outputs you approved):

```bash
python regression_test.py --save-baseline --docx /path/to/batch_review.docx
```

This parses the docx and saves one JSON snapshot per transcript to `tests/snapshots/T1_baseline.json`, `T2_baseline.json`, etc.

**Step 2: Run the current pipeline** and compare against baselines:

```bash
python regression_test.py --run --docx /path/to/batch_review.docx
```

**Step 3: Review the report:**

```
REGRESSION REPORT
T1 ✅ — no changes
T2 ⚠️  — 3 difference(s):
    ➕ ADDED:   [intake] berberine
    ➖ REMOVED: [symptom] sinusitis
    🔄 CHANGED: intake:vitamin D
       action: 'took' → 'did_not_take'
```

### Filtering to Specific Transcripts

```bash
python regression_test.py --run --docx /path/to/file.docx --ids T1 T4 T9
```

### Notes

- The `notes`, `time`, `start_time`, and `onset_time` fields are excluded from comparison — they legitimately vary between runs without indicating a regression.
- New snapshots for each run are saved alongside baselines as `T1_new.json` for manual inspection.
- The tool uses `pandoc` to parse docx files — it must be installed.
- Prefer **manual per-transcript review via Streamlit** for evaluating new transcripts. The regression tool is for confirming that code changes haven't broken previously correct outputs.

---

## 10. Design Principles & Key Decisions

### LLM layers vs. deterministic layers

The pipeline deliberately separates what LLMs do from what Python does:

- **LLMs** handle semantic judgment — understanding meaning, classifying mentions, deciding what is a symptom vs. a theory
- **Python** handles mechanical rules — temporal filtering, field validation, schema enforcement

This division exists because LLMs are unreliable for mechanical tasks (they will sometimes output `"unknown"` instead of `null`, or include extra fields) but essential for semantic tasks (no rule system can reliably distinguish a symptom from a general feeling).

### Chain-of-thought prompting

Both `mention.py` and `structure.py` prompt the model with explicit step-by-step reasoning questions before asking for output. This consistently outperforms a flat list of rules. The questions make the model's reasoning visible and auditable.

### Knowledge base design

The KB injects **reasons** into the validator's chain of thought rather than silently applying rules. This means:
- The validator can confirm a match before acting on it
- You can write nuanced rules ("only when mentioned as a group without individual names")
- The reasoning is traceable when something unexpected happens

### Ontological conservatism

The validator's scope is deliberately narrow: KB lookup and contradiction check only. It cannot make ontological decisions (e.g., "this should be an outcome not a symptom"). Those decisions belong to Gabriel and are encoded in schema rules, KB entries, and prompt instructions — not left to the validator's judgment.

### No overfitting to specific examples

Prompt examples use placeholder names (supplementA, symptomB, machineX) instead of real terms from the transcript. This prevents the model from pattern-matching on surface features of Gabriel's specific notes rather than learning the underlying rules.

### Pydantic as the schema contract

`models.py` is the single source of truth. The Anthropic tool schemas are generated from Pydantic models — not written by hand. This means the schema the model is constrained to and the schema the code validates against are always identical.

---

*For questions about schema decisions, ontology, or new entity types — these are Gabriel's domain. Implementation questions go to Gaye.*