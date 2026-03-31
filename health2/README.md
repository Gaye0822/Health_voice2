# Health Voice System — Technical Documentation

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Project Structure](#2-project-structure)
3. [Setup](#3-setup)
4. [The Pipeline — Stage by Stage](#4-the-pipeline--stage-by-stage)
5. [The Knowledge Base](#5-the-knowledge-base)
6. [User Interfaces](#6-user-interfaces)
7. [Entity Types Reference](#7-entity-types-reference)
8. [Design Principles](#8-design-principles)

---

## 1. What This System Does

The Health Voice System converts spoken health logs into structured, queryable records stored in a PostgreSQL database. You record a voice note describing what happened with your health — supplements taken, symptoms, measurements, activities — and the system:

1. Transcribes the audio (Whisper)
2. Normalizes transcription errors
3. Finds all health-relevant mentions in the text
4. Filters out noise (future plans, consultation relays, vague statements)
5. Structures each mention into a typed entity with defined fields
6. Validates entities against a knowledge base and the original transcript
7. Stores everything in the database

The system is intentionally conservative: it is better to miss an entity than to create a wrong one.

---

## 2. Project Structure

```
health-voice-system/
├── core/
│   ├── __init__.py
│   ├── db.py               — All database operations
│   ├── mention.py          — Stage 1: Mention extraction (LLM)
│   ├── mention_filter.py   — Stage 1b: Deterministic mention filter
│   ├── models.py           — Pydantic entity models (single source of truth)
│   ├── normalize.py        — Stage 0b: Transcription normalization (LLM)
│   ├── schema_enforcer.py  — Stage 3a: Deterministic field-level fixes
│   ├── structure.py        — Stage 2: Entity structuring (LLM)
│   ├── transcribe.py       — Stage 0: Whisper transcription
│   └── validate.py         — Stage 3: Validation (LLM)
│
├── app.py                  — Main UI: single voice note workflow
├── registry.py             — Knowledge Base Manager UI
├── requirements.txt
├── schema.sql              — Database table definitions
├── seed_data.sql           — Pre-loaded knowledge base and corrections
├── .env.example            — Environment variable template
├── setup.sh                — Automated setup script
└── README.md
```

---

## 3. Setup

Run the setup script — it handles everything automatically:

```bash
bash setup.sh
```

This will create a virtual environment, install dependencies, copy .env.example to .env, create the database tables, and load the knowledge base.

Before running, make sure PostgreSQL is running with a health_voice database and health_user created. Then fill in .env with your API keys.

For full manual setup instructions, see SETUP.md.

---

## 4. The Pipeline — Stage by Stage

```
Audio File
    |
[transcribe.py]        Stage 0   — Whisper ASR → raw text
    |
[normalize.py]         Stage 0b  — Fix transcription errors, flag uncertain terms
    |
[mention.py]           Stage 1   — Find all health-relevant mentions (LLM)
    |
[mention_filter.py]    Stage 1b  — Drop future plans, consultation relays (no LLM)
    |
[structure.py]         Stage 2   — Convert mentions to typed entity dicts (LLM)
    |
[schema_enforcer.py]   Stage 3a  — Deterministic field-level fixes (no LLM)
    |
[validate.py]          Stage 3   — KB lookup + transcript contradiction check (LLM)
    |
PostgreSQL Database
```

### Stage 0 — Transcription (transcribe.py)

Model: OpenAI Whisper (whisper-1)
Input: Audio file (.m4a, .mp3, .wav, .ogg)
Output: Raw transcript string

Whisper receives a domain hint mentioning supplements, biomarkers, devices, symptoms, and measurements to improve recognition of health-specific vocabulary.

---

### Stage 0b — Normalization (normalize.py)

Model: Claude Sonnet
Input: Raw transcript
Output: { normalized_text, low_confidence_segments }

Corrects ASR errors at two confidence levels:

High confidence (auto-fix): Correction is unambiguous from context — capitalization errors, obvious mishearings of known terms.

Low confidence (flagged for review): Term looks wrong but correct form is uncertain — brand names, supplement names, device names. Shown to the user for review before extraction proceeds.

Previously learned corrections from the corrections table are injected into the prompt so recurring Whisper errors on known terms are fixed automatically.

---

### Stage 1 — Mention Extraction (mention.py)

Model: Claude Sonnet (tool_use)
Input: Normalized transcript
Output: List of mention dicts

Each mention has: raw_mention, candidate_type, confidence, context, temporal_evidence.

Before assigning a type, the model works through three questions:
1. Did this actually happen? (not a future plan, not a provider recommendation)
2. Is this about the user's own body or actions? (not another person or equipment)
3. Is the label specific and resolvable? (can it be tracked over time?)

Temporal evidence values:
- explicit_today — clearly happened today
- active_regimen — ongoing habit, no today confirmation
- future_plan — not yet done
- consultation_relay — what a provider said
- unclear — not enough context

What is excluded: general feelings, energy/motivation states, sleep difficulty (only device-recorded sleep metrics are captured as measurements), vague discomfort, pathogen names used as cause/source context.

---

### Stage 1b — Mention Filter (mention_filter.py)

No LLM — pure Python.

- future_plan → always dropped
- consultation_relay → always dropped, except measurement type (test results are real data)
- active_regimen → kept only if a same-session signal is found near the mention
- unclear → same as active_regimen
- explicit_today → always kept
- theory, outside, context, other → always passed through

Same-session signals: "today", "this morning", "just took", "last night", "past few days", "yesterday", and similar.

---

### Stage 2 — Entity Structuring (structure.py)

Model: Claude Sonnet (tool_use)
Input: Filtered mention list + normalized transcript
Output: List of typed entity dicts

Converts each mention into a fully specified entity. The Pydantic EntityOutput model is converted to a JSON schema and passed as a tool definition, forcing schema-valid output.

Key rules:
- Only fill fields explicitly present in the transcript — everything else is null, never "unknown"
- Dose: only filled if a single unambiguous numeric value is stated. Ranges or vague quantities → null
- If a movement is described as part of a machine session, it goes in the machine notes — not as a separate activity
- Sleep difficulty is never a symptom
- Same entity mentioned twice → one record

---

### Stage 3a — Schema Enforcer (schema_enforcer.py)

No LLM — pure Python. Fixes mechanical issues before the validator:

1. Remove fields not in schema for that entity type
2. Replace string "unknown" with null
3. Set relative time phrases ("recently", "last week") to null
4. Vague dose quantities → null
5. Duration strings ("7 hours 20 minutes") → reformatted as "7h 20m"
6. Measurement string values → coerced to float; non-numeric → entity removed

---

### Stage 3 — Validation (validate.py)

Model: Claude Sonnet
Input: Schema-enforced entity list + transcript + knowledge base
Output: (validated_entities, changes_list)

Two jobs only:

Job 1 — Knowledge Base Lookup:
- Label matches registry entry with different type → correct the type
- Label matches known alias → replace with canonical form
- Label is in removal list → remove the entity
- Label has classification correction → apply it

Job 2 — Transcript Contradiction Check:
- For every entity, find the relevant passage and ask: does the transcript say this did NOT happen?
- Default is removal, not preservation — if uncertain, remove

theory, outside, and context types bypass the validator entirely. The validator cannot change types on its own judgment, add new entities, or change action fields.

---

## 5. The Knowledge Base

Managed via registry.py. Three entry types:

### Registry Entries

Canonical term definitions. When the validator sees any alias, it uses the canonical name and corrects the type if needed.

- Canonical name — the official label (e.g., QIAstat)
- Entity type — what type this is (e.g., test)
- Aliases — alternate spellings, ASR variants (e.g., QI stat, kaistat)
- Subtype / Description — optional context

### Removal Rules

Labels that should never become structured entities.

- Label — the term to never extract (e.g., wake up supplements)
- Reason — injected verbatim into the validator's chain of thought. Write it as an explanation, not just a category.
- Context hint — optional qualifier for when the rule applies

### Classification Corrections

Labels previously assigned the wrong entity type.

- Label — the misclassified term
- Correct entity type — what it should be
- Reason — injected into validator chain of thought

---

## 6. User Interfaces

### Main UI — app.py

```bash
streamlit run app.py
```

Five-step workflow:

Step 1 — Upload: Upload audio, Whisper transcribes.
Step 2 — Review Transcript: Review flagged terms, edit normalized text if needed.
Step 3 — Review Mentions: Change types or remove noise before structuring. Changes are saved to the knowledge base automatically.
Step 4 — Review Entities: Inspect structured records. Validator changes are shown.
Step 5 — Save: Write to database.

Note: If the UI shows stale output after a code change, hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+F5 (Windows).

---

### Knowledge Base Manager — registry.py

```bash
streamlit run registry.py
```

Three tabs: Registry, Removals, Classifications. All entries can be deleted individually.

---

## 7. Entity Types Reference

All types are defined in models.py (Pydantic). The tool schema sent to the API is generated from these models automatically.

### Event Layer

| Type | Key fields | Notes |
|---|---|---|
| intake | label, action (took/did_not_take), dose, unit, time, category | category: supplement / prescription / OTC / food |
| symptom | label, onset_time, severity, qualifier, duration | Clinical findings only |
| activity | label, start_time, duration, status | status: completed / planned / incomplete |
| machine | label, start_time, duration, status | Therapy equipment, PBM devices |
| device | label, start_time, status | Passive monitoring devices |
| measurement | metric, value (required float), unit, time, source | Objective numeric metrics only |
| meal | label, time, eaten_out, items | High-level only, not ingredient lists |
| intervention | label, start_date, end_date, status | Ongoing or completed protocols |
| outcome | linked_to, direction, qualifier, onset_time | Observed change linked to something |
| test | label, time, status, result | Diagnostic tests |

### Non-Event Layer

| Type | Purpose |
|---|---|
| context | Background information (infection source, environment) |
| theory | User speculation, causal explanations, provider recommendations not yet acted on |
| outside | Equipment issues, procurement notes, missed recordings |

---

## 8. Design Principles

**LLM layers vs. deterministic layers**
LLMs handle semantic judgment — understanding meaning, classifying mentions. Python handles mechanical rules — temporal filtering, field validation, schema enforcement. LLMs are unreliable for mechanical tasks but essential for semantic ones.

**Chain-of-thought prompting**
Both mention.py and structure.py prompt the model with explicit step-by-step reasoning questions. This consistently outperforms a flat list of rules.

**Knowledge base design**
The KB injects reasons into the validator's chain of thought rather than silently applying rules. The validator can confirm a match before acting, and rules can be context-sensitive.

**Validator scope is narrow**
The validator's job is KB lookup and contradiction check only. Ontological decisions are encoded in schema rules, KB entries, and prompt instructions — not left to the validator's judgment.

**Pydantic as the schema contract**
models.py is the single source of truth. The Anthropic tool schemas are generated from Pydantic models, so the schema the model is constrained to and the schema the code validates against are always identical.