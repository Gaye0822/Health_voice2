# Health Voice System — Setup Guide

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- An Anthropic API key
- An OpenAI API key (for Whisper transcription)

---

## Quick Setup (Recommended)

### Step 1 — Create the database

```bash
psql -U postgres
```

```sql
CREATE DATABASE health_voice;
CREATE USER health_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE health_voice TO health_user;
\q
```

### Step 2 — Fill in your credentials

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DB_NAME=health_voice
DB_USER=health_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

### Step 3 — Run the setup script

```bash
bash setup.sh
```

This will automatically:
- Create a Python virtual environment
- Install all dependencies
- Create the database tables
- Load the knowledge base and corrections

### Step 4 — Start the application

```bash
source venv/bin/activate
streamlit run app.py
```

---

## Manual Setup (if setup.sh fails)

### 1. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Database tables

```bash
psql -U health_user -d health_voice -f schema.sql
```

### 3. Knowledge base and corrections

```bash
psql -U health_user -d health_voice -f seed_data.sql
```

### 4. Start

```bash
streamlit run app.py
```

---

## Troubleshooting

**UI shows stale output after a change**
Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+F5` (Windows/Linux)

**Database connection error**
Check that PostgreSQL is running and that the credentials in `.env` match what you created in Step 1.

**psycopg2 install fails on Mac**
```bash
brew install libpq
pip install psycopg2-binary
```