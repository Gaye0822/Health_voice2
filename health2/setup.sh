#!/bin/bash
# setup.sh — Health Voice System setup script
# Run this once after cloning the repo.
# Usage: bash setup.sh

set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Health Voice System — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
echo "→ Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "  ✗ Python3 not found. Install from https://python.org"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  ✓ $PYTHON_VERSION"

# ── Check ffmpeg ──────────────────────────────────────────────────────────────
echo "→ Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
    echo "  ⚠ ffmpeg not found. Installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "  ✗ Homebrew not found. Install ffmpeg manually: https://ffmpeg.org"
        exit 1
    fi
else
    echo "  ✓ ffmpeg found"
fi

# ── Virtual environment ───────────────────────────────────────────────────────
echo "→ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ venv created"
else
    echo "  ✓ venv already exists"
fi

source venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo "→ Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  ✓ Dependencies installed"

# ── Environment file ──────────────────────────────────────────────────────────
echo "→ Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ .env created from .env.example"
    echo ""
    echo "  ⚠ ACTION REQUIRED: Open .env and fill in:"
    echo "    - OPENAI_API_KEY"
    echo "    - DB_PASSWORD"
    echo ""
else
    echo "  ✓ .env already exists"
fi

# ── Database setup ────────────────────────────────────────────────────────────
echo "→ Setting up database..."
echo ""
echo "  Please enter your PostgreSQL superuser name (e.g. your macOS username):"
read -r SUPERUSER

echo "  Creating database user and database..."

psql -U "$SUPERUSER" -c "CREATE USER health_user WITH PASSWORD 'health_pass';" 2>/dev/null && \
    echo "  ✓ User health_user created" || \
    echo "  ✓ User health_user already exists"

psql -U "$SUPERUSER" -c "CREATE DATABASE health_voice OWNER health_user;" 2>/dev/null && \
    echo "  ✓ Database health_voice created" || \
    echo "  ✓ Database health_voice already exists"

psql -U "$SUPERUSER" -c "GRANT ALL PRIVILEGES ON DATABASE health_voice TO health_user;" 2>/dev/null

echo "  Running schema..."
psql -U "$SUPERUSER" -d health_voice -f schema.sql -q
psql -U "$SUPERUSER" -d health_voice -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO health_user;" -q
psql -U "$SUPERUSER" -d health_voice -c "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO health_user;" -q
echo "  ✓ Schema created"

echo "  Seeding knowledge base..."
psql -U "$SUPERUSER" -d health_voice -f seed_data.sql -q
echo "  ✓ Knowledge base seeded"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Fill in .env with your API key and DB password"
echo ""
echo "  2. Start the Streamlit UI:"
echo "     source venv/bin/activate"
echo "     streamlit run app.py"
echo ""
echo "  3. Or start the automated watcher:"
echo "     source venv/bin/activate"
echo "     python icloud_watcher.py   # Terminal 1"
echo "     python watcher.py          # Terminal 2"
echo ""