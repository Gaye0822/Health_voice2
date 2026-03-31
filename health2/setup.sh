#!/bin/bash
# Health Voice System — Setup Script
# Run: bash setup.sh

set -e

echo "🔧 Health Voice System Setup"
echo "================================"

echo ""
echo "1. Checking Python version..."
python3 --version || { echo "❌ Python3 not found"; exit 1; }

echo ""
echo "2. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✅ venv active"

echo ""
echo "3. Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

echo ""
echo "4. Checking .env file..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env not found — copying .env.example"
    cp .env.example .env
    echo "❗ Edit .env and fill in your API keys before continuing"
else
    echo "✅ .env found"
fi

source .env 2>/dev/null || true

echo ""
echo "5. Creating database schema..."
if [ -z "$DB_NAME" ]; then
    echo "⚠️  No DB info in .env — skipping"
    echo "   Run manually: psql -U \$DB_USER -d \$DB_NAME -f schema.sql"
else
    PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -f schema.sql && \
        echo "✅ Schema created" || \
        echo "⚠️  Schema already exists or connection error"
fi

echo ""
echo "6. Loading knowledge base and corrections..."
if [ -z "$DB_NAME" ]; then
    echo "⚠️  No DB info in .env — skipping"
    echo "   Run manually: psql -U \$DB_USER -d \$DB_NAME -f seed_data.sql"
else
    PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -f seed_data.sql && \
        echo "✅ Seed data loaded" || \
        echo "⚠️  Seed data load failed or already loaded"
fi

echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "To start:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""