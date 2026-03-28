#!/bin/bash
# Health Voice System — Kurulum Scripti
# Çalıştır: bash setup.sh

set -e  # Hata olunca dur

echo "🔧 Health Voice System Kurulum"
echo "================================"

# ─────────────────────────────────────
# 1. Python kontrol
# ─────────────────────────────────────
echo ""
echo "1. Python versiyonu kontrol ediliyor..."
python3 --version || { echo "❌ Python3 bulunamadı"; exit 1; }

# ─────────────────────────────────────
# 2. Virtual environment
# ─────────────────────────────────────
echo ""
echo "2. Virtual environment oluşturuluyor..."
python3 -m venv venv
source venv/bin/activate
echo "✅ venv aktif"

# ─────────────────────────────────────
# 3. Bağımlılıklar
# ─────────────────────────────────────
echo ""
echo "3. Bağımlılıklar yükleniyor..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Bağımlılıklar yüklendi"

# ─────────────────────────────────────
# 4. .env kontrol
# ─────────────────────────────────────
echo ""
echo "4. .env dosyası kontrol ediliyor..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env bulunamadı — .env.example kopyalanıyor"
    cp .env.example .env
    echo "❗ .env dosyasını düzenle ve API key'lerini gir"
else
    echo "✅ .env mevcut"
fi

# ─────────────────────────────────────
# 5. DB schema
# ─────────────────────────────────────
echo ""
echo "5. Veritabanı schema oluşturuluyor..."
echo "   DB bilgilerini .env'den okuyorum..."

source .env 2>/dev/null || true

if [ -z "$DB_NAME" ]; then
    echo "⚠️  .env'de DB bilgisi yok — schema kurulumunu atla"
    echo "   Sonra manuel çalıştır: psql -U \$DB_USER -d \$DB_NAME -f schema.sql"
else
    PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -f schema.sql && \
        echo "✅ Schema oluşturuldu" || \
        echo "⚠️  Schema zaten mevcut veya bağlantı hatası"
fi

# ─────────────────────────────────────
# Tamamlandı
# ─────────────────────────────────────
echo ""
echo "================================"
echo "✅ Kurulum tamamlandı!"
echo ""
echo "Başlatmak için:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""