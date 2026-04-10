"""
pre_normalize.py — Deterministik pre-normalizasyon katmanı.

LLM'e gitmeden önce corrections tablosundaki bilinen hataları uygular.
- Yüksek benzerlik (≥90) → otomatik düzelt
- Orta benzerlik (70-90) → low confidence flag olarak işaretle
- Düşük benzerlik (<70) → dokunma

Bu katman normalize_transcript'ten önce çalışır.
"""

import re
from rapidfuzz import fuzz, process


def _get_corrections() -> list:
    """Corrections tablosundan bilinen düzeltmeleri çek."""
    try:
        from core.db import get_corrections_from_db
        return get_corrections_from_db()
    except Exception:
        try:
            from db import get_corrections_from_db
            return get_corrections_from_db()
        except Exception:
            return []


def _get_registry_aliases() -> list:
    """
    KB registry'den canonical terms ve mishearings çek.
    Mishearings: fonetik ASR hataları — fuzzy match için kullanılır.
    Aliases: açıklayıcı isimler — sadece LLM context için, fuzzy match'e girmez.
    """
    try:
        from core.db import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT original_text, corrected_value, example_before
               FROM knowledge_base
               WHERE correction_type = 'registry'
               ORDER BY created_at DESC"""
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        return []

    entries = []
    for row in rows:
        canonical, entity_type, before_json = row
        mishearings = []
        if before_json and isinstance(before_json, dict):
            mishearings = before_json.get("mishearings", [])
        if mishearings:  # Sadece mishearings varsa ekle
            entries.append({
                "canonical": canonical,
                "mishearings": mishearings,
                "entity_type": entity_type or ""
            })
    return entries


def _tokenize(text: str) -> list:
    """Metni kelime ve çok kelimeli token'lara böl."""
    words = text.split()
    tokens = []
    # Tek kelimeler
    for i, w in enumerate(words):
        tokens.append((w, i, i + 1))
    # İki kelimeli tokenlar
    for i in range(len(words) - 1):
        tokens.append((f"{words[i]} {words[i+1]}", i, i + 2))
    # Üç kelimeli tokenlar
    for i in range(len(words) - 2):
        tokens.append((f"{words[i]} {words[i+1]} {words[i+2]}", i, i + 3))
    return tokens


def apply_deterministic_corrections(raw_text: str) -> dict:
    """
    Transcript'e bilinen düzeltmeleri deterministik olarak uygular.

    Returns:
    {
        "text": str,                  # düzeltilmiş metin
        "applied": [...],             # uygulanan düzeltmeler
        "flagged": [...]              # flag edilmiş belirsiz terimler (low confidence)
    }
    """
    corrections = _get_corrections()
    registry = _get_registry_aliases()

    applied = []
    flagged = []
    text = raw_text

    # ── 1. Corrections tablosu — kesin bilinen hatalar ─────────────────────────
    # Bunlar kullanıcı tarafından onaylanmış düzeltmeler
    # Context-aware: sadece context_hint yoksa veya context uyuyorsa uygula
    for corr in corrections:
        original = corr.get("original", "")
        corrected = corr.get("corrected", "")
        context_hint = corr.get("context_hint", "")

        if not original or not corrected:
            continue
        if original.strip().lower() == corrected.strip().lower():
            continue

        # Çok kısa terimleri atla
        if len(original) < 3:
            continue

        # Eğer context_hint varsa, metnin o bağlamı içerip içermediğini kontrol et
        if context_hint:
            if context_hint.lower() not in text.lower():
                continue

        # Exact match — direkt düzelt
        if original.lower() in text.lower():
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(corrected, text)
                applied.append(f'"{original}" → "{corrected}" (corrections table)')
                continue

        # Fuzzy match — sadece çok kelimeli terimler için
        original_words = original.split()
        if len(original_words) < 2:
            continue  # Tek kelimeli terimler için fuzzy match yapma

        words = text.split()
        for i in range(len(words) - len(original_words) + 1):
            token = " ".join(words[i:i + len(original_words)])

            # Uzunluk oranı kontrolü
            len_ratio = len(token) / max(len(original), 1)
            if len_ratio < 0.7 or len_ratio > 1.4:
                continue

            # İlk kelime benzerliği yüksek olmalı
            first_word_score = fuzz.ratio(
                original_words[0].lower(),
                words[i].lower()
            )
            if first_word_score < 80:
                continue

            score = fuzz.ratio(original.lower(), token.lower())
            if score >= 90:
                text = text.replace(token, corrected, 1)
                applied.append(f'"{token}" → "{corrected}" (fuzzy {score:.0f}%, corrections table)')
                break
            elif score >= 80:
                flagged.append({
                    "original": token,
                    "suggested": corrected,
                    "context": " ".join(words[max(0, i-3):i+len(original_words)+3]),
                    "reason": f"Possible ASR error: similar to known correction '{original}' ({score:.0f}% match)"
                })

    # ── 2. KB Registry — sadece mishearings kullan ────────────────────────────
    # Mishearings (fonetik ASR hataları) → exact ve fuzzy match için
    # Aliases LLM context için kullanılır, fuzzy match'e girmez
    for entry in registry:
        canonical = entry["canonical"]
        mishearings = entry.get("mishearings", [])

        for mishearing in mishearings:
            if not mishearing or len(mishearing) < 3:
                continue
            mishearing_words = mishearing.split()
            use_fuzzy = len(mishearing_words) >= 2

            if mishearing.lower() in text.lower():
                pattern = re.compile(re.escape(mishearing), re.IGNORECASE)
                if pattern.search(text):
                    text = pattern.sub(canonical, text)
                    applied.append(f'"{mishearing}" → "{canonical}" (registry mishearing)')
                    continue

            if not use_fuzzy:
                continue

            words = text.split()
            for i in range(len(words) - len(mishearing_words) + 1):
                token = " ".join(words[i:i + len(mishearing_words)])
                len_ratio = len(token) / max(len(mishearing), 1)
                if len_ratio < 0.7 or len_ratio > 1.4:
                    continue
                first_word_score = fuzz.ratio(mishearing_words[0].lower(), words[i].lower())
                if first_word_score < 80:
                    continue
                if len(mishearing_words) >= 2:
                    last_word_score = fuzz.ratio(
                        mishearing_words[-1].lower(),
                        words[i + len(mishearing_words) - 1].lower()
                    )
                    if last_word_score < 70:
                        continue
                score = fuzz.ratio(mishearing.lower(), token.lower())
                if score >= 92:
                    text = text.replace(token, canonical, 1)
                    applied.append(f'"{token}" → "{canonical}" (fuzzy {score:.0f}%, registry mishearing)')
                    break
                elif score >= 85:
                    flagged.append({
                        "original": token,
                        "suggested": canonical,
                        "context": " ".join(words[max(0, i-3):i+len(mishearing_words)+3]),
                        "reason": f'Possible mishearing of "{canonical}" ({score:.0f}% match)'
                    })

    return {
        "text": text,
        "applied": applied,
        "flagged": flagged
    }