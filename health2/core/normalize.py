import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _get_knowledge_context() -> str:
    """Get corrections and registry from knowledge base for prompt injection."""
    try:
        from core.db import get_corrections_as_text, get_knowledge_for_prompt
        corrections = get_corrections_as_text()
        knowledge = get_knowledge_for_prompt()
        parts = []
        if corrections:
            parts.append(corrections)
        if knowledge:
            parts.append(knowledge)
        return "\n".join(parts)
    except Exception:
        return ""



def _detect_llm_corrections(working_text: str, normalized_text: str) -> list:
    """
    working_text ile normalized_text arasindaki farklari bulur.
    KB filtresi yok — LLM ne degistirdiyse hepsini yakalar.
    Sonuc applied_corrections'a girer, save_normalize_flags oradan KB kontrolu yapar.
    """
    if working_text == normalized_text:
        return []

    import difflib

    raw_words = working_text.split()
    norm_words = normalized_text.split()

    matcher = difflib.SequenceMatcher(
        None,
        [w.lower() for w in raw_words],
        [w.lower() for w in norm_words],
        autojunk=False
    )

    applied = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            original = " ".join(raw_words[i1:i2])
            corrected = " ".join(norm_words[j1:j2])
            applied.append(f'"{original}" \u2192 "{corrected}" (llm_normalize)')
            print(f"\u2699\ufe0f  llm_normalize: '{original}' \u2192 '{corrected}'")

    return applied



def normalize_transcript(raw_text: str) -> dict:
    """
    Normalizes a voice note transcript.

    Pipeline:
    1. Deterministik pre-normalizasyon — bilinen hataları Python'da düzelt
    2. LLM normalizasyon — geri kalanı ve belirsizlikleri halleder

    Returns:
    {
        "normalized_text": str,
        "low_confidence_segments": [...]
    }
    """
    # ── Step 1: Deterministik pre-normalizasyon ───────────────────────────────
    try:
        from core.pre_normalize import apply_deterministic_corrections
    except ImportError:
        try:
            from pre_normalize import apply_deterministic_corrections
        except ImportError:
            apply_deterministic_corrections = None

    pre_result = None
    if apply_deterministic_corrections:
        pre_result = apply_deterministic_corrections(raw_text)
        working_text = pre_result["text"]
        if pre_result["applied"]:
            print(f"⚙️  pre_normalize: {len(pre_result['applied'])} correction(s) applied:")
            for a in pre_result["applied"]:
                print(f"   → {a}")
    else:
        working_text = raw_text

    # ── Step 2: LLM normalizasyon ─────────────────────────────────────────────
    knowledge_context = _get_knowledge_context()
    kb_section = f"\n\n{knowledge_context}" if knowledge_context else ""

    system_prompt = f"""You are a transcription corrector for personal health voice notes.

ALWAYS respond in English regardless of the language used in the transcript or conversation.

The transcript was produced by Whisper automatic speech recognition. Your job is to:
1. Fix transcription errors you are highly confident about
2. Flag terms you are uncertain about so the user can review them

You must return a JSON object with two fields:
- "normalized_text" — the full transcript with high-confidence fixes applied
- "low_confidence_segments" — terms that need user review

─────────────────────────────────────────
HOW TO DECIDE WHAT TO FIX:
─────────────────────────────────────────

Fix a term directly (high confidence) when:
- The written form clearly does not match what was intended given the surrounding context
- The correct form is unambiguous based on context alone
- The correction is a simple capitalization or spacing fix for a known term

Flag a term (low confidence) when:
- A word or phrase seems out of place in a health context but you cannot be certain of the correct form
- A term could be a brand name, supplement, device, medication, or proper noun that Whisper may have misheard
- The context strongly suggests a health term but the written form is ambiguous or unusual
- You have a suggested correction but are not fully certain

Leave a term unchanged and do not flag it when:
- It is unusual but could legitimately be correct
- It is a known abbreviation or acronym used in health contexts
- The context does not give enough signal to judge

─────────────────────────────────────────
CRITICAL RULES:
─────────────────────────────────────────
- Never alter a term just because it looks unusual
- Never expand or modify abbreviations unless clearly wrong
- Never change sentence structure, grammar, or meaning
- Context is everything — the same word may be correct in one sentence and wrong in another
- When in doubt, flag rather than fix. Missing a correction is better than making a wrong one.
- Do not flag every unknown word — only flag when there is a genuine reason to suspect a transcription error{kb_section}

─────────────────────────────────────────
OUTPUT FORMAT — return ONLY this JSON:
─────────────────────────────────────────
{{
  "normalized_text": "full corrected transcript here",
  "low_confidence_segments": [
    {{
      "original": "exact word or phrase from transcript",
      "suggested": "your best guess correction, or null if no idea",
      "context": "5-8 surrounding words for display",
      "reason": "why you think this may be a transcription error"
    }}
  ]
}}

If there are no uncertain terms, return an empty list for low_confidence_segments."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": working_text}
        ]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        llm_low_conf = result.get("low_confidence_segments", [])

        # Filter out segments already handled by pre_normalize
        pre_applied_originals = set()
        pre_applied_corrected = set()
        if pre_result:
            for applied_msg in pre_result["applied"]:
                if applied_msg.startswith('"'):
                    parts = applied_msg.split('"')
                    if len(parts) >= 4:
                        original_term = parts[1].lower()
                        corrected_term = parts[3].lower()
                        pre_applied_originals.add(original_term)
                        pre_applied_corrected.add(corrected_term)

        print(f"⚙️  normalize filter: pre_applied={pre_applied_originals}")
        print(f"⚙️  normalize filter: pre_corrected={pre_applied_corrected}")
        print(f"⚙️  normalize filter: llm flagged={[s.get('original') for s in llm_low_conf]}")

        filtered_llm_low_conf = [
            seg for seg in llm_low_conf
            if seg.get("original", "").lower() not in pre_applied_originals
            and seg.get("original", "").lower() not in pre_applied_corrected
        ]

        print(f"⚙️  normalize filter: after filter={[s.get('original') for s in filtered_llm_low_conf]}")

        pre_flagged = pre_result["flagged"] if pre_result else []
        all_low_conf = pre_flagged + filtered_llm_low_conf

        # LLM'in normalized_text'e uyguladigi degisiklikleri yakala (difflib ile)
        normalized_text_final = result.get("normalized_text", working_text)
        llm_applied = _detect_llm_corrections(working_text, normalized_text_final)
        pre_applied = pre_result["applied"] if pre_result else []
        all_applied = pre_applied + llm_applied

        return {
            "normalized_text": normalized_text_final,
            "low_confidence_segments": all_low_conf,
            "applied_corrections": all_applied
        }
    except json.JSONDecodeError:
        print("⚠️ normalize_transcript JSON parse error")
        return {
            "normalized_text": working_text,
            "low_confidence_segments": pre_result["flagged"] if pre_result else [],
            "applied_corrections": pre_result["applied"] if pre_result else []
        }