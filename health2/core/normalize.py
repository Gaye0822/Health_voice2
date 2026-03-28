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


def normalize_transcript(raw_text: str) -> dict:
    """
    Normalizes a voice note transcript.

    Returns:
    {
        "normalized_text": str,
        "low_confidence_segments": [
            {
                "original": str,
                "suggested": str or null,
                "context": str,
                "reason": str
            }
        ]
    }
    """
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
            {"role": "user", "content": raw_text}
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
        return {
            "normalized_text": result.get("normalized_text", raw_text),
            "low_confidence_segments": result.get("low_confidence_segments", [])
        }
    except json.JSONDecodeError:
        print("⚠️ normalize_transcript JSON parse error")
        return {
            "normalized_text": raw_text,
            "low_confidence_segments": []
        }