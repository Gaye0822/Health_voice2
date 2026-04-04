"""
mention_filter.py — Deterministic mention filter

Runs between mention.py and structure.py.
Filters out mentions that should never become health events,
based on temporal_evidence and simple transcript signals.

No LLM involved — pure Python, fully testable.
"""

# Words/phrases that confirm something happened in the current session
SAME_SESSION_SIGNALS = [
    "today", "this morning", "this evening", "this afternoon", "this night",
    "tonight", "just now", "just took", "just did", "just finished",
    "took it", "took them", "took my", "took the",
    "did it", "did the", "did my",
    "right now", "a moment ago", "an hour ago", "two hours ago",
    "this session", "earlier today", "this week",  # "this week" is borderline but acceptable
    # Recent past — user reporting something from last night or recent days
    "last night", "last few nights", "last three nights", "last two nights",
    "past few days", "past few nights", "yesterday",
    # Ongoing symptoms confirmed as still present in this session
    "still have", "still keep", "still get", "still feeling", "still experiencing",
    "keep waking", "keep getting", "keeps happening", "still going on",
    "still messed up", "still broken", "still awful", "still terrible",
    # Recent repeated actions confirmed as current practice
    "a couple times now", "a few times now", "couple of times now",
    "been taking", "have been taking", "have taken",
]

# temporal_evidence values that should always be dropped
ALWAYS_DROP = {"future_plan", "consultation_relay"}

# temporal_evidence values that need same-session confirmation to pass
NEEDS_CONFIRMATION = {"active_regimen", "unclear"}


def has_same_session_signal(mention: dict, transcript: str) -> bool:
    """
    Check if a mention has same-session confirmation in its raw_mention,
    context, or the broader transcript.
    """
    raw = mention.get("raw_mention", "").lower()
    context = mention.get("context", "").lower()
    transcript_lower = transcript.lower()

    for signal in SAME_SESSION_SIGNALS:
        if signal in raw or signal in context:
            return True

    # Also check if the raw mention appears near a session signal in the transcript
    raw_mention = mention.get("raw_mention", "").lower()
    if raw_mention and raw_mention in transcript_lower:
        idx = transcript_lower.index(raw_mention)
        # Check 150 chars around the mention
        window = transcript_lower[max(0, idx - 150): idx + 150]
        for signal in SAME_SESSION_SIGNALS:
            if signal in window:
                return True

    return False


def filter_mentions(mentions: list, transcript: str) -> tuple:
    """
    Filter mentions before passing to structure.py.

    Returns:
        (kept_mentions, dropped_mentions)

    Rules:
    - future_plan → always drop
    - consultation_relay → always drop
    - active_regimen → keep only if same-session confirmation found
    - unclear → keep only if same-session confirmation found
    - explicit_today → always keep
    """
    kept = []
    dropped = []

    for mention in mentions:
        te = mention.get("temporal_evidence", "unclear")
        candidate_type = mention.get("candidate_type", "")

        # theory, outside, context pass through always — structure.py handles them
        if candidate_type in ("theory", "outside", "context"):
            kept.append(mention)
            continue

        # other → pass through UNLESS reasoning indicates a label specificity failure
        # If LLM already determined the label is vague/unresolvable, don't send to structure
        if candidate_type == "other":
            reasoning = mention.get("reasoning", "").lower()
            label_fail_signals = [
                "fails the label", "vague", "group label", "not resolvable",
                "cannot be tracked", "unresolvable", "not specific",
                "cannot determine", "label specificity"
            ]
            if any(signal in reasoning for signal in label_fail_signals):
                dropped.append({
                    **mention,
                    "_filter_reason": "other: label specificity failure in reasoning"
                })
            else:
                kept.append(mention)
            continue

        # Always drop — EXCEPT measurements from consultation_relay
        # (blood test results are real data points even in consultation notes)
        if te in ALWAYS_DROP:
            if te == "consultation_relay" and candidate_type == "measurement":
                kept.append(mention)
            else:
                dropped.append({**mention, "_filter_reason": f"temporal_evidence={te}"})
            continue

        # Needs same-session confirmation
        # Exception: outcome mentions don't need same-session confirmation —
        # they are observed directional changes, not actions that need to be confirmed today
        if te in NEEDS_CONFIRMATION:
            if candidate_type == "outcome":
                kept.append(mention)
                continue
            if has_same_session_signal(mention, transcript):
                kept.append(mention)
            else:
                dropped.append({
                    **mention,
                    "_filter_reason": f"temporal_evidence={te}, no same-session signal found"
                })
            continue

        # explicit_today — always keep
        kept.append(mention)

    return kept, dropped


def log_filtered(dropped: list):
    """Print dropped mentions for debugging."""
    if dropped:
        print(f"⚙️  mention_filter: dropped {len(dropped)} mention(s):")
        for m in dropped:
            reason = m.get("_filter_reason", "")
            raw = m.get("raw_mention", "")[:60]
            print(f"   - [{m.get('candidate_type')}] {raw!r} → {reason}")