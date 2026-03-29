#!/usr/bin/env python3
"""
regression_test.py — Pipeline regression test suite

Usage:
  # Save baseline from batch docx (old outputs)
  python regression_test.py --save-baseline --docx /path/to/batch1.docx

  # Run new pipeline and compare against baseline
  python regression_test.py --run --docx /path/to/batch1.docx

  # Run specific transcripts only
  python regression_test.py --run --docx /path/to/batch1.docx --ids T1 T4 T9
"""

import re
import json
import sys
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

SNAPSHOT_DIR = Path(__file__).parent / "tests" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Fields to ignore in comparison (these legitimately vary)
IGNORE_FIELDS = {"notes", "time", "start_time", "onset_time"}


# ─────────────────────────────────────────
# DOCX PARSER
# ─────────────────────────────────────────

def parse_docx(docx_path: str) -> dict:
    """
    Parse a batch docx file into a dict of {T_id: {transcript, old_entities}}.
    Uses pandoc for conversion.
    """
    import subprocess
    result = subprocess.run(
        ["pandoc", docx_path, "-t", "plain"],
        capture_output=True, text=True
    )
    content = result.stdout

    sections = re.split(r'(?m)^(T\d+[-])', content)

    transcripts = {}
    i = 1
    while i < len(sections) - 1:
        tid = sections[i].rstrip('-').strip()
        body = sections[i + 1]

        # Split transcript from entity output
        layer_match = re.search(
            r'\n(Event Layer|Nutrition Layer|Theory Layer|Outside Bucket)',
            body
        )
        if layer_match:
            transcript_text = body[:layer_match.start()].strip()
            entities_raw = body[layer_match.start():].strip()
        else:
            # Some transcripts skip layer headers — find first { as entity start
            brace_match = re.search(r'\n\{', body)
            if brace_match:
                transcript_text = body[:brace_match.start()].strip()
                entities_raw = body[brace_match.start():].strip()
            else:
                transcript_text = body.strip()
                entities_raw = ""

        transcripts[tid] = {
            "transcript": transcript_text,
            "old_entities": parse_entity_blocks(entities_raw)
        }
        i += 2

    return transcripts


def parse_entity_blocks(raw: str) -> list:
    """
    Parse entity blocks from docx output.
    Uses brace matching to handle nested content in field values.
    """
    entities = []
    i = 0
    while i < len(raw):
        start = raw.find('{', i)
        if start == -1:
            break
        # Find matching closing brace
        depth = 0
        end = start
        for j in range(start, len(raw)):
            if raw[j] == '{':
                depth += 1
            elif raw[j] == '}':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end > start:
            block = raw[start+1:end]
            entity = _parse_block(block)
            if entity and 'type' in entity:
                entities.append(entity)
        i = end + 1

    return entities


def _parse_block(block: str) -> dict:
    """Parse a single entity block in either compact or multi-line format."""
    # Try compact JSON first
    json_str = '{' + block + '}'
    json_str = re.sub(r':\s*NULL\b', ': null', json_str)
    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Multi-line format: keys and values on separate lines
    # Tokenize: quoted strings, NULL, and bare numbers
    content = block.replace('\n', ' ')
    token_pattern = re.compile(r'"([^"]*)"|(\bNULL\b)|(-?\d+(?:\.\d+)?)')
    raw_tokens = token_pattern.findall(content)

    all_tokens = []
    for quoted, null, number in raw_tokens:
        if null:
            all_tokens.append(None)
        elif number and not quoted:
            try:
                all_tokens.append(int(number))
            except ValueError:
                all_tokens.append(float(number))
        else:
            all_tokens.append(quoted)

    entity = {}
    i = 0
    while i < len(all_tokens) - 1:
        key = all_tokens[i]
        val = all_tokens[i + 1]
        if isinstance(key, str) and key:
            entity[key] = val
        i += 2

    return entity if entity else {}


# ─────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────

def run_pipeline(transcript: str) -> list:
    """Run the current pipeline on a transcript text."""
    sys.path.insert(0, str(Path(__file__).parent))

    from core.mention import extract_mentions
    from core.structure import structure_mentions
    from core.validate import validate_entities

    mentions = extract_mentions(transcript)
    entities = structure_mentions(mentions, transcript)
    entities, _ = validate_entities(entities, transcript)
    return entities


# ─────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────

def entity_key(entity: dict) -> str:
    """Unique key for an entity — type + label/metric."""
    t = entity.get("type", "?")
    label = entity.get("label", entity.get("metric", entity.get("raw_text", "?")[:40]))
    return f"{t}:{label}"


def compare_entities(old: list, new: list) -> dict:
    """
    Compare old and new entity lists.
    Returns dict with added, removed, changed lists.
    """
    old_keys = {entity_key(e): e for e in old}
    new_keys = {entity_key(e): e for e in new}

    added = [new_keys[k] for k in new_keys if k not in old_keys]
    removed = [old_keys[k] for k in old_keys if k not in new_keys]
    changed = []

    for k in old_keys:
        if k in new_keys:
            old_e = old_keys[k]
            new_e = new_keys[k]
            diffs = {}
            all_fields = set(old_e) | set(new_e)
            for field in all_fields:
                if field in IGNORE_FIELDS:
                    continue
                old_val = old_e.get(field)
                new_val = new_e.get(field)
                if old_val != new_val:
                    diffs[field] = {"old": old_val, "new": new_val}
            if diffs:
                changed.append({"entity": k, "diffs": diffs})

    return {"added": added, "removed": removed, "changed": changed}


def print_comparison(tid: str, result: dict):
    """Print comparison results for one transcript."""
    added = result["added"]
    removed = result["removed"]
    changed = result["changed"]

    total = len(added) + len(removed) + len(changed)
    if total == 0:
        print(f"  {tid} ✅ — no changes")
        return

    print(f"  {tid} ⚠️  — {total} difference(s):")

    for e in added:
        label = entity_key(e)
        print(f"    ➕ ADDED:   [{e.get('type')}] {e.get('label', e.get('metric', '?'))}")

    for e in removed:
        print(f"    ➖ REMOVED: [{e.get('type')}] {e.get('label', e.get('metric', '?'))}")

    for c in changed:
        print(f"    🔄 CHANGED: {c['entity']}")
        for field, vals in c["diffs"].items():
            print(f"       {field}: {vals['old']!r} → {vals['new']!r}")


# ─────────────────────────────────────────
# SAVE BASELINE
# ─────────────────────────────────────────

def save_baseline(docx_path: str, ids: list = None):
    """Parse old outputs from docx and save as baseline snapshots."""
    print(f"📂 Parsing {docx_path}...")
    data = parse_docx(docx_path)

    saved = 0
    for tid, content in sorted(data.items()):
        if ids and tid not in ids:
            continue
        snapshot = {
            "transcript": content["transcript"],
            "entities": content["old_entities"]
        }
        path = SNAPSHOT_DIR / f"{tid}_baseline.json"
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        print(f"  ✅ Saved baseline: {tid} ({len(content['old_entities'])} entities)")
        saved += 1

    print(f"\n✅ {saved} baseline(s) saved to {SNAPSHOT_DIR}")


# ─────────────────────────────────────────
# RUN REGRESSION
# ─────────────────────────────────────────

def run_regression(docx_path: str, ids: list = None):
    """Run pipeline on transcripts and compare against baselines."""
    print(f"📂 Parsing {docx_path}...")
    data = parse_docx(docx_path)

    results = {}
    has_diff = False

    for tid, content in sorted(data.items()):
        if ids and tid not in ids:
            continue

        baseline_path = SNAPSHOT_DIR / f"{tid}_baseline.json"
        if not baseline_path.exists():
            print(f"  {tid} ⏭️  — no baseline, skipping (run --save-baseline first)")
            continue

        with open(baseline_path) as f:
            baseline = json.load(f)

        print(f"\n  Running pipeline for {tid}...")
        try:
            new_entities = run_pipeline(content["transcript"])
        except Exception as e:
            print(f"  {tid} ❌ — pipeline error: {e}")
            continue

        # Save new output
        new_path = SNAPSHOT_DIR / f"{tid}_new.json"
        with open(new_path, 'w') as f:
            json.dump({"transcript": content["transcript"], "entities": new_entities}, f, indent=2)

        # Compare
        result = compare_entities(baseline["entities"], new_entities)
        results[tid] = result

        if any(result[k] for k in ("added", "removed", "changed")):
            has_diff = True

    # Print summary
    print("\n" + "=" * 60)
    print("REGRESSION REPORT")
    print("=" * 60)
    for tid in sorted(results.keys()):
        print_comparison(tid, results[tid])

    print()
    if has_diff:
        print("⚠️  Differences found — review above")
    else:
        print("✅ All transcripts match baseline")

    return results


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline regression test suite")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Parse docx old outputs and save as baselines")
    parser.add_argument("--run", action="store_true",
                        help="Run pipeline and compare against baselines")
    parser.add_argument("--docx", required=True,
                        help="Path to batch docx file")
    parser.add_argument("--ids", nargs="+",
                        help="Specific transcript IDs to process (e.g. T1 T4 T9)")

    args = parser.parse_args()

    if args.save_baseline:
        save_baseline(args.docx, args.ids)
    elif args.run:
        run_regression(args.docx, args.ids)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()