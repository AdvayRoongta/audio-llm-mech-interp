#!/usr/bin/env python3
"""
Merge the three category datasets (emotional_prosody, speaker_identity,
sarcasm_pacing_v2) into one combined dataset for cross-model inference.

Does NOT force emotion-label vocabularies together (prosody's 8 lowercase
RAVDESS labels, speaker_identity's 2-way male/female, and sarcasm's 10
Capitalized MUStARD++ labels are genuinely different label spaces per
category -- forcing them into one taxonomy would misrepresent the task).
Each item already carries its own `options` dict and `correct_letter`, so
per-item scoring never needs a shared vocabulary.

What this script DOES reconcile:
  - text field naming: `transcript` (prosody/speaker_identity) vs `sentence`
    (sarcasm) -> unified `text` field added to every item (originals kept).
  - adds `source_file` so every item traces back to which build script
    produced it.
  - verifies the "spine" fields every item must have for downstream code
    to run in category-agnostic loops: id, category, condition, modality,
    audio_path, prompt, options, correct_letter, ground_truth_label.
  - guarantees global ID uniqueness by prefixing with category, even though
    the audit found no raw collisions.

Usage (run from repo root):
    python build/merge_datasets.py \
        --prosody data/emotional_prosody_examples.json \
        --speaker-identity data/speaker_identity_examples.json \
        --sarcasm data/sarcasm_pacing_examples_v2.json \
        --out data/combined_examples.json
"""

import argparse
import json
from collections import Counter, defaultdict

SPINE_FIELDS = [
    "id", "category", "condition", "modality",
    "audio_path", "prompt", "options", "correct_letter", "ground_truth_label",
]

CONDITION_ORDER = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]
TEXT_FIELD_CANDIDATES = ["transcript", "sentence"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reconcile_item(item, source_file):
    """Return a copy of item with unified `text` and `source_file` fields."""
    out = dict(item)
    text_val = None
    for field in TEXT_FIELD_CANDIDATES:
        if field in out and out[field]:
            text_val = out[field]
            break
    out["text"] = text_val  # may be None for audio_only items, that's expected
    out["source_file"] = source_file
    out["global_id"] = f"{out.get('category', 'unknown')}::{out['id']}"
    return out


def check_spine(items, label, errors):
    missing_counter = Counter()
    for it in items:
        for field in SPINE_FIELDS:
            if field not in it or it[field] in (None, ""):
                missing_counter[field] += 1
    if missing_counter:
        for field, n in missing_counter.items():
            errors.append(f"{label}: {n} item(s) missing/empty spine field '{field}'")


def check_id_uniqueness(items, errors):
    ids = [it["global_id"] for it in items]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        errors.append(f"global_id collisions found: {dupes[:10]}{'...' if len(dupes) > 10 else ''}")


def summarize(items):
    by_cat = defaultdict(lambda: defaultdict(int))
    for it in items:
        by_cat[it["category"]][it["condition"]] += 1
    return by_cat


def print_report(items, errors):
    line = "=" * 64
    print(line)
    print("COMBINED DATASET -- MERGE REPORT")
    print(line)
    print(f"Total items: {len(items)}")

    by_cat = summarize(items)
    for cat in sorted(by_cat):
        print(f"\n  {cat}")
        for cond in CONDITION_ORDER:
            n = by_cat[cat].get(cond, 0)
            if n:
                print(f"    {cond:<16} {n}")

    print("\nSOURCE FILE COUNTS")
    src_counts = Counter(it["source_file"] for it in items)
    for src, n in src_counts.items():
        print(f"  {src:<45} {n}")

    print("\nTEXT FIELD RECONCILIATION")
    missing_text = sum(1 for it in items if it["text"] is None)
    print(f"  items with unified 'text' populated: {len(items) - missing_text}")
    print(f"  items with text=None (expected for audio_only cells): {missing_text}")

    print("\nERRORS")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
    else:
        print("  none -- safe to use combined_examples.json")
    print(line)


def main():
    ap = argparse.ArgumentParser(description="Merge the three category datasets into one.")
    ap.add_argument("--prosody", default="data/emotional_prosody_examples.json")
    ap.add_argument("--speaker-identity", default="data/speaker_identity_examples.json")
    ap.add_argument("--sarcasm", default="data/sarcasm_pacing_examples_v2.json")
    ap.add_argument("--out", default="data/combined_examples.json")
    args = ap.parse_args()

    sources = [
        (args.prosody, "emotional_prosody_examples.json"),
        (args.speaker_identity, "speaker_identity_examples.json"),
        (args.sarcasm, "sarcasm_pacing_examples_v2.json"),
    ]

    all_items = []
    errors = []
    for path, label in sources:
        try:
            raw_items = load(path)
        except FileNotFoundError:
            errors.append(f"could not find {path} -- run its build script first")
            continue
        except json.JSONDecodeError as e:
            errors.append(f"{path} is not valid JSON: {e}")
            continue
        reconciled = [reconcile_item(it, label) for it in raw_items]
        check_spine(reconciled, label, errors)
        all_items.extend(reconciled)

    check_id_uniqueness(all_items, errors)

    print_report(all_items, errors)

    if errors:
        print(f"\nNOT writing {args.out} -- fix the errors above and re-run.")
        raise SystemExit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_items)} items to {args.out}")


if __name__ == "__main__":
    main()
