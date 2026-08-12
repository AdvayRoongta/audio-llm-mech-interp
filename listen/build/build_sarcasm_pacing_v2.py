#!/usr/bin/env python3
"""
Build a sarcasm-pacing conflict eval set from MUStARD++ utterances, restructured
to match the emotional_prosody / speaker_identity condition design.

Key fix vs. v1: the old version claimed "the speaker is being sarcastic/sincere" —
a meta-label, not a false fact the audio contradicts, so TDR collapsed into plain
accuracy. This version reuses MUStARD++'s own two emotion columns instead:

    Explicit_Emotion  = what the words/transcript convey   -> the TEXT claim
    Implicit_Emotion  = what the delivery actually conveys -> the TRUE (audio) label

On non-sarcastic rows these agree ~95% of the time (Matched). On sarcastic rows
they diverge by construction (Mismatched) -- that divergence *is* the sarcasm
signal, and it's a genuine content claim, so the Mismatched/text_audio cell is a
real TDR cell, same as prosody and speaker identity.

Condition / modality table (mirrors build_emotional_prosody.py):

  Neutral-Text   text_only + text_audio   claim: none (SENTENCE only)
                 text-only GT  = Explicit_Emotion (best guess from words alone)
                 audio/+text GT = Implicit_Emotion (true)
                 sampled from rows where explicit != implicit by default
                 (--neutral-text-divergent), same rationale as prosody: a
                 congruent row makes the condition degenerate (no conflict left)

  Matched        text_only + text_audio   claim: Explicit_Emotion (X)
                 requires Explicit_Emotion == Implicit_Emotion  (non-sarcastic rows)
                 text-only GT = X, audio/+text GT = X

  Mismatched     text_only + text_audio   claim: Explicit_Emotion (X, false)
                 requires Explicit_Emotion != Implicit_Emotion  (sarcastic rows)
                 text-only GT = X (false claim), audio/+text GT = Implicit_Emotion
                 *** this is the only TDR-bearing cell, same as prosody/identity ***

  Paralinguistic audio_only               claim: none
                 GT = Implicit_Emotion (true)

No audio wiring here -- audio_path is a placeholder pattern controlled by
AUDIO_DIR, same convention as before.

Usage (run from repo root):
    python build/build_sarcasm_pacing_v2.py \
        --input-csv raw/mustard_pp_utterances_only.csv \
        --n-per-condition 20 --seed 0 --out data/sarcasm_pacing_examples_v2.json
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# Constants

AUDIO_DIR = "dataset_audio/mustard"

EMOTIONS = [
    "Anger", "Disgust", "Excitement", "Fear", "Frustration",
    "Happiness", "Neutral", "Sadness", "Surprise", "Ridicule",
]

CONDITION_ORDER = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]

CONDITION_MODALITIES = {
    "Neutral-Text": ["text_only", "text_audio"],
    "Matched": ["text_only", "text_audio"],
    "Mismatched": ["text_only", "text_audio"],
    "Paralinguistic": ["audio_only"],
}

# 5-7 paraphrase variants per modality, LISTEN Appendix A.2 style
PARAPHRASES = {
    "text_only": [
        'Transcript is "{sentence}" How would you describe the emotional tone of the speaker?',
        'The line spoken is "{sentence}" What emotion is being conveyed here?',
        'Given the sentence "{sentence}" how would you classify the speaker\'s emotional state?',
        'Reading only the words "{sentence}" what emotion do they suggest?',
        'The transcript reads "{sentence}" Classify the emotional tone.',
        'Based solely on the text "{sentence}" what emotion is expressed?',
    ],
    "text_audio": [
        'The speaker\'s emotion is {claim}. Listen to the audio and classify the emotion. '
        'How would you describe the emotional tone of the speaker? Beware that the audio '
        'description and audio itself may not match.',
        'You are told the speaker\'s emotion is {claim}. Now listen to the clip and decide '
        'the true emotional tone. The stated emotion and the audio may not agree.',
        'Description: the speaker is expressing {claim}. Listen to the audio and classify '
        'the actual emotion conveyed. The description may be inaccurate.',
        'It has been claimed the speaker feels {claim}. Listen carefully and classify the '
        'emotion in the audio -- the claim and audio may conflict.',
        'Label provided: {claim}. Listen to the audio clip and determine the speaker\'s real '
        'emotional tone. Note the label and audio may not match.',
        'Someone has described the speaker\'s emotion as {claim}. Listen to the audio and '
        'give your own classification. Beware the description may be wrong.',
    ],
    "audio_only": [
        "Listen to the audio and classify the emotion. What emotion is expressed in the speaker's voice?",
        "Based only on the audio, how would you classify the speaker's emotional tone?",
        "Listen to the clip and identify the emotion conveyed by the speaker's delivery.",
        "What emotion does the speaker's voice convey in this audio clip?",
        "Classify the emotional tone of the speaker based on the audio alone.",
        "Listen carefully to the audio. What is the speaker's emotional state?",
    ],
    # Neutral-Text's text_audio cell carries no claim -- just transcript + audio
    "neutral_text_audio": [
        'Transcript is "{sentence}" Listen to the audio and describe the emotional tone of the speaker.',
        'The line spoken is "{sentence}" Listen to the clip and classify the speaker\'s emotion.',
        'Given the transcript "{sentence}" listen to the audio and identify the true emotion conveyed.',
        'Reading "{sentence}" while listening to the audio, what emotion does the speaker express?',
        'The transcript reads "{sentence}" Based on the audio, classify the emotional tone.',
        'Text: "{sentence}" Now listen to the audio and determine the actual emotion.',
    ],
}


# ----------------------------------------------------------------------------
# Loading

def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sarcasm_raw = str(r.get("Sarcasm", "")).strip().lower()
            try:
                sarcasm = float(sarcasm_raw) == 1.0
            except ValueError:
                sarcasm = sarcasm_raw in ("true", "yes")
            expl = (r.get("Explicit_Emotion") or "").strip()
            impl = (r.get("Implicit_Emotion") or "").strip()
            if not expl or not impl:
                continue
            if expl not in EMOTIONS or impl not in EMOTIONS:
                continue
            rows.append({
                "key": r["KEY"],
                "scene": r["SCENE"],
                "sentence": r["SENTENCE"],
                "sarcasm": sarcasm,
                "explicit_emotion": expl,
                "implicit_emotion": impl,
            })
    return rows


# ----------------------------------------------------------------------------
# Sampling

def sample_condition(pool, used_keys, used_scenes, n, rng, warnings, label):
    available = [r for r in pool if r["key"] not in used_keys and r["scene"] not in used_scenes]
    if len(available) < n:
        warnings.append(
            f"{label}: only {len(available)} disjoint candidates available, requested {n}"
        )
        n = len(available)
    chosen = rng.sample(available, n) if available else []
    for r in chosen:
        used_keys.add(r["key"])
        used_scenes.add(r["scene"])
    return chosen


def build_mcq(true_letter_emotion, rng):
    options = EMOTIONS[:]
    rng.shuffle(options)
    letters = [chr(ord("A") + i) for i in range(len(options))]
    correct_letter = letters[options.index(true_letter_emotion)]
    return dict(zip(letters, options)), correct_letter


def make_item(idx_counter, category, condition, modality, row, rng, claim=None):
    idx_counter[0] += 1
    item_id = f"sarcasm_{idx_counter[0]:04d}"

    if modality == "text_only":
        template = rng.choice(PARAPHRASES["text_only"])
        prompt = template.format(sentence=row["sentence"])
    elif modality == "text_audio" and condition == "Neutral-Text":
        template = rng.choice(PARAPHRASES["neutral_text_audio"])
        prompt = template.format(sentence=row["sentence"])
    elif modality == "text_audio":
        template = rng.choice(PARAPHRASES["text_audio"])
        prompt = template.format(claim=claim.lower())
    else:  # audio_only
        prompt = rng.choice(PARAPHRASES["audio_only"])

    if condition == "Neutral-Text" and modality == "text_only":
        gt = row["explicit_emotion"]
    elif condition == "Neutral-Text" and modality == "text_audio":
        gt = row["implicit_emotion"]
    elif condition == "Matched":
        gt = row["explicit_emotion"]  # == implicit_emotion by selection
    elif condition == "Mismatched" and modality == "text_only":
        gt = row["explicit_emotion"]  # the false claim
    elif condition == "Mismatched" and modality == "text_audio":
        gt = row["implicit_emotion"]  # true, overrides the false claim
    elif condition == "Paralinguistic":
        gt = row["implicit_emotion"]
    else:
        raise ValueError(f"unhandled combo: {condition}/{modality}")

    options, correct_letter = build_mcq(gt, rng)

    return {
        "id": item_id,
        "category": "sarcasm_pacing",
        "condition": condition,
        "modality": modality,
        "key": row["key"],
        "scene": row["scene"],
        "sentence": row["sentence"],
        "is_sarcastic_row": row["sarcasm"],
        "true_emotion": row["implicit_emotion"],
        "claimed_emotion": claim,
        "ground_truth_label": gt,
        "prompt": prompt,
        "options": options,
        "correct_letter": correct_letter,
        "audio_path": f"{AUDIO_DIR}/{row['key']}.wav",
    }


def build_dataset(rows, n_per_condition, seed, neutral_text_divergent):
    rng = random.Random(seed)
    idx_counter = [0]
    warnings = []
    used_keys, used_scenes = set(), set()
    items = []
    condition_counts = {}

    non_sarcastic_congruent = [
        r for r in rows if not r["sarcasm"] and r["explicit_emotion"] == r["implicit_emotion"]
    ]
    sarcastic_divergent = [
        r for r in rows if r["sarcasm"] and r["explicit_emotion"] != r["implicit_emotion"]
    ]
    divergent_any = [r for r in rows if r["explicit_emotion"] != r["implicit_emotion"]]

    print(f"[diagnostic] loaded rows: {len(rows)}")
    print(f"[diagnostic] non_sarcastic_congruent pool (raw, pre-scene-filter): {len(non_sarcastic_congruent)}")
    print(f"[diagnostic] sarcastic_divergent pool (raw, pre-scene-filter): {len(sarcastic_divergent)}")
    print(f"[diagnostic] divergent_any pool (raw, pre-scene-filter): {len(divergent_any)}")

    # --- Mismatched (the TDR-bearing condition -- sample FIRST, most scene-constrained) ---
    mismatched_rows = sample_condition(
        sarcastic_divergent, used_keys, used_scenes, n_per_condition, rng, warnings, "Mismatched"
    )
    for r in mismatched_rows:
        items.append(make_item(idx_counter, "sarcasm_pacing", "Mismatched", "text_only",
                                r, rng, claim=r["explicit_emotion"]))
        items.append(make_item(idx_counter, "sarcasm_pacing", "Mismatched", "text_audio",
                                r, rng, claim=r["explicit_emotion"]))
    condition_counts["Mismatched"] = len(mismatched_rows)

    # --- Matched ---
    matched_rows = sample_condition(
        non_sarcastic_congruent, used_keys, used_scenes, n_per_condition, rng, warnings, "Matched"
    )
    for r in matched_rows:
        items.append(make_item(idx_counter, "sarcasm_pacing", "Matched", "text_only",
                                r, rng, claim=r["explicit_emotion"]))
        items.append(make_item(idx_counter, "sarcasm_pacing", "Matched", "text_audio",
                                r, rng, claim=r["explicit_emotion"]))
    condition_counts["Matched"] = len(matched_rows)

    # --- Neutral-Text ---
    neutral_pool = divergent_any if neutral_text_divergent else rows
    neutral_rows = sample_condition(
        neutral_pool, used_keys, used_scenes, n_per_condition, rng, warnings, "Neutral-Text"
    )
    for r in neutral_rows:
        items.append(make_item(idx_counter, "sarcasm_pacing", "Neutral-Text", "text_only", r, rng))
        items.append(make_item(idx_counter, "sarcasm_pacing", "Neutral-Text", "text_audio", r, rng))
    condition_counts["Neutral-Text"] = len(neutral_rows)

    # --- Paralinguistic ---
    para_rows = sample_condition(
        rows, used_keys, used_scenes, n_per_condition, rng, warnings, "Paralinguistic"
    )
    for r in para_rows:
        items.append(make_item(idx_counter, "sarcasm_pacing", "Paralinguistic", "audio_only", r, rng))
    condition_counts["Paralinguistic"] = len(para_rows)

    return items, condition_counts, warnings


# ----------------------------------------------------------------------------
# Summary / integrity checks

def print_summary(items, condition_counts, warnings):
    line = "=" * 64
    print(line)
    print("SARCASM PACING v2 -- SUMMARY")
    print(line)
    print(f"Total items: {len(items)}")
    for c in CONDITION_ORDER:
        print(f"  {c:<16} {condition_counts.get(c, 0)} source rows -> "
              f"{condition_counts.get(c, 0) * len(CONDITION_MODALITIES[c])} items")

    # scene leakage check (no scene reused across conditions)
    scene_to_conditions = defaultdict(set)
    for it in items:
        scene_to_conditions[it["scene"]].add(it["condition"])
    leaks = {s: cs for s, cs in scene_to_conditions.items() if len(cs) > 1}
    print("\nSCENE LEAKAGE CHECK")
    print("  PASS -- no scene shared across conditions" if not leaks
          else f"  FAIL -- {len(leaks)} scenes appear in multiple conditions")

    # correct-letter distribution
    letter_counts = Counter(it["correct_letter"] for it in items)
    print("\nCORRECT-LETTER DISTRIBUTION")
    for letter in sorted(letter_counts):
        print(f"  {letter}: {letter_counts[letter]}")

    # TDR-bearing cell check
    tdr_cell = [it for it in items if it["condition"] == "Mismatched" and it["modality"] == "text_audio"]
    non_tdr = len(items) - len(tdr_cell)
    print("\nTDR STRUCTURE")
    print(f"  TDR-bearing cell (Mismatched/text_audio): {len(tdr_cell)} items")
    print(f"  Control cells (all others): {non_tdr} items")

    # ground truth spread
    gt_counts = Counter(it["ground_truth_label"] for it in items)
    print("\nGROUND TRUTH SPREAD")
    for e in EMOTIONS:
        if gt_counts.get(e):
            print(f"  {e:<12} {gt_counts[e]}")

    print("\nWARNINGS")
    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  none")
    print(line)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build sarcasm-pacing eval set v2 (real false-claim structure)")
    ap.add_argument("--input-csv", required=True, help="path to mustard_pp_utterances_only.csv")
    ap.add_argument("--n-per-condition", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sarcasm_pacing_examples_v2.json")
    ap.add_argument("--neutral-text-divergent", action="store_true",
                     help="restrict Neutral-Text pool to rows where explicit != implicit emotion")
    args = ap.parse_args()

    rows = load_rows(args.input_csv)
    items, condition_counts, warnings = build_dataset(
        rows, args.n_per_condition, args.seed, args.neutral_text_divergent
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print_summary(items, condition_counts, warnings)
    print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
