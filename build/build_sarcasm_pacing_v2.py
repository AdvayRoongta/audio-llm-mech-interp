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

ANSWER_INSTRUCTION = "Answer with the letter of the correct option."

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
    # claim_last (default order): claim sentence leads, audio instruction follows --
    # the claim is the thing immediately preceding the instruction/question.
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
    # claim_first order: same six framings, but the audio instruction leads and the
    # claim sentence is appended last -- the claim is now the thing immediately
    # preceding the answer, mirroring the claim_last variant with position swapped.
    "text_audio_claim_first": [
        'Listen to the audio and classify the emotion. How would you describe the emotional '
        'tone of the speaker? Beware that the audio description and audio itself may not '
        'match. You are told the speaker\'s emotion is {claim}.',
        'Now listen to the clip and decide the true emotional tone. The stated emotion and '
        'the audio may not agree. You are told the speaker\'s emotion is {claim}.',
        'Listen to the audio and classify the actual emotion conveyed. The description may '
        'be inaccurate. Description: the speaker is expressing {claim}.',
        'Listen carefully and classify the emotion in the audio -- the claim and audio may '
        'conflict. It has been claimed the speaker feels {claim}.',
        'Listen to the audio clip and determine the speaker\'s real emotional tone. Note the '
        'label and audio may not match. Label provided: {claim}.',
        'Listen to the audio and give your own classification. Beware the description may be '
        'wrong. Someone has described the speaker\'s emotion as {claim}.',
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


def ground_truth_for(condition, modality, row):
    if condition == "Neutral-Text" and modality == "text_only":
        return row["explicit_emotion"]
    if condition == "Neutral-Text" and modality == "text_audio":
        return row["implicit_emotion"]
    if condition == "Matched":
        return row["explicit_emotion"]  # == implicit_emotion by selection
    if condition == "Mismatched" and modality == "text_only":
        return row["explicit_emotion"]  # the false claim
    if condition == "Mismatched" and modality == "text_audio":
        return row["implicit_emotion"]  # true, overrides the false claim
    if condition == "Paralinguistic":
        return row["implicit_emotion"]
    raise ValueError(f"unhandled combo: {condition}/{modality}")


def make_item(idx_counter, category, condition, modality, row, rng, claim=None, order=None,
              options=None, correct_letter=None):
    """
    `order` is None for items with no competing claim (Neutral-Text,
    Paralinguistic) and for text_only claim-condition items, which have a
    fixed single layout and are stamped "claim_last" by the caller without
    being duplicated. text_audio claim-condition items (Matched/Mismatched)
    are called twice by the caller, once per order value, and share one
    `options`/`correct_letter` pair (computed by the caller) so the MCQ
    lettering is identical across the claim_first/claim_last pair -- only
    the prompt text differs.
    """
    idx_counter[0] += 1
    if modality == "text_audio" and order is not None:
        item_id = f"sarcasm_{order}_{idx_counter[0]:04d}"
    else:
        item_id = f"sarcasm_{idx_counter[0]:04d}"

    if modality == "text_only":
        template = rng.choice(PARAPHRASES["text_only"])
        prompt = template.format(sentence=row["sentence"])
    elif modality == "text_audio" and condition == "Neutral-Text":
        template = rng.choice(PARAPHRASES["neutral_text_audio"])
        prompt = template.format(sentence=row["sentence"])
    elif modality == "text_audio":
        key = "text_audio_claim_first" if order == "claim_first" else "text_audio"
        template = rng.choice(PARAPHRASES[key])
        prompt = template.format(claim=claim.lower())
    else:  # audio_only
        prompt = rng.choice(PARAPHRASES["audio_only"])

    gt = ground_truth_for(condition, modality, row)
    if options is None or correct_letter is None:
        options, correct_letter = build_mcq(gt, rng)

    option_lines = "\n".join(f"{l}. {v}" for l, v in sorted(options.items()))
    prompt = f"{prompt}\n\n{option_lines}\n\n{ANSWER_INSTRUCTION}"

    return {
        "id": item_id,
        "category": "sarcasm_pacing",
        "condition": condition,
        "modality": modality,
        "order": order,
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
                                r, rng, claim=r["explicit_emotion"], order="claim_last"))
        # shared options/correct_letter across the order pair -- only the
        # prompt text differs between claim_last and claim_first
        gt = ground_truth_for("Mismatched", "text_audio", r)
        shared_options, shared_letter = build_mcq(gt, rng)
        for order in ("claim_last", "claim_first"):
            items.append(make_item(idx_counter, "sarcasm_pacing", "Mismatched", "text_audio",
                                    r, rng, claim=r["explicit_emotion"], order=order,
                                    options=shared_options, correct_letter=shared_letter))
    condition_counts["Mismatched"] = len(mismatched_rows)

    # --- Matched ---
    matched_rows = sample_condition(
        non_sarcastic_congruent, used_keys, used_scenes, n_per_condition, rng, warnings, "Matched"
    )
    for r in matched_rows:
        items.append(make_item(idx_counter, "sarcasm_pacing", "Matched", "text_only",
                                r, rng, claim=r["explicit_emotion"], order="claim_last"))
        # shared options/correct_letter across the order pair -- only the
        # prompt text differs between claim_last and claim_first
        gt = ground_truth_for("Matched", "text_audio", r)
        shared_options, shared_letter = build_mcq(gt, rng)
        for order in ("claim_last", "claim_first"):
            items.append(make_item(idx_counter, "sarcasm_pacing", "Matched", "text_audio",
                                    r, rng, claim=r["explicit_emotion"], order=order,
                                    options=shared_options, correct_letter=shared_letter))
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
    items_per_condition = Counter(it["condition"] for it in items)
    for c in CONDITION_ORDER:
        print(f"  {c:<16} {condition_counts.get(c, 0)} source rows -> "
              f"{items_per_condition.get(c, 0)} items")

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

    # order breakdown (Matched/Mismatched, text_audio duplicated; text_only stamped claim_last)
    print("\nORDER BREAKDOWN (Matched/Mismatched, text_audio duplicated; text_only stamped claim_last)")
    order_grid = Counter(
        (it["condition"], it["modality"], it["order"])
        for it in items if it["order"] is not None
    )
    for condition in ["Matched", "Mismatched"]:
        for modality in CONDITION_MODALITIES[condition]:
            for order in ("claim_last", "claim_first"):
                n = order_grid.get((condition, modality, order), 0)
                if n:
                    print(f"  {condition:<16} {modality:<12} {order:<12} {n}")

    # claim_last/claim_first pairing check
    print("\nCLAIM_LAST/CLAIM_FIRST PAIRING CHECK")
    pairing_ok = True
    for condition in ["Matched", "Mismatched"]:
        last_keys = {
            it["key"] for it in items
            if it["condition"] == condition and it["modality"] == "text_audio"
            and it["order"] == "claim_last"
        }
        first_keys = {
            it["key"] for it in items
            if it["condition"] == condition and it["modality"] == "text_audio"
            and it["order"] == "claim_first"
        }
        if last_keys != first_keys:
            pairing_ok = False
            print(f"  FAIL {condition}: claim_last/claim_first key sets differ "
                  f"({len(last_keys)} vs {len(first_keys)})")
        else:
            print(f"  OK   {condition}: {len(last_keys)} rows, fully paired 1:1")
    if not pairing_ok:
        print("  FAIL -- order pairing broken, investigate before using this dataset")

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
    ap.add_argument("--n-per-condition", type=int, default=133)
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
