#!/usr/bin/env python3
"""
Build a sarcasm-conflict evaluation dataset from MUStARD++ labeled utterances,
following the condition x modality structure of the LISTEN benchmark
("Do Audio LLMs Really LISTEN, or Just Transcribe?", EACL 2026).

Conditions (row-level selection filters):
    Neutral-Text     Sarcasm==0 and Explicit_Emotion=="Neutral"
    Matched          Sarcasm==0 and Explicit_Emotion==Implicit_Emotion
    Mismatched       Sarcasm==1
    Paralinguistic   any row (audio-only condition)

Ground truth:
    text_only                -> Neutral-Text: "Neutral" (fixed); else Explicit_Emotion
    audio_only / text_audio  -> Implicit_Emotion

Usage (run from repo root):
    python build/build_sarcasm_pacing_dataset.py \
        --input-csv raw/mustard_pp_utterances_only.csv \
        --n-per-condition 20 --seed 42 --out data/sarcasm_pacing_examples.json
"""

import argparse
import json
import random
import string
import sys
from collections import Counter, defaultdict

import pandas as pd

# --------------------------------------------------------------------------
# Label set (10 classes). Ridicule only ever appears as Implicit_Emotion.
# --------------------------------------------------------------------------
EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Excitement",
    "Fear",
    "Frustration",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Ridicule",
]
LETTERS = list(string.ascii_uppercase[: len(EMOTION_LABELS)])  # A..J

CATEGORY = "sarcasm_pacing"
AUDIO_DIR = "dataset_audio/mustard"  # placeholder root; real wiring happens later

CONDITIONS = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]
CONDITION_MODALITIES = {
    "Neutral-Text": ["text_only", "audio_only", "text_audio"],
    "Matched": ["text_only", "audio_only", "text_audio"],
    "Mismatched": ["text_only", "audio_only", "text_audio"],
    "Paralinguistic": ["audio_only"],
}

# --------------------------------------------------------------------------
# Paraphrase pools (LISTEN Appendix A.2 style). One is sampled per item.
# --------------------------------------------------------------------------
PARAPHRASES = {
    "text_only": [
        "Based on the transcript above, which emotion is the speaker expressing?",
        "Read the transcript and classify the speaker's emotion.",
        "From the wording of the transcript alone, how would you describe the speaker's emotional state?",
        "Considering only what is written, what emotion does the speaker convey?",
        "Judging by the transcript, which of the following best captures the speaker's emotion?",
        "Using the text above, identify the emotion of the speaker.",
        "What emotion does the transcript suggest the speaker is feeling?",
    ],
    "audio_only": [
        "What emotion is expressed in the speaker's voice?",
        "How would you describe the emotional tone of the speaker?",
        "Listen to the recording and classify the speaker's emotion.",
        "Based on how the speaker sounds, which emotion are they conveying?",
        "From the speaker's vocal delivery, identify the emotion being expressed.",
        "Which emotion best matches the way the speaker sounds in this clip?",
        "Attend to the speaker's tone of voice and classify their emotion.",
    ],
    "text_audio": [
        "Listen to the audio and classify the emotion the speaker is actually expressing.",
        "Using both the transcript and the recording, which emotion is the speaker conveying?",
        "Attend to how the line is delivered and identify the speaker's emotion.",
        "Given the transcript and the audio, how would you describe the speaker's emotional tone?",
        "Classify the speaker's emotion based on the recording, not only on the words.",
        "What emotion is the speaker expressing in this utterance?",
        "Taking the delivery into account, which emotion best describes the speaker?",
    ],
}

MISMATCH_WARNING = (
    "Beware that the transcript and the audio itself may not match."
)

MCQ_INSTRUCTION = "Answer with the letter of the correct option only."


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def render_options(options):
    return "\n".join(f"{ltr}. {options[ltr]}" for ltr in LETTERS)


def build_prompt(modality, transcript, paraphrase, options, warn):
    parts = []
    if modality == "text_only":
        parts.append(f'Transcript: "{transcript}"')
        parts.append(paraphrase)
    elif modality == "audio_only":
        parts.append(paraphrase)
    elif modality == "text_audio":
        parts.append(f'Transcript: "{transcript}"')
        if warn:
            parts.append(f"{paraphrase} {MISMATCH_WARNING}")
        else:
            parts.append(paraphrase)
    else:
        raise ValueError(f"unknown modality: {modality}")
    parts.append("Options:\n" + render_options(options))
    parts.append(MCQ_INSTRUCTION)
    return "\n\n".join(parts)


def shuffled_options(rng):
    labels = EMOTION_LABELS[:]
    rng.shuffle(labels)
    return {ltr: lbl for ltr, lbl in zip(LETTERS, labels)}


def letter_for(options, label):
    for ltr, lbl in options.items():
        if lbl == label:
            return ltr
    raise ValueError(f"label {label!r} not in option set")


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def ground_truth(condition, modality, explicit, implicit):
    if modality == "text_only":
        return "Neutral" if condition == "Neutral-Text" else explicit
    return implicit  # audio_only, text_audio


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
NEUTRAL_TEXT_DIVERGENT_ONLY = False  # set by --neutral-text-divergent


def condition_pool(df, condition):
    if condition == "Neutral-Text":
        pool = df[(df["Sarcasm"] == 0) & (df["Explicit_Emotion"] == "Neutral")]
        if NEUTRAL_TEXT_DIVERGENT_ONLY:
            pool = pool[pool["Implicit_Emotion"] != "Neutral"]
        return pool
    if condition == "Matched":
        return df[
            (df["Sarcasm"] == 0)
            & (df["Explicit_Emotion"] == df["Implicit_Emotion"])
        ]
    if condition == "Mismatched":
        return df[df["Sarcasm"] == 1]
    if condition == "Paralinguistic":
        return df
    raise ValueError(condition)


def sample_conditions(df, n, rng, warnings):
    """Sample n rows per condition, disjoint by KEY where the pool allows."""
    used_keys = set()
    selected = {}
    # Order matters: narrowest pool first so the tight conditions get first pick.
    order = sorted(CONDITIONS, key=lambda c: len(condition_pool(df, c)))
    for cond in order:
        pool = condition_pool(df, cond)
        avail = pool[~pool["KEY"].isin(used_keys)]
        if len(avail) >= n:
            picked = avail.sample(n=n, random_state=rng.randrange(2**31))
        else:
            warnings.append(
                f"[{cond}] pool exhausted: {len(avail)} unused rows available "
                f"for n={n}; allowing reuse of rows already assigned to another "
                f"condition (total pool = {len(pool)})."
            )
            extra = n - len(avail)
            reused = pool[pool["KEY"].isin(used_keys)]
            if len(avail) + len(reused) < n:
                warnings.append(
                    f"[{cond}] total pool ({len(pool)}) smaller than n={n}; "
                    f"emitting {len(pool)} items instead."
                )
                picked = pool
            else:
                picked = pd.concat(
                    [
                        avail,
                        reused.sample(n=extra, random_state=rng.randrange(2**31)),
                    ]
                )
        used_keys.update(picked["KEY"].tolist())
        selected[cond] = picked
    return {c: selected[c] for c in CONDITIONS}


# --------------------------------------------------------------------------
# Item generation
# --------------------------------------------------------------------------
def build_items(selected, rng):
    items = []
    for cond in CONDITIONS:
        rows = selected[cond]
        for i, (_, row) in enumerate(rows.iterrows()):
            explicit = row["Explicit_Emotion"]
            implicit = row["Implicit_Emotion"]
            transcript = str(row["SENTENCE"]).strip()
            key = row["KEY"]
            for modality in CONDITION_MODALITIES[cond]:
                gt = ground_truth(cond, modality, explicit, implicit)
                if gt not in EMOTION_LABELS:
                    raise ValueError(
                        f"ground truth {gt!r} for {key} not in label set"
                    )
                options = shuffled_options(rng)
                paraphrase = rng.choice(PARAPHRASES[modality])
                prompt = build_prompt(
                    modality,
                    transcript,
                    paraphrase,
                    options,
                    warn=(cond in ("Matched", "Mismatched")),
                )
                items.append(
                    {
                        "id": f"{CATEGORY}_{cond.lower().replace('-', '_')}_"
                        f"{modality}_{i:03d}",
                        "category": CATEGORY,
                        "condition": cond,
                        "modality": modality,
                        "source_key": key,
                        "source_scene": row["SCENE"],
                        "speaker": row["SPEAKER"],
                        "show": row["SHOW"],
                        "sarcasm": int(row["Sarcasm"]),
                        "transcript": transcript if modality != "audio_only" else "",
                        "audio_path": (
                            f"{AUDIO_DIR}/{key}.wav"
                            if modality != "text_only"
                            else ""
                        ),
                        "prompt": prompt,
                        "options": options,
                        "correct_letter": letter_for(options, gt),
                        "ground_truth_label": gt,
                        "explicit_emotion": explicit,
                        "implicit_emotion": implicit,
                    }
                )
    return items


# --------------------------------------------------------------------------
# Summary / integrity checks
# --------------------------------------------------------------------------
def summarize(items, selected, df, n, warnings):
    print("\n=== ITEM COUNTS ===")
    counts = Counter((it["condition"], it["modality"]) for it in items)
    print(f"{'condition':<16}{'text_only':>11}{'audio_only':>12}{'text_audio':>12}{'total':>8}")
    for cond in CONDITIONS:
        row = [counts.get((cond, m), 0) for m in ("text_only", "audio_only", "text_audio")]
        print(f"{cond:<16}{row[0]:>11}{row[1]:>12}{row[2]:>12}{sum(row):>8}")
    print(f"{'TOTAL':<16}{'':>11}{'':>12}{'':>12}{len(items):>8}")

    print("\n=== POOL SIZES ===")
    for cond in CONDITIONS:
        pool = condition_pool(df, cond)
        print(f"{cond:<16} pool={len(pool):<6} sampled={len(selected[cond]):<4} (target n={n})")

    print("\n=== LEAKAGE CHECK (SCENE overlap across conditions) ===")
    scenes = {c: set(selected[c]["SCENE"]) for c in CONDITIONS}
    clean = True
    for a in range(len(CONDITIONS)):
        for b in range(a + 1, len(CONDITIONS)):
            ca, cb = CONDITIONS[a], CONDITIONS[b]
            ov = scenes[ca] & scenes[cb]
            if ov:
                clean = False
                print(f"  OVERLAP {ca} x {cb}: {len(ov)} scene(s) -> {sorted(ov)[:5]}")
    print("  no SCENE appears in more than one condition." if clean
          else "  WARNING: scene-level overlap detected (see above).")

    keys = [k for c in CONDITIONS for k in selected[c]["KEY"]]
    dup = [k for k, v in Counter(keys).items() if v > 1]
    print(f"  KEY reuse across conditions: {len(dup)}" + (f" -> {dup[:5]}" if dup else ""))

    print("\n=== CORRECT-LETTER DISTRIBUTION (shuffle sanity) ===")
    lc = Counter(it["correct_letter"] for it in items)
    print("  " + "  ".join(f"{l}:{lc.get(l, 0)}" for l in LETTERS))

    print("\n=== PARAPHRASE COVERAGE ===")
    per_mod = defaultdict(set)
    for it in items:
        line = it["prompt"].split("\n\nOptions:")[0].splitlines()[-1]
        per_mod[it["modality"]].add(line.replace(MISMATCH_WARNING, "").strip())
    for m in ("text_only", "audio_only", "text_audio"):
        print(f"  {m:<12} {len(per_mod[m])}/{len(PARAPHRASES[m])} variants used")

    print("\n=== GROUND-TRUTH SPREAD ===")
    for cond in CONDITIONS:
        sub = [it for it in items if it["condition"] == cond]
        t = Counter(it["ground_truth_label"] for it in sub if it["modality"] == "text_only")
        a = Counter(it["ground_truth_label"] for it in sub if it["modality"] == "audio_only")
        print(f"  {cond}")
        if t:
            print(f"    text  GT: {dict(t)}")
        print(f"    audio GT: {dict(a)}")
        if cond in ("Neutral-Text", "Matched", "Mismatched"):
            div = sum(
                1 for _, r in selected[cond].iterrows()
                if (("Neutral" if cond == "Neutral-Text" else r["Explicit_Emotion"])
                    != r["Implicit_Emotion"])
            )
            print(f"    rows where text GT != audio GT: {div}/{len(selected[cond])}")

    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print("  " + w)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", default="raw/mustard_pp_utterances_only.csv")
    ap.add_argument("--n-per-condition", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sarcasm_pacing_examples.json")
    ap.add_argument(
        "--neutral-text-divergent",
        action="store_true",
        help="restrict the Neutral-Text pool to rows where Implicit_Emotion != "
             "Neutral, so text GT and audio GT actually differ (pool shrinks to 24)",
    )
    args = ap.parse_args()

    global NEUTRAL_TEXT_DIVERGENT_ONLY
    NEUTRAL_TEXT_DIVERGENT_ONLY = args.neutral_text_divergent

    rng = random.Random(args.seed)
    df = pd.read_csv(args.input_csv)

    required = {"SCENE", "KEY", "SENTENCE", "SPEAKER", "SHOW",
                "Sarcasm", "Implicit_Emotion", "Explicit_Emotion"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"missing columns in {args.input_csv}: {sorted(missing)}")

    df = df.dropna(subset=["SENTENCE", "Sarcasm",
                           "Implicit_Emotion", "Explicit_Emotion"]).copy()
    df["Sarcasm"] = df["Sarcasm"].astype(int)

    bad = set(df["Implicit_Emotion"]) | set(df["Explicit_Emotion"])
    bad -= set(EMOTION_LABELS)
    if bad:
        sys.exit(f"unexpected emotion labels in CSV: {sorted(bad)}")

    warnings = []
    selected = sample_conditions(df, args.n_per_condition, rng, warnings)
    items = build_items(selected, rng)

    with open(args.out, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    summarize(items, selected, df, args.n_per_condition, warnings)
    print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
