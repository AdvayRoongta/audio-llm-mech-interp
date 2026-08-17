#!/usr/bin/env python3
"""
Build an emotional-prosody conflict evaluation set from the RAVDESS filename
convention, structured after the LISTEN benchmark's condition x modality split.

No input files required: the RAVDESS naming scheme is fully deterministic, so the
theoretical file inventory is enumerated synthetically. Audio is wired in later
via `audio_path`.

Usage (run from repo root):
    python build/build_emotional_prosody.py --n-per-condition 20 --seed 0 \
        --out data/emotional_prosody_examples.json
"""

import argparse
import json
import random
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# RAVDESS convention
# ----------------------------------------------------------------------------

MODALITY_CODE = "03"        # audio-only
VOCAL_CHANNEL_CODE = "01"   # speech

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
EMOTION_LABELS = list(EMOTIONS.values())          # canonical MCQ option set (8)

INTENSITIES = ["01", "02"]                        # neutral only has "01"
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
REPETITIONS = ["01", "02"]
ACTORS = [f"{i:02d}" for i in range(1, 25)]       # odd = male, even = female

AUDIO_DIR = "dataset_audio/ravdess"

# Neutral is excluded as a *true* emotion by default: a neutral-voiced clip makes
# the Neutral-Text condition degenerate (text GT == audio GT), so there is no
# conflict left to measure. It stays in the MCQ option set and in the pool of
# false claims for the Mismatched condition. Override with --include-neutral.
DEFAULT_TRUE_EMOTIONS = [e for e in EMOTION_LABELS if e != "neutral"]

# The proposal's table shows the transcript only in the Neutral-Text cell. LISTEN's
# text-only modality presents the transcript in every condition, and the RAVDESS
# statements are lexically emotion-free either way. Set to False to match the
# proposal table exactly (claim-only prompts for Matched/Mismatched).
SHOW_TRANSCRIPT_WITH_CLAIM = True

# ----------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------

# Which modalities each condition is instantiated in. Mirrors the two GT columns
# of the proposal table ("Text-only GT" / "Audio-only|Text+Audio GT"): the three
# text-bearing conditions get a text-only control plus the text+audio trial, and
# the Paralinguistic condition is the audio-only baseline.
CONDITION_MODALITIES = {
    "Neutral-Text":   ["text_only", "text_audio"],
    "Matched":        ["text_only", "text_audio"],
    "Mismatched":     ["text_only", "text_audio"],
    "Paralinguistic": ["audio_only"],
}
CONDITION_ORDER = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]

CLAIM_CONDITIONS = {"Matched", "Mismatched"}
CLAIM_TEMPLATE = "The speaker's emotion is {emotion}."
CONFLICT_WARNING = "Beware that the audio description and the audio itself may not match."
ANSWER_INSTRUCTION = "Answer with the letter of the correct option."

# LISTEN Appendix A.2-style phrasing variants, sampled per item.
PARAPHRASES = {
    "text_only": [
        "Based on the transcript, how would you describe the emotional tone of the speaker?",
        "Reading the text above, what emotion is the speaker most likely expressing?",
        "From the text alone, classify the speaker's emotional state.",
        "Given the information above, which emotion best characterizes the speaker?",
        "Judging from the text, what is the speaker's emotional tone?",
        "Using only the text provided, identify the emotion of the speaker.",
    ],
    "audio_only": [
        "Listen to the audio and classify the emotion in the speaker's voice.",
        "What emotion is expressed in the speaker's voice?",
        "Based on the recording, how would you describe the speaker's emotional tone?",
        "Attend to the vocal delivery and identify the speaker's emotion.",
        "From the audio alone, which emotion best describes the speaker?",
        "Listen carefully and determine the emotional state of the speaker.",
    ],
    "text_audio": [
        "Listen to the audio and classify the emotion. How would you describe the emotional tone of the speaker?",
        "Considering both the text and the recording, which emotion is the speaker expressing?",
        "Listen to the audio and identify the speaker's emotion.",
        "Using the audio and the text above, classify the speaker's emotional state.",
        "After listening to the recording, how would you describe the speaker's emotional tone?",
        "Attend to the recording and determine which emotion best describes the speaker.",
    ],
}

# ----------------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------------


def build_inventory():
    """Enumerate the full theoretical RAVDESS speech inventory (no files needed)."""
    items = []
    for actor in ACTORS:
        gender = "male" if int(actor) % 2 == 1 else "female"
        for emo_code, emo_label in EMOTIONS.items():
            intensities = ["01"] if emo_label == "neutral" else INTENSITIES
            for intensity in intensities:
                for stmt_code, stmt_text in STATEMENTS.items():
                    for rep in REPETITIONS:
                        stem = "-".join([
                            MODALITY_CODE, VOCAL_CHANNEL_CODE, emo_code,
                            intensity, stmt_code, rep, actor,
                        ])
                        items.append({
                            "filename": f"{stem}.wav",
                            "actor_id": int(actor),
                            "actor_gender": gender,
                            "true_emotion": emo_label,
                            "intensity": "normal" if intensity == "01" else "strong",
                            "statement_code": stmt_code,
                            "transcript": stmt_text,
                            "repetition": int(rep),
                        })
    return items


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


def emotion_quotas(n, emotions, rng):
    """Split n as evenly as possible across emotions; remainder goes to a random subset."""
    base, extra = divmod(n, len(emotions))
    quotas = {e: base for e in emotions}
    for e in rng.sample(list(emotions), extra):
        quotas[e] += 1
    return quotas


def sample_condition(inventory, used_filenames, n, emotions, condition, rng, warnings):
    """Pick n unused clips for one condition, balanced across emotion and gender."""
    by_key = defaultdict(list)
    by_emotion = defaultdict(list)
    for item in inventory:
        if item["filename"] in used_filenames:
            continue
        by_key[(item["true_emotion"], item["actor_gender"])].append(item)
        by_emotion[item["true_emotion"]].append(item)

    for emo in emotions:
        if len(by_emotion[emo]) < 2:
            warnings.append(
                f"[{condition}] only {len(by_emotion[emo])} unused clip(s) available "
                f"for emotion '{emo}' — pool is too thin for balanced sampling."
            )

    quotas = emotion_quotas(n, emotions, rng)
    genders = ["male", "female"]
    gender_cursor = rng.randrange(2)  # alternates globally so totals stay even
    chosen = []

    for emo in sorted(emotions):
        for _ in range(quotas[emo]):
            want = genders[gender_cursor % 2]
            gender_cursor += 1
            pool = [i for i in by_key[(emo, want)] if i["filename"] not in used_filenames]
            if not pool:
                other = genders[(gender_cursor) % 2]
                pool = [i for i in by_key[(emo, other)] if i["filename"] not in used_filenames]
                if pool:
                    warnings.append(
                        f"[{condition}] no unused '{want}' clip for '{emo}'; "
                        f"substituted '{other}'."
                    )
            if not pool:
                warnings.append(
                    f"[{condition}] exhausted pool for emotion '{emo}' — "
                    f"produced fewer items than requested."
                )
                continue
            pick = rng.choice(pool)
            used_filenames.add(pick["filename"])
            chosen.append(pick)

    rng.shuffle(chosen)
    return chosen


# ----------------------------------------------------------------------------
# Item construction
# ----------------------------------------------------------------------------


def build_prompt(condition, modality, transcript, claimed_emotion, rng, order="claim_last"):
    """
    `order` only matters for text_audio items in a claim condition:
    "claim_last" (default) keeps the claim in the leading transcript block,
    right before the question -- today's behavior. "claim_first" instead
    appends the claim after the question/conflict-warning block, so the
    question is the thing immediately preceding it. text_only always uses
    the transcript-block placement regardless of `order`.
    """
    blocks = []
    text_block = []
    trailing_claim = None

    if modality in ("text_only", "text_audio"):
        show_transcript = (
            condition not in CLAIM_CONDITIONS or SHOW_TRANSCRIPT_WITH_CLAIM
        )
        if show_transcript:
            text_block.append(f'Transcript: "{transcript}."')
        if condition in CLAIM_CONDITIONS:
            claim_line = CLAIM_TEMPLATE.format(emotion=claimed_emotion)
            if modality == "text_audio" and order == "claim_first":
                trailing_claim = claim_line
            else:
                text_block.append(claim_line)
    if text_block:
        blocks.append("\n".join(text_block))

    question = rng.choice(PARAPHRASES[modality])
    if modality == "text_audio" and condition in CLAIM_CONDITIONS:
        question = f"{question}\n{CONFLICT_WARNING}"
    blocks.append(question)

    if trailing_claim is not None:
        blocks.append(trailing_claim)
    return blocks


def build_options(rng):
    labels = EMOTION_LABELS[:]
    rng.shuffle(labels)
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    return dict(zip(letters, labels))


def ground_truth_for(condition, modality, true_emotion, claimed_emotion):
    if modality == "text_only":
        if condition == "Neutral-Text":
            return "neutral"          # lexically neutral transcript, no audio
        return claimed_emotion        # the text channel asserts the label
    return true_emotion               # audio_only and text_audio follow the voice


def make_item(idx, condition, modality, clip, claimed_emotion, rng, order=None,
              options=None, correct_letter=None):
    """
    `order` is None for items with no competing claim (Neutral-Text,
    Paralinguistic) and for text_only claim-condition items, which have a
    fixed single layout and are stamped "claim_last" by the caller without
    being duplicated. text_audio claim-condition items are called twice by
    the caller, once per order value, and share one `options`/`correct_letter`
    pair (computed by the caller) so the MCQ lettering is identical across
    the claim_first/claim_last pair -- only the prompt text differs.
    """
    prompt_blocks = build_prompt(
        condition, modality, clip["transcript"], claimed_emotion, rng,
        order=order or "claim_last",
    )
    if options is None:
        options = build_options(rng)
    gt = ground_truth_for(condition, modality, clip["true_emotion"], claimed_emotion)
    if correct_letter is None:
        correct_letter = next(l for l, v in options.items() if v == gt)

    option_lines = "\n".join(f"{l}. {v}" for l, v in sorted(options.items()))
    prompt = "\n\n".join(prompt_blocks + [option_lines, ANSWER_INSTRUCTION])

    slug = condition.lower().replace("-", "_")
    if modality == "text_audio" and order is not None:
        item_id = f"ep_{slug}_{modality}_{order}_{idx:03d}"
    else:
        item_id = f"ep_{slug}_{modality}_{idx:03d}"
    return {
        "id": item_id,
        "category": "emotional_prosody",
        "condition": condition,
        "modality": modality,
        "order": order,
        "ravdess_filename": clip["filename"],
        "actor_id": clip["actor_id"],
        "actor_gender": clip["actor_gender"],
        "transcript": clip["transcript"],
        "audio_path": f"{AUDIO_DIR}/{clip['filename']}",
        "prompt": prompt,
        "options": options,
        "correct_letter": correct_letter,
        "ground_truth_label": gt,
        "true_emotion": clip["true_emotion"],
        "claimed_emotion": claimed_emotion,
        "intensity": clip["intensity"],
    }


def build_dataset(n_per_condition, seed, include_neutral):
    rng = random.Random(seed)
    inventory = build_inventory()
    true_emotions = (
        EMOTION_LABELS[:] if include_neutral else DEFAULT_TRUE_EMOTIONS[:]
    )
    pool = [i for i in inventory if i["true_emotion"] in true_emotions]

    used_filenames = set()
    warnings = []
    condition_files = {}
    items = []

    for condition in CONDITION_ORDER:
        clips = sample_condition(
            pool, used_filenames, n_per_condition, true_emotions,
            condition, rng, warnings,
        )
        condition_files[condition] = {c["filename"] for c in clips}

        # One claim per clip, shared across that clip's modalities so the
        # text-only control and the text+audio trial are properly paired.
        claims = {}
        for clip in clips:
            if condition == "Mismatched":
                alternatives = [e for e in EMOTION_LABELS if e != clip["true_emotion"]]
                claims[clip["filename"]] = rng.choice(alternatives)
            elif condition == "Matched":
                claims[clip["filename"]] = clip["true_emotion"]
            else:
                claims[clip["filename"]] = None

        for modality in CONDITION_MODALITIES[condition]:
            for idx, clip in enumerate(clips, start=1):
                claimed = claims[clip["filename"]]
                if modality == "text_audio" and condition in CLAIM_CONDITIONS:
                    # One options/correct_letter draw shared by both order
                    # variants, so the pair differs only in prompt text.
                    gt = ground_truth_for(condition, modality, clip["true_emotion"], claimed)
                    shared_options = build_options(rng)
                    shared_letter = next(l for l, v in shared_options.items() if v == gt)
                    for order in ("claim_last", "claim_first"):
                        items.append(
                            make_item(idx, condition, modality, clip, claimed, rng,
                                      order=order, options=shared_options,
                                      correct_letter=shared_letter)
                        )
                elif modality == "text_only" and condition in CLAIM_CONDITIONS:
                    items.append(
                        make_item(idx, condition, modality, clip, claimed, rng,
                                  order="claim_last")
                    )
                else:
                    items.append(
                        make_item(idx, condition, modality, clip, claimed, rng)
                    )

    return items, inventory, condition_files, warnings


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------


def print_summary(items, inventory, condition_files, warnings):
    line = "-" * 68
    print(line)
    print(f"Theoretical RAVDESS inventory: {len(inventory)} filenames")
    print(f"Generated items:               {len(items)}")
    print(line)

    print("\nItems per condition x modality")
    grid = Counter((i["condition"], i["modality"]) for i in items)
    for condition in CONDITION_ORDER:
        for modality in CONDITION_MODALITIES[condition]:
            print(f"  {condition:<16} {modality:<12} {grid[(condition, modality)]:>4}")

    print("\nOrder breakdown (Matched/Mismatched, text_audio duplicated; text_only stamped claim_last)")
    order_grid = Counter(
        (i["condition"], i["modality"], i["order"])
        for i in items if i["order"] is not None
    )
    for condition in ["Matched", "Mismatched"]:
        for modality in CONDITION_MODALITIES[condition]:
            for order in ("claim_last", "claim_first"):
                n = order_grid.get((condition, modality, order), 0)
                if n:
                    print(f"  {condition:<16} {modality:<12} {order:<12} {n:>4}")

    print("\n[CHECK] claim_last/claim_first pairing (Matched/Mismatched text_audio)")
    pairing_ok = True
    for condition in ["Matched", "Mismatched"]:
        last_files = {
            i["ravdess_filename"] for i in items
            if i["condition"] == condition and i["modality"] == "text_audio"
            and i["order"] == "claim_last"
        }
        first_files = {
            i["ravdess_filename"] for i in items
            if i["condition"] == condition and i["modality"] == "text_audio"
            and i["order"] == "claim_first"
        }
        if last_files != first_files:
            pairing_ok = False
            print(f"  FAIL {condition}: claim_last/claim_first filename sets differ "
                  f"({len(last_files)} vs {len(first_files)})")
        else:
            print(f"  OK   {condition}: {len(last_files)} clips, fully paired 1:1")
    if not pairing_ok:
        print("  FAIL -- order pairing broken, investigate before using this dataset")

    print("\nGround-truth label distribution (per condition)")
    for condition in CONDITION_ORDER:
        labels = Counter(
            i["ground_truth_label"] for i in items if i["condition"] == condition
        )
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        print(f"  {condition:<16} {rendered}")

    print("\nTrue-emotion distribution (per condition, unique clips)")
    for condition in CONDITION_ORDER:
        seen, counts = set(), Counter()
        for i in items:
            if i["condition"] == condition and i["ravdess_filename"] not in seen:
                seen.add(i["ravdess_filename"])
                counts[i["true_emotion"]] += 1
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {condition:<16} {rendered}")

    print("\nClaimed-emotion distribution (Mismatched)")
    claims = Counter(
        i["claimed_emotion"] for i in items if i["condition"] == "Mismatched"
    )
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(claims.items())))

    print("\nGender balance (unique clips per condition)")
    for condition in CONDITION_ORDER:
        seen, counts = set(), Counter()
        for i in items:
            if i["condition"] == condition and i["ravdess_filename"] not in seen:
                seen.add(i["ravdess_filename"])
                counts[i["actor_gender"]] += 1
        total = sum(counts.values())
        print(f"  {condition:<16} male={counts['male']}, female={counts['female']} "
              f"(n={total})")

    print("\nActor coverage")
    actors = sorted({i["actor_id"] for i in items})
    print(f"  {len(actors)} / 24 actors used")

    print("\nFilename disjointness across conditions")
    overlap_found = False
    for a in range(len(CONDITION_ORDER)):
        for b in range(a + 1, len(CONDITION_ORDER)):
            ca, cb = CONDITION_ORDER[a], CONDITION_ORDER[b]
            shared = condition_files[ca] & condition_files[cb]
            if shared:
                overlap_found = True
                print(f"  OVERLAP {ca} / {cb}: {len(shared)} shared filename(s)")
    if not overlap_found:
        print("  OK — no filename appears in more than one condition")

    total_unique = len({i["ravdess_filename"] for i in items})
    expected = sum(len(v) for v in condition_files.values())
    print(f"  {total_unique} unique filenames across {expected} sampled clips")

    print("\nCorrect-letter distribution (shuffle sanity check)")
    letters = Counter(i["correct_letter"] for i in items)
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(letters.items())))

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nWARNINGS\n  none")
    print(line)


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Generate an emotional-prosody conflict eval set from the "
                    "RAVDESS filename convention (no source files required)."
    )
    ap.add_argument("--n-per-condition", type=int, default=133,
                    help="unique clips sampled per condition (default: 133)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument("--out", default="data/emotional_prosody_examples.json",
                    help="output JSON path")
    ap.add_argument("--include-neutral", action="store_true",
                    help="allow neutral as a true emotion (off by default)")
    args = ap.parse_args()

    items, inventory, condition_files, warnings = build_dataset(
        args.n_per_condition, args.seed, args.include_neutral
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print_summary(items, inventory, condition_files, warnings)
    print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
