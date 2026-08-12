#!/usr/bin/env python3
"""
Build a speaker-identity (gender) conflict evaluation set from the RAVDESS
filename convention, structured after the LISTEN benchmark's condition x modality
split.

No input files required: the RAVDESS naming scheme is deterministic, so the
theoretical file inventory is enumerated synthetically. Audio is wired in later
via `audio_path`.

Confound control
----------------
Speaker gender is the target variable here, so it must not covary with anything
else. Sampling is therefore done in MATCHED PAIRS: each sampled cell fixes
(emotion, intensity, statement, repetition) and draws one male actor and one
female actor from it. Gender is then orthogonal to emotion, statement, intensity
and repetition BY CONSTRUCTION, not by luck of the draw. This is verified and
printed in the summary.

Usage (run from repo root):
    python build/build_speaker_identity.py --n-per-condition 20 --seed 0 \
        --out data/speaker_identity_examples.json
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
EMOTION_LABELS = list(EMOTIONS.values())

INTENSITIES = ["01", "02"]                    # neutral only has "01"
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
REPETITIONS = ["01", "02"]
ACTORS = [f"{i:02d}" for i in range(1, 25)]   # odd = male, even = female

GENDERS = ["male", "female"]
OPPOSITE = {"male": "female", "female": "male"}

AUDIO_DIR = "dataset_audio/ravdess"

# The proposal's table shows the transcript only in the Neutral-Text cell.
# LISTEN's text-bearing modalities present it in every condition, and neither
# RAVDESS statement contains any gendered lexical cue. Set False to match the
# proposal table literally (claim-only prompts for Matched/Mismatched).
SHOW_TRANSCRIPT_WITH_CLAIM = True

# ----------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------

# Neutral-Text has no text-only cell: the transcript carries zero gender
# information, so a text-only trial has no determinable answer, and the option
# set here is strictly 2-way (male/female) with no "cannot determine" choice.
# Adding one would change the guess baseline from 50% to 33% and break
# comparability with every other cell. The condition therefore exists only as
# text+audio, testing whether gender-free lexical context perturbs the audio read.
CONDITION_MODALITIES = {
    "Neutral-Text":   ["text_audio"],
    "Matched":        ["text_only", "text_audio"],
    "Mismatched":     ["text_only", "text_audio"],
    "Paralinguistic": ["audio_only"],
}
CONDITION_ORDER = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]

CLAIM_CONDITIONS = {"Matched", "Mismatched"}
CLAIM_TEMPLATE = "The speaker is an adult {gender}."
CONFLICT_WARNING = "Beware that the audio description and the audio itself may not match."
ANSWER_INSTRUCTION = "Answer with the letter of the correct option."

# LISTEN Appendix A.2-style phrasing variants, sampled per item.
PARAPHRASES = {
    "text_only": [
        "Based on the information above, what is the speaker's gender identity?",
        "From the text alone, classify the gender identity of the speaker.",
        "Reading the text above, is the speaker male or female?",
        "Given the information provided, which gender identity best describes the speaker?",
        "Using only the text, identify the speaker's gender identity.",
        "Judging from the text, what is the gender identity of the speaker?",
    ],
    "audio_only": [
        "Listen to the audio and classify the speaker's gender identity.",
        "What is the gender identity of the speaker in the recording?",
        "Based on the voice in the recording, is the speaker male or female?",
        "Attend to the vocal characteristics and identify the speaker's gender identity.",
        "From the audio alone, which gender identity best describes the speaker?",
        "Listen carefully and determine the gender identity of the speaker.",
    ],
    "text_audio": [
        "Listen to the audio and classify the gender identity. What is the speaker's gender identity?",
        "Considering both the text and the recording, is the speaker male or female?",
        "Listen to the audio and identify the speaker's gender identity.",
        "Using the audio and the text above, classify the speaker's gender identity.",
        "After listening to the recording, what is the gender identity of the speaker?",
        "Attend to the recording and determine which gender identity describes the speaker.",
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
                            "emotion": emo_label,
                            "intensity": "normal" if intensity == "01" else "strong",
                            "intensity_code": intensity,
                            "statement_code": stmt_code,
                            "transcript": stmt_text,
                            "repetition": int(rep),
                        })
    return items


def index_inventory(inventory):
    """(emotion, intensity_code, statement_code, repetition, gender) -> [items]"""
    idx = defaultdict(list)
    for item in inventory:
        key = (item["emotion"], item["intensity_code"], item["statement_code"],
               item["repetition"], item["actor_gender"])
        idx[key].append(item)
    return idx


# ----------------------------------------------------------------------------
# Paired sampling (the confound fix)
# ----------------------------------------------------------------------------


def emotion_quotas(n_cells, emotions, rng):
    """Split n_cells as evenly as possible across emotions."""
    base, extra = divmod(n_cells, len(emotions))
    quotas = {e: base for e in emotions}
    for e in rng.sample(list(emotions), extra):
        quotas[e] += 1
    return quotas


def cell_variants(emotion, rng):
    """All (intensity, statement, repetition) combos for an emotion, shuffled."""
    intensities = ["01"] if emotion == "neutral" else INTENSITIES
    combos = [
        (i, s, r)
        for i in intensities
        for s in STATEMENTS
        for r in [1, 2]
    ]
    rng.shuffle(combos)
    return combos


def pick_least_used(pool, actor_usage, rng):
    """Draw from pool, preferring actors seen least often so far."""
    lowest = min(actor_usage[i["actor_id"]] for i in pool)
    candidates = [i for i in pool if actor_usage[i["actor_id"]] == lowest]
    return rng.choice(candidates)


def sample_condition(idx, used_filenames, actor_usage, n, condition, rng, warnings):
    """
    Draw n clips for one condition as n//2 gender-matched pairs.

    Each pair shares emotion, intensity, statement and repetition, differing only
    in actor gender. Gender is therefore statistically independent of every other
    sampled variable within the condition. Actors are drawn least-used-first so no
    single voice dominates its gender class.
    """
    if n % 2 != 0:
        warnings.append(
            f"[{condition}] --n-per-condition={n} is odd; matched-pair sampling "
            f"requires an even n. Using {n - 1} to preserve gender orthogonality."
        )
        n = n - 1

    n_cells = n // 2
    quotas = emotion_quotas(n_cells, EMOTION_LABELS, rng)
    chosen = []
    statement_cursor = rng.randrange(2)  # keeps statements balanced across cells

    for emotion in sorted(EMOTION_LABELS):
        drawn = 0
        combos = cell_variants(emotion, rng)
        # prefer combos whose statement matches the alternating cursor
        for _ in range(quotas[emotion]):
            want_stmt = ["01", "02"][statement_cursor % 2]
            statement_cursor += 1
            ordered = (
                [c for c in combos if c[1] == want_stmt]
                + [c for c in combos if c[1] != want_stmt]
            )
            placed = False
            for intensity, stmt, rep in ordered:
                male_pool = [
                    i for i in idx[(emotion, intensity, stmt, rep, "male")]
                    if i["filename"] not in used_filenames
                ]
                female_pool = [
                    i for i in idx[(emotion, intensity, stmt, rep, "female")]
                    if i["filename"] not in used_filenames
                ]
                if male_pool and female_pool:
                    m = pick_least_used(male_pool, actor_usage, rng)
                    f = pick_least_used(female_pool, actor_usage, rng)
                    used_filenames.add(m["filename"])
                    used_filenames.add(f["filename"])
                    actor_usage[m["actor_id"]] += 1
                    actor_usage[f["actor_id"]] += 1
                    chosen.extend([m, f])
                    drawn += 2
                    placed = True
                    break
            if not placed:
                warnings.append(
                    f"[{condition}] no unused male/female pair left for emotion "
                    f"'{emotion}' — condition is short by 2 items."
                )
        if quotas[emotion] and drawn < 2:
            warnings.append(
                f"[{condition}] emotion '{emotion}' contributed only {drawn} "
                f"item(s) — pool is too thin for balanced sampling."
            )

    rng.shuffle(chosen)
    return chosen


# ----------------------------------------------------------------------------
# Item construction
# ----------------------------------------------------------------------------


def build_prompt(condition, modality, transcript, claimed_gender, rng):
    blocks = []
    text_block = []

    if modality in ("text_only", "text_audio"):
        show_transcript = (
            condition not in CLAIM_CONDITIONS or SHOW_TRANSCRIPT_WITH_CLAIM
        )
        if show_transcript:
            text_block.append(f'Transcript: "{transcript}."')
        if condition in CLAIM_CONDITIONS:
            text_block.append(CLAIM_TEMPLATE.format(gender=claimed_gender))
    if text_block:
        blocks.append("\n".join(text_block))

    question = rng.choice(PARAPHRASES[modality])
    if modality == "text_audio" and condition in CLAIM_CONDITIONS:
        question = f"{question}\n{CONFLICT_WARNING}"
    blocks.append(question)
    return blocks


def build_options(rng):
    labels = GENDERS[:]
    rng.shuffle(labels)
    return {"A": labels[0], "B": labels[1]}


def ground_truth_for(condition, modality, true_gender, claimed_gender):
    if modality == "text_only":
        return claimed_gender     # only claim conditions have a text-only cell
    return true_gender            # audio_only and text_audio follow the voice


def make_item(idx_num, condition, modality, clip, claimed_gender, rng):
    prompt_blocks = build_prompt(
        condition, modality, clip["transcript"], claimed_gender, rng
    )
    options = build_options(rng)
    gt = ground_truth_for(condition, modality, clip["actor_gender"], claimed_gender)
    correct_letter = next(l for l, v in options.items() if v == gt)

    option_lines = "\n".join(f"{l}. {v}" for l, v in sorted(options.items()))
    prompt = "\n\n".join(prompt_blocks + [option_lines, ANSWER_INSTRUCTION])

    slug = condition.lower().replace("-", "_")
    return {
        "id": f"si_{slug}_{modality}_{idx_num:03d}",
        "category": "speaker_identity",
        "condition": condition,
        "modality": modality,
        "ravdess_filename": clip["filename"],
        "actor_id": clip["actor_id"],
        "actor_gender": clip["actor_gender"],
        "transcript": clip["transcript"],
        "audio_path": f"{AUDIO_DIR}/{clip['filename']}",
        "prompt": prompt,
        "options": options,
        "correct_letter": correct_letter,
        "ground_truth_label": gt,
        "true_gender": clip["actor_gender"],
        "claimed_gender": claimed_gender,
        "emotion": clip["emotion"],
        "intensity": clip["intensity"],
    }


def build_dataset(n_per_condition, seed):
    rng = random.Random(seed)
    inventory = build_inventory()
    idx = index_inventory(inventory)

    used_filenames = set()
    actor_usage = Counter({int(a): 0 for a in ACTORS})
    warnings = []
    condition_files = {}
    items = []

    for condition in CONDITION_ORDER:
        clips = sample_condition(
            idx, used_filenames, actor_usage, n_per_condition,
            condition, rng, warnings
        )
        condition_files[condition] = {c["filename"] for c in clips}

        claims = {}
        for clip in clips:
            if condition == "Mismatched":
                claims[clip["filename"]] = OPPOSITE[clip["actor_gender"]]
            elif condition == "Matched":
                claims[clip["filename"]] = clip["actor_gender"]
            else:
                claims[clip["filename"]] = None

        for modality in CONDITION_MODALITIES[condition]:
            for i, clip in enumerate(clips, start=1):
                items.append(
                    make_item(i, condition, modality, clip,
                              claims[clip["filename"]], rng)
                )

    return items, inventory, condition_files, warnings


# ----------------------------------------------------------------------------
# Summary + balance verification
# ----------------------------------------------------------------------------


def unique_clips(items, condition=None):
    seen, out = set(), []
    for i in items:
        if condition and i["condition"] != condition:
            continue
        if i["ravdess_filename"] in seen:
            continue
        seen.add(i["ravdess_filename"])
        out.append(i)
    return out


def print_summary(items, inventory, condition_files, warnings):
    line = "-" * 72
    failures = []

    print(line)
    print(f"Theoretical RAVDESS inventory: {len(inventory)} filenames")
    print(f"Generated items:               {len(items)}")
    print(line)

    print("\nItems per condition x modality")
    grid = Counter((i["condition"], i["modality"]) for i in items)
    for condition in CONDITION_ORDER:
        for modality in CONDITION_MODALITIES[condition]:
            print(f"  {condition:<16} {modality:<12} {grid[(condition, modality)]:>4}")

    print("\nGround-truth label distribution (per condition)")
    for condition in CONDITION_ORDER:
        labels = Counter(
            i["ground_truth_label"] for i in items if i["condition"] == condition
        )
        print(f"  {condition:<16} " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())))

    print("\nClaimed-gender distribution (claim conditions, unique clips)")
    for condition in ["Matched", "Mismatched"]:
        counts = Counter(c["claimed_gender"] for c in unique_clips(items, condition))
        print(f"  {condition:<16} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # --- CHECK 1: equal male/female within every condition ------------------
    print("\n[CHECK 1] Gender balance within each condition (unique clips)")
    for condition in CONDITION_ORDER:
        counts = Counter(c["true_gender"] for c in unique_clips(items, condition))
        ok = counts["male"] == counts["female"]
        status = "OK " if ok else "FAIL"
        if not ok:
            failures.append(f"gender imbalance in {condition}: {dict(counts)}")
        print(f"  {status} {condition:<16} male={counts['male']}, female={counts['female']}")

    # --- CHECK 2: gender independent of emotion -----------------------------
    print("\n[CHECK 2] Gender x emotion cross-tab (all conditions, unique clips)")
    tab = defaultdict(Counter)
    for c in unique_clips(items):
        tab[c["emotion"]][c["true_gender"]] += 1
    print(f"  {'emotion':<12} {'male':>5} {'female':>7}   balanced?")
    for emotion in sorted(tab):
        m, f = tab[emotion]["male"], tab[emotion]["female"]
        ok = m == f
        if not ok:
            failures.append(f"gender x emotion imbalance for '{emotion}': m={m}, f={f}")
        print(f"  {emotion:<12} {m:>5} {f:>7}   {'OK' if ok else 'FAIL'}")

    # --- CHECK 3: gender independent of statement ---------------------------
    print("\n[CHECK 3] Gender x statement cross-tab (all conditions, unique clips)")
    tab = defaultdict(Counter)
    for c in unique_clips(items):
        tab[c["transcript"]][c["true_gender"]] += 1
    for stmt in sorted(tab):
        m, f = tab[stmt]["male"], tab[stmt]["female"]
        ok = m == f
        if not ok:
            failures.append(f"gender x statement imbalance for '{stmt}': m={m}, f={f}")
        print(f"  {'OK ' if ok else 'FAIL'} \"{stmt}\"  male={m}, female={f}")

    # --- CHECK 4: gender independent of intensity ---------------------------
    print("\n[CHECK 4] Gender x intensity cross-tab (all conditions, unique clips)")
    tab = defaultdict(Counter)
    for c in unique_clips(items):
        tab[c["intensity"]][c["true_gender"]] += 1
    for intensity in sorted(tab):
        m, f = tab[intensity]["male"], tab[intensity]["female"]
        ok = m == f
        if not ok:
            failures.append(f"gender x intensity imbalance for '{intensity}': m={m}, f={f}")
        print(f"  {'OK ' if ok else 'FAIL'} {intensity:<8} male={m}, female={f}")

    # --- CHECK 5: gender x emotion within each condition --------------------
    print("\n[CHECK 5] Gender x emotion within each condition")
    for condition in CONDITION_ORDER:
        tab = defaultdict(Counter)
        for c in unique_clips(items, condition):
            tab[c["emotion"]][c["true_gender"]] += 1
        bad = [e for e in tab if tab[e]["male"] != tab[e]["female"]]
        if bad:
            failures.append(f"gender x emotion imbalance inside {condition}: {bad}")
            print(f"  FAIL {condition:<16} unbalanced emotions: {bad}")
        else:
            print(f"  OK  {condition:<16} {len(tab)} emotion cell(s), "
                  f"all gender-paired")

    # --- CHECK 6: actor coverage --------------------------------------------
    print("\n[CHECK 6] Actor coverage")
    actors = Counter(c["actor_id"] for c in unique_clips(items))
    male_actors = {a for a in actors if a % 2 == 1}
    female_actors = {a for a in actors if a % 2 == 0}
    print(f"  {len(actors)} / 24 actors used "
          f"({len(male_actors)} male, {len(female_actors)} female)")
    print(f"  clips per actor: min={min(actors.values())}, max={max(actors.values())}")

    # --- CHECK 7: no filename leakage across conditions ---------------------
    print("\n[CHECK 7] Filename disjointness across conditions")
    overlap = False
    for a in range(len(CONDITION_ORDER)):
        for b in range(a + 1, len(CONDITION_ORDER)):
            ca, cb = CONDITION_ORDER[a], CONDITION_ORDER[b]
            shared = condition_files[ca] & condition_files[cb]
            if shared:
                overlap = True
                failures.append(f"filename leakage {ca}/{cb}: {len(shared)} shared")
                print(f"  FAIL {ca} / {cb}: {len(shared)} shared filename(s)")
    if not overlap:
        print("  OK  no filename appears in more than one condition")
    total_unique = len({i["ravdess_filename"] for i in items})
    sampled = sum(len(v) for v in condition_files.values())
    print(f"  {total_unique} unique filenames across {sampled} sampled clips")

    # --- CHECK 8: option-order shuffle --------------------------------------
    print("\n[CHECK 8] Correct-letter distribution (2-way shuffle sanity)")
    letters = Counter(i["correct_letter"] for i in items)
    total = sum(letters.values())
    print("  " + ", ".join(f"{k}={v} ({v / total:.0%})" for k, v in sorted(letters.items())))

    print("\nWARNINGS")
    if warnings:
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("  none")

    print("\nBALANCE VERDICT")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("  PASS — gender is orthogonal to condition, emotion, statement "
              "and intensity")
    print(line)


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Generate a speaker-identity conflict eval set from the "
                    "RAVDESS filename convention (no source files required)."
    )
    ap.add_argument("--n-per-condition", type=int, default=20,
                    help="unique clips per condition; must be even (default: 20)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument("--out", default="data/speaker_identity_examples.json",
                    help="output JSON path")
    args = ap.parse_args()

    items, inventory, condition_files, warnings = build_dataset(
        args.n_per_condition, args.seed
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print_summary(items, inventory, condition_files, warnings)
    print(f"\nWrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
