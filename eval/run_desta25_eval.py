#!/usr/bin/env python3
"""
DeSTA2.5-Audio (DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B) inference for the
audio-text dominance benchmark.

Mirrors run_qwen2audio_eval.py's structure: same CLI shape, same
checkpoint/resume pattern, same scoring (score_item/compute_tdr/
condition_breakdown, incl. the by_order claim_first/claim_last breakdown),
same audit_dataset() pre-flight, same per-item result record shape
(including the "order" field), same parse_answer_letter (with the
isupper() fix already included from the start -- see below). Only the
model-loading/message API and text_only handling differ.

This is a from-scratch rewrite, NOT a resume of
archive/old_stuff/desta25-listen-eval/test_desta25.py. That script used a
different Runner-class pattern and scored free-text generation against raw
ground_truth rather than MCQ letters -- that entire approach, and any of its
prior results, are fully superseded and are not referenced here.

--------------------------------------------------------------------------- #
Setup
--------------------------------------------------------------------------- #

1. Clone and install the desta package (it's a real pip package, not a
   sys.path hack -- unlike SALMONN, nothing needs to stay on disk at
   runtime beyond the installed package and the HF-downloaded weights):

     git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
     cd DeSTA2.5-Audio
     pip install -e .

2. Model weights download automatically on first run via
   DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
   (HF hub download, cached under $HF_HOME). The underlying LLM is
   DeSTA-ntu/Llama-3.1-8B-Instruct (a DeSTA-ntu redistribution, not
   meta-llama/... directly) -- this MAY require HF_TOKEN / license
   acceptance depending on how that repo is gated; set HF_TOKEN if the
   download 401s.

3. Environment: setup.py only pins transformers>=4.49.0 and uses stock
   AutoModelForCausalLM/WhisperForConditionalGeneration/AutoProcessor (no
   vendored/copied HF internals like SALMONN). This should work fine in the
   same environment already used for run_qwen2audio_eval.py -- no isolated
   venv required here.

4. First call to model.generate() lazily runs
   torch.hub.load('snakers4/silero-vad', model='silero_vad') internally
   (used to VAD-gate ASR transcription of audio-bearing items). This needs
   network/torch-hub access at least once -- it happens unconditionally on
   the first generate() call, even for a text_only-only run, before the
   audio/no-audio branch is decided.

Usage:
  py run_desta25_eval.py --dry-run                 # validate dataset, no model load
  py run_desta25_eval.py --experiments all --limit 5

--------------------------------------------------------------------------- #
Precision: bfloat16, not float16 -- deliberate, not an oversight
--------------------------------------------------------------------------- #

DeSTA25AudioModel.__init__ (desta/models/modeling_desta25.py) hard-codes:

    self.llm_model = AutoModelForCausalLM.from_pretrained(
        self.config.llm_model_id, torch_dtype=torch.bfloat16, ...
    )

This is not exposed as a kwarg anywhere (their own from_pretrained
classmethod only forwards cache_dir/token, never dtype), and there is NO
quantization option anywhere in that file -- no load_in_8bit/load_in_4bit/
bitsandbytes reference at all, so there's nothing to disable on that front.

This project's "fp16 only" rule is about blocking quantization (4-bit/8-bit
precision loss), and bfloat16 is a full 16-bit non-quantized format, so it's
accepted here as compliant with the intent of that rule even though it's not
literally torch.float16. We deliberately do NOT force a post-load
.to(torch.float16) conversion:
  - bf16 -> fp16 downcast risks overflow on an 8B model's outlier
    activations (bf16's exponent range matches fp32; fp16's is much
    narrower), which could silently corrupt outputs.
  - It would run the model outside the precision regime its authors
    actually trained and evaluated it in -- a bigger deviation from "use
    the model correctly" than accepting its native dtype.

A runtime assertion (in load_model()) confirms the loaded LLM actually is
bfloat16 -- not accidentally fp32, not silently something else -- so this
stays enforced, not just documented.

--------------------------------------------------------------------------- #
text_only handling: genuine no-audio generation, NOT a silence placeholder
--------------------------------------------------------------------------- #

Unlike SALMONN, DeSTA2.5's generate() has a real text-only code path.
Internally it branches on whether any message's "audios" list is non-empty:
if none are provided, it skips the Whisper encoder and VAD entirely and
runs plain self.llm_model.generate() on the tokenized chat text ("if no
audios are provided, it's identical to the original LLM generation" -- their
own comment). So for modality == "text_only" here, messages are built
WITHOUT "<|AUDIO|>" in the content and WITHOUT an "audios" key at all --
this is architecturally the same kind of genuinely-audio-free inference as
Qwen2-Audio's text_only, not SALMONN's silence-WAV workaround. (An internal
assertion requires len(audios) == content.count("<|AUDIO|>"), so omitting
both together is required and correct.)

Hard constraints (same spirit as the other two scripts):
  * No quantization, ever -- see the bf16 section above for why bf16 is
    accepted as this model's compliant native precision.
  * No hardcoded MCQ option count. Letter sets are derived per item.
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CATEGORIES = ["emotional_prosody", "speaker_identity", "sarcasm_pacing"]
CONDITIONS = ["Neutral-Text", "Matched", "Mismatched", "Paralinguistic"]
MODALITIES = ["text_only", "audio_only", "text_audio"]

LETTERS = list(string.ascii_uppercase)

AUDIO_LOCATOR = "<|AUDIO|>"
SYSTEM_PROMPT = "Focus on the audio clips and instructions."

# Per-category field names are NOT unified in the dataset. Candidates are tried
# in order and the resolved key is reported by audit_dataset() so it can be
# verified against real items before any GPU time is spent.
TRUE_KEY_CANDIDATES = {
    "speaker_identity": ["true_gender", "true_label", "gender"],
    "emotional_prosody": ["true_emotion", "true_label", "emotion"],
    "sarcasm_pacing": ["true_emotion", "true_label", "implicit_emotion", "Implicit_Emotion"],
}
CLAIMED_KEY_CANDIDATES = {
    "speaker_identity": ["claimed_gender", "claimed_label", "text_claim"],
    "emotional_prosody": ["claimed_emotion", "claimed_label", "text_claim"],
    "sarcasm_pacing": ["claimed_emotion", "claimed_label", "explicit_emotion", "Explicit_Emotion"],
}

# --------------------------------------------------------------------------- #
# Small helpers (shared with run_qwen2audio_eval.py's conventions)
# --------------------------------------------------------------------------- #


def norm(text):
    """Lowercase, strip punctuation, collapse whitespace. For value matching."""
    if text is None:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def first_present_key(item, candidates):
    for key in candidates:
        if key in item and item[key] is not None:
            return key
    return None


def item_uid(item, index):
    """Stable identifier used for checkpoint/resume keying."""
    for key in ("id", "item_id", "uid"):
        if item.get(key) is not None:
            return str(item[key])
    return "{}:{}".format(item.get("category", "unknown"), index)


def resolve_audio_path(raw_path, input_path):
    """Try the path as given, then relative to the dataset file, then cwd."""
    if not raw_path:
        return None
    candidates = [raw_path]
    if not os.path.isabs(raw_path):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(input_path)), raw_path))
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_path))), raw_path))
        candidates.append(os.path.join(os.getcwd(), raw_path))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Option handling -- fully dynamic, never assumes 2-way
# --------------------------------------------------------------------------- #


def build_option_map(item):
    """
    Return OrderedDict of letter -> option text for this item.

    Accepts `options` as either a dict {"A": "male", ...} or a list. List entries
    may or may not already carry their letter prefix ("A. male" / "A) male" / "male").
    Option count comes from the item, never from a constant.
    """
    options = item.get("options")
    out = OrderedDict()

    if isinstance(options, dict):
        for key in sorted(options.keys(), key=lambda k: str(k).upper()):
            out[str(key).strip().upper()] = str(options[key]).strip()
        return out

    if isinstance(options, list):
        for i, opt in enumerate(options):
            text = str(opt).strip()
            m = re.match(r"^\(?([A-Za-z])\)?\s*[\.\)\:\-]\s*(.+)$", text)
            if m:
                out[m.group(1).upper()] = m.group(2).strip()
            else:
                if i >= len(LETTERS):
                    raise ValueError("more options than letters available")
                out[LETTERS[i]] = text
        return out

    raise ValueError("item has no usable `options` field")


def resolve_value_to_letter(option_map, value):
    """Map a raw label value (e.g. 'angry', 'male') onto its option letter."""
    if value is None:
        return None
    v = norm(value)
    if not v:
        return None
    for letter, text in option_map.items():
        if norm(text) == v:
            return letter
    # tolerant fallback: unique substring containment
    hits = [l for l, t in option_map.items() if v and (v in norm(t) or norm(t) in v)]
    if len(hits) == 1:
        return hits[0]
    return None


def parse_answer_letter(raw_output, option_map):
    """
    Extract the answer letter from raw model text. Returns None if unparseable.
    Valid letters are exactly the keys of this item's option map.
    """
    if raw_output is None:
        return None
    valid = set(option_map.keys())
    text = str(raw_output).strip()
    if not text:
        return None

    # 1. whole output is just a letter, optionally wrapped/punctuated
    m = re.fullmatch(r"\(?\[?([A-Za-z])\]?\)?[\.\)\:\,]?", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 2. explicit "answer is X" / "answer: X" / "option (X)"
    m = re.search(r"(?:answer|option|choice|response)\s*(?:is|:|=)?\s*\(?\[?([A-Za-z])\]?\)?\b", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3. leading letter followed by a delimiter, e.g. "C. angry"
    m = re.match(r"^\s*\(?\[?([A-Za-z])\]?\)?\s*[\.\)\:\-]\s+", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 4. first standalone token that is a valid letter. Uppercase only:
    # re.findall splits contractions on the apostrophe ("I'm" -> "I","m";
    # "I'd" -> "I","d"), and lowercase remnants like that (or the article
    # "a") can coincidentally match a valid option letter before the real
    # answer is reached. Answer letters are written capitalized in practice;
    # a lowercase match here is virtually always a false positive.
    for tok in re.findall(r"\b([A-Za-z])\b", text):
        if tok.isupper() and tok in valid:
            return tok

    # 5. last resort: the output names exactly one option's text
    hits = [l for l, t in option_map.items() if norm(t) and norm(t) in norm(text)]
    if len(hits) == 1:
        return hits[0]

    return None


def _ensure_terse_instruction(prompt):
    """
    Append an explicit terse-answer instruction if the prompt doesn't already
    carry equivalent text. In practice every build script already appends
    "Answer with the letter of the correct option." to every prompt, so this
    is normally a no-op -- it's defensive, not load-bearing.
    """
    if re.search(r"answer with the letter", prompt, re.I):
        return prompt
    return prompt.rstrip() + "\nAnswer with the letter of the correct option only."


# --------------------------------------------------------------------------- #
# Dataset loading + pre-flight audit
# --------------------------------------------------------------------------- #


def load_dataset(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("items", "examples", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("expected a list of items in {}".format(input_path))
    return data


def select_items(data, category, limit, seed):
    items = [(i, it) for i, it in enumerate(data) if it.get("category") == category]
    if limit is not None and limit > 0:
        rng = random.Random(seed)
        if len(items) > limit:
            # deterministic subset, but keeps original order for readable logs
            picked = sorted(rng.sample(range(len(items)), limit))
            items = [items[j] for j in picked]
    return items


def audit_dataset(data, categories, input_path):
    """
    Report exactly which per-category key names were found, and whether every
    mismatched item can resolve a text-claimed letter. Run this before the GPU
    window opens -- it is the check that the TDR logic is reading real fields.
    """
    print("\n=== dataset audit ===")
    ok = True
    for cat in categories:
        items = [it for it in data if it.get("category") == cat]
        if not items:
            print("[{}] NO ITEMS FOUND".format(cat))
            ok = False
            continue

        true_key = first_present_key(items[0], TRUE_KEY_CANDIDATES[cat])
        claim_key = first_present_key(items[0], CLAIMED_KEY_CANDIDATES[cat])
        # a neutral/paralinguistic item may lack claimed_*, so scan for it
        if claim_key is None:
            for it in items:
                claim_key = first_present_key(it, CLAIMED_KEY_CANDIDATES[cat])
                if claim_key:
                    break

        opt_counts = Counter()
        cond_counts = Counter()
        mod_counts = Counter()
        unresolved_claim = 0
        claim_equals_truth = 0
        missing_audio_field = 0
        missing_audio_file = 0
        n_mismatched = 0

        for it in items:
            cond_counts[it.get("condition")] += 1
            mod_counts[it.get("modality")] += 1
            try:
                omap = build_option_map(it)
                opt_counts[len(omap)] += 1
            except Exception:
                opt_counts["BROKEN"] += 1
                ok = False
                continue
            if it.get("modality") in ("audio_only", "text_audio"):
                if not it.get("audio_path"):
                    missing_audio_field += 1
                elif resolve_audio_path(it["audio_path"], input_path) is None:
                    missing_audio_file += 1
            if it.get("condition") == "Mismatched":
                n_mismatched += 1
                claimed = it.get(claim_key) if claim_key else None
                cl = resolve_value_to_letter(omap, claimed)
                if cl is None:
                    unresolved_claim += 1
                elif it.get("modality") == "text_audio" and cl == str(it.get("correct_letter", "")).strip().upper():
                    # text_only's ground truth IS the text claim by design (no
                    # audio to contradict it), so claim==truth there is expected,
                    # not a bug. Only text_audio actually pits claim vs audio
                    # truth, so only there does claim==truth indicate a real
                    # data problem (the "mismatch" isn't a mismatch after all).
                    claim_equals_truth += 1

        print("[{}] n={}".format(cat, len(items)))
        print("   true key   : {}".format(true_key or "!! NOT FOUND !!"))
        print("   claimed key: {}".format(claim_key or "!! NOT FOUND !!"))
        print("   option counts seen: {}".format(dict(opt_counts)))
        print("   conditions: {}".format(dict(cond_counts)))
        print("   modalities: {}".format(dict(mod_counts)))
        print("   mismatched items: {}".format(n_mismatched))
        print("   unresolved claimed->letter: {}".format(unresolved_claim))
        print("   text_audio: claimed letter == correct letter (data bug if >0): {}".format(claim_equals_truth))
        print("   audio modality items with no audio_path: {}".format(missing_audio_field))
        print("   audio_path set but file not found on disk: {}".format(missing_audio_file))

        if claim_key is None and n_mismatched:
            print("   ERROR: no claimed_* key found but mismatched items exist -> TDR is undefined")
            ok = False
        if unresolved_claim or claim_equals_truth or missing_audio_field or missing_audio_file:
            ok = False
    print("=== audit {} ===\n".format("PASSED" if ok else "FLAGGED ISSUES"))
    return ok


# --------------------------------------------------------------------------- #
# Model loading -- bfloat16 (DeSTA2.5's hard-coded native precision; see the
# module docstring for why this is accepted instead of forcing fp16)
# --------------------------------------------------------------------------- #


def load_model(args):
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. An 8B-parameter model on CPU will be extremely slow.")

    from desta import DeSTA25AudioModel

    print("loading DeSTA2.5-Audio from: {}".format(args.model_path))
    model = DeSTA25AudioModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    # DeSTA25AudioModel.__init__ hard-codes torch_dtype=torch.bfloat16 for the
    # LLM with no override path (see module docstring "Precision" section).
    # There is no quantization kwarg anywhere in that class either -- nothing
    # to disable there. This assertion just confirms the load actually landed
    # in bf16 (not fp32, not silently something else), not that we chose it.
    llm_dtype = next(model.llm_model.parameters()).dtype
    assert llm_dtype == torch.bfloat16, (
        "LLM did not load in the expected bfloat16 (got {}) -- DeSTA2.5's "
        "loading code may have changed; investigate before trusting results.".format(llm_dtype)
    )
    print("LLM dtype: {} (DeSTA2.5's native precision, hard-coded upstream)".format(llm_dtype))

    return model


# --------------------------------------------------------------------------- #
# Two inference entry points (mirrors run_qwen2audio_eval.py's shape)
# --------------------------------------------------------------------------- #


def inference_with_audio_clean(model, item, max_new_tokens=64, audio_path=None):
    """
    Audio-bearing items: modality == 'audio_only' or 'text_audio'.

    The prompt is item['prompt'] verbatim (plus a terse-answer instruction if
    not already present) -- for audio_only that prompt already carries no
    text claim, for text_audio it already carries the claim. This function
    does not construct or reorder any claim text itself.

    Uses the item's real audio_path directly -- generate() requires a real
    file path (os.path.exists check inside desta's own code) and one already
    exists on disk for these items, so no temp file is needed.
    """
    prompt = _ensure_terse_instruction(item["prompt"])
    path = audio_path or item.get("audio_path")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": AUDIO_LOCATOR + "\n" + prompt,
            "audios": [{"audio": path, "text": None}],
        },
    ]
    outputs = model.generate(
        messages=messages, do_sample=False, top_p=1.0, temperature=1.0, max_new_tokens=max_new_tokens
    )
    raw = outputs.text[0].strip()
    return raw, parse_answer_letter(raw, build_option_map(item))


def inference_with_text_clean(model, item, max_new_tokens=64):
    """
    Text-only items: modality == 'text_only'.

    Unlike SALMONN, DeSTA2.5's generate() has a genuine text-only code path
    (see module docstring): no "<|AUDIO|>" in the content, no "audios" key
    at all -> it skips the speech encoder/VAD entirely and runs plain LLM
    generation on the tokenized chat text. This is real audio-free
    inference, not a silence placeholder.
    """
    prompt = _ensure_terse_instruction(item["prompt"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    outputs = model.generate(
        messages=messages, do_sample=False, top_p=1.0, temperature=1.0, max_new_tokens=max_new_tokens
    )
    raw = outputs.text[0].strip()
    return raw, parse_answer_letter(raw, build_option_map(item))


# --------------------------------------------------------------------------- #
# Scoring -- flat, every branch reachable (identical to run_qwen2audio_eval.py)
# --------------------------------------------------------------------------- #


def score_item(item, parsed_letter, option_map, claim_key):
    """
    parse -> compare to correct_letter -> correct / incorrect / invalid.
    Then, separately and only for mismatched items, compare to the resolved
    text-claimed letter to classify follow_source.
    """
    correct_letter = str(item.get("correct_letter", "")).strip().upper() or None
    claimed_value = item.get(claim_key) if claim_key else None
    claimed_letter = resolve_value_to_letter(option_map, claimed_value)

    if parsed_letter is None:
        correctness = "invalid"
    elif correct_letter is None:
        correctness = "invalid"
    elif parsed_letter == correct_letter:
        correctness = "correct"
    else:
        correctness = "incorrect"

    if item.get("condition") != "Mismatched":
        follow_source = "not_applicable"
    elif parsed_letter is None:
        follow_source = "invalid"
    elif claimed_letter is None:
        follow_source = "claim_unresolved"
    elif parsed_letter == claimed_letter:
        follow_source = "text_claim"
    elif parsed_letter == correct_letter:
        follow_source = "audio_truth"
    else:
        follow_source = "neither"

    return {
        "correct_letter": correct_letter,
        "claimed_value": claimed_value,
        "claimed_letter": claimed_letter,
        "correctness": correctness,
        "follow_source": follow_source,
        "pred_label": option_map.get(parsed_letter) if parsed_letter else None,
        "true_label": option_map.get(correct_letter) if correct_letter else None,
    }


# --------------------------------------------------------------------------- #
# Metrics (identical to run_qwen2audio_eval.py)
# --------------------------------------------------------------------------- #


def _prf(gold, pred, labels):
    per_class = {}
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return per_class


def compute_metrics(records):
    """
    LISTEN-standard metrics over items with a parseable answer.
    weighted accuracy = overall accuracy; UAR = mean per-class recall.
    micro F1 equals accuracy in the single-label multiclass case.
    Invalid responses are excluded from these and reported separately, plus
    accuracy_invalid_as_wrong for the stricter reading.
    """
    scored = [r for r in records if r.get("correctness") in ("correct", "incorrect")]
    n_invalid = sum(1 for r in records if r.get("correctness") == "invalid")
    n_error = sum(1 for r in records if r.get("correctness") == "error")

    if not scored:
        return {
            "n_total": len(records), "n_scored": 0, "n_invalid": n_invalid, "n_error": n_error,
            "weighted_accuracy": None, "uar": None, "macro_f1": None, "micro_f1": None,
            "per_class": {},
        }

    gold = [r["true_label"] for r in scored]
    pred = [r["pred_label"] for r in scored]
    labels = sorted({l for l in gold + pred if l is not None})
    per_class = _prf(gold, pred, labels)

    n_correct = sum(1 for g, p in zip(gold, pred) if g == p)
    wa = n_correct / len(scored)
    gold_labels = sorted({g for g in gold if g is not None})
    uar = sum(per_class[l]["recall"] for l in gold_labels) / len(gold_labels) if gold_labels else None
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels) if labels else None

    denom = len(scored) + n_invalid + n_error
    return {
        "n_total": len(records),
        "n_scored": len(scored),
        "n_invalid": n_invalid,
        "n_error": n_error,
        "weighted_accuracy": wa,
        "uar": uar,
        "macro_f1": macro_f1,
        "micro_f1": wa,
        "accuracy_invalid_as_wrong": n_correct / denom if denom else None,
        "per_class": per_class,
    }


def compute_tdr(records):
    """TDR over mismatched-condition items only, with a modality breakdown
    and (for text_audio) a claim_first/claim_last order breakdown."""
    mism = [r for r in records if r.get("condition") == "Mismatched"]
    counts = Counter(r.get("follow_source") for r in mism)
    usable = counts["text_claim"] + counts["audio_truth"] + counts["neither"]
    overall = counts["text_claim"] / usable if usable else None

    def _bucket(sub):
        c = Counter(r.get("follow_source") for r in sub)
        u = c["text_claim"] + c["audio_truth"] + c["neither"]
        return {
            "n": len(sub),
            "n_usable": u,
            "text_claim": c["text_claim"],
            "audio_truth": c["audio_truth"],
            "neither": c["neither"],
            "invalid": c["invalid"],
            "claim_unresolved": c["claim_unresolved"],
            "tdr": (c["text_claim"] / u) if u else None,
        }

    by_mod = {mod: _bucket([r for r in mism if r.get("modality") == mod]) for mod in MODALITIES}

    # order only varies within text_audio; other modalities carry order=None
    # and are excluded here by construction (orders_present stays empty).
    text_audio_mism = [r for r in mism if r.get("modality") == "text_audio"]
    orders_present = sorted({r.get("order") for r in text_audio_mism if r.get("order")})
    by_order = {order: _bucket([r for r in text_audio_mism if r.get("order") == order])
                for order in orders_present}

    return {
        "n_mismatched": len(mism),
        "n_usable": usable,
        "follow_text": counts["text_claim"],
        "follow_audio": counts["audio_truth"],
        "follow_neither": counts["neither"],
        "invalid": counts["invalid"],
        "claim_unresolved": counts["claim_unresolved"],
        "tdr": overall,
        "by_modality": by_mod,
        "by_order": by_order,
    }


def condition_breakdown(records):
    out = {}
    for cond in CONDITIONS:
        sub = [r for r in records if r.get("condition") == cond]
        if not sub:
            continue
        entry = compute_metrics(sub)
        # order split only where it's meaningful: Matched/Mismatched text_audio.
        # Neutral-Text/Paralinguistic/text_only carry order=None or a single
        # fixed value and are left alone -- no by_order key is added for them.
        text_audio_sub = [r for r in sub if r.get("modality") == "text_audio"]
        orders_present = sorted({r.get("order") for r in text_audio_sub if r.get("order")})
        if orders_present:
            entry["by_order"] = {
                "text_audio": {
                    order: compute_metrics([r for r in text_audio_sub if r.get("order") == order])
                    for order in orders_present
                }
            }
        out[cond] = entry
    return out


# --------------------------------------------------------------------------- #
# Checkpoint / resume (identical to run_qwen2audio_eval.py, desta25_ prefix)
# --------------------------------------------------------------------------- #


def checkpoint_path(output_dir, category):
    return os.path.join(output_dir, "desta25_{}_progress.jsonl".format(category))


def load_checkpoint(path):
    records, done = [], set()
    if not os.path.isfile(path):
        return records, done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print("  skipping malformed checkpoint line")
                continue
            records.append(rec)
            done.add(rec.get("uid"))
    return records, done


def append_checkpoint(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# Run one category
# --------------------------------------------------------------------------- #


def run_category(model, data, category, args, timestamp):
    items = select_items(data, category, args.limit, args.seed)
    if not items:
        print("[{}] no items, skipping".format(category))
        return None

    claim_key = None
    for _, it in items:
        claim_key = first_present_key(it, CLAIMED_KEY_CANDIDATES[category])
        if claim_key:
            break
    print("[{}] {} items | claimed key: {}".format(category, len(items), claim_key))

    ckpt = checkpoint_path(args.output_dir, category)
    records, done = ([], set())
    if not args.no_resume:
        records, done = load_checkpoint(ckpt)
        if done:
            print("[{}] resuming, {} items already done".format(category, len(done)))

    total = len(items)
    t0 = time.time()

    for n, (idx, item) in enumerate(items, 1):
        uid = item_uid(item, idx)
        if uid in done:
            continue

        base = {
            "uid": uid,
            "index": idx,
            "category": item.get("category"),
            "condition": item.get("condition"),
            "modality": item.get("modality"),
            "audio_path": item.get("audio_path"),
            "order": item.get("order"),
        }

        try:
            option_map = build_option_map(item)
        except Exception as exc:
            rec = dict(base, raw_output=None, parsed_letter=None, correctness="error",
                       follow_source="error", error="bad options: {}".format(exc))
            records.append(rec)
            append_checkpoint(ckpt, rec)
            print("  [{}/{}] {} ERROR bad options".format(n, total, uid))
            continue

        modality = item.get("modality")
        raw, parsed, error = None, None, None

        try:
            if modality == "text_only":
                raw, parsed = inference_with_text_clean(model, item, args.max_new_tokens)
            elif modality in ("audio_only", "text_audio"):
                path = resolve_audio_path(item.get("audio_path"), args.input)
                if path is None:
                    error = "audio file missing: {}".format(item.get("audio_path"))
                else:
                    raw, parsed = inference_with_audio_clean(
                        model, item, args.max_new_tokens, audio_path=path
                    )
            else:
                error = "unknown modality: {}".format(modality)
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            if args.verbose_errors:
                traceback.print_exc()

        if error is not None:
            rec = dict(base, raw_output=raw, parsed_letter=None, correctness="error",
                       follow_source="error", error=error,
                       correct_letter=str(item.get("correct_letter", "")).strip().upper() or None,
                       claimed_value=item.get(claim_key) if claim_key else None,
                       claimed_letter=None, pred_label=None, true_label=None)
            print("  [{}/{}] {} ERROR {}".format(n, total, uid, error))
        else:
            scored = score_item(item, parsed, option_map, claim_key)
            rec = dict(base, raw_output=raw, parsed_letter=parsed, error=None, **scored)
            print("  [{}/{}] {} {} -> {} ({}, {})".format(
                n, total, uid, modality, parsed, scored["correctness"], scored["follow_source"]))

        records.append(rec)
        append_checkpoint(ckpt, rec)

    elapsed = time.time() - t0
    print("[{}] finished {} records in {:.1f}s".format(category, len(records), elapsed))

    result_file = os.path.join(args.output_dir, "desta25_{}_{}.json".format(category, timestamp))
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "input": args.input,
            "timestamp": timestamp,
            "claimed_key": claim_key,
            "n_records": len(records),
            "records": records,
        }, f, indent=2, ensure_ascii=False)
    print("[{}] wrote {}".format(category, result_file))

    return {
        "category": category,
        "result_file": result_file,
        "claimed_key": claim_key,
        "elapsed_seconds": elapsed,
        "metrics": compute_metrics(records),
        "by_condition": condition_breakdown(records),
        "tdr": compute_tdr(records),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DeSTA2.5-Audio audio-text dominance evaluation (bfloat16 native, no quantization)"
    )
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=CATEGORIES + ["all"],
                   help="which categories to run")
    p.add_argument("--input", default="data/combined_examples.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--model-path", default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="cap items per category (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoint files")
    p.add_argument("--dry-run", action="store_true", help="audit dataset only, do not load the model")
    p.add_argument("--verbose-errors", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    categories = CATEGORIES if "all" in args.experiments else list(dict.fromkeys(args.experiments))
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_dataset(args.input)
    print("loaded {} items from {}".format(len(data), args.input))
    audit_ok = audit_dataset(data, categories, args.input)

    if args.dry_run:
        print("dry run complete, model not loaded.")
        return 0 if audit_ok else 1

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = load_model(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for cat in categories:
        print("\n===== {} =====".format(cat))
        summaries[cat] = run_category(model, data, cat, args, timestamp)

    summary_file = os.path.join(args.output_dir, "desta25_run_{}_summary.json".format(timestamp))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "input": args.input,
            "seed": args.seed,
            "limit": args.limit,
            "experiments": categories,
            "per_category": summaries,
        }, f, indent=2, ensure_ascii=False)

    print("\n===== summary =====")
    for cat, s in summaries.items():
        if not s:
            continue
        m, t = s["metrics"], s["tdr"]
        print("{}: n={} scored={} invalid={} error={}".format(
            cat, m["n_total"], m["n_scored"], m["n_invalid"], m["n_error"]))
        print("   WA={} UAR={} macroF1={} microF1={}".format(
            _fmt(m["weighted_accuracy"]), _fmt(m["uar"]), _fmt(m["macro_f1"]), _fmt(m["micro_f1"])))
        print("   TDR={} (text={} audio={} neither={} of {} mismatched)".format(
            _fmt(t["tdr"]), t["follow_text"], t["follow_audio"], t["follow_neither"], t["n_mismatched"]))
    print("wrote {}".format(summary_file))
    return 0


def _fmt(x):
    return "n/a" if x is None else "{:.4f}".format(x)


if __name__ == "__main__":
    sys.exit(main())
