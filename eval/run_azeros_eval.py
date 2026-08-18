#!/usr/bin/env python3
"""
AZeroS (AudenAI/azeros) inference for the audio-text dominance benchmark.

Mirrors run_phi4mm_eval.py's structure: same CLI shape, same checkpoint/resume
pattern, same scoring (score_item/compute_tdr/condition_breakdown, incl. the
by_order claim_first/claim_last breakdown), same audit_dataset() pre-flight,
same dynamic build_option_map/parse_answer_letter. Only the model-loading/
prompt API and modality handling differ.

--------------------------------------------------------------------------- #
Repo / precision / architecture -- confirmed via direct source inspection,
not the model card alone
--------------------------------------------------------------------------- #

  Repo: AudenAI/azeros on Hugging Face. Genuinely self-contained -- weight
  files (pretrained.pt + per-submodule weights under llm/, speech_encoder/,
  paraling_encoder/) live directly in the Hub repo. No Drive/Dropbox manual
  assembly step (unlike LTU and GAMA, both ruled out earlier for exactly
  that reason).

  Precision: fp16, confirmed in two independent places -- config.json's LLM
  sub-config states "dtype": "float16", AND the actual loading code
  (examples/azeros/model.py) instantiates the LLM with
  torch_dtype=torch.float16 and explicitly casts encoder outputs to fp16
  before they reach the LLM's embedding space (encoder_outs.to(torch.float16)).

  No Whisper anywhere in this checkpoint. model.py's set_speech_encoder()
  branches on `"whisper" in model_type`; the real config.json has
  speech_encoder_config.model_type == paraling_encoder_config.model_type ==
  "zipformer", so that branch is provably dead code here -- confirmed by
  reading the actual source (auden/models/zipformer/model.py), not just
  trusting the model card's claim. Whisper support exists elsewhere in the
  auden package (auden/models/whisper_encoder/) for OTHER models in the
  Auden family, but AudenAI/azeros itself never touches it.

--------------------------------------------------------------------------- #
Environment pins -- REQUIRED, confirmed via source inspection, not a guess
--------------------------------------------------------------------------- #

  The `auden` package itself (not on PyPI as a standalone install):
    git clone https://github.com/AudenAI/Auden.git
    pip install -e Auden/

  k2 -- a HARD, unconditional import-time dependency, not just for
  training/decoding as the README's softer phrasing implies. auden's
  Zipformer encoder (auden/models/zipformer/modules/scaling.py) does:

      try:
          import k2
      except Exception as e:
          raise ImportError("k2 is required for Zipformer custom ops ...")

  This fires just importing the encoder module, before any inference call.
  k2 has NO Windows wheels -- only "Pre-compiled CUDA wheels (Linux)" and
  "Pre-compiled ROCm wheels (Linux)" exist per k2's own install docs. This is
  why this script was never run locally (Windows, RTX 5070 Ti) -- it can only
  run somewhere Linux, i.e. Modal. Verified k2 wheel/torch/cuda combination
  (see modal_azeros_eval.py for the exact pin):
    k2==1.24.4.dev20250714+cuda12.1.torch2.4.1

--------------------------------------------------------------------------- #
No text_only support -- architectural, not a missing feature
--------------------------------------------------------------------------- #

  AzerosModel.generate() has no code path for "no audio at all": inside
  preprocess_text_and_audio(), batch_size is derived from
  audio_features.size(0), and generate() only ever sets x/x_lens via one of
  three audio-bearing branches (precomputed features, an extract_feature
  encoder method, or a whisper log-mel fallback -- itself unreachable here).
  There is no branch for "text only, no audio tensor at all". Rather than
  feed a text_only item's own audio_path anyway (which would defeat that
  condition's entire controlled-design point -- text_only exists specifically
  to test judgment from text alone, with no audio access), this script leaves
  text_only unsupported: inference_with_text_clean() raises NotImplementedError,
  which run_category()'s existing per-item error handling records as a clean,
  explicit "error" record rather than a crash or a silently-corrupted trial.

--------------------------------------------------------------------------- #
Chat / audio API -- confirmed against examples/azeros/model.py and README,
not guessed
--------------------------------------------------------------------------- #

  messages = [[{"role": "user", "content": f"{model.audio_token_wrapped} {prompt}"}]]
  outputs = model.generate([audio_path], messages, max_new_tokens=..., do_sample=False, num_beams=1)

  `model.audio_token_wrapped` is a real attribute set in AzerosModel.__init__
  (not the class name suggested by some secondary sources -- the real class
  in the actual source is `AzerosModel`, not "AZerosModel"). generate()
  natively batches (list of paths + list of message-lists); this script calls
  it one item at a time for clean per-item checkpointing, matching every
  other eval script in this repo.

Usage (Modal only -- see modal_azeros_eval.py; local runs need Linux for k2):
  py run_azeros_eval.py --dry-run                 # validate dataset, no model load
  py run_azeros_eval.py --sanity-check             # 5-example smoke test, no checkpointing
  py run_azeros_eval.py --experiments all --limit 5
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

# 15 examples: (condition, category, count). Weighted toward
# sarcasm_pacing/Mismatched/text_audio (2 distinct items, not the same item
# resampled) -- that cell produced a long, hedging response ending in a
# trailing "...option is:\n\nA" pattern in the first 5-example check, caught
# only by parse_answer_letter's loose "first standalone uppercase token"
# fallback (strategy 4), not one of the primary/explicit strategies (1-3).
# 5 sarcasm_pacing total (all 4 conditions covered, Mismatched doubled), 5
# emotional_prosody (all 4 conditions, Mismatched doubled), 5 speaker_identity
# (all 4 conditions, Mismatched doubled) = 15. Audio-bearing modalities only:
# AZeroS has no text_only path (see module docstring).
SANITY_CHECK_PLAN = [
    ("Mismatched", "sarcasm_pacing", 2),
    ("Neutral-Text", "sarcasm_pacing", 1),
    ("Matched", "sarcasm_pacing", 1),
    ("Paralinguistic", "sarcasm_pacing", 1),
    ("Neutral-Text", "emotional_prosody", 1),
    ("Matched", "emotional_prosody", 1),
    ("Mismatched", "emotional_prosody", 2),
    ("Paralinguistic", "emotional_prosody", 1),
    ("Neutral-Text", "speaker_identity", 1),
    ("Matched", "speaker_identity", 1),
    ("Mismatched", "speaker_identity", 2),
    ("Paralinguistic", "speaker_identity", 1),
]

# --------------------------------------------------------------------------- #
# Small helpers
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
# Option handling -- fully dynamic, never assumes a fixed option count
# --------------------------------------------------------------------------- #


def build_option_map(item):
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
    if value is None:
        return None
    v = norm(value)
    if not v:
        return None
    for letter, text in option_map.items():
        if norm(text) == v:
            return letter
    hits = [l for l, t in option_map.items() if v and (v in norm(t) or norm(t) in v)]
    if len(hits) == 1:
        return hits[0]
    return None


def parse_answer_letter_debug(raw_output, option_map):
    """
    Same matching logic as parse_answer_letter, but also reports which
    strategy matched: "1_exact" / "2_answer_is" / "3_leading_delim" are
    primary (the model's output was already unambiguous); "4_standalone_token"
    / "5_option_text" are loose fallbacks that scan the whole response for
    something letter-shaped. A response that only resolves via a fallback
    strategy is exactly the case worth double-checking by hand -- the
    fallback can pick the wrong letter out of a rambling response just as
    easily as the right one. parse_answer_letter() below is a thin wrapper
    over this, so production scoring can't drift from what the sanity check
    reports.
    """
    if raw_output is None:
        return None, None
    valid = set(option_map.keys())
    text = str(raw_output).strip()
    if not text:
        return None, None

    m = re.fullmatch(r"\(?\[?([A-Za-z])\]?\)?[\.\)\:\,]?", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "1_exact"

    m = re.search(r"(?:answer|option|choice|response)\s*(?:is|:|=)?\s*\(?\[?([A-Za-z])\]?\)?\b", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "2_answer_is"

    m = re.match(r"^\s*\(?\[?([A-Za-z])\]?\)?\s*[\.\)\:\-]\s+", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "3_leading_delim"

    for tok in re.findall(r"\b([A-Za-z])\b", text):
        if tok.isupper() and tok in valid:
            return tok, "4_standalone_token"

    hits = [l for l, t in option_map.items() if norm(t) and norm(t) in norm(text)]
    if len(hits) == 1:
        return hits[0], "5_option_text"

    return None, None


def parse_answer_letter(raw_output, option_map):
    return parse_answer_letter_debug(raw_output, option_map)[0]


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
            picked = sorted(rng.sample(range(len(items)), limit))
            items = [items[j] for j in picked]
    return items


def audit_dataset(data, categories, input_path):
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
# Model loading -- fp16 only, no quantization code path
# --------------------------------------------------------------------------- #


def _fixed_forward_speech_encoder(x, x_lens, model, projector, model_type):
    """
    Patched replacement for AzerosModel.forward_speech_encoder (a
    @staticmethod on the real class in examples/azeros/model.py). This is
    the exact, unmodified logic from that method -- the only change is
    encoder_output.encoder_out / .encoder_out_lens (attribute access) ->
    encoder_output["encoder_out"] / ["encoder_out_lens"] (subscript).

    Real bug, confirmed live: ZipformerEncoderModel.forward() defaults to
    return_dict=True and returns a plain dict, which doesn't support
    attribute access. The reference model.py assumes something HF-ModelOutput-
    shaped (supports both). Every audio-bearing inference call fails with
    "AttributeError: 'dict' object has no attribute 'encoder_out'" without
    this patch -- confirmed against a live 5-example Modal run before this
    fix existed.
    """
    import torch

    if "whisper" in model_type:
        x = x.transpose(1, 2)  # (N, C, T)
        x = x[:, :, :3000]
        encoder_outs = model(x)[0]
        feature_lens = ((x_lens - 1) // 2 + 1)
        feature_lens = torch.where(feature_lens > 1500, 1500, feature_lens)
        encoder_outs = encoder_outs[:, :feature_lens.max()]
    else:
        x_lens = torch.where(x_lens > x.size(1), x.size(1), x_lens)
        encoder_output = model(x, x_lens)
        encoder_outs = encoder_output["encoder_out"]
        feature_lens = encoder_output["encoder_out_lens"]

    encoder_outs, feature_lens = projector(encoder_outs, feature_lens)
    encoder_outs = encoder_outs.to(torch.float16)
    return encoder_outs, feature_lens


def load_model(model_path="AudenAI/azeros", examples_dir="/opt/Auden/examples/azeros"):
    """
    examples_dir must contain model.py/model_config.py/utils.py from
    AudenAI/Auden's examples/azeros/ -- these are example driver files, not
    part of the pip-installed `auden` package itself, so they're not
    importable without this path being added to sys.path first. In the Modal
    image this is where the repo gets git-cloned; see modal_azeros_eval.py.
    """
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. fp16 on CPU will be slow/unstable.")

    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from model import AzerosModel  # noqa: E402  (must follow the sys.path insert above)

    AzerosModel.forward_speech_encoder = staticmethod(_fixed_forward_speech_encoder)

    print("loading AZeroS: {}".format(model_path))
    # strict=False: pretrained.pt only contains the two trained projector
    # modules (config.json's exclude_from_checkpoint = ["llm",
    # "speech_encoder", "paraling_encoder"] -- those frozen submodules load
    # separately, right after, from their own subdirectories). The default
    # strict=True checks the *first* load_state_dict call against the whole
    # model and fails with "Missing key(s)" for every llm.*/speech_encoder.*/
    # paraling_encoder.* parameter, before that second load ever happens.
    model = AzerosModel.from_pretrained(model_path, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    llm_dtype = next(model.llm.parameters()).dtype
    assert llm_dtype == torch.float16, "LLM did not load in fp16: {}".format(llm_dtype)
    print("LLM dtype: {}".format(llm_dtype))
    return model


GENERATE_CONFIG = {
    "num_beams": 1,
    "do_sample": False,
    "min_length": 1,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "top_p": None,
    "top_k": None,
    "temperature": None,
}


# --------------------------------------------------------------------------- #
# Two inference entry points (LISTEN shape)
# --------------------------------------------------------------------------- #


def inference_with_audio_clean(model, item, max_new_tokens=64, audio_path=None):
    """Audio-bearing items: modality == 'audio_only' or 'text_audio'."""
    path = audio_path or item.get("audio_path")
    prompt = item["prompt"]
    content = "{} {}".format(model.audio_token_wrapped, prompt)
    messages = [[{"role": "user", "content": content}]]

    outputs = model.generate(
        [path], messages, max_new_tokens=max_new_tokens, **GENERATE_CONFIG
    )
    raw = str(outputs[0]).strip()
    return raw, parse_answer_letter(raw, build_option_map(item))


def inference_with_text_clean(model, item, max_new_tokens=64):
    """
    Not supported -- see module docstring "No text_only support". Raises so
    run_category()'s existing per-item try/except records this as a clean
    "error" outcome, same mechanism used for any other inference failure.
    """
    raise NotImplementedError(
        "AZeroS has no text-only inference path: generate() requires an "
        "audio_input for every call (batch_size is derived from "
        "audio_features.size(0)). Feeding the item's own audio_path anyway "
        "would defeat the text_only condition's controlled design."
    )


# --------------------------------------------------------------------------- #
# Scoring -- flat, every branch reachable
# --------------------------------------------------------------------------- #


def score_item(item, parsed_letter, option_map, claim_key):
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
# Metrics
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
# Checkpoint / resume
# --------------------------------------------------------------------------- #


def checkpoint_path(output_dir, category):
    return os.path.join(output_dir, "azeros_{}_progress.jsonl".format(category))


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

    result_file = os.path.join(args.output_dir, "azeros_{}_{}.json".format(category, timestamp))
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "model_path": args.model_path,
            "dtype": "float16",
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
# 5-example sanity check -- no checkpointing, just format/parseability/timing
# --------------------------------------------------------------------------- #


def select_sanity_examples(items):
    selected = []
    for condition, category, count in SANITY_CHECK_PLAN:
        candidates = [
            it for it in items
            if it["condition"] == condition
            and it["category"] == category
            and it["modality"] in ("text_audio", "audio_only")
        ]
        if len(candidates) < count:
            raise SystemExit(
                f"Only {len(candidates)} audio-bearing example(s) for "
                f"condition={condition!r} category={category!r}, need {count} "
                f"-- SANITY_CHECK_PLAN doesn't match this dataset."
            )
        # distinct items, not the same row repeated, when count > 1
        selected.extend(candidates[:count])
    return selected


def run_sanity_check(model, data, args):
    import torch

    examples = select_sanity_examples(data)

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM allocated={allocated_gb:.2f} GB reserved={reserved_gb:.2f} GB "
          f"(GPU total={total_gb:.2f} GB)\n")

    results = []
    for i, item in enumerate(examples, start=1):
        path = resolve_audio_path(item.get("audio_path"), args.input)
        start = time.perf_counter()
        try:
            raw, parsed = inference_with_audio_clean(model, item, args.max_new_tokens, audio_path=path)
            _, strategy = parse_answer_letter_debug(raw, build_option_map(item))
            error = None
        except Exception as exc:
            raw, parsed, strategy = None, None, None
            error = "{}: {}".format(type(exc).__name__, exc)
        elapsed = time.perf_counter() - start

        print(f"--- [{i}/{len(examples)}] {item['id']} "
              f"({item['category']}/{item['condition']}/{item['modality']}) ---")
        print(f"  audio: {item.get('audio_path')}")
        print(f"  correct_letter: {item.get('correct_letter')}  options: {item.get('options')}")
        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  raw output: {raw!r}")
            fallback_tag = " <-- FALLBACK" if strategy and strategy.startswith(("4", "5")) else ""
            print(f"  parsed letter: {parsed!r}  parseable: {parsed is not None}  strategy: {strategy}{fallback_tag}")
        print(f"  time: {elapsed:.2f}s\n")

        results.append({
            "id": item["id"],
            "category": item["category"],
            "condition": item["condition"],
            "raw_output": raw,
            "parsed_letter": parsed,
            "parseable": parsed is not None,
            "strategy": strategy,
            "is_fallback": bool(strategy) and strategy.startswith(("4", "5")),
            "error": error,
            "seconds": round(elapsed, 2),
        })

    n_parseable = sum(r["parseable"] for r in results)
    n_fallback = sum(r["is_fallback"] for r in results)
    print("=" * 60)
    print(f"SANITY CHECK SUMMARY: {n_parseable}/{len(results)} responses parsed into a valid MCQ letter, "
          f"{n_fallback}/{len(results)} only via a fallback strategy")
    for r in results:
        status = "OK" if r["parseable"] else ("ERROR" if r["error"] else "UNPARSEABLE")
        tag = " [FALLBACK]" if r["is_fallback"] else ""
        print(f"  [{status:11}] {r['category']:<18} {r['condition']:<15} {r['seconds']:6.2f}s  "
              f"-> {r['parsed_letter']}{tag}")

    if n_fallback:
        print(f"\n{n_fallback} response(s) only resolved via a fallback strategy (4_standalone_token "
              f"or 5_option_text) rather than a clean/explicit match. Worth checking these by hand -- "
              f"a fallback can pick the wrong letter out of a rambling response as easily as the right one:")
        for r in results:
            if r["is_fallback"]:
                print(f"  {r['id']} ({r['category']}/{r['condition']}) strategy={r['strategy']}: {r['raw_output']!r}")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AZeroS audio-text dominance evaluation (fp16 only)")
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=CATEGORIES + ["all"],
                   help="which categories to run")
    p.add_argument("--input", default="data/combined_examples.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--model-path", default="AudenAI/azeros")
    p.add_argument("--examples-dir", default="/opt/Auden/examples/azeros",
                   help="dir containing model.py/model_config.py/utils.py from AudenAI/Auden")
    p.add_argument("--limit", type=int, default=None, help="cap items per category (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoint files")
    p.add_argument("--dry-run", action="store_true", help="audit dataset only, do not load the model")
    p.add_argument("--sanity-check", action="store_true",
                   help="load the model and run only the 5-example smoke test, no checkpointing")
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

    model = load_model(args.model_path, args.examples_dir)

    if args.sanity_check:
        run_sanity_check(model, data, args)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for cat in categories:
        print("\n===== {} =====".format(cat))
        summaries[cat] = run_category(model, data, cat, args, timestamp)

    summary_file = os.path.join(args.output_dir, "azeros_run_{}_summary.json".format(timestamp))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model_path": args.model_path,
            "dtype": "float16",
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
