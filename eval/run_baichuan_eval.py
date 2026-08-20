#!/usr/bin/env python3
"""
Baichuan-Omni-1.5 (baichuan-inc/Baichuan-Omni-1d5) inference for the
audio-text dominance benchmark.

Mirrors run_phi4mm_eval.py's / run_azeros_eval.py's structure: same CLI
shape, same checkpoint/resume pattern, same scoring (score_item/compute_tdr/
condition_breakdown, incl. the by_order claim_first/claim_last breakdown),
same audit_dataset() pre-flight, same per-item result record shape
(including the "order" field), same dynamic build_option_map/
parse_answer_letter (never assumes an option count). Only the
model-loading/prompt API differs.

--------------------------------------------------------------------------- #
Third non-Whisper-encoder model in this project (after Phi-4-multimodal and
AZeroS). Baichuan-Omni-1.5 uses its own 8-layer RVQ-based
Baichuan-Audio-Tokenizer (12.5Hz frame rate), not Whisper.

Known, accepted limitation: Baichuan-Omni-1.5's LLM backbone is Qwen2.5-7B --
the same lineage as Qwen2-Audio's backbone in this same benchmark. This is a
known confound (shared text-LLM ancestry between two of the models under
test), not a bug in this script and not something to route around here.

--------------------------------------------------------------------------- #
Facts below confirmed against Baichuan-Omni-1.5's own source, not guessed.
Sources:
  https://github.com/baichuan-inc/Baichuan-Omni-1.5/blob/main/web_demo/base_asr_demo.py
  https://github.com/baichuan-inc/Baichuan-Omni-1.5/blob/main/web_demo/constants.py
  https://github.com/baichuan-inc/Baichuan-Omni-1.5/blob/main/web_demo/s2s_gradio_demo_cosy_multiturn.py
  https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5/blob/main/processor_omni.py
--------------------------------------------------------------------------- #

  Precision: bfloat16 -- hard constraint per this project, and also the
  model's own reference demos' default (`torch_dtype=torch.bfloat16` in
  base_asr_demo.py's load_model_tokenizer()). No quantization kwarg appears
  anywhere in the reference loading code. load_model() below asserts the
  loaded weights are actually torch.bfloat16 (same pattern as
  run_desta25_eval.py's / run_audex_eval.py's bf16 asserts) rather than
  trusting the kwarg silently did what it claims.

  trust_remote_code=True is required for BOTH the tokenizer and the model --
  confirmed directly from base_asr_demo.py's load_model_tokenizer(), which
  is the only place this project could confirm it (neither the HF model
  card nor the top-level README state it explicitly).

  Model class: AutoModelForCausalLM (not a dedicated Omni class) -- the
  omni/audio behavior comes from Baichuan's custom modeling_omni.py
  registered via trust_remote_code, not a different transformers entry
  point.

  Setup: after `from_pretrained`, the reference code calls
  `model.bind_processor(tokenizer, training=False, relative_path='/')`
  before any generation -- this wires up `model.processor`, which is what
  actually reads audio files off disk and builds model inputs. Skipping
  this call is not an option; nothing in the demo scripts generates
  without it.

  Audio input format -- NOT a separate audio array/tensor argument like the
  other five models in this project. Audio is referenced *inline in the
  prompt text* as a JSON path reference bracketed by two special tokens:

      audio_start_token + json.dumps({"path": audio_path}) + audio_end_token

  where audio_start_token/audio_end_token come from
  tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
  / audio_end_token_id. model.processor([prompt_text]) parses this out of
  the text via regex, loads the actual audio file at that path, and returns
  an OmniProcessorOutput with input_ids/attention_mask/audios/
  encoder_length/bridge_length -- the latter three are None when no audio
  tag is present in the prompt (confirmed in processor_omni.py).

  Chat template -- role-prefixed, NOT ChatML: role_prefix = {"system":
  "<B_SYS>", "user": "<C_Q>", "assistant": "<C_A>"} (from the reference
  repo's own web_demo/constants.py), concatenated as
  "<B_SYS>{system}<C_Q>{user content}<C_A>" with nothing after the trailing
  assistant-role tag (preprocess_messages() in
  s2s_gradio_demo_cosy_multiturn.py).

  text_only IS supported natively -- unlike AZeroS and Audex (this project's
  other two non-Whisper-encoder models), Baichuan-Omni-1.5's own processor
  gracefully handles a prompt with no audio tag: audio extraction in
  processor_omni.py only triggers on a regex match against
  audio_start_tag/audio_end_tag, and audios/encoder_length/bridge_length
  simply stay None otherwise. There is no NotImplementedError carve-out for
  text_only items in this script -- all three modalities go through
  inference_clean() below via the same code path, differing only in whether
  an audio JSON reference is embedded in the prompt.

  Sampling: this project's benchmark needs a low-variance single-letter
  answer, not conversational diversity, so this script defaults to greedy
  decoding (do_sample=False) for consistency with every other model script
  in this repo (Qwen2-Audio/SALMONN/Phi-4mm all default to greedy here) --
  NOT the demo scripts' own do_sample=True/temperature=0.5-0.8 defaults,
  which are tuned for open-ended chat/ASR, not MCQ answering. Sampling
  params are still exposed as CLI flags in case a real run wants to compare.

Usage:
  py run_baichuan_eval.py --dry-run                 # validate dataset, no model load
  py run_baichuan_eval.py --experiments all --limit 5
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

SYSTEM_PROMPT = "You are a helpful assistant."

# From baichuan-inc/Baichuan-Omni-1.5's own web_demo/constants.py
ROLE_PREFIX = {
    "system": "<B_SYS>",
    "user": "<C_Q>",
    "assistant": "<C_A>",
}

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
# Model loading -- bfloat16 only, no quantization code path anywhere in this
# file
# --------------------------------------------------------------------------- #


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. A 7B+ omni model on CPU is not viable.")

    print("loading tokenizer: {}".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, model_max_length=128000)

    print("loading model in bfloat16: {} (trust_remote_code=True)".format(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = True

    # Wires up model.processor -- the reference demos never generate without
    # this call first. Not optional.
    model.bind_processor(tokenizer, training=False, relative_path="/")

    # There is no quantization kwarg anywhere in the reference loading code
    # this was confirmed against (base_asr_demo.py's load_model_tokenizer()).
    # This assertion confirms the load actually landed in bf16 -- not fp32,
    # not silently int8/4-bit -- not that a quantization option was disabled,
    # because none exists here to disable.
    model_dtype = next(model.parameters()).dtype
    assert model_dtype == torch.bfloat16, (
        "model did not load in the expected bfloat16 (got {}) -- investigate "
        "before trusting results.".format(model_dtype)
    )
    print("model dtype: {}".format(model_dtype))

    audio_start_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
    audio_end_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)

    return model, tokenizer, audio_start_token, audio_end_token


def build_prompt(item, audio_start_token, audio_end_token, audio_path=None):
    """
    "<B_SYS>{system}<C_Q>{user content}<C_A>" -- role_prefix format from
    Baichuan-Omni-1.5's own web_demo/constants.py, built the same way
    preprocess_messages() in s2s_gradio_demo_cosy_multiturn.py does.

    When audio_path is given, the audio is referenced inline as a JSON path
    reference bracketed by the model's own audio start/end tokens (see
    module docstring) -- not passed as a separate array. When audio_path is
    None (text_only items), no audio tag is emitted at all; the reference
    processor handles that gracefully (see module docstring "text_only IS
    supported natively").
    """
    user_content = ""
    if audio_path:
        user_content += audio_start_token + json.dumps({"path": audio_path}, ensure_ascii=False) + audio_end_token
    user_content += item["prompt"]
    return (ROLE_PREFIX["system"] + SYSTEM_PROMPT
            + ROLE_PREFIX["user"] + user_content
            + ROLE_PREFIX["assistant"])


def _generate(model, tokenizer, prompt_text, max_new_tokens, do_sample,
              top_k, top_p, temperature, repetition_penalty):
    """
    model.processor(...) parses any embedded audio JSON reference out of the
    prompt text, loads the audio off disk, and returns input_ids/
    attention_mask/audios/encoder_length/bridge_length -- the latter three
    None for a text-only prompt. model.generate() here is Baichuan's own
    custom override (accepts audios/encoder_length/bridge_length beyond the
    standard HF generate() signature), confirmed against
    web_demo/base_asr_demo.py's exact call shape.
    """
    import torch

    ret = model.processor([prompt_text])
    input_ids = ret.input_ids.cuda()
    attention_mask = ret.attention_mask.cuda()
    audios = ret.audios.cuda() if ret.audios is not None else None
    encoder_length = ret.encoder_length.cuda() if ret.encoder_length is not None else None
    bridge_length = ret.bridge_length.cuda() if ret.bridge_length is not None else None

    with torch.no_grad():
        predicted_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            audios=audios,
            encoder_length=encoder_length,
            bridge_length=bridge_length,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            repetition_penalty=repetition_penalty,
        )

    raw = tokenizer.batch_decode(
        predicted_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return raw.strip()


# --------------------------------------------------------------------------- #
# Single inference entry point -- text_only, audio_only and text_audio all
# go through this (Baichuan's processor handles the text-only case natively,
# see module docstring; there is no NotImplementedError carve-out here,
# unlike run_azeros_eval.py / run_audex_eval.py)
# --------------------------------------------------------------------------- #


def inference_clean(model, tokenizer, audio_start_token, audio_end_token, item,
                     max_new_tokens, do_sample, top_k, top_p, temperature,
                     repetition_penalty, audio_path=None):
    prompt_text = build_prompt(item, audio_start_token, audio_end_token, audio_path=audio_path)
    raw = _generate(model, tokenizer, prompt_text, max_new_tokens, do_sample,
                     top_k, top_p, temperature, repetition_penalty)
    return raw, parse_answer_letter(raw, build_option_map(item))


# --------------------------------------------------------------------------- #
# Scoring -- flat, every branch reachable
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
    return os.path.join(output_dir, "baichuan_{}_progress.jsonl".format(category))


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


def run_category(model, tokenizer, audio_start_token, audio_end_token, data, category, args, timestamp):
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
                raw, parsed = inference_clean(
                    model, tokenizer, audio_start_token, audio_end_token, item,
                    args.max_new_tokens, args.do_sample, args.top_k, args.top_p,
                    args.temperature, args.repetition_penalty, audio_path=None)
            elif modality in ("audio_only", "text_audio"):
                path = resolve_audio_path(item.get("audio_path"), args.input)
                if path is None:
                    error = "audio file missing: {}".format(item.get("audio_path"))
                else:
                    raw, parsed = inference_clean(
                        model, tokenizer, audio_start_token, audio_end_token, item,
                        args.max_new_tokens, args.do_sample, args.top_k, args.top_p,
                        args.temperature, args.repetition_penalty, audio_path=path)
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

    result_file = os.path.join(args.output_dir, "baichuan_{}_{}.json".format(category, timestamp))
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
    p = argparse.ArgumentParser(description="Baichuan-Omni-1.5 audio-text dominance evaluation (bfloat16 only)")
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=CATEGORIES + ["all"],
                   help="which categories to run")
    p.add_argument("--input", default="data/combined_examples.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--model-path", default="baichuan-inc/Baichuan-Omni-1d5")
    p.add_argument("--limit", type=int, default=None, help="cap items per category (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="200, not the 64 every other model script in this repo uses -- "
                        "confirmed necessary by real smoke-test evidence: Baichuan-Omni-1.5 "
                        "reasons in prose before stating its final letter ('Based on the "
                        "information provided... Answer: A. male'), and at 64 tokens most "
                        "audio-bearing responses were cut off mid-reasoning before ever "
                        "reaching the letter, showing up as spurious 'invalid' results that "
                        "had nothing to do with the audio pipeline or the model's actual "
                        "answer.")
    p.add_argument("--do-sample", action="store_true",
                   help="off by default -- greedy decoding, matching every other model "
                        "script in this repo for a low-variance MCQ answer (NOT this "
                        "model's own demo scripts' do_sample=True default, which is tuned "
                        "for open-ended chat/ASR)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
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

    model, tokenizer, audio_start_token, audio_end_token = load_model(args.model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for cat in categories:
        print("\n===== {} =====".format(cat))
        summaries[cat] = run_category(model, tokenizer, audio_start_token, audio_end_token,
                                       data, cat, args, timestamp)

    summary_file = os.path.join(args.output_dir, "baichuan_run_{}_summary.json".format(timestamp))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "input": args.input,
            "seed": args.seed,
            "limit": args.limit,
            "do_sample": args.do_sample,
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
