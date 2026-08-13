#!/usr/bin/env python3
"""
SALMONN-13B inference for the audio-text dominance benchmark.

Mirrors run_qwen2audio_eval.py's structure as closely as SALMONN's different
loading/inference API allows: same CLI shape, same checkpoint/resume pattern,
same scoring (score_item/compute_tdr/condition_breakdown, incl. the by_order
claim_first/claim_last breakdown), same audit_dataset() pre-flight, same
per-item result record shape (including the "order" field). See that file
for the reasoning behind the scoring logic; only the model-loading and
inference layer differs here.

--------------------------------------------------------------------------- #
Setup
--------------------------------------------------------------------------- #

1. Clone SALMONN's inference branch (SALMONN lives on a branch, not main):

     git clone -b salmonn https://github.com/bytedance/SALMONN.git

2. Download the 4 artifacts SALMONN needs (there is no single HF checkpoint):

     - Whisper-large-v2 (whisper_path)
         https://huggingface.co/openai/whisper-large-v2
     - Fine-tuned BEATs_iter3+ (AS2M) (cpt2) (beats_path)
         **MANUAL DOWNLOAD REQUIRED** -- this one is hosted on OneDrive, not
         HuggingFace or git, and will not come down via a normal HF/git pull:
         https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea
     - Vicuna-13B-v1.1 (llama_path)
         https://huggingface.co/lmsys/vicuna-13b-v1.1
     - salmonn_v1.pth (ckpt)
         https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth

3. SALMONN needs its OWN Python environment -- do not run it in the same env
   as run_qwen2audio_eval.py. Its requirements.txt pins torch==2.0.1,
   transformers==4.28.0, peft==0.3.0, and models/modeling_llama.py /
   models/modeling_whisper.py are vendored copies from that transformers era
   (importing internal paths like transformers.modeling_utils). This has
   been smoke-tested: a venv built with exactly those pins imports both
   vendored files cleanly. Build it the same way, e.g.:

     py -3.10 -m venv salmonn-eval
     salmonn-eval\\Scripts\\python.exe -m pip install torch==2.0.1 \\
         --index-url https://download.pytorch.org/whl/cu118   # or /cpu
     salmonn-eval\\Scripts\\python.exe -m pip install \\
         transformers==4.28.0 sentencepiece==0.1.97 accelerate==0.20.3 \\
         peft==0.3.0 soundfile librosa

   Do NOT try to run this script under the same interpreter/env used for
   run_qwen2audio_eval.py's newer transformers -- it is very likely to break
   against SALMONN's vendored model code.

Usage (from within the SALMONN-pinned venv):
  py run_salmonn_eval.py --dry-run                 # validate dataset, no model load
  py run_salmonn_eval.py --experiments all --limit 5 \\
      --salmonn-repo /path/to/SALMONN \\
      --whisper-path /path/to/whisper-large-v2 \\
      --beats-path /path/to/BEATs_iter3_plus_AS2M.pt \\
      --llama-path /path/to/vicuna-13b-v1.1 \\
      --ckpt /path/to/salmonn_v1.pth

Hard constraints (same spirit as run_qwen2audio_eval.py, SALMONN-specific
enforcement is stricter because the model itself has an 8-bit code path):
  * fp16 only. SALMONN's models/salmonn.py has a `low_resource` flag that,
    when True, loads the LLM with load_in_8bit=True -- 8-bit quantization,
    FORBIDDEN in this project under any circumstances, on any hardware.
    low_resource=False is hard-coded below. There is no CLI flag, config
    option, or fallback that can enable low_resource=True or any other
    quantization path. If SALMONN-13B does not fit in fp16 on the target
    hardware (a single A100, 40GB or 80GB), this script errors clearly --
    it does not silently reduce precision to fit.
  * Generation runs under torch.cuda.amp.autocast(dtype=torch.float16), the
    repo's own documented fp16 path (see SALMONN/cli_inference.py).
  * No hardcoded MCQ option count. Letter sets are derived per item.

SALMONN-specific design note -- "text_only" means SILENT audio, not absent
audio: SALMONN's generate() (models/salmonn.py) has no text-only code path
at all -- it unconditionally encodes a spectrogram before wrapping the
prompt, unlike Qwen2-Audio's chat template, which truly omits audio for a
text-only turn. So for modality == "text_only" here, a 1-second silent WAV
is generated on the fly and fed through the normal audio pipeline as a
placeholder -- it carries no audio signal, but it is not the same code path
as Qwen2-Audio's genuine no-audio text_only inference. Keep this distinction
in mind when comparing text_only results across the two models.
"""

import argparse
import json
import os
import random
import re
import string
import sys
import tempfile
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

# SALMONN model config -- mirrors configs/decode_config.yaml's `model:` section
# in the SALMONN repo. The 4 artifact paths (llama_path/whisper_path/beats_path/
# ckpt) and low_resource are NOT here; they're set explicitly in load_model()
# so low_resource's hard-coded False stays maximally visible/grep-able.
SALMONN_MODEL_DEFAULTS = {
    "freeze_whisper": True,
    "freeze_beats": True,
    "use_speech_Qformer": True,
    "freeze_speech_QFormer": False,
    "window_level_Qformer": True,
    "num_speech_query_token": 1,
    "second_per_window": 0.333333,
    "second_stride": 0.333333,
    "speech_llama_proj_model": "",
    "freeze_speech_llama_proj": False,
    "lora": True,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "multi_prompt": True,
    # Deliberately left empty: generate() always takes an explicit `prompts=`
    # argument and never touches self.prompt_dict, so these (which point at
    # repo-relative files like prompts/train_prompt.json in decode_config.yaml)
    # are not needed here and would just be a cwd-relative-path fragility for
    # no benefit.
    "prompt_path": "",
    "test_prompt_path": "",
    "max_txt_len": 300,
    "end_sym": "</s>",
}

PROMPT_TEMPLATE = "USER: {}\nASSISTANT:"

# Mirrors configs/decode_config.yaml's `generate:` section. max_new_tokens is
# the one field driven by --max-new-tokens; the rest match SALMONN's own
# documented decode defaults and are not CLI-tunable, to keep this close to
# "the repo's own documented fp16 path" rather than inventing new behavior.
GENERATE_CFG_DEFAULTS = {
    "num_beams": 4,
    "do_sample": False,
    "min_length": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
}

SILENCE_DURATION_SECONDS = 1.0

# SALMONN's own generate() decodes via llama_tokenizer.batch_decode(...) without
# skip_special_tokens=True (see models/salmonn.py), so raw output can carry
# these artifacts. Stripped before parsing/reporting.
SPECIAL_TOKEN_ARTIFACTS = ["</s>", "<s>", "[PAD]", "<unk>", "<pad>"]

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

    SALMONN tends to be more verbose/chatty than Qwen2-Audio by default, so
    this robust multi-strategy parse matters more here, not less.
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


def _clean_llm_output(text):
    """Strip special-token artifacts SALMONN's own decode call doesn't skip."""
    if text is None:
        return text
    cleaned = text
    for tok in SPECIAL_TOKEN_ARTIFACTS:
        cleaned = cleaned.replace(tok, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


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
# Model loading -- fp16 only, low_resource=False hard-coded
# --------------------------------------------------------------------------- #


def load_model(args):
    import torch

    if not torch.cuda.is_available():
        # SALMONN's own generate() hard-codes stop_words_ids = [torch.tensor([2]).cuda()]
        # (models/salmonn.py) -- there is no CPU fallback, so fail fast with a
        # clear message rather than crashing mid-generate.
        raise SystemExit(
            "SALMONN requires CUDA: its generate() hard-codes .cuda() internally "
            "with no CPU code path. No CUDA device visible."
        )

    if args.salmonn_repo not in sys.path:
        sys.path.insert(0, args.salmonn_repo)

    from transformers import WhisperFeatureExtractor
    from models.salmonn import SALMONN

    model_cfg = dict(SALMONN_MODEL_DEFAULTS)
    model_cfg["llama_path"] = args.llama_path
    model_cfg["whisper_path"] = args.whisper_path
    model_cfg["beats_path"] = args.beats_path
    model_cfg["ckpt"] = args.ckpt
    model_cfg["prompt_template"] = PROMPT_TEMPLATE

    # HARD-CODED. Loading in 8-bit (load_in_8bit=True, triggered by
    # low_resource=True in SALMONN/models/salmonn.py) is FORBIDDEN in this
    # project under any circumstances, on any hardware. Do not add a CLI
    # flag, config option, or fallback that could set this True -- if fp16
    # doesn't fit, this script should error, not silently quantize.
    model_cfg["low_resource"] = False
    assert model_cfg["low_resource"] is False

    print("loading SALMONN-13B (fp16, low_resource=False) -- ckpt: {}".format(args.ckpt))
    model = SALMONN.from_config(model_cfg)
    model.to(args.device)
    model.eval()

    llm_dtype = next(model.llama_model.parameters()).dtype
    assert llm_dtype == torch.float16, (
        "LLM did not load in fp16 (got {}) -- aborting. Quantization is "
        "forbidden in this project.".format(llm_dtype)
    )
    assert model.low_resource is False, "low_resource became True somewhere -- aborting."
    print("LLM dtype: {} | low_resource: {}".format(llm_dtype, model.low_resource))

    wav_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
    return model, wav_processor


def _write_silence_wav(sample_rate, duration_seconds=SILENCE_DURATION_SECONDS):
    """
    SALMONN-specific: a placeholder for text_only items (see module docstring
    for why -- SALMONN's generate() has no text-only code path, so a silent
    clip is fed through the normal audio pipeline instead).
    """
    import numpy as np
    import soundfile as sf

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    silence = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    sf.write(path, silence, sample_rate)
    return path


def _run_generate(model, samples, prompt, max_new_tokens):
    import torch

    from utils import prepare_one_sample  # noqa: F401 (imported for side-effect check at call sites)

    wrapped_prompt = PROMPT_TEMPLATE.format("<Speech><SpeechHere></Speech> " + prompt.strip())
    generate_cfg = dict(GENERATE_CFG_DEFAULTS)
    generate_cfg["max_new_tokens"] = max_new_tokens

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model.generate(samples, generate_cfg, prompts=[wrapped_prompt])
    return outputs[0]


# --------------------------------------------------------------------------- #
# Two inference entry points (mirrors run_qwen2audio_eval.py's shape)
# --------------------------------------------------------------------------- #


def inference_with_audio_clean(model, wav_processor, item, max_new_tokens=64, audio_path=None):
    """
    Audio-bearing items: modality == 'audio_only' or 'text_audio'.

    The prompt is item['prompt'] verbatim (plus a terse-answer instruction if
    not already present) -- for audio_only that prompt already carries no
    text claim, for text_audio it already carries the claim. This function
    does not construct or reorder any claim text itself.

    Uses the item's real audio_path directly. Unlike the text_only path
    below, no temp file is needed here -- prepare_one_sample() takes a file
    path, and a real audio file already exists on disk for these items.
    """
    from utils import prepare_one_sample

    prompt = _ensure_terse_instruction(item["prompt"])
    path = audio_path or item.get("audio_path")
    samples = prepare_one_sample(path, wav_processor)
    raw = _run_generate(model, samples, prompt, max_new_tokens)
    cleaned = _clean_llm_output(raw)
    return cleaned, parse_answer_letter(cleaned, build_option_map(item))


def inference_with_text_clean(model, wav_processor, item, max_new_tokens=64):
    """
    Text-only items: modality == 'text_only'.

    SALMONN-specific: unlike Qwen2-Audio's inference_with_text_clean, which
    truly skips audio, SALMONN's generate() has no text-only code path at
    all -- it always encodes a spectrogram before wrapping the prompt. So
    here "text_only" means a 1-second SILENT placeholder WAV is written to a
    temp file and fed through the normal audio pipeline, not the total
    absence of an audio path. Keep this distinction in mind for cross-model
    comparison against Qwen2-Audio's genuinely audio-free text_only results.
    """
    from utils import prepare_one_sample

    prompt = _ensure_terse_instruction(item["prompt"])
    sr = wav_processor.sampling_rate
    silence_path = _write_silence_wav(sr)
    try:
        samples = prepare_one_sample(silence_path, wav_processor)
        raw = _run_generate(model, samples, prompt, max_new_tokens)
    finally:
        try:
            os.remove(silence_path)
        except OSError:
            pass
    cleaned = _clean_llm_output(raw)
    return cleaned, parse_answer_letter(cleaned, build_option_map(item))


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
# Checkpoint / resume (identical to run_qwen2audio_eval.py, salmonn_ prefix)
# --------------------------------------------------------------------------- #


def checkpoint_path(output_dir, category):
    return os.path.join(output_dir, "salmonn_{}_progress.jsonl".format(category))


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


def run_category(model, wav_processor, data, category, args, timestamp):
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
                raw, parsed = inference_with_text_clean(model, wav_processor, item, args.max_new_tokens)
            elif modality in ("audio_only", "text_audio"):
                path = resolve_audio_path(item.get("audio_path"), args.input)
                if path is None:
                    error = "audio file missing: {}".format(item.get("audio_path"))
                else:
                    raw, parsed = inference_with_audio_clean(
                        model, wav_processor, item, args.max_new_tokens, audio_path=path
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

    result_file = os.path.join(args.output_dir, "salmonn_{}_{}.json".format(category, timestamp))
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "ckpt": args.ckpt,
            "llama_path": args.llama_path,
            "whisper_path": args.whisper_path,
            "beats_path": args.beats_path,
            "dtype": "float16",
            "low_resource": False,
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
        description="SALMONN-13B audio-text dominance evaluation "
                    "(fp16 only, low_resource=False hard-coded)"
    )
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=CATEGORIES + ["all"],
                   help="which categories to run")
    p.add_argument("--input", default="data/combined_examples.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--salmonn-repo", default=None,
                   help="path to the cloned SALMONN repo (salmonn branch); required unless --dry-run")
    p.add_argument("--llama-path", default=None,
                   help="path to Vicuna-13B-v1.1; required unless --dry-run")
    p.add_argument("--whisper-path", default=None,
                   help="path to Whisper-large-v2; required unless --dry-run")
    p.add_argument("--beats-path", default=None,
                   help="path to the fine-tuned BEATs checkpoint (manual OneDrive download); "
                        "required unless --dry-run")
    p.add_argument("--ckpt", default=None,
                   help="path to salmonn_v1.pth; required unless --dry-run")
    p.add_argument("--device", default="cuda:0")
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

    missing = [name for name, val in [
        ("--salmonn-repo", args.salmonn_repo),
        ("--llama-path", args.llama_path),
        ("--whisper-path", args.whisper_path),
        ("--beats-path", args.beats_path),
        ("--ckpt", args.ckpt),
    ] if not val]
    if missing:
        print("ERROR: missing required argument(s) for a real run: {}".format(", ".join(missing)))
        return 1

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, wav_processor = load_model(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for cat in categories:
        print("\n===== {} =====".format(cat))
        summaries[cat] = run_category(model, wav_processor, data, cat, args, timestamp)

    summary_file = os.path.join(args.output_dir, "salmonn_run_{}_summary.json".format(timestamp))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "ckpt": args.ckpt,
            "llama_path": args.llama_path,
            "whisper_path": args.whisper_path,
            "beats_path": args.beats_path,
            "dtype": "float16",
            "low_resource": False,
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
