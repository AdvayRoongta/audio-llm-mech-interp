#!/usr/bin/env python3
"""
NVIDIA Nemotron-Labs-Audex-30B-A3B (nvidia/Nemotron-Labs-Audex-30B-A3B)
inference for the audio-text dominance benchmark.

Mirrors run_phi4mm_eval.py's / run_azeros_eval.py's structure: same CLI
shape, same checkpoint/resume pattern, same scoring (score_item/compute_tdr/
condition_breakdown, incl. the by_order claim_first/claim_last breakdown),
same audit_dataset() pre-flight, same per-item result record shape
(including the "order" field), same dynamic build_option_map/
parse_answer_letter (never assumes an option count). Only the
model-loading/prompt API and text_only handling differ.

--------------------------------------------------------------------------- #
Model facts -- confirmed against NVIDIA's own HF repo and its own reference
inference script, not guessed. Sources:
  https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B
  https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B/raw/main/inference_scripts_hf/inference_hf.py
  https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B/raw/main/inference_scripts_hf/audio_utils.py
  https://arxiv.org/abs/2607.05196  (Audex paper)
--------------------------------------------------------------------------- #

  Architecture: 30B-parameter MoE, 3B active params, built on
  Nemotron-Cascade-2 (a hybrid Mamba/Transformer text backbone) with an
  NV-Whisper audio encoder bolted on via a learned sound-embedding scheme
  (<sound> placeholder expanded to N <so_embedding> tokens bracketed by
  <so_start>/<so_end>, N = num_whisper_clips * config.sound_embedding_size).

  Precision: bfloat16 -- hard constraint per this project, and also what
  NVIDIA's own inference_hf.py defaults --torch-dtype to. No quantization
  code path exists anywhere in this file; load_model() asserts the loaded
  weights are actually torch.bfloat16 (same pattern as run_desta25_eval.py's
  bf16 assert) rather than trusting the kwarg silently did what it claims.

  Backend: "native", not "legacy". NVIDIA's inference_hf.py auto-selects
  between two backends: a "legacy" HF AutoModelForCausalLM path (any
  transformers >= 4.53.0, tested at 4.53.3) and a "native" path that requires
  transformers >= 5.14.0 plus mamba-ssm==2.3.1 / causal-conv1d==1.6.2.post1
  compiled against the container's CUDA/torch, and loads a dedicated
  NemotronHAudexForConditionalGeneration class straight out of the
  checkpoint folder's modeling_nemotron_h_audio_native.py (bypassing the
  transformers AutoModel registry entirely -- see load_native_model()
  below). This project's environment is pinned to the native path
  specifically (see modal_audex_eval.py's image), so that is what this
  script always uses -- there is no --hf-backend=legacy fallback exposed
  here on purpose, to avoid silently running a different code path than the
  one this environment was actually built and smoke-tested against.

  Reused, not reimplemented: audio_utils.py (feature extraction, the
  <sound>-placeholder expansion math, the ChatML prompt template, the
  thinking-vs-instruct prefix, ID/thinking-trace stripping) and the
  model-loading helpers in inference_hf.py (validate_native_kernels,
  load_native_module, load_native_model, resolve_hf_backend) are NVIDIA's
  own code, fetched from the model's HF repo at run time (see
  modal_audex_eval.py) and imported directly rather than hand-copied here.
  Re-deriving the sound-embedding-count math or the ChatML template by hand
  would risk a subtle mismatch against what the model was actually trained
  on; importing NVIDIA's own tested implementation avoids that class of bug
  entirely. This mirrors how run_salmonn_eval.py imports bytedance/SALMONN's
  own model code rather than reimplementing SALMONN.

  Chat template / thinking mode -- from audio_utils.build_prompt_template():
    reasoning=True:  "<|im_start|>user\\n<sound>\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"
    reasoning=False: "<|im_start|>user\\n<sound>\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think></think>"
  This project always uses reasoning=False (instruct/non-thinking mode for
  audio tasks, per the model card's own recommendation) -- the "<think></think>"
  prefix with no trailing newline, exactly as documented.

  Sampling: temperature=0.7, top_p=0.9, top_k=0 -- NVIDIA's own documented
  recipe for audio-understanding tasks (inference_example.sh uses these
  exact values). This makes generation stochastic (do_sample=True), so
  args.seed is applied via torch.manual_seed() before the run, same as
  every other script in this repo.

  text_only -- a genuinely separate model/code path, not a variant of the
  audio-conditioned one. NVIDIA's own build_prompt_template()/
  expand_sound_placeholder() (used by inference_with_audio_clean() below)
  require exactly one <sound> placeholder and a real audio_path to compute
  the sound-embedding count from actual Whisper clip features -- there is no
  text-only mode of that function. Instead NVIDIA ships a dedicated
  text-only entry point: per the model card, "text reasoning follows
  Nemotron-Cascade-2-30B-A3B directly", exposed as checkpoint_folder_textonly
  (plain ChatML, no <sound> tag, no audio tensor) and demonstrated in
  inference_scripts_vllm/textonly_scripts/run_text_vllm_example.py. That
  reference script happens to use vLLM, but the checkpoint itself needs no
  new serving stack: diffing checkpoint_folder_textonly/
  model.safetensors.index.json against checkpoint_folder_full's shows every
  one of its 6243 tensor keys resolves to the identical shard filename in
  both checkpoints (checkpoint_folder_full's 490 extra tensors -- audio
  encoder/projector -- live only in the separate model-audio.safetensors,
  which the text-only index never references). So this project loads the
  text-only backbone as a plain AutoModelForCausalLM (trust_remote_code,
  class NemotronHForCausalLM per checkpoint_folder_textonly/config.json's
  auto_map) against those same already-downloaded shards -- see
  modal_audex_eval.py's _resolve_textonly_artifacts(). No second ~60GB
  download, no vLLM, same causal-conv1d/mamba-ssm kernels already required
  by the native audio backend (confirmed by checkpoint_folder_textonly's own
  modeling_nemotron_h.py importing from the same two packages).

  Text-only prompt format -- built directly, not reimplemented from a guess:
  audio_utils.DEFAULT_SYSTEM_PROMPT (already fetched for the audio path, and
  byte-identical to run_text_vllm_example.py's own SYSTEM_PROMPT constant)
  + a user turn carrying item['prompt'] verbatim (same "pass the item's own
  prompt through unmodified" convention inference_with_audio_clean() uses)
  + "<|im_start|>assistant\\n</think>\\n" as the instruct/non-thinking
  prefix, matching run_text_vllm_example.py's build_chatml_prompt(think=False)
  exactly. Built directly rather than via tokenizer.apply_chat_template(),
  because checkpoint_folder_textonly's own chat_template.jinja defaults
  enable_thinking to True (would prepend "<think>\\n" instead) -- per
  NVIDIA's instruct-mode note, this project always wants the non-thinking
  "</think>\\n" prefix, same choice inference_with_audio_clean() makes for
  audio tasks via reasoning=False.

Usage:
  py run_audex_eval.py --dry-run                 # validate dataset, no model load
  py run_audex_eval.py --experiments all --limit 5 \\
      --model-path /local/checkpoint_folder_full \\
      --inference-utils-dir /local/inference_scripts_hf
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


def select_items(data, category, limit, seed, modality_filter=None):
    items = [
        (i, it) for i, it in enumerate(data)
        if it.get("category") == category
        and (modality_filter is None or it.get("modality") == modality_filter)
    ]
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
# Model loading -- bfloat16 only, native Nemotron-H fast path, no
# quantization code path anywhere in this file
# --------------------------------------------------------------------------- #


def load_model(model_path, inference_utils_dir, refresh_code_cache=True):
    """Load Audex via NVIDIA's own native-backend loading code.

    model_path         -- local directory containing the downloaded
                           checkpoint_folder_full/ contents (config, tokenizer,
                           modeling_nemotron_h_audio_native.py, safetensors,
                           audio_preprocessor/). Resolving this from the HF
                           repo is the Modal wrapper's job (modal_audex_eval.py),
                           same division of labor as run_salmonn_eval.py's
                           --ckpt/--beats-path args.
    inference_utils_dir -- local directory containing NVIDIA's own
                           inference_hf.py + audio_utils.py (also fetched by
                           the Modal wrapper). Imported directly, not
                           reimplemented -- see module docstring.
    """
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. A 30B-parameter model on CPU is not viable.")

    if inference_utils_dir not in sys.path:
        sys.path.insert(0, inference_utils_dir)
    import inference_hf as audex_hf  # noqa: E402  (NVIDIA's own script, fetched at run time)

    print("loading Audex (native backend) from: {}".format(model_path))
    model, tokenizer, feature_extractor, config = audex_hf.load_model(
        model_path,
        device_map="single",
        device="cuda:0",
        torch_dtype="bfloat16",
        tp_plan="none",
        refresh_code_cache=refresh_code_cache,
        hf_backend="native",
    )

    # audex_hf.load_model() passes dtype=torch.bfloat16 into from_pretrained,
    # but this project treats "we asked for bf16" and "it actually loaded in
    # bf16" as two different claims -- same posture as run_desta25_eval.py's
    # bf16 assert. There is no quantization kwarg anywhere in NVIDIA's
    # load_native_model()/load_model() (no BitsAndBytesConfig, no int8/int4
    # path at all), so this assertion is confirming the load landed correctly,
    # not guarding against a quantization option that doesn't exist here.
    model_dtype = next(model.parameters()).dtype
    assert model_dtype == torch.bfloat16, (
        "model did not load in the expected bfloat16 (got {}) -- Audex's "
        "native loading path may have changed; investigate before trusting "
        "results.".format(model_dtype)
    )
    print("model dtype: {} (native Nemotron-H fast path)".format(model_dtype))

    return model, tokenizer, feature_extractor, config, audex_hf


def load_text_only_model(textonly_model_path, inference_utils_dir):
    """Load Audex's dedicated text-only backbone (NemotronHForCausalLM,
    plain HF AutoModelForCausalLM/trust_remote_code -- not the audio native
    conditional-generation class). textonly_model_path is the merged local
    directory modal_audex_eval.py's _resolve_textonly_artifacts() assembles:
    checkpoint_folder_textonly's own small config/tokenizer/index files plus
    the already-downloaded checkpoint_folder_full shard weights those files
    reference (same backbone, see module docstring's text_only note). Kept
    fully independent of load_model() above -- this function does not
    require the audio-conditioned model to have been loaded, so a
    text_only-only run never has to load (and pay GPU memory/time for) the
    audio model at all. bf16 only, same hard constraint as load_model().
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. A 30B-parameter model on CPU is not viable.")

    if inference_utils_dir not in sys.path:
        sys.path.insert(0, inference_utils_dir)
    # Reused, not reimplemented -- same native-kernel presence check
    # (causal-conv1d/mamba-ssm) NVIDIA's own audio-path loader uses;
    # checkpoint_folder_textonly's modeling_nemotron_h.py imports from the
    # identical two packages (confirmed by inspecting that file directly).
    import inference_hf as audex_hf  # noqa: E402
    audex_hf.validate_native_kernels()

    print("loading Audex text-only backbone (NemotronHForCausalLM) from: {}".format(textonly_model_path))
    tokenizer = AutoTokenizer.from_pretrained(textonly_model_path, trust_remote_code=True)

    # checkpoint_folder_textonly declares vocab_size=131072; the shared shard
    # files it points at (see modal_audex_eval.py's _resolve_textonly_artifacts())
    # were saved against checkpoint_folder_full's larger, multimodal-extended
    # vocab_size=205312, so lm_head.weight/backbone.embeddings.weight -- the
    # only two tensors whose shape depends on vocab_size -- come out oversized
    # for this config and transformers refuses to load them by default.
    # Confirmed by diffing both checkpoints' tokenizer.json vocabs directly:
    # of the base 131072 ids, all but 12 map to the identical token string in
    # both (those 12 are cosmetic <SPECIAL_NN> placeholder names in reserved
    # slots that checkpoint_folder_full's tokenizer repurposes for multimodal
    # tokens -- same underlying trained embedding row), and every one of
    # checkpoint_folder_full's ~74.7k extra added tokens is a strictly higher
    # id (>=131072). So ids 0..131071 are the identical, correctly-trained
    # rows this text-only model needs -- just truncated, not reinitialized.
    # ignore_mismatched_sizes=True lets the rest of the checkpoint load
    # normally and only reinits these two tensors (to the correct [131072, ...]
    # shape, randomly); immediately overwritten below with the real sliced
    # rows read directly from the shard files, so nothing stays randomly
    # initialized.
    model = AutoModelForCausalLM.from_pretrained(
        textonly_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        ignore_mismatched_sizes=True,
    )

    vocab_size = model.config.vocab_size
    with open(os.path.join(textonly_model_path, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    for tensor_name, module in (
        ("backbone.embeddings.weight", model.backbone.embeddings),
        ("lm_head.weight", model.lm_head),
    ):
        shard_path = os.path.join(textonly_model_path, weight_map[tensor_name])
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            oversized = sf.get_tensor(tensor_name)
        sliced = oversized[:vocab_size].to(dtype=torch.bfloat16, device=module.weight.device)
        if sliced.shape != module.weight.shape:
            raise RuntimeError(
                "sliced {} shape {} does not match model's {} shape {} -- the "
                "checkpoint's vocab layout may not be the simple prefix this "
                "code assumes; investigate before trusting results.".format(
                    tensor_name, tuple(sliced.shape), tensor_name, tuple(module.weight.shape))
            )
        module.weight.data.copy_(sliced)
        del oversized, sliced
    print("re-sliced lm_head.weight / backbone.embeddings.weight to vocab_size={} "
          "from the shared checkpoint shards".format(vocab_size))

    model.eval()

    # Same posture as load_model()'s bf16 assert: confirm the load actually
    # landed in bf16 rather than trusting the kwarg silently did what it claims.
    model_dtype = next(model.parameters()).dtype
    assert model_dtype == torch.bfloat16, (
        "text-only model did not load in the expected bfloat16 (got {}) -- "
        "investigate before trusting results.".format(model_dtype)
    )
    print("text-only model dtype: {}".format(model_dtype))

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Two inference entry points (mirrors run_phi4mm_eval.py's / run_azeros_eval.py's shape)
# --------------------------------------------------------------------------- #


def inference_with_audio_clean(model, tokenizer, feature_extractor, config, audex_hf,
                                item, sample_rate, max_new_tokens,
                                temperature, top_p, top_k, audio_path=None):
    """
    Audio-bearing items: modality == 'audio_only' or 'text_audio'.

    The prompt is item['prompt'] verbatim -- for audio_only that prompt
    already carries no text claim, for text_audio it already carries the
    claim. This function does not construct or reorder any claim text
    itself. Feature extraction, the <sound> placeholder expansion, the
    ChatML/instruct-mode template, and generation all go through NVIDIA's
    own audex_hf.generate_one() (see module docstring) -- reasoning=False
    selects the "<think></think>" (no newline) instruct-mode prefix for
    audio tasks, per the model card's own recommendation.
    """
    prompt = item["prompt"]
    path = audio_path or item.get("audio_path")

    raw = audex_hf.generate_one(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config,
        audio_path=path,
        prompt=prompt,
        sample_rate=sample_rate,
        reasoning=False,
        prompt_repitition="none",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return raw, parse_answer_letter(raw, build_option_map(item))


def inference_with_text_clean(text_model, text_tokenizer, item,
                               max_new_tokens, temperature, top_p, top_k):
    """
    Text-only items: modality == 'text_only'. Uses Audex's dedicated
    text-only entry point (see module docstring's text_only note) -- a
    fully separate model/code path from inference_with_audio_clean(), not a
    variant of it. No <sound> token, no audio tensor anywhere in this
    function.

    Prompt: audio_utils.DEFAULT_SYSTEM_PROMPT + a user turn carrying
    item['prompt'] verbatim + "</think>\n" as the assistant-turn prefix,
    built directly rather than via tokenizer.apply_chat_template() --
    matches NVIDIA's own inference_scripts_vllm/textonly_scripts/
    run_text_vllm_example.py::build_chatml_prompt(think=False) exactly (see
    module docstring for why apply_chat_template's default would differ).
    Sampling kwargs mirror inference_hf.py::generate_one()'s do_sample/
    temperature/top_p/top_k handling so text_only generation is governed by
    the same sampling settings as the audio-conditioned conditions, for a
    controlled comparison across modalities (this project's CLI args, not
    NVIDIA's own MMLU-Pro reasoning-benchmark defaults).
    """
    import torch
    import audio_utils  # already on sys.path via load_text_only_model()

    prompt = (
        audio_utils.DEFAULT_SYSTEM_PROMPT
        + "<|im_start|>user\n"
        + item["prompt"]
        + audio_utils.IM_END_TOKEN + "\n"
        + "<|im_start|>assistant\n"
        + "</think>\n"
    )

    tokenized = text_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized.input_ids.to(text_model.device)
    attention_mask = (
        tokenized.attention_mask.to(text_model.device)
        if "attention_mask" in tokenized
        else torch.ones_like(input_ids)
    )

    eos_token_id = text_tokenizer.convert_tokens_to_ids(audio_utils.IM_END_TOKEN)
    if eos_token_id is None or eos_token_id == text_tokenizer.unk_token_id:
        eos_token_id = getattr(text_model.config, "eos_token_id", None)

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    do_sample = temperature != 1.0 or 0.0 < top_p < 1.0 or top_k > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "eos_token_id": eos_token_id,
        "pad_token_id": text_tokenizer.pad_token_id or getattr(text_model.config, "pad_token_id", 0),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        if top_p > 0.0:
            generation_kwargs["top_p"] = top_p
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    response = text_tokenizer.decode(new_tokens, skip_special_tokens=False)
    raw = response.split(audio_utils.IM_END_TOKEN, 1)[0].strip()
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
    return os.path.join(output_dir, "audex_{}_progress.jsonl".format(category))


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


def run_category(model, tokenizer, feature_extractor, config, audex_hf,
                  text_model, text_tokenizer, data, category, args, timestamp):
    items = select_items(data, category, args.limit, args.seed, args.modality_filter)
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
                if text_model is None:
                    raise RuntimeError(
                        "text_only item encountered but no text-only model was loaded "
                        "for this run (--textonly-model-path missing)."
                    )
                raw, parsed = inference_with_text_clean(
                    text_model, text_tokenizer, item,
                    args.max_new_tokens, args.temperature, args.top_p, args.top_k,
                )
            elif modality in ("audio_only", "text_audio"):
                path = resolve_audio_path(item.get("audio_path"), args.input)
                if path is None:
                    error = "audio file missing: {}".format(item.get("audio_path"))
                else:
                    raw, parsed = inference_with_audio_clean(
                        model, tokenizer, feature_extractor, config, audex_hf, item,
                        args.sample_rate, args.max_new_tokens,
                        args.temperature, args.top_p, args.top_k, audio_path=path,
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

    result_file = os.path.join(args.output_dir, "audex_{}_{}.json".format(category, timestamp))
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "backend": "native",
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
    p = argparse.ArgumentParser(description="Audex (Nemotron-Labs-Audex-30B-A3B) audio-text dominance evaluation (bf16, native backend only)")
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=CATEGORIES + ["all"],
                   help="which categories to run")
    p.add_argument("--input", default="data/combined_examples.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--model-path", default=None,
                   help="local directory containing checkpoint_folder_full/'s contents "
                        "(resolved from the HF repo by the Modal wrapper). Not required for "
                        "--dry-run, which never loads the model; required otherwise.")
    p.add_argument("--inference-utils-dir", default=None,
                   help="local directory containing NVIDIA's inference_hf.py + audio_utils.py "
                        "(resolved from the HF repo by the Modal wrapper). Not required for "
                        "--dry-run, which never loads the model; required otherwise.")
    p.add_argument("--textonly-model-path", default=None,
                   help="local directory containing Audex's dedicated text-only checkpoint "
                        "(checkpoint_folder_textonly's config/tokenizer merged with the shared "
                        "checkpoint_folder_full shard weights -- resolved by the Modal wrapper's "
                        "_resolve_textonly_artifacts()). Only required if the selected items "
                        "include modality == 'text_only'.")
    p.add_argument("--modality-filter", default=None, choices=MODALITIES,
                   help="restrict item selection to a single modality (e.g. text_only), "
                        "applied before --limit. Used to target a specific rerun without "
                        "touching already-completed items of other modalities.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="NVIDIA's documented recipe for audio-understanding tasks")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="NVIDIA's documented recipe for audio-understanding tasks")
    p.add_argument("--top-k", type=int, default=0)
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

    if not args.inference_utils_dir:
        raise SystemExit(
            "--inference-utils-dir is required for a real run (only --dry-run can "
            "omit it). The Modal wrapper resolves it from the HF repo automatically "
            "-- see modal_audex_eval.py."
        )

    # Which modalities the selected items actually need a model for -- decided
    # up front so (a) we only require CLI args this invocation actually needs,
    # and (b) a text_only-only run never loads the ~65GB audio-conditioned
    # model at all (loading both the audio-native model and the text-only
    # backbone simultaneously would not fit in a single A100-80GB's memory).
    needed_modalities = set()
    for cat in categories:
        items = select_items(data, cat, args.limit, args.seed, args.modality_filter)
        needed_modalities.update(it.get("modality") for _, it in items)
    need_audio_model = bool(needed_modalities & {"audio_only", "text_audio"})
    need_text_model = "text_only" in needed_modalities

    if need_audio_model and not args.model_path:
        raise SystemExit(
            "--model-path is required: the selected items include audio_only/text_audio "
            "modalities. The Modal wrapper resolves this from the HF repo automatically."
        )
    if need_text_model and not args.textonly_model_path:
        raise SystemExit(
            "--textonly-model-path is required: the selected items include text_only "
            "modality. The Modal wrapper resolves this from the HF repo automatically."
        )

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, feature_extractor, config, audex_hf = (None, None, None, None, None)
    if need_audio_model:
        model, tokenizer, feature_extractor, config, audex_hf = load_model(
            args.model_path, args.inference_utils_dir)

    text_model, text_tokenizer = (None, None)
    if need_text_model:
        text_model, text_tokenizer = load_text_only_model(
            args.textonly_model_path, args.inference_utils_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = {}
    for cat in categories:
        print("\n===== {} =====".format(cat))
        summaries[cat] = run_category(model, tokenizer, feature_extractor, config, audex_hf,
                                       text_model, text_tokenizer, data, cat, args, timestamp)

    summary_file = os.path.join(args.output_dir, "audex_run_{}_summary.json".format(timestamp))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "backend": "native",
            "input": args.input,
            "seed": args.seed,
            "limit": args.limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
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
