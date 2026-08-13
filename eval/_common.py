"""Shared plumbing for the mechanistic experiments (exp1-exp4).

These experiments ask *why* the model behaves the way run_qwen2audio_eval.py
measures, so they deliberately reuse that script's definitions of an item, a
prompt, an option map and a correct answer. Anything imported from R below is
there so the two never drift apart.

Two deliberate differences from run_qwen2audio_eval.py:

  * Scoring is by option log-prob, not by generating text and parsing a letter.
    Letters are single tokens, so one forward scores every option, nothing can
    fail to parse, and we get a graded score per option rather than a hard
    label -- which is what the patching experiments need.

  * The model can be loaded with eager attention (for attention weights and
    gradients) and optionally in 4-bit (to fit a 15 GB T4). The main eval
    asserts fp16 on purpose; here it is a flag, and every result records which
    was used.

Run from the repo root:  python eval/exp1_capability_screen.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_qwen2audio_eval as R  # noqa: E402  (same directory)

DEFAULT_DATASET = "data/combined_examples.json"
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

SYSTEM_PROMPT = R.SYSTEM_PROMPT
CATEGORIES = R.CATEGORIES
CONDITIONS = R.CONDITIONS
MODALITIES = R.MODALITIES


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def load_items(path=DEFAULT_DATASET, require_audio=False):
    """Items with audio_path resolved to something that exists on disk.

    dataset_audio/ is gitignored, so on a fresh clone the audio is absent and
    every audio-bearing item resolves to None. Callers that need audio should
    pass require_audio=True and get told immediately rather than 300 items in.
    """
    data = R.load_dataset(path)
    items = data if isinstance(data, list) else data.get("examples", data)
    out = []
    for i, item in enumerate(items):
        item = dict(item)
        item["_uid"] = R.item_uid(item, i)
        raw = item.get("audio_path")
        item["_audio"] = R.resolve_audio_path(raw, path) if raw else None
        out.append(item)

    if require_audio:
        need = [i for i in out if i.get("modality") in ("audio_only", "text_audio")]
        missing = [i for i in need if not i["_audio"]]
        if missing:
            raise SystemExit(
                "{} of {} audio-bearing items have no audio on disk (e.g. {}).\n"
                "dataset_audio/ is gitignored -- put the corpus in place first."
                .format(len(missing), len(need), missing[0].get("audio_path")))
    return out


def select(items, category=None, condition=None, modality=None, order=None):
    """Filter by any combination of fields. None means 'don't care'."""
    def keep(i):
        return ((category is None or i.get("category") == category)
                and (condition is None or i.get("condition") == condition)
                and (modality is None or i.get("modality") == modality)
                and (order is None or i.get("order") == order))
    return [i for i in items if keep(i)]


def option_map(item):
    return R.build_option_map(item)


def correct_letter(item):
    return item.get("correct_letter")


def claimed_letter(item):
    """Letter matching the text's claim, for TDR. None when there is no claim.

    CLAIMED_KEY_CANDIDATES is keyed by category and holds *field names*, so it
    has to be indexed by category first and then read off the item -- the same
    two steps run_qwen2audio_eval does.
    """
    candidates = R.CLAIMED_KEY_CANDIDATES.get(item.get("category"), [])
    key = R.first_present_key(item, candidates)
    if key is None:
        return None
    value = item.get(key)
    if value is None:
        return None
    return R.resolve_value_to_letter(option_map(item), value)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(eager=False, four_bit=False, verbose=True):
    """Load Qwen2-Audio. Returns (model, processor, info).

    eager    -- required for attention weights and for gradients through them;
                sdpa returns an EMPTY attentions tuple with no error.
    four_bit -- the model is 16.8 GB in fp16 and a T4 has 15 GB. Quantization
                perturbs activations, so `info` records it and every result
                file should carry that through.
    """
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    kwargs = {"low_cpu_mem_usage": True,
              "device_map": "auto" if torch.cuda.is_available() else None}
    if eager:
        kwargs["attn_implementation"] = "eager"
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()

    info = {"model_path": MODEL_PATH, "dtype": str(model.dtype),
            "quantization": "4bit" if four_bit else "fp16",
            "attn_implementation": "eager" if eager else "default",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if verbose:
        print("loaded {} [{}, {}]".format(MODEL_PATH, info["quantization"],
                                          info["attn_implementation"]))
    return model, processor, info


def audio_tower(model):
    """Whisper encoder. Newer transformers nests it under model.model."""
    return getattr(model, "audio_tower", None) or model.model.audio_tower


def projector(model):
    """Encoder d_model -> LM hidden size."""
    return (getattr(model, "multi_modal_projector", None)
            or model.model.multi_modal_projector)


def decoder_layers(model):
    import torch
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "language_model.layers", "model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList) and len(obj):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


# --------------------------------------------------------------------------- #
# encoding -- must match run_qwen2audio_eval's prompt construction exactly
# --------------------------------------------------------------------------- #
def encode(model, processor, item, fixed_seconds=None):
    """Model inputs for one item, built the same way the main eval builds them.

    fixed_seconds pads/trims the waveform so every clip yields the same number
    of audio tokens. Patching needs that: otherwise position i means different
    things in the clean and corrupt runs.
    """
    import numpy as np
    import torch

    prompt = item["prompt"]
    has_audio = item.get("modality") in ("audio_only", "text_audio")

    if has_audio:
        import librosa
        path = item.get("_audio") or item.get("audio_path")
        sr = processor.feature_extractor.sampling_rate
        wav, _ = librosa.load(path, sr=sr)
        if fixed_seconds is not None:
            want = int(fixed_seconds * sr)
            wav = wav[:want] if len(wav) >= want else np.pad(wav, (0, want - len(wav)))
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "audio", "audio_url": path},
                                         {"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                             tokenize=False)
        inputs = R._processor_call(processor, text, [wav], sr)
    else:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                             tokenize=False)
        inputs = R._processor_call(processor, text, None, None)

    device = next(model.parameters()).device
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(device)
            if v.dtype in (torch.float32, torch.float64):
                v = v.to(model.dtype)
        moved[k] = v
    return moved


def audio_span(inputs, audio_token_id):
    """(first, last) index of the audio tokens inside input_ids."""
    pos = (inputs["input_ids"][0] == audio_token_id).nonzero().flatten()
    if not len(pos):
        return None
    return int(pos[0]), int(pos[-1])


def audio_token_id(model):
    return model.config.audio_token_index


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def letter_token_ids(processor, letters):
    """Token ids for each option letter, bare and space-prefixed.

    The chat template ends with 'assistant\\n', so the model emits the bare
    letter; the space-prefixed form is scored too and the larger taken, so a
    template change can't silently zero out the scores.
    """
    tok = processor.tokenizer
    out = {}
    for L in letters:
        ids = []
        for variant in (L, " " + L):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        if not ids:
            raise ValueError("option letter {!r} is not a single token".format(L))
        out[L] = ids
    return out


def score_letters(model, processor, inputs, letters):
    """{letter: log-prob} from a single forward pass."""
    import torch

    ids = letter_token_ids(processor, letters)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return {L: float(max(lp[i] for i in variants)) for L, variants in ids.items()}


def predict_letter(model, processor, inputs, letters):
    scores = score_letters(model, processor, inputs, letters)
    return max(scores, key=scores.get), scores


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #
class ResidualCache:
    """Capture each decoder layer's output hidden states."""

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.acts = {}
        self._handles = []

    def __enter__(self):
        def mk(i):
            def hook(mod, inp, out):
                self.acts[i] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        self._handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


class PatchResidual:
    """Overwrite hidden states at `positions` for the layers in `source`.

    source: {layer_index: tensor of shape (n_positions, hidden)}
    """

    def __init__(self, model, source, positions):
        self.layers = decoder_layers(model)
        self.source = source
        self.lo, self.hi = positions
        self._handles = []

    def __enter__(self):
        def mk(i):
            def hook(mod, inp, out):
                if i not in self.source:
                    return out
                is_tuple = isinstance(out, tuple)
                h = (out[0] if is_tuple else out).clone()
                if h.shape[1] <= self.hi:      # decode step; nothing to patch
                    return out
                h[:, self.lo:self.hi + 1, :] = self.source[i].to(h.dtype).to(h.device)
                return (h,) + out[1:] if is_tuple else h
            return hook
        self._handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """Wilson interval -- n is small here, so the normal approximation lies."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def binom_p(k, n, p0):
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(stats.binomtest(k, n, p0, alternative="two-sided").pvalue)


def accuracy_block(records, chance=None):
    """accuracy + CI + prediction spread. The spread is what exposes a model
    that just answers 'C' every time."""
    n = len(records)
    k = sum(1 for r in records if r["correct"])
    lo, hi = wilson_ci(k, n)
    spread = {}
    for r in records:
        spread[r["prediction"]] = spread.get(r["prediction"], 0) + 1
    out = {"n": n, "correct": k, "accuracy": (k / n) if n else 0.0,
           "ci95": [lo, hi], "prediction_counts": spread}
    if chance is not None and n:
        out["chance"] = chance
        out["p_vs_chance"] = binom_p(k, n, chance)
    return out


def tdr_block(records):
    """Text-dominance on mismatched items.

    'neither' is reported because LISTEN found most conflict answers match
    neither cue; a bare TDR hides that entirely.
    """
    n = len(records)
    if not n:
        return {}
    text = sum(1 for r in records if r.get("claimed") and r["prediction"] == r["claimed"])
    audio = sum(1 for r in records if r["prediction"] == r["correct_letter"])
    return {"n": n, "tdr": text / n, "audio_rate": audio / n,
            "neither": (n - text - audio) / n}


def set_seed(s=0):
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass
