"""Shared plumbing for the mechanistic experiments (exp1-exp6).

These experiments ask *why* the model behaves the way the per-model eval
scripts measure, so they deliberately reuse those scripts' definitions of an
item, a prompt, an option map and a correct answer. Anything imported from the
backend's `R` below is there so the two never drift apart.

Two deliberate differences from the behavioural eval scripts:

  * Scoring is by option log-prob, not by generating text and parsing a letter.
    Letters are single tokens, so one forward scores every option, nothing can
    fail to parse, and we get a graded score per option rather than a hard
    label -- which is what the patching experiments need.

  * The model can be loaded with eager attention (for attention weights and
    gradients) and optionally in 4-bit (to fit a 15 GB T4). The main eval
    asserts fp16 on purpose; here it is a flag, and every result records which
    was used. (The SALMONN backend refuses 4-bit outright.)

Two backends, selected by the MECH_BACKEND environment variable:

    qwen     (default)  Qwen2-Audio-7B-Instruct   -- _backend_qwen.py
    salmonn             SALMONN-13B               -- _backend_salmonn.py

Everything model-specific lives in those modules. What stays here is what does
not depend on the model: dataset loading, option/label resolution, the residual
hooks (which need only a layer list), and the statistics. exp1-exp6 import this
module and nothing else, so adding a third model means adding a backend, not
touching an experiment.

Run from the repo root:  python eval/exp1_capability_screen.py
"""
import importlib
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND = os.environ.get("MECH_BACKEND", "qwen").strip().lower()
_BACKENDS = {"qwen": "_backend_qwen", "salmonn": "_backend_salmonn"}
if BACKEND not in _BACKENDS:
    raise SystemExit("MECH_BACKEND={!r} is not one of {}"
                     .format(BACKEND, ", ".join(sorted(_BACKENDS))))

_B = importlib.import_module(_BACKENDS[BACKEND])
R = _B.R

DEFAULT_DATASET = "data/combined_examples.json"
MODEL_PATH = _B.MODEL_PATH

SYSTEM_PROMPT = _B.SYSTEM_PROMPT
CATEGORIES = _B.CATEGORIES
CONDITIONS = _B.CONDITIONS
MODALITIES = _B.MODALITIES


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def load_items(path=DEFAULT_DATASET, require_audio=False, categories=None):
    """Items with audio_path resolved to something that exists on disk.

    dataset_audio/ is gitignored, so a corpus is often only partly present --
    RAVDESS is redistributable and MUStARD is not, so in practice the two
    RAVDESS categories are available and sarcasm_pacing is not.

    require_audio therefore *drops* audio-bearing items whose file is missing
    and reports what went, rather than refusing to run. It only aborts when
    nothing usable is left, so a partial corpus still lets you evaluate the
    categories you do have. Nothing is dropped silently.
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

    if categories:
        out = [i for i in out if i.get("category") in categories]

    if not require_audio:
        return out

    def needs_audio(i):
        return i.get("modality") in ("audio_only", "text_audio")

    missing = [i for i in out if needs_audio(i) and not i["_audio"]]
    if missing:
        by_cat = {}
        for i in missing:
            by_cat[i.get("category")] = by_cat.get(i.get("category"), 0) + 1
        print("WARNING: dropping {} items with no audio on disk:".format(len(missing)))
        for cat, n in sorted(by_cat.items()):
            total = sum(1 for i in out if i.get("category") == cat and needs_audio(i))
            print("   {:<20} {}/{} audio items missing".format(cat, n, total))
        print("   example: {}".format(missing[0].get("audio_path")))
        out = [i for i in out if not (needs_audio(i) and not i["_audio"])]

    if not any(needs_audio(i) for i in out):
        raise SystemExit(
            "No audio-bearing item has a file on disk. dataset_audio/ is "
            "gitignored -- put the corpus in place first.")
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
    two steps the behavioural eval does.
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
# model -- delegated to the active backend
# --------------------------------------------------------------------------- #
def load_model(eager=False, four_bit=False, verbose=True):
    """(model, processor, info). `info` becomes the "run" block of every
    results/*.json, so its keys are the same whichever backend is active."""
    return _B.load_model(eager=eager, four_bit=four_bit, verbose=verbose)


def audio_tower(model):
    return _B.audio_tower(model)


def projector(model):
    return _B.projector(model)


def decoder_layers(model):
    return _B.decoder_layers(model)


def projected_audio(model, processor, inputs, n_valid_frames):
    """Mean-pooled vector the language model actually receives for this clip.

    exp2 compares encoder-last against this to test whether the projection
    stage loses paralinguistic signal. What counts as "the projection stage"
    is architecture-specific, so each backend defines it.
    """
    return _B.projected_audio(model, processor, inputs, n_valid_frames)


def encode(model, processor, item, fixed_seconds=None):
    """Model inputs for one item, built the way that model's eval builds them.

    fixed_seconds pads/trims the waveform so every clip yields the same number
    of audio tokens. Patching needs that: otherwise position i means different
    things in the clean and corrupt runs. (SALMONN's audio token count is
    already constant, but the flag is still honoured there for parity.)
    """
    return _B.encode(model, processor, item, fixed_seconds=fixed_seconds)


def audio_span(inputs, audio_token_id_):
    """(first, last) index of the audio tokens in the sequence."""
    return _B.audio_span(inputs, audio_token_id_)


def audio_token_id(model):
    return _B.audio_token_id(model)


def find_span(processor, inputs, text):
    """Token span (first, last) of a prompt substring, or None."""
    return _B.find_span(processor, inputs, text)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def letter_token_ids(processor, letters):
    return _B.letter_token_ids(processor, letters)


def score_letters(model, processor, inputs, letters):
    """{letter: log-prob} from a single forward pass."""
    return _B.score_letters(model, processor, inputs, letters)


def predict_letter(model, processor, inputs, letters):
    scores = score_letters(model, processor, inputs, letters)
    return max(scores, key=scores.get), scores


# --------------------------------------------------------------------------- #
# hooks -- model-agnostic: they need a layer list and nothing else
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
