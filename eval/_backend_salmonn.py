"""SALMONN-13B backend for the mechanistic experiments (exp1-exp6).

Selected with MECH_BACKEND=salmonn. exp1-exp6 are NOT modified: they keep
calling C.load_model / C.encode / C.decoder_layers / C.audio_span /
C.score_letters and `model(**inputs)`, and this module makes SALMONN answer to
that shape.

Model loading, the prompt, and the audio pipeline are taken from
run_salmonn_eval.py rather than reinvented, so a mech-interp number and the
2026-08-14 behavioural number describe the same model on the same input.

Three things differ from Qwen2-Audio and drive the whole design:

1. SALMONN has no audio placeholder token. Qwen splices audio into `input_ids`
   at `audio_token_index`; SALMONN concatenates *embeddings*:

       [BOS] [p_before] [speech x N] [p_after]

   built by prompt_wrap() splitting the prompt on "<SpeechHere>". So there is
   no input_ids to search, and `model(**inputs)` has to become a call on
   inputs_embeds.

   But exp3 (:80, :405, :446) and exp6 (:163, :185, :304, :318) all read
   inputs["input_ids"] for sequence length and span arithmetic. Rather than
   patch six scripts, encode() returns a SHADOW input_ids alongside the
   embeddings: real Vicuna ids at the text positions, SENTINEL repeated across
   the N audio positions. find_span, audio_span and exp3's position_groups
   then work unchanged, and the shim never embeds the shadow.

2. The audio span is a FIXED width. With window_level_Qformer=True,
   second_per_window == second_stride == 0.333333, and Whisper always padding
   to 1500 frames, kernel = stride = round(1500*0.333333/30) = 17, so
   L = (1500-17)//17 + 1 = 88 windows, times num_speech_query_token=1
   => 88 audio tokens for EVERY clip regardless of duration. Qwen needs
   fixed_seconds to align clean/corrupt pairs; here alignment is free. The
   count is still computed from the model's own config and asserted against a
   real forward at load time, never hard-coded.

3. Vicuna-13B has 40 decoder layers / hidden 5120, not Qwen's 32 / 4096, and
   the LLM is PEFT-wrapped (lora=True), so the layer list is at
   llama_model.model.model.layers. Any exp4/5/6 layer argument tuned for Qwen
   (l_src=31) must be rescaled by the caller; nothing here silently remaps it.

Quantization is refused outright, matching run_salmonn_eval.py: no 4-bit, and
low_resource/8-bit stays off.

Artifact paths come from the environment, because C.load_model's signature is
fixed by the experiment scripts:

    SALMONN_REPO          /opt/SALMONN            (bytedance/SALMONN, salmonn branch)
    SALMONN_WHISPER_PATH  openai/whisper-large-v2
    SALMONN_LLAMA_PATH    lmsys/vicuna-13b-v1.1
    SALMONN_BEATS_PATH    /vol/salmonn_artifacts/beats.pt
    SALMONN_CKPT          .../salmonn_v1.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_salmonn_eval as R  # noqa: E402  (same directory)

MODEL_PATH = "tsinghua-ee/SALMONN (SALMONN-13B)"

# run_salmonn_eval.py drives the model with PROMPT_TEMPLATE ("USER: {}\n
# ASSISTANT:") and no system turn -- SALMONN has no system role. Kept empty so
# the mech-interp prompt is identical to the behavioural one.
SYSTEM_PROMPT = ""
CATEGORIES = R.CATEGORIES
CONDITIONS = R.CONDITIONS
MODALITIES = R.MODALITIES

# Stands in for the audio positions in the shadow input_ids. Negative so it can
# never collide with a real token id, and so anything that tries to embed it
# fails loudly instead of quietly resolving to some real word.
SENTINEL = -1

_ENV = {
    "salmonn_repo": ("SALMONN_REPO", "/opt/SALMONN"),
    "whisper_path": ("SALMONN_WHISPER_PATH", "openai/whisper-large-v2"),
    "llama_path": ("SALMONN_LLAMA_PATH", "lmsys/vicuna-13b-v1.1"),
    "beats_path": ("SALMONN_BEATS_PATH", "/vol/salmonn_artifacts/beats.pt"),
    "ckpt": ("SALMONN_CKPT", None),
}


# --------------------------------------------------------------------------- #
# processor -- gives the exps the two attributes they actually use
# --------------------------------------------------------------------------- #
class SalmonnProcessor:
    """`processor.tokenizer` and `processor.feature_extractor`, nothing more.

    exp6 calls processor.tokenizer.decode; find_span and letter_token_ids call
    processor.tokenizer.encode; encode() reads
    processor.feature_extractor.sampling_rate. SALMONN keeps those on two
    separate objects, so this just puts them where the shared code looks.
    """

    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor


# --------------------------------------------------------------------------- #
# the shim -- SALMONN wearing an HF CausalLM's call signature
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


class SalmonnShim:
    """Presents SALMONN's LLM as `model(**inputs) -> .logits`.

    Deliberately not an nn.Module: the exps only ever need parameters(), dtype,
    generate() and __call__, and subclassing nn.Module around an already-built
    module would put the weights in two module trees at once.
    """

    def __init__(self, salmonn, tokenizer):
        self.salmonn = salmonn
        self.llama_model = salmonn.llama_model
        self.tokenizer = tokenizer
        self.sentinel = SENTINEL
        self.n_audio = n_audio_tokens(salmonn)

    # -- plumbing the exps touch ------------------------------------------- #
    def parameters(self):
        return self.salmonn.parameters()

    def eval(self):
        self.salmonn.eval()
        return self

    def zero_grad(self, set_to_none=True):
        """exp3's attribution patching calls this before each backward pass."""
        self.salmonn.zero_grad(set_to_none=set_to_none)

    def __getattr__(self, name):
        """Fall through to the wrapped SALMONN for anything not defined here.

        Only reached when normal lookup fails, so it never shadows the methods
        above. Without it every nn.Module method an experiment happens to need
        (zero_grad was the first) is a separate crash on a loaded 13B model --
        an expensive way to discover a one-line gap.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            salmonn = object.__getattribute__(self, "salmonn")
        except AttributeError:
            raise AttributeError(name)
        return getattr(salmonn, name)

    @property
    def dtype(self):
        return next(self.llama_model.parameters()).dtype

    @property
    def device(self):
        return next(self.llama_model.parameters()).device

    def embed_tokens(self, ids):
        """LoRA adds a wrapper level, exactly as SALMONN's own code handles it."""
        m = self.llama_model
        return (m.model.model.embed_tokens(ids) if self.salmonn.lora
                else m.model.embed_tokens(ids))

    # -- the two entry points ---------------------------------------------- #
    def _embeds(self, inputs):
        """(inputs_embeds, attention_mask, leftover kwargs).

        Handles exp6's continuation case: freeform_fluency rebuilds input_ids
        as cat([ctx_ids, generated]) and re-scores it (exp6:316-322). When the
        passed input_ids is longer than the cached embeddings, the extra tail
        is real token ids, so embed it and append. The shadow prefix is never
        embedded.
        """
        torch = _torch()
        inputs = dict(inputs)
        embeds = inputs.pop("inputs_embeds", None)
        mask = inputs.pop("attention_mask", None)
        ids = inputs.pop("input_ids", None)

        # Keys encode() carries for exp2's encoder probe. They are inputs to
        # the audio front-end, not to the LM, so they must be dropped before
        # the call -- llama_model() would reject them as unexpected kwargs.
        inputs.pop("input_features", None)
        inputs.pop("feature_attention_mask", None)
        inputs.pop("_speech_embeds", None)

        if embeds is None:
            if ids is None:
                raise ValueError("SalmonnShim needs inputs_embeds or input_ids")
            embeds = self.embed_tokens(ids)
        elif ids is not None and ids.shape[1] > embeds.shape[1]:
            tail = ids[:, embeds.shape[1]:]
            if bool((tail < 0).any()):
                raise ValueError("shadow sentinel found past the prompt; "
                                 "only real token ids can be appended")
            embeds = torch.cat([embeds, self.embed_tokens(tail)], dim=1)

        if mask is None or mask.shape[1] != embeds.shape[1]:
            mask = torch.ones(embeds.shape[:2], dtype=torch.long,
                              device=embeds.device)
        return embeds, mask, inputs

    def __call__(self, **inputs):
        embeds, mask, rest = self._embeds(inputs)
        return self.llama_model(inputs_embeds=embeds, attention_mask=mask, **rest)

    def generate(self, **kw):
        """Same return shape as an HF generate() on input_ids.

        Driven from inputs_embeds with no input_ids, transformers returns ONLY
        the newly generated tokens -- there is no prompt to echo back. But
        exp6:304 does `out[0, inputs["input_ids"].shape[1]:]`, which on that
        return value slices away every generated token and yields nothing. So
        the prompt columns are prepended back as sentinels: the slice then
        lands exactly where exp6 expects, and the padding is never decoded.
        """
        torch = _torch()
        embeds, mask, rest = self._embeds(kw)
        prompt_len = embeds.shape[1]
        rest.pop("use_cache", None)
        out = self.llama_model.generate(inputs_embeds=embeds,
                                        attention_mask=mask, **rest)
        pad = torch.full((out.shape[0], prompt_len), self.sentinel,
                         dtype=out.dtype, device=out.device)
        return torch.cat([pad, out], dim=1)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def n_audio_tokens(salmonn):
    """How many LLM positions one clip becomes. Constant -- see module docstring.

    Derived from the model's own config rather than hard-coded to 88, so a
    different second_per_window in SALMONN_MODEL_DEFAULTS cannot silently
    desynchronise the span from reality. load_model asserts it against a real
    forward.
    """
    per_window = salmonn.speech_query_tokens.shape[1]
    if not getattr(salmonn, "window_level_Qformer", False):
        return per_window
    kernel = round(1500 * salmonn.second_per_window / 30.0)
    stride = round(1500 * salmonn.second_stride / 30.0)
    return ((1500 - kernel) // stride + 1) * per_window


def _artifact_args(device):
    ns = argparse.Namespace(device=device)
    missing = []
    for attr, (env, default) in _ENV.items():
        value = os.environ.get(env, default)
        if not value:
            missing.append(env)
        setattr(ns, attr, value)
    if missing:
        raise SystemExit(
            "SALMONN backend needs these environment variables: {}.\n"
            "modal/modal_app_salmonn.py sets them; if you are running locally, "
            "see eval/run_salmonn_eval.py's docstring for what each artifact is."
            .format(", ".join(missing)))
    if not os.path.isfile(ns.beats_path):
        raise SystemExit(
            "BEATs checkpoint not found at {}. It is distributed on OneDrive "
            "only (link in eval/run_salmonn_eval.py's docstring); download it, "
            "then:\n  modal volume put salmonn-artifacts <path> beats.pt"
            .format(ns.beats_path))
    return ns


def load_model(eager=False, four_bit=False, verbose=True):
    """Load SALMONN-13B. Returns (shim, processor, info).

    `info` carries the same keys the Qwen backend emits, so every results/*.json
    keeps the identical "run" block shape.

    eager    -- accepted and recorded, but a no-op: SALMONN pins
                transformers==4.28.0, whose LlamaAttention has no sdpa path, so
                attention is eager already. Nothing to switch on.
    four_bit -- refused. run_salmonn_eval.py forbids quantization for this model
                (its low_resource/8-bit path included), and mechanistic deltas
                measured on perturbed activations are not comparable to the
                fp16 behavioural run.
    """
    import torch

    if four_bit:
        raise SystemExit(
            "4-bit is refused for SALMONN. Quantization is forbidden for this "
            "model (see eval/run_salmonn_eval.py's load_model) -- run on a GPU "
            "that fits fp16 instead.")

    args = _artifact_args("cuda" if torch.cuda.is_available() else "cpu")
    model, wav_processor = R.load_model(args)
    processor = SalmonnProcessor(model.llama_tokenizer, wav_processor)
    shim = SalmonnShim(model, model.llama_tokenizer)

    layers = decoder_layers(shim)
    info = {
        "model_path": MODEL_PATH,
        "dtype": str(shim.dtype),
        "quantization": "fp16",
        "attn_implementation": "eager",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ckpt": args.ckpt,
        "llama_path": args.llama_path,
        "whisper_path": args.whisper_path,
        "beats_path": args.beats_path,
        "low_resource": False,
        "n_layers": len(layers),
        "n_audio_tokens": shim.n_audio,
    }
    if verbose:
        print("loaded SALMONN-13B [{} layers, {} audio tokens/clip, {}]"
              .format(len(layers), shim.n_audio, shim.dtype))
    _self_check(shim, processor)
    return shim, processor, info


def _self_check(model, processor):
    """Fail at load time, not 40 minutes into a sweep.

    Every one of these is a thing that would otherwise produce plausible-looking
    numbers from a broken setup.
    """
    letters = ["A", "B", "C", "D", "E"]
    try:
        letter_token_ids(processor, letters)
    except ValueError as exc:
        raise SystemExit(
            "Vicuna's SentencePiece tokenizer does not give single tokens for "
            "the option letters ({}). Option scoring would be wrong; fix "
            "letter_token_ids before running.".format(exc))

    layers = decoder_layers(model)
    if len(layers) < 2:
        raise SystemExit("found {} decoder layers -- layer lookup is wrong"
                         .format(len(layers)))
    print("self-check ok: {} layers, letters A-E are single tokens"
          .format(len(layers)))


def audio_tower(model):
    """SALMONN has TWO audio encoders, unlike Qwen's single Whisper tower.

    Returns the Whisper encoder, which is the one exp2 probes, and exposes BEATs
    beside it. exp2 currently probes a single tower; probing the BEATs branch
    and the post-Q-Former output is a separate change.
    """
    return model.salmonn.speech_encoder


def beats_tower(model):
    return getattr(model.salmonn, "beats", None)


def projector(model):
    """Q-Former + linear, encoder width -> Vicuna's 5120."""
    return model.salmonn.speech_llama_proj


def projected_audio(model, processor, inputs, n_valid_frames):
    """Mean-pooled representation the LM actually receives, for exp2.

    This is where SALMONN and Qwen genuinely differ, and the difference is not
    plumbing. Qwen's multi_modal_projector sits directly on its Whisper tower,
    so exp2 could call proj(tower_out). SALMONN's speech_llama_proj sits after
    the Q-Former, which consumes Whisper (1280) and BEATs (768) CONCATENATED to
    2048 -- handing it raw Whisper output is a shape error, and would not mean
    the same thing even if the dimensions lined up.

    So "post-projector" here is the output of the whole audio front-end:
    Whisper + BEATs -> window-level Q-Former -> speech_llama_proj, i.e. the 88
    x 5120 block that gets spliced into the prompt. That is the quantity exp2's
    encoder-last-vs-post-projector comparison is actually about, and it keeps
    the two backends answering the same question.

    n_valid_frames is ignored: the Q-Former already compresses the whole clip
    to a fixed 88 tokens, so there is no padding to pool across -- the Whisper
    padding was consumed upstream. Pooling a prefix here would silently discard
    part of the clip.
    """
    embeds = inputs.get("_speech_embeds")
    if embeds is None:
        raise RuntimeError("encode() did not cache _speech_embeds")
    return embeds[0].float().mean(0).cpu().numpy()


def decoder_layers(model):
    import torch
    salmonn = getattr(model, "salmonn", model)
    llama = salmonn.llama_model
    for path in ("model.model.layers",       # PEFT-wrapped (lora=True)
                 "model.layers",             # plain LlamaForCausalLM
                 "base_model.model.model.layers"):
        obj = llama
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList) and len(obj):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate SALMONN's decoder layers")


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def _waveform(item, sample_rate, fixed_seconds):
    """Mono float32 at sample_rate, matching prepare_one_sample's conditioning.

    Done in memory rather than through run_salmonn_eval's temp-file resample
    path, so fixed_seconds can be applied to the samples themselves.
    text_only items get the same 1s silence the behavioural run uses -- SALMONN
    has no audio-free code path.
    """
    import numpy as np

    has_audio = item.get("modality") in ("audio_only", "text_audio")
    if has_audio:
        import librosa
        path = item.get("_audio") or item.get("audio_path")
        wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    else:
        wav = np.zeros(int(sample_rate * R.SILENCE_DURATION_SECONDS),
                       dtype=np.float32)

    if fixed_seconds is not None:
        want = int(fixed_seconds * sample_rate)
        wav = wav[:want] if len(wav) >= want else np.pad(wav, (0, want - len(wav)))

    # prepare_one_sample: pad to >= 1s, truncate to <= 30s.
    if len(wav) < sample_rate:
        wav = np.pad(wav, (0, sample_rate - len(wav)))
    return np.asarray(wav[: sample_rate * 30], dtype=np.float32)


def encode(model, processor, item, fixed_seconds=None):
    """Build [BOS][p_before][speech xN][p_after] plus the shadow input_ids.

    fixed_seconds is honoured for parity with the Qwen runs, but the audio
    token count does not depend on it -- Whisper pads to 30s and the windowed
    Q-Former emits a constant N either way.
    """
    import torch

    salmonn = model.salmonn
    tok = processor.tokenizer
    sr = processor.feature_extractor.sampling_rate
    device = model.device

    wav = _waveform(item, sr, fixed_seconds)
    spectrogram = processor.feature_extractor(
        wav, sampling_rate=sr, return_tensors="pt")["input_features"].to(device)

    # float64, NOT float32, and this is load-bearing. BEATs.preprocess scales
    # the raw waveform by 2**15 before ta_kaldi.fbank; under SALMONN's
    # maybe_autocast(float16) a float32 input is downcast to fp16, and the
    # power spectrum of a ~7.5e3 signal (~5.7e7) blows past fp16's 65504
    # ceiling -> inf -> NaN, which BEATs' self-attention then spreads across
    # every output position. Measured: 129024/129024 values NaN, while Whisper
    # on the same clip was clean.
    #
    # autocast only downcasts float32, so float64 passes through at full
    # precision. SALMONN's own prepare_one_sample() gets this for free because
    # sf.read returns float64; we load with librosa (float32) so the waveform
    # can be trimmed in memory for fixed_seconds, and have to restore it here.
    # The spectrogram stays float32 -- that is what prepare_one_sample feeds
    # the Whisper extractor too.
    raw_wav = torch.from_numpy(wav.astype("float64")).unsqueeze(0).to(device)
    padding_mask = torch.zeros(len(wav), dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        speech_embeds, _ = salmonn.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=padding_mask)
    n_audio = speech_embeds.shape[1]
    if n_audio != model.n_audio:
        raise RuntimeError(
            "audio token count {} != the {} derived from config -- the span "
            "arithmetic in exp3-exp6 would be wrong".format(n_audio, model.n_audio))

    # Identical wrapping to run_salmonn_eval._run_generate, so the mech-interp
    # prompt is the behavioural prompt.
    prompt = R._ensure_terse_instruction(item["prompt"])
    wrapped = R.PROMPT_TEMPLATE.format(
        "<Speech><SpeechHere></Speech> " + prompt.strip())
    p_before, p_after = wrapped.split("<SpeechHere>")

    before = tok(p_before, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    after = tok(p_after, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    bos = torch.full((1, 1), tok.bos_token_id, dtype=torch.long, device=device)

    embeds = torch.cat([model.embed_tokens(bos),
                        model.embed_tokens(before),
                        speech_embeds.to(model.dtype),
                        model.embed_tokens(after)], dim=1)

    shadow = torch.cat([
        bos,
        before,
        torch.full((1, n_audio), SENTINEL, dtype=torch.long, device=device),
        after,
    ], dim=1)

    # input_features / feature_attention_mask are Qwen's key names for the raw
    # mel input and its valid-frame mask. exp2 reads both to probe the encoder
    # layer by layer, so they are surfaced here under the same names. The
    # Whisper extractor pads every clip to 30 s / 3000 frames, and pooling
    # across that padding washes out a 3 s clip, so the mask has to be real:
    # one frame per 160 samples (10 ms hop), clamped to what the extractor
    # emitted.
    n_valid = min(int(len(wav) // 160), spectrogram.shape[-1])
    feat_mask = torch.zeros((1, spectrogram.shape[-1]), dtype=torch.long, device=device)
    feat_mask[0, :max(1, n_valid)] = 1

    return {
        "inputs_embeds": embeds,
        "attention_mask": torch.ones(embeds.shape[:2], dtype=torch.long, device=device),
        "input_ids": shadow,
        "input_features": spectrogram,
        "feature_attention_mask": feat_mask,
        # What the LM actually receives for this clip, cached so exp2's
        # post-projection probe does not have to re-run the encoders.
        "_speech_embeds": speech_embeds,
    }


def audio_span(inputs, audio_token_id_):
    """(first, last) audio position. Contiguous by construction."""
    pos = (inputs["input_ids"][0] == audio_token_id_).nonzero().flatten()
    if not len(pos):
        return None
    first, last = int(pos[0]), int(pos[-1])
    if last - first + 1 != len(pos):
        raise RuntimeError("audio positions are not contiguous -- encode() is wrong")
    return first, last


def audio_token_id(model):
    return SENTINEL


def find_span(processor, inputs, text):
    """Same contract as the Qwen backend: token span of a prompt substring.

    Operates on the shadow input_ids. The sentinel run never matches a real
    token subsequence, so the audio block can't be found by accident.
    """
    ids = inputs["input_ids"][0].tolist()
    tok = processor.tokenizer
    text = (text or "").strip()
    if len(text) < 2:
        return None
    for variant in (text, " " + text, "\n" + text):
        sub = tok.encode(variant, add_special_tokens=False)
        for lo in (0, 1):
            for hi in (0, 1):
                s = sub[lo:len(sub) - hi] if hi else sub[lo:]
                if len(s) < 2:
                    continue
                for i in range(len(ids) - len(s) + 1):
                    if ids[i:i + len(s)] == s:
                        return i, i + len(s) - 1
    return None


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def letter_token_ids(processor, letters):
    """Token ids per option letter.

    Vicuna's SentencePiece tokenizer is not Qwen's BPE: it prepends a word
    boundary, so "A" already encodes as the single piece "_A" and an explicitly
    space-prefixed " A" usually splits in two. Both are tried and only
    single-token encodings are kept, same as the Qwen backend; _self_check
    turns "neither form is a single token" into a load-time error rather than a
    silently mis-scored sweep.
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
