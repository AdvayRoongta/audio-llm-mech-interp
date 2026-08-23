"""Qwen2-Audio backend for the mechanistic experiments.

Every function here was moved verbatim out of _common.py when the backend split
happened; nothing about the Qwen path changed. The regression test is that
exp1-exp6 reproduce their existing results/*.json byte-for-byte.

_common.py dispatches to this module by default (MECH_BACKEND=qwen).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_qwen2audio_eval as R  # noqa: E402  (same directory)

MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

SYSTEM_PROMPT = R.SYSTEM_PROMPT
CATEGORIES = R.CATEGORIES
CONDITIONS = R.CONDITIONS
MODALITIES = R.MODALITIES


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


def projected_audio(model, processor, inputs, n_valid_frames):
    """Mean-pooled post-projector vector -- what the LM actually receives.

    Qwen has one encoder feeding one projector, so this is literally
    projector(audio_tower(feats)), pooled over the valid frames. Factored out
    of exp2 so the same question can be asked of an architecture whose
    projector does not sit directly on the encoder (see the SALMONN backend).
    """
    import torch

    tower, proj = audio_tower(model), projector(model)
    feats = inputs["input_features"].to(model.dtype)
    with torch.no_grad():
        out = tower(feats, output_hidden_states=False)
        projected = proj(out.last_hidden_state)
    n = max(1, min(n_valid_frames, projected.shape[1]))
    return projected[0, :n].float().mean(0).cpu().numpy()


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


def audio_span(inputs, audio_token_id_):
    """(first, last) index of the audio tokens inside input_ids."""
    pos = (inputs["input_ids"][0] == audio_token_id_).nonzero().flatten()
    if not len(pos):
        return None
    return int(pos[0]), int(pos[-1])


def audio_token_id(model):
    return model.config.audio_token_index


def find_span(processor, inputs, text):
    """Token span (first, last) of a prompt substring, or None.

    Tokenizing a fragment in isolation can differ from its tokenization in
    context at the edges -- a leading space merges, a trailing period splits --
    so this retries progressively trimmed variants rather than giving up on the
    first miss. Returns None instead of guessing when nothing matches.
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
