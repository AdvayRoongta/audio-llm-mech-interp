"""Exercise _backend_salmonn's shim + encode against a fake SALMONN.

No GPU, no weights. Verifies exactly the things that would otherwise produce
plausible-but-wrong numbers on a real run: the shadow input_ids layout, the
audio span, the exp6 continuation-scoring path, and the generate() padding.
"""
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, "eval")
import _backend_salmonn as B

VOCAB, HID, NAUD = 200, 16, 88


class FakeTok:
    """SentencePiece-ish: word-boundary prefix, so bare 'A' is one token."""
    bos_token_id = 1

    def _ids(self, text):
        return [10 + (ord(c) % 100) for c in text if not c.isspace()] or [10]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        class O:
            pass
        o = O()
        o.input_ids = torch.tensor([self._ids(text)], dtype=torch.long)
        return o

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class FakeLlama(nn.Module):
    """PEFT layout: .model.model.{layers,embed_tokens}"""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Linear(HID, HID) for _ in range(40)])
        inner.embed_tokens = nn.Embedding(VOCAB, HID)
        wrapper = nn.Module()
        wrapper.model = inner
        self.model = wrapper
        self.head = nn.Linear(HID, VOCAB)

    # STRICT on purpose: no **kw catch-all. encode() returns front-end keys
    # (input_features, feature_attention_mask, _speech_embeds) that are inputs
    # to the audio encoders, not the LM. If the shim ever stops popping one,
    # the real llama_model would raise an unexpected-kwarg TypeError -- so this
    # fake must raise too, or the test silently passes on broken code.
    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=None):
        class Out:
            pass
        o = Out()
        o.logits = self.head(inputs_embeds)
        return o

    def generate(self, inputs_embeds=None, attention_mask=None,
                 max_new_tokens=5, do_sample=None):
        # transformers returns ONLY new tokens when driven by inputs_embeds
        b = inputs_embeds.shape[0]
        return torch.arange(50, 50 + max_new_tokens).unsqueeze(0).repeat(b, 1)


class FakeSalmonn:
    lora = True
    window_level_Qformer = True
    second_per_window = 0.333333
    second_stride = 0.333333

    def __init__(self):
        self.llama_model = FakeLlama()
        self.llama_tokenizer = FakeTok()
        self.speech_query_tokens = torch.zeros(1, 1, 768)
        self.speech_encoder = nn.Identity()
        self.beats = nn.Identity()
        self.speech_llama_proj = nn.Linear(768, HID)

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        return torch.randn(1, NAUD, HID), torch.ones(1, NAUD, dtype=torch.long)

    def parameters(self):
        return self.llama_model.parameters()

    def eval(self):
        return self

    def zero_grad(self, set_to_none=True):
        self.zeroed = True


class FakeFE:
    sampling_rate = 16000

    def __call__(self, wav, sampling_rate=None, return_tensors=None):
        return {"input_features": torch.randn(1, 80, 3000)}


def main():
    salmonn = FakeSalmonn()
    model = B.SalmonnShim(salmonn, salmonn.llama_tokenizer)
    proc = B.SalmonnProcessor(salmonn.llama_tokenizer, FakeFE())

    fails = []

    def check(name, cond, detail=""):
        print("{:<52} {}".format(name, "OK" if cond else "FAIL " + detail))
        if not cond:
            fails.append(name)

    check("n_audio_tokens == 88", model.n_audio == 88, str(model.n_audio))
    check("decoder_layers finds 40 (PEFT layout)",
          len(B.decoder_layers(model)) == 40)

    item = {"modality": "text_only", "prompt": "The speaker sounds happy. Which emotion?"}
    inputs = B.encode(model, proc, item)

    seq = inputs["input_ids"].shape[1]
    check("embeds and shadow input_ids same length",
          inputs["inputs_embeds"].shape[1] == seq)
    check("attention_mask same length", inputs["attention_mask"].shape[1] == seq)
    check("encode exposes exp2's encoder-probe keys",
          {"input_features", "feature_attention_mask", "_speech_embeds"} <= set(inputs))
    check("feature mask marks only valid mel frames",
          0 < int(inputs["feature_attention_mask"].sum()) < inputs["input_features"].shape[-1],
          str(int(inputs["feature_attention_mask"].sum())))

    span = B.audio_span(inputs, B.audio_token_id(model))
    check("audio span width == 88", span[1] - span[0] + 1 == 88, str(span))

    ids = inputs["input_ids"][0].tolist()
    check("shadow starts with BOS", ids[0] == FakeTok.bos_token_id)
    check("sentinels are exactly the audio block",
          ids.count(B.SENTINEL) == 88
          and all(v == B.SENTINEL for v in ids[span[0]:span[1] + 1]))
    check("no sentinel outside the span",
          B.SENTINEL not in ids[:span[0]] and B.SENTINEL not in ids[span[1] + 1:])

    # find_span must locate real text and never wander into the audio block
    got = B.find_span(proc, inputs, "Which emotion?")
    check("find_span locates a text substring", got is not None, str(got))
    if got:
        check("find_span result is outside the audio span",
              got[0] > span[1] or got[1] < span[0], str(got))

    # forward through the shim
    out = model(**inputs, use_cache=False)
    check("forward returns logits (1,S,V)",
          tuple(out.logits.shape) == (1, seq, VOCAB), str(tuple(out.logits.shape)))

    scores = B.score_letters(model, proc, inputs, ["A", "B", "C"])
    check("score_letters returns a logprob per letter",
          set(scores) == {"A", "B", "C"} and all(v < 0 for v in scores.values()),
          str(scores))

    # exp6 continuation: input_ids longer than the cached embeddings
    new_tokens = torch.tensor([[60, 61, 62]])
    full = torch.cat([inputs["input_ids"], new_tokens], dim=1)
    kw = dict(inputs)
    kw["input_ids"] = full
    kw["attention_mask"] = torch.ones_like(full)
    out2 = model(**kw, use_cache=False)
    check("exp6 continuation extends the sequence by 3",
          out2.logits.shape[1] == seq + 3, str(out2.logits.shape))

    # a sentinel past the prompt must be refused, not silently embedded
    bad = torch.cat([inputs["input_ids"], torch.tensor([[B.SENTINEL]])], dim=1)
    kwb = dict(inputs)
    kwb["input_ids"] = bad
    try:
        model(**kwb)
        check("sentinel past the prompt is refused", False)
    except ValueError:
        check("sentinel past the prompt is refused", True)

    # exp3's attribution patching calls model.zero_grad() before each backward
    model.zero_grad(set_to_none=True)
    check("zero_grad reaches the wrapped model",
          getattr(salmonn, "zeroed", False))
    check("__getattr__ falls through for undefined attrs",
          model.lora is True and model.second_stride == FakeSalmonn.second_stride)

    # generate(): exp6 slices out[0, prompt_len:] and must get the new tokens
    gen = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    sliced = gen[0, inputs["input_ids"].shape[1]:]
    check("generate() output is prompt_len + new", gen.shape[1] == seq + 5,
          str(gen.shape))
    check("exp6's slice recovers exactly the 5 new tokens",
          sliced.tolist() == list(range(50, 55)), str(sliced.tolist()))

    # fixed_seconds must not change the audio token count
    i2 = B.encode(model, proc, dict(item), fixed_seconds=3.0)
    s2 = B.audio_span(i2, B.audio_token_id(model))
    check("fixed_seconds leaves the span width at 88",
          s2[1] - s2[0] + 1 == 88, str(s2))

    print("\n{} checks, {} failures".format(20, len(fails)))
    if fails:
        print("FAILED:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
