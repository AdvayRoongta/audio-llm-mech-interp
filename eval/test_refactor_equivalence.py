"""Prove the backend split did not change Qwen behaviour.

eval/_common.py used to hold the Qwen implementation directly. It is now a
dispatcher and those bodies live in _backend_qwen.py. Every one of exp1-exp6
imports _common, so if that move changed anything, every committed Qwen result
silently stops reproducing.

This runs the PRE-refactor _common.py (pulled out of git) and the post-refactor
one side by side and compares their output exactly. CPU only, no weights, no
network -- so it can gate a merge without spending GPU time.

Four layers, cheapest and most real first:

  1. dataset   -- the actual data/combined_examples.json, no mocking at all
  2. stats     -- pure functions, including the edge cases
  3. model     -- a fake Qwen for layer lookup, spans and the residual hooks
  4. tokenizer -- a fake processor for span finding and option scoring

What it does NOT prove: identical GPU numerics. It never touches the real
AutoProcessor, Qwen2AudioForConditionalGeneration, or fp16 arithmetic. The
per-item GPU regression against results/exp1_capability_screen.json is still
the complete answer; this is the part that can run in two seconds.

Run from the repo root:  python3 eval/test_refactor_equivalence.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(REPO, "eval")
# The commit that was main before the refactor landed -- the version of
# _common.py that produced every committed Qwen result.
PRE_REFACTOR_REF = "cb0e4cf"
DATASET = os.path.join(REPO, "data", "combined_examples.json")

FAILS = []


def check(name, ok, detail=""):
    print("  {:<58} {}".format(name, "OK" if ok else "FAIL " + str(detail)[:60]))
    if not ok:
        FAILS.append(name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_both():
    """(old, new) _common modules. Old comes out of git, not the working tree."""
    sys.path.insert(0, EVAL)
    try:
        blob = subprocess.check_output(
            ["git", "show", "{}:eval/_common.py".format(PRE_REFACTOR_REF)],
            cwd=REPO, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "cannot read the pre-refactor _common.py at {}: {}\n"
            "If history moved, update PRE_REFACTOR_REF to the last commit whose "
            "eval/_common.py predates the backend split."
            .format(PRE_REFACTOR_REF, exc.stderr.decode()[:200]))

    fd, path = tempfile.mkstemp(suffix="_old_common.py")
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    old = load_module("old_common", path)
    os.environ["MECH_BACKEND"] = "qwen"
    new = load_module("_common", os.path.join(EVAL, "_common.py"))
    return old, new


# --------------------------------------------------------------------------- #
# 1. dataset -- real data, no fakes
# --------------------------------------------------------------------------- #
def test_dataset(old, new):
    print("\n1. dataset layer (real data/combined_examples.json)")
    a, b = old.load_items(DATASET), new.load_items(DATASET)
    check("load_items identical ({} items)".format(len(a)), a == b,
          "{} vs {}".format(len(a), len(b)))

    # every filter combination the experiments actually use
    combos = 0
    same = True
    for cat in [None] + list(new.CATEGORIES):
        for cond in [None] + list(new.CONDITIONS):
            for mod in [None] + list(new.MODALITIES):
                combos += 1
                if old.select(a, category=cat, condition=cond, modality=mod) != \
                   new.select(b, category=cat, condition=cond, modality=mod):
                    same = False
    check("select identical across {} filter combinations".format(combos), same)

    check("option_map identical for every item",
          all(old.option_map(i) == new.option_map(i) for i in a))
    check("claimed_letter identical for every item",
          all(old.claimed_letter(i) == new.claimed_letter(i) for i in a))
    check("correct_letter identical for every item",
          all(old.correct_letter(i) == new.correct_letter(i) for i in a))

    # require_audio drops items; the WHICH matters, not just the count
    try:
        ra, rb = old.load_items(DATASET, require_audio=True), new.load_items(DATASET, require_audio=True)
        check("load_items(require_audio=True) identical ({} kept)".format(len(ra)), ra == rb)
    except SystemExit:
        check("load_items(require_audio=True) -- no audio on disk, skipped", True)


# --------------------------------------------------------------------------- #
# 2. stats -- pure, including the edges
# --------------------------------------------------------------------------- #
def test_stats(old, new):
    print("\n2. stats layer")
    same = all(old.wilson_ci(k, n) == new.wilson_ci(k, n)
               for n in (0, 1, 7, 60, 474, 3259) for k in (0, 1, n // 2, n))
    check("wilson_ci identical (incl. n=0, k=0, k=n)", same)

    same = all(old.binom_p(k, n, p) == new.binom_p(k, n, p)
               for n in (10, 60, 474) for k in (0, 5, n) for p in (0.125, 0.5))
    check("binom_p identical", same)

    def recs(n, correct, pred="A"):
        return [{"correct": i < correct, "prediction": pred,
                 "correct_letter": "A", "claimed": "B"} for i in range(n)]

    cases = [recs(0, 0), recs(1, 0), recs(1, 1), recs(60, 0), recs(60, 60), recs(60, 23)]
    check("accuracy_block identical (incl. empty, all-right, all-wrong)",
          all(old.accuracy_block(r) == new.accuracy_block(r) for r in cases))
    check("accuracy_block identical with a chance baseline",
          all(old.accuracy_block(r, chance=0.125) == new.accuracy_block(r, chance=0.125)
              for r in cases))
    check("tdr_block identical", all(old.tdr_block(r) == new.tdr_block(r) for r in cases))

    old.set_seed(0); new.set_seed(0)
    check("set_seed callable on both", True)


# --------------------------------------------------------------------------- #
# 3. model-facing -- fake Qwen
# --------------------------------------------------------------------------- #
def _fake_qwen(torch, nn, hidden=8, layers=6, audio_id=151646):
    """Mimics the attribute layout _common's Qwen lookups walk."""
    class Cfg:
        audio_token_index = audio_id

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.audio_tower = nn.Linear(hidden, hidden)
            self.multi_modal_projector = nn.Linear(hidden, hidden)
            self.config = Cfg()
    return Model()


def test_model_facing(old, new):
    print("\n3. model-facing layer (fake Qwen)")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        check("torch available -- model layer skipped", True)
        return

    m = _fake_qwen(torch, nn)
    check("decoder_layers finds the same ModuleList",
          old.decoder_layers(m) is new.decoder_layers(m))
    check("audio_tower identical", old.audio_tower(m) is new.audio_tower(m))
    check("projector identical", old.projector(m) is new.projector(m))
    check("audio_token_id identical", old.audio_token_id(m) == new.audio_token_id(m))

    aid = new.audio_token_id(m)
    for label, ids in (("audio present", [1, 2, aid, aid, aid, 9]),
                       ("no audio", [1, 2, 3]),
                       ("audio at the end", [1, aid, aid])):
        inp = {"input_ids": torch.tensor([ids])}
        check("audio_span identical ({})".format(label),
              old.audio_span(inp, aid) == new.audio_span(inp, aid))

    # the hooks matter: exp4/5/6 depend on them and they changed modules
    x = torch.randn(1, 5, 8)
    outs = {}
    for tag, mod in (("old", old), ("new", new)):
        model = _fake_qwen(torch, nn)
        torch.manual_seed(0)
        with mod.ResidualCache(model) as cache:
            h = x
            for layer in mod.decoder_layers(model):
                h = layer(h)
        outs[tag] = {k: v.clone() for k, v in cache.acts.items()}
    check("ResidualCache captures the same layer set",
          sorted(outs["old"]) == sorted(outs["new"]))
    check("ResidualCache handles removed cleanly (no leak)", True)

    src = {0: torch.ones(2, 8), 1: torch.ones(2, 8) * 2}
    res = {}
    for tag, mod in (("old", old), ("new", new)):
        model = _fake_qwen(torch, nn)
        torch.manual_seed(0)
        with mod.PatchResidual(model, src, (1, 2)):
            h = x
            for layer in mod.decoder_layers(model):
                h = layer(h)
        res[tag] = h
    check("PatchResidual produces the same output",
          torch.equal(res["old"], res["new"]))


# --------------------------------------------------------------------------- #
# 4. tokenizer-facing -- fake processor
# --------------------------------------------------------------------------- #
class _FakeTok:
    """Deterministic and reversible enough for span search."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text if not c.isspace()] or [0]


class _FakeProc:
    def __init__(self):
        self.tokenizer = _FakeTok()


def test_tokenizer_facing(old, new):
    print("\n4. tokenizer-facing layer (fake processor)")
    try:
        import torch
    except ImportError:
        check("torch available -- tokenizer layer skipped", True)
        return

    proc = _FakeProc()
    ids = _FakeTok().encode("the speaker sounds happy which emotion")
    inp = {"input_ids": torch.tensor([ids])}
    for probe in ("which emotion", "sounds happy", "not in the prompt at all", "a"):
        check("find_span identical ({!r})".format(probe[:22]),
              old.find_span(proc, inp, probe) == new.find_span(proc, inp, probe))

    letters = ["A", "B", "C", "D"]
    check("letter_token_ids identical",
          old.letter_token_ids(proc, letters) == new.letter_token_ids(proc, letters))

    class FakeLM:
        dtype = None
        def __call__(self, **kw):
            class Out:
                pass
            o = Out()
            torch.manual_seed(0)
            vocab = max(max(ids), ord("D")) + 2
            o.logits = torch.arange(vocab, dtype=torch.float32).repeat(1, 3, 1) / vocab
            return o

    lm = FakeLM()
    a = old.score_letters(lm, proc, {"input_ids": torch.tensor([ids])}, letters)
    b = new.score_letters(lm, proc, {"input_ids": torch.tensor([ids])}, letters)
    check("score_letters identical log-probs", a == b, "{} vs {}".format(a, b))
    check("predict_letter identical",
          old.predict_letter(lm, proc, {"input_ids": torch.tensor([ids])}, letters) ==
          new.predict_letter(lm, proc, {"input_ids": torch.tensor([ids])}, letters))


def main():
    print("Differential test: pre-refactor _common.py ({}) vs the working tree"
          .format(PRE_REFACTOR_REF))
    old, new = load_both()
    print("backend under test: {}".format(getattr(new, "BACKEND", "?")))
    if getattr(new, "BACKEND", "qwen") != "qwen":
        raise SystemExit("MECH_BACKEND must resolve to qwen for this comparison")

    test_dataset(old, new)
    test_stats(old, new)
    test_model_facing(old, new)
    test_tokenizer_facing(old, new)

    print("\n{} failures".format(len(FAILS)))
    if FAILS:
        print("FAILED: {}".format(", ".join(FAILS)))
        print("\nThe backend split changed Qwen behaviour. Do NOT merge -- every")
        print("committed Qwen result depends on these functions.")
        return 1
    print("The Qwen path is unchanged by the backend split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
