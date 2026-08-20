"""Experiment 5 -- why did back-patching do nothing?

exp4 applied Cho et al.'s (arXiv:2606.18924) intervention -- cache the audio
span's hidden states at a late layer, write them into an early layer -- and got
0.000 -> 0.000 on emotional_prosody across all nine configurations, plus active
harm on speaker_identity (0.075 -> 0.050, capability 1.000 -> 0.875).

Three explanations survive exp3's attribution data, and they imply different
next interventions, so this experiment separates them rather than assuming one:

  content/depth   The audio span's influence peaks at layer 9 and holds only
                  0.5% of its attribution at layers 29-31. Back-patching sources
                  from 31, i.e. from a spent buffer. Predicts: sourcing from
                  ~L9 should move the needle.

  floor           Cho et al. lifted 0.35 -> 0.46. Our baseline is 0.000. If the
                  intervention amplifies existing audio influence rather than
                  creating it, zero times anything is zero. Predicts: nothing
                  helps, at any source layer.

  norm mismatch   Residual-stream norms grow with depth. Writing L31 states into
                  L0 injects vectors far larger than early layers expect. Cho et
                  al. specify direct replacement with no scaling and we followed
                  that, so our null could be an artifact of scale rather than of
                  content. Predicts: norm-matched patching rescues it.

Three tests, cheapest first:

  A  norm profile      one forward. ||h_L|| over the audio span, per layer.
  B  source sweep      src in {4,9,13,16,20,24,28,31} -> dst 0. Does sourcing
                       from the peak beat sourcing from the end?
  C  norm-matched      the best src from B, rescaled to the destination layer's
                       own norm before writing.

Every test carries exp4's two controls: capability on Matched items must not
drop, and results are reported against the unpatched baseline.

Deliberately subset-sized. Detecting movement away from 0.000 does not need 266
items, and this is a falsification test, not a result.

Run from the repo root:
    python eval/exp5_patch_diagnosis.py --category emotional_prosody
    python eval/exp5_patch_diagnosis.py --category speaker_identity --limit 40
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


# --------------------------------------------------------------------------- #
# patching with an optional norm match
# --------------------------------------------------------------------------- #
class PatchSpan:
    """Overwrite the audio span at chosen layers, optionally norm-matched.

    _common.PatchResidual does the raw version; this adds the rescale so the
    norm-mismatch hypothesis can be tested without touching validated code.

    match_norm rescales each patched token to the L2 norm the destination layer
    already had at that position, so the injected content keeps its direction
    but not its (much larger) late-layer magnitude.
    """

    def __init__(self, model, source, span, match_norm=False):
        self.layers = C.decoder_layers(model)
        self.source = source
        self.lo, self.hi = span
        self.match_norm = match_norm
        self._handles = []

    def __enter__(self):
        import torch

        def mk(i):
            def hook(mod, inp, out):
                if i not in self.source:
                    return out
                is_tuple = isinstance(out, tuple)
                h = (out[0] if is_tuple else out).clone()
                if h.shape[1] <= self.hi:          # decode step, nothing to patch
                    return out
                new = self.source[i].to(h.dtype).to(h.device)
                if self.match_norm:
                    cur = h[0, self.lo:self.hi + 1, :]
                    cur_n = cur.norm(dim=-1, keepdim=True)
                    new_n = new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    new = new * (cur_n / new_n)
                h[:, self.lo:self.hi + 1, :] = new
                return (h,) + out[1:] if is_tuple else h
            return hook

        self._handles = [l.register_forward_hook(mk(i))
                         for i, l in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# one item, many source layers, one cached forward
# --------------------------------------------------------------------------- #
def cache_item(model, processor, item, fixed_seconds):
    """Encode once and cache every layer's hidden states.

    exp4 re-ran the caching forward for every configuration. Caching once per
    item and reusing it across source layers nearly halves the sweep: 1 + k
    forwards instead of 2k.
    """
    import torch

    inputs = C.encode(model, processor, item, fixed_seconds=fixed_seconds)
    span = C.audio_span(inputs, C.audio_token_id(model))
    if span is None:
        return None, None, None
    with C.ResidualCache(model) as cache:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    return inputs, span, cache.acts


def predict_with(model, processor, item, inputs, span, acts,
                 src, dst, window, n_layers, match_norm=False):
    letters = list(C.option_map(item).keys())
    if acts is None or src not in acts:
        return C.predict_letter(model, processor, inputs, letters)[0]
    donor = acts[src][:, span[0]:span[1] + 1, :][0]
    source = {dst + i: donor for i in range(window) if dst + i < n_layers}
    with PatchSpan(model, source, span, match_norm=match_norm):
        return C.predict_letter(model, processor, inputs, letters)[0]


def score(records):
    blk = C.accuracy_block(records)
    lo, hi = blk["ci95"]
    return blk["accuracy"], lo, hi


def evaluate(prepared, predict_fn):
    out = []
    for item, inputs, span, acts in prepared:
        pred = predict_fn(item, inputs, span, acts)
        out.append({"prediction": pred,
                    "correct_letter": C.correct_letter(item),
                    "claimed": C.claimed_letter(item),
                    "correct": pred == C.correct_letter(item)})
    return out


# --------------------------------------------------------------------------- #
# test A -- norm profile
# --------------------------------------------------------------------------- #
def test_a_norms(model, processor, prepared, args):
    print("\n" + "=" * 74)
    print("TEST A -- residual norm over the audio span, by layer")
    print("=" * 74)
    print("If ||h_31|| >> ||h_0||, writing L31 states into L0 is a large")
    print("perturbation regardless of content, and the exp4 null may be scale.\n")

    import torch

    n_layers = len(C.decoder_layers(model))
    sums = {l: 0.0 for l in range(n_layers)}
    n = 0
    for item, inputs, span, acts in prepared[:args.norm_items]:
        if acts is None:
            continue
        for l in range(n_layers):
            h = acts[l][0, span[0]:span[1] + 1, :].float()
            sums[l] += float(h.norm(dim=-1).mean())
        n += 1
    if not n:
        print("  no usable items"); return {}

    prof = {l: sums[l] / n for l in sums}
    base = prof[0] if prof[0] else 1.0
    print("  {:<7}{:>12}{:>12}".format("layer", "mean ||h||", "x L0"))
    for l in sorted(prof):
        if l in (0, 2, 4, 9, 13, 16, 20, 24, 28, n_layers - 1):
            print("  L{:<6}{:>12.1f}{:>12.1f}".format(l, prof[l], prof[l] / base))
    ratio = prof[n_layers - 1] / base
    print("\n  ||h_{}|| / ||h_0|| = {:.1f}x".format(n_layers - 1, ratio))
    if ratio > 5:
        print("  -> large. Raw replacement injects out-of-scale vectors; test C matters.")
    else:
        print("  -> modest. Scale is unlikely to explain the exp4 null on its own.")
    return {"profile": prof, "late_over_early": ratio}


# --------------------------------------------------------------------------- #
# test B -- source-layer sweep
# --------------------------------------------------------------------------- #
def test_b_source_sweep(model, processor, prep_mis, prep_mat, args, n_layers):
    print("\n" + "=" * 74)
    print("TEST B -- does the source layer matter?")
    print("=" * 74)
    print("exp3 put the audio span's peak influence at layer 9 and ~0.5% of it")
    print("at layers 29-31. Cho et al. source from the end. If depth is the")
    print("story, sourcing near the peak should beat sourcing near the end.\n")

    base_mis = evaluate(prep_mis, lambda it, i, s, a:
                        C.predict_letter(model, processor, i,
                                         list(C.option_map(it).keys()))[0])
    base_mat = evaluate(prep_mat, lambda it, i, s, a:
                        C.predict_letter(model, processor, i,
                                         list(C.option_map(it).keys()))[0])
    b_audio, b_lo, b_hi = score(base_mis)
    b_cap, _, _ = score(base_mat)
    b_tdr = C.tdr_block(base_mis)
    print("  baseline audio accuracy {:.3f} [{:.2f},{:.2f}]   capability {:.3f}   TDR {:.3f}"
          .format(b_audio, b_lo, b_hi, b_cap, b_tdr.get("tdr", float("nan"))))

    # Identity control: patching layer d with layer d's own cached output is a
    # no-op. If it does not reproduce the baseline exactly, the patching
    # machinery is wrong and every null below is uninterpretable. This is the
    # same role exp3's causal-only mask control plays.
    ident = evaluate(prep_mis, lambda it, i, s, a: predict_with(
        model, processor, it, i, s, a, args.dst, args.dst, 1, n_layers))
    i_audio, _, _ = score(ident)
    ok = abs(i_audio - b_audio) < 1e-9
    print("\n  identity control (src=dst={}): audio {:.3f} vs baseline {:.3f}  {}"
          .format(args.dst, i_audio, b_audio, "OK" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit(
            "Identity patch did not reproduce the baseline. The hook is not "
            "applying as intended; fix that before reading any result here.")

    print("\n  {:>5}{:>9}{:>16}{:>12}{:>8}".format(
        "src", "audio", "95% CI", "capability", "TDR"))
    rows = []
    for src in args.sources:
        if src == args.dst:
            continue          # that is the identity control, already run
        if src >= n_layers:
            continue
        mis = evaluate(prep_mis, lambda it, i, s, a, S=src: predict_with(
            model, processor, it, i, s, a, S, args.dst, args.window, n_layers))
        mat = evaluate(prep_mat, lambda it, i, s, a, S=src: predict_with(
            model, processor, it, i, s, a, S, args.dst, args.window, n_layers))
        a_, lo, hi = score(mis)
        cap, _, _ = score(mat)
        t = C.tdr_block(mis)
        rows.append({"src": src, "audio": a_, "ci95": [lo, hi],
                     "capability": cap, "tdr": t.get("tdr")})
        print("  {:>5}{:>9.3f}   [{:.2f},{:.2f}]{:>12.3f}{:>8.3f}".format(
            src, a_, lo, hi, cap, t.get("tdr") or float("nan")))

    return {"baseline": {"audio": b_audio, "capability": b_cap,
                         "tdr": b_tdr.get("tdr")},
            "identity_control": {"src": args.dst, "audio": i_audio,
                                 "reproduces_baseline": ok},
            "sweep": rows}


# --------------------------------------------------------------------------- #
# test C -- norm-matched patching
# --------------------------------------------------------------------------- #
def test_c_norm_matched(model, processor, prep_mis, prep_mat, args, n_layers, best_src):
    print("\n" + "=" * 74)
    print("TEST C -- norm-matched patching from src={}".format(best_src))
    print("=" * 74)
    print("Same donor states, rescaled to the destination layer's own norm, so")
    print("direction is preserved and magnitude is not. If this works where raw")
    print("replacement did not, the exp4 null was scale, not content.\n")

    mis = evaluate(prep_mis, lambda it, i, s, a: predict_with(
        model, processor, it, i, s, a, best_src, args.dst, args.window,
        n_layers, match_norm=True))
    mat = evaluate(prep_mat, lambda it, i, s, a: predict_with(
        model, processor, it, i, s, a, best_src, args.dst, args.window,
        n_layers, match_norm=True))
    a_, lo, hi = score(mis)
    cap, _, _ = score(mat)
    t = C.tdr_block(mis)
    print("  audio {:.3f} [{:.2f},{:.2f}]   capability {:.3f}   TDR {:.3f}".format(
        a_, lo, hi, cap, t.get("tdr") or float("nan")))
    return {"src": best_src, "audio": a_, "ci95": [lo, hi],
            "capability": cap, "tdr": t.get("tdr")}


# --------------------------------------------------------------------------- #
def verdict(res_a, res_b, res_c, b_cap):
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    base = res_b["baseline"]["audio"]
    sweep = res_b["sweep"]
    if not sweep:
        print("  no sweep rows"); return "inconclusive"

    best = max(sweep, key=lambda r: r["audio"])
    late = max((r for r in sweep if r["src"] >= 28), key=lambda r: r["audio"], default=None)
    gain = best["audio"] - base

    print("  baseline           {:.3f}".format(base))
    print("  best source (L{})   {:.3f}  ({:+.3f})".format(best["src"], best["audio"], gain))
    if late:
        print("  late source (L{})   {:.3f}   <- what exp4/Cho et al. use".format(
            late["src"], late["audio"]))
    print("  norm-matched       {:.3f}  ({:+.3f})".format(
        res_c["audio"], res_c["audio"] - base))

    broke = best["capability"] < b_cap - 0.05
    helped = gain > 0.05
    norm_helped = (res_c["audio"] - base) > 0.05

    print()
    if broke and helped:
        out = ("AMBIGUOUS: the best source gained {:+.3f} but capability fell "
               "{:.3f} -> {:.3f}. A gain bought by damaging the model is not a fix."
               .format(gain, b_cap, best["capability"]))
    elif helped and late and best["audio"] > late["audio"] + 0.05:
        out = ("DEPTH: sourcing near the attribution peak (L{}) beats sourcing from "
               "the end (L{}). exp4's null was aiming at a spent buffer, not a "
               "property of paralinguistic content.".format(best["src"], late["src"]))
    elif helped:
        out = ("SOMETHING MOVES, but not specifically from the peak -- the gain is "
               "not concentrated at the layer exp3 predicted.")
    elif norm_helped:
        out = ("SCALE: raw replacement failed and norm-matched replacement did not. "
               "Cho et al.'s method needs rescaling for this model, and the exp4 "
               "null was an artifact.")
    else:
        out = ("FLOOR / READOUT: no source layer and no rescaling moves it. "
               "Availability is not the lever. The remaining target is the text "
               "circuit at the generation position, where exp3 measured a 16:1 "
               "text-to-audio attribution ratio.")
    print("  " + out)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", default="emotional_prosody", choices=C.CATEGORIES)
    ap.add_argument("--limit", type=int, default=60,
                    help="items per condition; 60 is enough to see movement off a floor")
    ap.add_argument("--sources", type=int, nargs="+",
                    default=[4, 9, 13, 16, 20, 24, 28, 31])
    ap.add_argument("--dst", type=int, default=0)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--fixed-seconds", type=float, default=3.0)
    ap.add_argument("--norm-items", type=int, default=10,
                    help="items averaged for the norm profile; it is very stable")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    C.set_seed(args.seed)

    items = C.load_items(args.input, require_audio=True)
    mism = [i for i in C.select(items, category=args.category, condition="Mismatched",
                                modality="text_audio") if i.get("_audio")][:args.limit]
    matched = [i for i in C.select(items, category=args.category, condition="Matched",
                                   modality="text_audio") if i.get("_audio")][:args.limit]
    if not mism:
        raise SystemExit("no mismatched text_audio items for " + args.category)
    print("{}: {} mismatched, {} matched (capability control)".format(
        args.category, len(mism), len(matched)))

    model, processor, info = C.load_model(four_bit=args.four_bit)
    n_layers = len(C.decoder_layers(model))
    print("{} decoder layers".format(n_layers))

    print("\ncaching hidden states once per item ...")
    prep_mis, prep_mat = [], []
    for dest, src_items in ((prep_mis, mism), (prep_mat, matched)):
        for n, it in enumerate(src_items, 1):
            inputs, span, acts = cache_item(model, processor, it, args.fixed_seconds)
            dest.append((it, inputs, span, acts))
            if n % 20 == 0:
                print("  {}/{}".format(n, len(src_items)))

    res_a = test_a_norms(model, processor, prep_mis, args)
    res_b = test_b_source_sweep(model, processor, prep_mis, prep_mat, args, n_layers)
    best_src = max(res_b["sweep"], key=lambda r: r["audio"])["src"] if res_b["sweep"] else 9
    res_c = test_c_norm_matched(model, processor, prep_mis, prep_mat, args,
                                n_layers, best_src)
    text = verdict(res_a, res_b, res_c, res_b["baseline"]["capability"])

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       "exp5_patch_diagnosis_{}.json".format(args.category))
    with open(out, "w") as f:
        json.dump({"run": dict(info, category=args.category, seed=args.seed,
                               limit=args.limit, sources=args.sources,
                               dst=args.dst, window=args.window,
                               fixed_seconds=args.fixed_seconds,
                               n_mismatched=len(mism), n_matched=len(matched)),
                   "test_a_norms": {k: (v if k != "profile" else
                                        {str(a): b for a, b in v.items()})
                                    for k, v in res_a.items()},
                   "test_b_source_sweep": res_b,
                   "test_c_norm_matched": res_c,
                   "verdict": text}, f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
