"""Experiment 4 -- does back-patching transfer to paralinguistic conflicts?

Cho et al. (arXiv:2606.18924) showed the audio information is suppressed, not
erased: deleting the text circuit lifted lexical audio accuracy 0.33 -> 0.82.
But ablation is not a usable fix -- it destroys free-form coherence. Their
working intervention was back-patching: take the audio span's hidden states
from a late layer, write them into an early layer, run once. Lexical audio
accuracy 0.35 -> 0.46 with text capability unharmed.

The rationale is that by the late layers the audio representation has matured
into the text-aligned semantic space, so re-injecting it early gives the model
more opportunities to use it.

Their hyperparameters are settled (late -> early, ~31 -> 0-4, window 1/3/5,
single iteration; iterating hurts). So this does not redo the grid. It tests
whether the effect transfers to paralinguistic conflicts, with a small
confirmatory sweep around their optimum.

Two controls decide whether a gain counts:

  capability   accuracy on Matched items must not drop. If it does, we broke
               the model rather than fixing it, and the audio gain is noise.
  mid-layer    patching from a middle layer instead of a late one must do
               worse. If it does just as well, "late -> early" is not the
               mechanism and any story about matured representations is wrong.

Run from the repo root:
    python eval/exp4_backpatching.py --category emotional_prosody
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def predict_plain(model, processor, item, fixed_seconds):
    letters = list(C.option_map(item).keys())
    inputs = C.encode(model, processor, item, fixed_seconds=fixed_seconds)
    return C.predict_letter(model, processor, inputs, letters)[0]


def predict_backpatched(model, processor, item, src, dst, window, fixed_seconds, n_layers):
    """Predict with the audio span's layer-`src` states written into `dst`."""
    letters = list(C.option_map(item).keys())
    inputs = C.encode(model, processor, item, fixed_seconds=fixed_seconds)
    span = C.audio_span(inputs, C.audio_token_id(model))
    if span is None:
        return C.predict_letter(model, processor, inputs, letters)[0]

    with C.ResidualCache(model) as cache:
        import torch
        with torch.no_grad():
            model(**inputs, use_cache=False)
    late = cache.acts[src][:, span[0]:span[1] + 1, :]

    source = {dst + i: late[0] for i in range(window) if dst + i < n_layers}
    with C.PatchResidual(model, source, span):
        return C.predict_letter(model, processor, inputs, letters)[0]


def evaluate(items, predict_fn):
    records = []
    for it in items:
        pred = predict_fn(it)
        records.append({"uid": it["_uid"], "prediction": pred,
                        "correct_letter": C.correct_letter(it),
                        "claimed": C.claimed_letter(it),
                        "correct": pred == C.correct_letter(it),
                        "n_options": len(C.option_map(it))})
    return records


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", default="emotional_prosody", choices=C.CATEGORIES)
    ap.add_argument("--fixed-seconds", type=float, default=3.0)
    ap.add_argument("--dst", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--limit", type=int, help="cap items per condition")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    C.set_seed(args.seed)

    items = C.load_items(args.input, require_audio=True)
    mismatched = [i for i in C.select(items, category=args.category, condition="Mismatched",
                                      modality="text_audio") if i.get("_audio")]
    matched = [i for i in C.select(items, category=args.category, condition="Matched",
                                   modality="text_audio") if i.get("_audio")]
    if args.limit:
        mismatched, matched = mismatched[:args.limit], matched[:args.limit]
    if not mismatched:
        raise SystemExit("no mismatched text_audio items for " + args.category)
    print("{}: {} mismatched (target), {} matched (capability control)".format(
        args.category, len(mismatched), len(matched)))

    model, processor, info = C.load_model(four_bit=args.four_bit)
    n_layers = len(C.decoder_layers(model))
    src = n_layers - 1
    print("{} decoder layers; source = {}".format(n_layers, src))

    base_mis = evaluate(mismatched, lambda it: predict_plain(model, processor, it,
                                                             args.fixed_seconds))
    base_mat = evaluate(matched, lambda it: predict_plain(model, processor, it,
                                                          args.fixed_seconds))
    b_audio = C.accuracy_block(base_mis)["accuracy"]
    b_cap = C.accuracy_block(base_mat)["accuracy"] if base_mat else float("nan")
    b_tdr = C.tdr_block(base_mis)

    print("\nbaseline audio accuracy (mismatched) {:.3f}".format(b_audio))
    print("baseline capability     (matched)    {:.3f}".format(b_cap))
    print("baseline TDR                         {:.3f}".format(b_tdr["tdr"]))

    print("\n" + "=" * 74)
    print("SWEEP around Cho et al.'s optimum (late -> early, single iteration)")
    print("=" * 74)
    print("{:>4} {:>4} {:>3} {:>9} {:>10} {:>11} {:>7}".format(
        "src", "dst", "w", "audio", "d(audio)", "capability", "TDR"))
    rows = []
    for dst in args.dst:
        for w in args.windows:
            mis = evaluate(mismatched, lambda it, d=dst, ww=w: predict_backpatched(
                model, processor, it, src, d, ww, args.fixed_seconds, n_layers))
            mat = evaluate(matched, lambda it, d=dst, ww=w: predict_backpatched(
                model, processor, it, src, d, ww, args.fixed_seconds, n_layers)) if matched else []
            a = C.accuracy_block(mis)["accuracy"]
            cap = C.accuracy_block(mat)["accuracy"] if mat else float("nan")
            t = C.tdr_block(mis)
            rows.append({"src": src, "dst": dst, "window": w, "audio": a,
                         "capability": cap, "tdr": t["tdr"]})
            print("{:>4} {:>4} {:>3} {:>9.3f} {:>+10.3f} {:>11.3f} {:>7.3f}".format(
                src, dst, w, a, a - b_audio, cap, t["tdr"]))

    best = max(rows, key=lambda r: r["audio"])

    mid = n_layers // 2
    ctrl = evaluate(mismatched, lambda it: predict_backpatched(
        model, processor, it, mid, best["dst"], best["window"], args.fixed_seconds, n_layers))
    c_audio = C.accuracy_block(ctrl)["accuracy"]

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("best: src={} dst={} window={}   audio {:.3f} -> {:.3f} ({:+.3f})".format(
        best["src"], best["dst"], best["window"], b_audio, best["audio"],
        best["audio"] - b_audio))
    print("mid-layer source control (src={}): audio {:.3f}".format(mid, c_audio))
    print("capability {:.3f} -> {:.3f} ({:+.3f})".format(
        b_cap, best["capability"], best["capability"] - b_cap))

    if best["capability"] == best["capability"] and best["capability"] < b_cap - 0.05:
        verdict = "REJECT: capability dropped -- the patch broke the model, not the suppression."
    elif best["audio"] <= c_audio:
        verdict = "REJECT: a mid-layer source does as well, so late->early is not the mechanism."
    elif best["audio"] > b_audio:
        verdict = "TRANSFERS: +{:.3f} audio accuracy with capability intact.".format(
            best["audio"] - b_audio)
    else:
        verdict = "NO EFFECT: back-patching does not transfer to this conflict type."
    print("\n" + verdict)
    print("\nIf it transfers, check whether the destination layers coincide with the")
    print("text circuit exp3 found -- that would say back-patching overwrites the")
    print("suppression pathway, which is the mechanism claim worth publishing.")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "exp4_backpatching_{}.json".format(args.category))
    with open(out, "w") as f:
        json.dump({"run": dict(info, category=args.category, seed=args.seed,
                               fixed_seconds=args.fixed_seconds, n_layers=n_layers,
                               n_mismatched=len(mismatched), n_matched=len(matched)),
                   "baseline": {"audio": b_audio, "capability": b_cap, "tdr": b_tdr},
                   "sweep": rows, "best": best,
                   "mid_layer_control": {"src": mid, "audio": c_audio},
                   "verdict": verdict}, f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
