"""Experiment 1 -- audio-only capability screen, and the recency control.

The gate for experiments 2-4. If the model cannot name the emotion from audio
with no text claim present, then a failure under conflict does not show
suppression -- it shows the signal never arrived. In that case the honest claim
is insufficient capability, and the mechanistic experiments have nothing to
localize.

The dataset already carries the conditions this needs:

    Paralinguistic / audio_only   the capability floor -- no text claim at all
    Neutral-Text                  non-committal text; LISTEN found this alone degrades
    Matched                       both cues agree -- the ceiling
    Mismatched / text_audio       the conflict, in both claim orders

Scoring is by option log-prob rather than generate-then-parse: option letters
are single tokens, so one forward scores every option, nothing can fail to
parse, and abstention cannot be confused with a parsing failure. That makes
these numbers comparable to run_qwen2audio_eval.py but not identical to it --
that script measures what the model *says*, this one measures what it *ranks*.

Because Mismatched/text_audio exists in both claim_first and claim_last, TDR is
reported per order. Cho et al. (arXiv:2606.18924) found Qwen shows a recency
effect on lexical conflicts, so a TDR quoted without the order split is not
interpretable.

Run from the repo root:
    python eval/exp1_capability_screen.py
    python eval/exp1_capability_screen.py --category emotional_prosody --4bit
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def evaluate(model, processor, items, verbose_every=25):
    """Score every item by option log-prob. One forward each."""
    records = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        letters = list(C.option_map(item).keys())
        inputs = C.encode(model, processor, item)
        pred, scores = C.predict_letter(model, processor, inputs, letters)
        gold = C.correct_letter(item)
        records.append({
            "uid": item["_uid"],
            "category": item.get("category"),
            "condition": item.get("condition"),
            "modality": item.get("modality"),
            "order": item.get("order"),
            "prediction": pred,
            "correct_letter": gold,
            "claimed": C.claimed_letter(item),
            "correct": pred == gold,
            "n_options": len(letters),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        })
        if verbose_every and (n % verbose_every == 0 or n == len(items)):
            acc = sum(r["correct"] for r in records) / len(records)
            print("  [{:>4}/{}] running acc {:.3f}  ({:.0f}s)".format(
                n, len(items), acc, time.time() - t0))
    return records


def report_screen(records):
    """The gate. Everything downstream depends on this passing."""
    print("\n" + "=" * 74)
    print("CAPABILITY SCREEN -- Paralinguistic / audio_only, no text claim")
    print("=" * 74)
    verdicts = {}
    present = [r for r in records if r["condition"] == "Paralinguistic"]
    if not present:
        print("\nNo Paralinguistic/audio_only items in this selection, so the screen")
        print("did not run. That is not a failed screen -- it is an unrun one.")
        print("Drop --limit, or widen --category, to include them.")
        return verdicts
    for cat in C.CATEGORIES:
        rs = [r for r in records if r["category"] == cat and r["condition"] == "Paralinguistic"]
        if not rs:
            continue
        chance = 1.0 / rs[0]["n_options"]
        blk = C.accuracy_block(rs, chance=chance)
        lo, hi = blk["ci95"]
        p = blk.get("p_vs_chance")
        passed = blk["accuracy"] > chance and (p is not None and p < 0.05)
        verdicts[cat] = passed
        print("\n{:<20} acc {:.3f}  95% CI [{:.3f}, {:.3f}]  chance {:.3f}  p {}".format(
            cat, blk["accuracy"], lo, hi, chance,
            "{:.4f}".format(p) if p is not None else "n/a"))
        print("{:<20} predictions {}".format("", blk["prediction_counts"]))
        print("{:<20} -> {}".format(
            "", "PASS: signal survives the encoder; conflict failures are suppression"
                if passed else
                "FAIL: at chance -- cannot separate encoder loss from suppression"))
    return verdicts


def report_conditions(records):
    print("\n" + "=" * 74)
    print("ACCURACY BY CONDITION")
    print("=" * 74)
    print("{:<20} {:<14} {:<11} {:>4} {:>7} {:>16} {:>8}".format(
        "category", "condition", "modality", "n", "acc", "95% CI", "chance"))
    for cat in C.CATEGORIES:
        for cond in C.CONDITIONS:
            for mod in C.MODALITIES:
                rs = [r for r in records if r["category"] == cat
                      and r["condition"] == cond and r["modality"] == mod]
                if not rs:
                    continue
                chance = 1.0 / rs[0]["n_options"]
                blk = C.accuracy_block(rs, chance=chance)
                lo, hi = blk["ci95"]
                print("{:<20} {:<14} {:<11} {:>4} {:>7.3f} [{:>6.3f},{:>6.3f}] {:>8.3f}".format(
                    cat, cond, mod, blk["n"], blk["accuracy"], lo, hi, chance))


def report_tdr(records):
    """TDR split by claim order -- the recency control."""
    print("\n" + "=" * 74)
    print("TEXT DOMINANCE on Mismatched / text_audio, by claim order")
    print("=" * 74)
    print("{:<20} {:<12} {:>4} {:>7} {:>8} {:>9}".format(
        "category", "order", "n", "TDR", "audio", "neither"))
    for cat in C.CATEGORIES:
        base = [r for r in records if r["category"] == cat
                and r["condition"] == "Mismatched" and r["modality"] == "text_audio"]
        if not base:
            continue
        for order in ("claim_first", "claim_last", None):
            rs = base if order is None else [r for r in base if r["order"] == order]
            if not rs:
                continue
            t = C.tdr_block(rs)
            print("{:<20} {:<12} {:>4} {:>7.3f} {:>8.3f} {:>9.3f}".format(
                cat, order or "ALL", t["n"], t["tdr"], t["audio_rate"], t["neither"]))

        first = C.tdr_block([r for r in base if r["order"] == "claim_first"])
        last = C.tdr_block([r for r in base if r["order"] == "claim_last"])
        if first and last:
            delta = last["tdr"] - first["tdr"]
            print("{:<20} {:<12} {:>4} {:>7} recency effect (last - first): {:+.3f}".format(
                "", "", "", "", delta))
            if abs(delta) >= 0.10:
                print("{:<20} -> order matters; do not quote a pooled TDR for this category"
                      .format(""))
    print("\n'neither' is reported because LISTEN found most conflict answers match")
    print("neither cue -- a bare TDR hides that entirely.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", choices=C.CATEGORIES, help="restrict to one category")
    ap.add_argument("--limit", type=int, help="cap items (smoke test)")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load in 4-bit to fit a 15 GB T4; recorded in the output")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    C.set_seed(args.seed)

    items = C.load_items(args.input, require_audio=True)
    if args.category:
        items = C.select(items, category=args.category)
    if args.limit:
        items = items[:args.limit]
    print("{} items from {}".format(len(items), args.input))

    model, processor, info = C.load_model(four_bit=args.four_bit)
    records = evaluate(model, processor, items)

    verdicts = report_screen(records)
    report_conditions(records)
    report_tdr(records)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "exp1_capability_screen.json")
    with open(out, "w") as f:
        json.dump({"run": dict(info, scoring="option_logprob", seed=args.seed,
                               dataset=args.input, n_items=len(records)),
                   "screen_verdicts": verdicts,
                   "records": records}, f, indent=2)
    print("\nwrote {}".format(out))

    if verdicts and not any(verdicts.values()):
        print("\nNO category passed the screen. Experiments 2-4 would be measuring")
        print("a signal the model never had. Fix capability before localizing.")
    elif verdicts:
        passed = [c for c, ok in verdicts.items() if ok]
        print("\nPassed the screen: {}. Localize suppression in those only."
              .format(", ".join(passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
