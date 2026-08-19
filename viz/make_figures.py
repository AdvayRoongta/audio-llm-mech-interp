#!/usr/bin/env python3
"""Build the paper figures from whatever is in results/.

Reads two result formats, because we have both:

  * the behavioural runs (run_*_eval.py)     -> *_summary.json
  * the mechanistic experiments (exp*.py)    -> exp{1,2,3,4}_*.json

Every figure is skipped with a note if its inputs are missing, so this is safe
to run mid-sweep. Vector PDF for the paper plus PNG for slides.

Usage, from the repo root:

    python3 viz/make_figures.py
    python3 viz/make_figures.py --results results --out figures

Figures:

  fig1_condition_ladder   the suppression story: audio alone -> +neutral text
                          -> +false claim, per category
  fig2_tdr_by_model       text-dominance across models, with Wilson intervals
  fig3_recency            claim_first vs claim_last, the order control
  fig4_encoder_probe      where the signal lives in the audio encoder
  fig5_circuit_depth      the mechanism: audio influence peaks early, text late
  fig6_backpatching       the intervention that does not transfer
"""
import argparse
import glob
import json
import math
import os

# Headless by default: this usually runs over ssh or in CI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORIES = ["emotional_prosody", "speaker_identity", "sarcasm_pacing"]
PRETTY = {"emotional_prosody": "Emotional prosody",
          "speaker_identity": "Speaker identity",
          "sarcasm_pacing": "Sarcasm / pacing"}
# One colour per role, reused across figures so the paper reads as one system.
C_AUDIO, C_TEXT, C_CHANCE, C_NEUTRAL = "#2b6cb0", "#c53030", "#718096", "#2f855a"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """Wilson interval. The normal approximation misbehaves at proportions
    near 0 and 1, and several of these land at exactly 0.0 or 1.0."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def err_bars(k, n):
    """(point, lower_err, upper_err) with the errors clamped non-negative.

    The Wilson bounds can land a few 1e-16 the wrong side of the point estimate
    at p = 0 or p = 1 -- exactly where several of these results sit -- and
    matplotlib rejects a negative yerr outright.
    """
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    lo, hi = wilson(k, n)
    return p, max(0.0, p - lo), max(0.0, hi - p)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "{}.{}".format(name, ext)))
    plt.close(fig)
    print("  wrote {}/{}.pdf (+.png)".format(out_dir, name))


def skip(name, why):
    print("  skip {}: {}".format(name, why))


def exp1_records(results_dir):
    d = load(os.path.join(results_dir, "exp1_capability_screen.json"))
    return d.get("records") if d else None


def behavioural_summaries(results_dir):
    """{model_name: summary_dict} from the *_summary.json files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_summary.json"))):
        d = load(path)
        if not d or "per_category" not in d:
            continue
        name = os.path.basename(path).replace("_summary.json", "")
        out[name] = d
    return out


# --------------------------------------------------------------------------- #
# fig 1 -- the condition ladder
# --------------------------------------------------------------------------- #
def fig_condition_ladder(results_dir, out_dir):
    recs = exp1_records(results_dir)
    if not recs:
        return skip("fig1_condition_ladder", "no exp1_capability_screen.json")

    # Audio-bearing conditions only. text_only items have no audio to ignore,
    # and for Mismatched their "correct" answer IS the claim, so pooling them
    # in would wash the effect out entirely.
    ladder = [("Paralinguistic", "audio_only", "audio alone"),
              ("Neutral-Text", "text_audio", "+ neutral text"),
              ("Matched", "text_audio", "+ matching text"),
              ("Mismatched", "text_audio", "+ false claim")]

    cats = [c for c in CATEGORIES if any(r["category"] == c for r in recs)]
    if not cats:
        return skip("fig1_condition_ladder", "no categories present")

    fig, axes = plt.subplots(1, len(cats), figsize=(3.2 * len(cats), 3.0), squeeze=False)
    for ax, cat in zip(axes[0], cats):
        xs, ys, los, his, chance = [], [], [], [], None
        for cond, mod, label in ladder:
            sub = [r for r in recs if r["category"] == cat
                   and r["condition"] == cond and r["modality"] == mod]
            if not sub:
                continue
            k = sum(bool(r["correct"]) for r in sub)
            p, elo, ehi = err_bars(k, len(sub))
            xs.append(label)
            ys.append(p)
            los.append(elo)
            his.append(ehi)
            chance = 1.0 / sub[0]["n_options"]
        if not xs:
            continue
        ax.errorbar(range(len(xs)), ys, yerr=[los, his], marker="o", ms=5,
                    lw=1.6, capsize=3, color=C_AUDIO)
        if chance:
            ax.axhline(chance, ls="--", lw=1, color=C_CHANCE, label="chance")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(PRETTY.get(cat, cat))
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("accuracy vs. the audio")
    axes[0][-1].legend(loc="upper right")
    fig.suptitle("Adding text degrades audio judgement; a false claim erases it", y=1.02)
    save(fig, out_dir, "fig1_condition_ladder")


# --------------------------------------------------------------------------- #
# fig 2 -- TDR across models
# --------------------------------------------------------------------------- #
def fig_tdr_by_model(results_dir, out_dir):
    sums = behavioural_summaries(results_dir)
    if not sums:
        return skip("fig2_tdr_by_model", "no *_summary.json files")

    models = sorted(sums)
    cats = [c for c in CATEGORIES if any(c in s["per_category"] for s in sums.values())]
    if not cats:
        return skip("fig2_tdr_by_model", "summaries carry no categories")

    fig, ax = plt.subplots(figsize=(1.9 * len(cats) + 2.4, 3.2))
    width = 0.8 / max(1, len(models))
    plotted, stale = 0, []
    for i, model in enumerate(models):
        ys, los, his, xs = [], [], [], []
        for j, cat in enumerate(cats):
            blk = sums[model]["per_category"].get(cat, {})
            # text_audio only, and no fallback to the pooled 'tdr' field: that
            # one mixes in text_only items whose correct answer IS the claim by
            # construction, which inflates the rate. A summary written before
            # by_modality existed simply cannot answer this question, so it is
            # reported as stale rather than quietly plotted wrong.
            ta = (((blk.get("tdr") or {}).get("by_modality") or {})
                  .get("text_audio") or {})
            n, t = ta.get("n_usable"), ta.get("text_claim")
            if not n:
                stale.append("{}/{}".format(model, cat))
                continue
            p, elo, ehi = err_bars(t, n)
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(p); los.append(elo); his.append(ehi)
        if xs:
            plotted += 1
            ax.bar(xs, ys, width=width * 0.9, label=model,
                   yerr=[los, his], capsize=2, error_kw={"lw": 0.8})
    if not plotted:
        plt.close(fig)
        return skip("fig2_tdr_by_model",
                    "no summary carries tdr.by_modality.text_audio -- these "
                    "predate that field; rerun the behavioural eval")
    if stale:
        print("    note: no text_audio TDR for {}".format(", ".join(stale[:6])))
    ax.axhline(0.5, ls=":", lw=1, color=C_CHANCE)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([PRETTY.get(c, c) for c in cats])
    ax.set_ylabel("text-dominance rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("How often the model believes the text over the audio")
    ax.legend(ncol=min(3, len(models)), loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "fig2_tdr_by_model")


# --------------------------------------------------------------------------- #
# fig 3 -- recency
# --------------------------------------------------------------------------- #
def fig_recency(results_dir, out_dir):
    recs = exp1_records(results_dir)
    sums = behavioural_summaries(results_dir)
    series = {}

    if recs:
        for cat in CATEGORIES:
            base = [r for r in recs if r["category"] == cat
                    and r["condition"] == "Mismatched" and r["modality"] == "text_audio"]
            got = {}
            for order in ("claim_first", "claim_last"):
                sub = [r for r in base if r.get("order") == order]
                if sub:
                    t = sum(1 for r in sub if r.get("claimed")
                            and r["prediction"] == r["claimed"])
                    got[order] = (t, len(sub))
            if len(got) == 2:
                series.setdefault("exp1 (log-prob)", {})[cat] = got

    for model, s in sums.items():
        for cat, blk in s["per_category"].items():
            by_order = ((blk.get("tdr") or {}).get("by_order")) or {}
            got = {o: (v.get("text_claim"), v.get("n_usable"))
                   for o, v in by_order.items() if v.get("n_usable")}
            if len(got) == 2:
                series.setdefault(model, {})[cat] = got

    if not series:
        return skip("fig3_recency", "no per-order TDR found")

    cats = sorted({c for v in series.values() for c in v})
    fig, axes = plt.subplots(1, len(cats), figsize=(2.9 * len(cats), 3.0),
                             squeeze=False, sharey=True)
    for ax, cat in zip(axes[0], cats):
        for name, per_cat in sorted(series.items()):
            got = per_cat.get(cat)
            if not got:
                continue
            xs, ys, los, his = [], [], [], []
            for j, order in enumerate(("claim_first", "claim_last")):
                if order not in got:
                    continue
                t, n = got[order]
                p_, elo, ehi = err_bars(t, n)
                xs.append(j); ys.append(p_)
                los.append(elo); his.append(ehi)
            if xs:
                ax.errorbar(xs, ys, yerr=[los, his], marker="o", ms=4,
                            lw=1.4, capsize=3, label=name)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["claim first", "claim last"])
        ax.set_title(PRETTY.get(cat, cat))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("text-dominance rate")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("Does claim position matter? (recency control)", y=1.02)
    save(fig, out_dir, "fig3_recency")


# --------------------------------------------------------------------------- #
# fig 4 -- encoder probe
# --------------------------------------------------------------------------- #
def fig_encoder_probe(results_dir, out_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "exp2_encoder_probe*.json")))
    blocks = []
    for path in files:
        d = load(path)
        for r in (d or {}).get("results", []):
            blocks.append((r, (d.get("run") or {})))
    if not blocks:
        return skip("fig4_encoder_probe", "no exp2_encoder_probe*.json")

    fig, axes = plt.subplots(1, len(blocks), figsize=(3.4 * len(blocks), 3.0),
                             squeeze=False)
    for ax, (r, run) in zip(axes[0], blocks):
        layers = r["layers"]
        enc = layers[:-1]
        proj = layers[-1]
        xs = [x["layer"] for x in enc]
        ax.plot(xs, [x["accuracy"] for x in enc], marker="o", ms=2.5, lw=1.5,
                color=C_AUDIO, label="linear probe")
        ax.plot(xs, [x["shuffled"] for x in enc], ls=":", lw=1.2,
                color=C_CHANCE, label="shuffled labels")
        ax.axhline(r["chance"], ls="--", lw=1, color=C_TEXT, label="chance")
        ax.scatter([xs[-1] + 1], [proj["accuracy"]], marker="D", s=30,
                   color=C_NEUTRAL, zorder=5, label="post-projector")
        ax.set_xlabel("audio encoder layer")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        cv = r.get("cv_scheme", "")
        ax.set_title("{}\n{} clips, {}".format(
            PRETTY.get(r["category"], r["category"]), r["n_clips"], cv), fontsize=9)
    axes[0][0].set_ylabel("probe accuracy")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("The signal survives the encoder and the projector", y=1.03)
    save(fig, out_dir, "fig4_encoder_probe")


# --------------------------------------------------------------------------- #
# fig 5 -- circuit depth (the mechanism figure)
# --------------------------------------------------------------------------- #
def fig_circuit_depth(results_dir, out_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "exp3_circuits_*.json")))
    if not files:
        return skip("fig5_circuit_depth", "no exp3_circuits_*.json")

    fig, axes = plt.subplots(1, len(files), figsize=(3.6 * len(files), 3.0),
                             squeeze=False, sharey=True)
    for ax, path in zip(axes[0], files):
        d = load(path)
        if not d:
            continue
        cat = d["run"].get("category", os.path.basename(path))
        for key, colour, label in (("attr_text", C_TEXT, "text circuit"),
                                   ("attr_audio", C_AUDIO, "audio circuit")):
            per_layer = {}
            for k, v in (d.get(key) or {}).items():
                layer = int(k.split("|")[0])
                per_layer[layer] = per_layer.get(layer, 0.0) + abs(v)
            if not per_layer:
                continue
            total = sum(per_layer.values()) or 1.0
            xs = sorted(per_layer)
            ax.plot(xs, [per_layer[x] / total for x in xs], marker="o", ms=2.5,
                    lw=1.5, color=colour, label=label)
        ax.set_xlabel("language model layer")
        ax.set_title(PRETTY.get(cat, cat))
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("share of total |attribution|")
    axes[0][-1].legend(loc="upper left")
    fig.suptitle("Audio influence peaks early and fades; text takes over late", y=1.02)
    save(fig, out_dir, "fig5_circuit_depth")


# --------------------------------------------------------------------------- #
# fig 6 -- back-patching
# --------------------------------------------------------------------------- #
def fig_backpatching(results_dir, out_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "exp4_backpatching_*.json")))
    if not files:
        return skip("fig6_backpatching", "no exp4_backpatching_*.json")

    fig, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 3.0),
                             squeeze=False, sharey=True)
    for ax, path in zip(axes[0], files):
        d = load(path)
        if not d:
            continue
        cat = d["run"].get("category", "")
        base = d.get("baseline", {})
        rows = d.get("sweep", [])
        for w in sorted({r["window"] for r in rows}):
            sub = sorted([r for r in rows if r["window"] == w], key=lambda r: r["dst"])
            ax.plot([r["dst"] for r in sub], [r["audio"] for r in sub],
                    marker="o", ms=4, lw=1.4, label="window {}".format(w))
        if base.get("audio") is not None:
            ax.axhline(base["audio"], ls="--", lw=1.2, color=C_TEXT,
                       label="baseline")
        ctrl = d.get("mid_layer_control") or {}
        if ctrl.get("audio") is not None:
            ax.axhline(ctrl["audio"], ls=":", lw=1.2, color=C_CHANCE,
                       label="mid-layer source")
        ax.set_xlabel("destination layer")
        ax.set_ylim(-0.02, max(0.2, 1.15 * max(
            [r["audio"] for r in rows] + [base.get("audio", 0)] or [0.2])))
        ax.set_title(PRETTY.get(cat, cat))
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("audio accuracy under conflict")
    axes[0][-1].legend(loc="upper right")
    fig.suptitle("Back-patching does not recover the audio", y=1.02)
    save(fig, out_dir, "fig6_backpatching")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isdir(args.results):
        raise SystemExit("no such directory: {}".format(args.results))

    print("reading {}/ -> writing {}/".format(args.results, args.out))
    for fn in (fig_condition_ladder, fig_tdr_by_model, fig_recency,
               fig_encoder_probe, fig_circuit_depth, fig_backpatching):
        try:
            fn(args.results, args.out)
        except Exception as exc:                      # keep going; one bad
            print("  ERROR in {}: {}: {}".format(     # figure must not block
                fn.__name__, type(exc).__name__, exc))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
