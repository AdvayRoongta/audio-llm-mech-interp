#!/usr/bin/env python3
"""The mechanistic figures, in the shapes this literature actually uses.

`make_figures.py` covers the behavioural story and the four headline plots.
This file adds the figure *types* that mech-interp papers are expected to
carry, and reads the result files that script never opens -- exp5, exp6, and
everything under `results/salmonn/`.

Each figure below names the convention it follows and the prior work that
established it, because a reviewer reads a figure type before reading its
caption, and a familiar shape is doing half the explaining:

  m1c_attribution_grid    layer x position-group causal map      ROME (Meng
                          et al. 2022) fig 2; Chen et al. 2026 fig 3, the
                          audio-LALM version of the same grid
  m2_faithfulness         faithfulness vs. circuit sparsity      ACDC (Conmy
                          et al. 2023), AtP* (Kramar et al. 2024)
  m3_group_magnitude      per-position-group modality bars       Cho et al.
                          2026; Geva et al. 2023
  m4_circuit_overlap      IoU with bootstrap CI vs. chance       Cho et al. 2026
  m5_probe_depth          layer-wise probe + selectivity         Pasad et al.
                          2021; Hewitt & Liang 2019 for the control
  m6_margin_distribution  per-item logit-difference spread       Zhang & Nanda
                          2024, "report the distribution, not the mean"
  m7_backpatch_grid       intervention sweep grid                Meng et al. 2022
  m8_source_sweep         effect vs. collateral damage           our exp5
  m9_norm_profile         residual-norm growth with depth        StreamingLLM
                          (Xiao et al. 2023)
  m10_knockout_bands      attention knockout by layer band       Geva et al. 2023
                          fig 3
  m11_routing             the routing bisection                  Geva et al. 2023
  m12_generative_fix      free-form intervention + controls      ours
  m13_generation_quality  fluency / repetition / length          answers Cho
                          et al.'s degeneration finding
  m14_outcome_mass        three-way follow-rate across models    LISTEN
  m15_capability_gate     the audio-only screen that gates all   LISTEN

Usage, from the repo root:

    python3 viz/make_mechinterp_figures.py
    python3 viz/make_mechinterp_figures.py --out figures --only m1 m6

Every figure is skipped with a printed reason if its inputs are absent, so
this is safe to run against a partial results tree.
"""
import argparse
import glob
import json
import math
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise SystemExit("needs matplotlib and numpy:  pip3 install matplotlib numpy")

# --------------------------------------------------------------------------- #
# house style -- deliberately identical to make_figures.py so the two scripts
# produce one visual system rather than two.
# --------------------------------------------------------------------------- #
CATEGORIES = ["emotional_prosody", "speaker_identity", "sarcasm_pacing"]
PRETTY = {"emotional_prosody": "Emotional prosody",
          "speaker_identity": "Speaker identity",
          "sarcasm_pacing": "Sarcasm / pacing"}
C_AUDIO, C_TEXT, C_CHANCE, C_NEUTRAL = "#2b6cb0", "#c53030", "#718096", "#2f855a"

# Position groups in sequence order, so the y-axis of the causal map reads as a
# spatial map of the prompt the way Chen et al. lay theirs out.
GROUPS = ["audio", "claim", "query", "options", "gen"]
GROUP_LABEL = {"audio": "audio tokens", "claim": "text claim", "query": "question",
               "options": "answer options", "gen": "answer position"}

# Where each model's results live. Qwen writes to the top level for historical
# reasons; every later model gets its own subdirectory.
MODEL_ROOTS = [("Qwen2-Audio-7B", "results"),
               ("SALMONN-13B", os.path.join("results", "salmonn"))]

# SALMONN is excluded from the mechanistic analysis: it does not clear the
# capability screen, and a model that cannot do the task from audio cannot be
# shown to be suppressing audio. So the mechanism figures are Qwen-only by
# default. Override with --models to draw a panel per model again.
#
# Two figures deliberately ignore this filter, because for them SALMONN is not
# a mechanistic claim:
#   m15  the capability screen -- it IS the evidence for the exclusion, so
#        dropping the excluded model would remove the argument
#   m14  behavioural follow-rates, a different section of the paper, where
#        SALMONN appears in the results table alongside every other model
MECH_MODELS = ["Qwen2-Audio-7B"]

# sarcasm_pacing is run but not reported: its audio does not settle a single
# correct label (removing the claim lifts audio accuracy only to 0.233 against
# 0.100 chance), and its free-form descriptions name one identifiable label
# only ~10 times in 30, so the intervention's intervals overlap the baseline.
# Underpowered rather than negative -- excluded from the mechanism figures for
# the same reason it is excluded from the write-up. --categories overrides.
# m14 and m15 ignore this filter: m15 is part of why sarcasm is set aside, and
# m14 is behavioural.
MECH_CATEGORIES = ["emotional_prosody", "speaker_identity"]

# Circuits drawn in the heatmaps. The audio panels are near-blank on a scale
# shared with the text panels, which is the finding but costs half the figure
# width to show it -- and a blank panel reads as a rendering failure as easily
# as a result. m3 carries the text-vs-audio comparison numerically, so the
# heatmaps show the text circuit alone. Set to ("text", "audio") to restore.
HEATMAP_CIRCUITS = ("text",)


def roots(every=False):
    """Model roots for a figure. `every=True` opts out of the mech filter."""
    if every:
        return MODEL_ROOTS
    return [(m, r) for m, r in MODEL_ROOTS if m in MECH_MODELS]

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
    """Wilson interval; the normal approximation fails at the 0.0 and 1.0 rates
    this project hits constantly."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def err_bars(k, n):
    """(point, lower_err, upper_err), clamped non-negative -- matplotlib
    rejects a negative yerr and Wilson can land 1e-16 the wrong side at p=0."""
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


def found(results_dir, pattern):
    return sorted(glob.glob(os.path.join(results_dir, pattern)))


def cat_of(d, path):
    """Category for a result file, from the run header or the filename."""
    c = (d.get("run") or {}).get("category")
    return c or os.path.basename(path).rsplit("_", 1)[-1].replace(".json", "")


def per_model(pattern, every=False, all_cats=False):
    """[(model_name, path, parsed)] for every model root carrying `pattern`.

    Filtered to MECH_CATEGORIES unless `all_cats`. A file whose category cannot
    be determined is kept -- better a stray panel than a silently dropped one.
    """
    out = []
    for model, root in roots(every):
        for path in found(root, pattern):
            d = load(path)
            if not d:
                continue
            cat = cat_of(d, path)
            if not all_cats and cat in CATEGORIES and cat not in MECH_CATEGORIES:
                continue
            out.append((model, path, d))
    return out


def grid_from_attr(attr, n_layers):
    """{'layer|submodule|group': value} -> {submodule: (len(GROUPS), n_layers)}.

    Returns absolute attribution: magnitude is the comparison the paper makes
    (text 3.29 vs audio 0.18), and a signed map would let two components of
    opposite sign look like a quiet region.
    """
    out = {sub: np.zeros((len(GROUPS), n_layers)) for sub in ("attn", "mlp")}
    for key, val in attr.items():
        layer, sub, group = key.split("|")
        if sub in out and group in GROUPS and int(layer) < n_layers:
            out[sub][GROUPS.index(group), int(layer)] = abs(val)
    return out


def chance_iou(n_components, top_frac):
    """Expected IoU of two independent top-k sets drawn from n components.

    E|A n B| = k^2/n and |A u B| = 2k - k^2/n, so IoU = k / (2n - k). With
    n=64 and k=floor(0.2*64)=12 this gives 0.103, the value the methods note
    quotes for Qwen -- worth reproducing rather than hardcoding, since SALMONN
    has 40 layers and therefore a different chance level.
    """
    k = int(top_frac * n_components)
    return k / (2 * n_components - k) if n_components and k else None


# --------------------------------------------------------------------------- #
# m1c -- all eight panels in one grid
# --------------------------------------------------------------------------- #
def m1c_attribution_grid(out_dir):
    """Every category and circuit in one figure, on one colour scale.

    The only heatmap. One panel per category, attention and MLP summed, one
    shared vmax so the categories are comparable by eye. Chen et al. facet all
    four of their attributes into a single grid for the same reason.

    An earlier version also wrote per-category files with attn and mlp split
    into separate rows. Dropped -- but the split carried a real result worth
    keeping in the text: text mass at the answer position is 72% attention
    (15.1 vs 6.0) while audio mass at the audio tokens is 71% MLP (1.44 vs
    0.58). The claim is routed to the readout; the acoustics are processed in
    place and largely not moved.
    """
    blocks = per_model("exp3_circuits_*.json")
    if not blocks:
        return skip("m1c_attribution_grid", "no exp3_circuits_*.json")

    cols = []                       # (model, cat, circuit_label, grids, cmap)
    vmax = 0.0
    for model, path, d in blocks:
        cat = cat_of(d, path)
        n_layers = 1 + max(int(k.split("|")[0]) for k in d["attr_text"])
        for key, label, cmap in (("attr_text", "text", "Reds"),
                                 ("attr_audio", "audio", "Blues")):
            if label not in HEATMAP_CIRCUITS:
                continue
            g = grid_from_attr(d[key], n_layers)
            vmax = max(vmax, (g["attn"] + g["mlp"]).max())
            cols.append((model, cat, label, g, cmap))
    vmax = vmax or 1.0

    # One row: attention and MLP summed per (layer, group). The split lives in
    # m1's per-category files; here the question is where the mass sits, not
    # which mechanism carries it.
    fig, axes = plt.subplots(1, len(cols), figsize=(2.55 * len(cols), 2.5),
                             squeeze=False, sharey=True)
    for c, (model, cat, label, grids, cmap) in enumerate(cols):
        ax = axes[0][c]
        im = ax.imshow(grids["attn"] + grids["mlp"], aspect="auto",
                       origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(len(GROUPS)))
        if c == 0:
            ax.set_yticklabels([GROUP_LABEL[g] for g in GROUPS], fontsize=6.5)
        # circuit subtitle only when there is more than one to tell apart
        if len(HEATMAP_CIRCUITS) > 1:
            ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("layer", fontsize=8)
        ax.tick_params(labelsize=7)
        if label == HEATMAP_CIRCUITS[0]:
            span = 1.02 if len(HEATMAP_CIRCUITS) > 1 else 0.5
            ax.text(span, 1.04 if len(HEATMAP_CIRCUITS) == 1 else 1.16,
                    PRETTY.get(cat, cat), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.20, wspace=0.16)
    # Two colourbars, not one. The panels use two colormaps, so a single bar
    # necessarily shows one of them and silently mislabels the other half of
    # the figure -- the limits are shared, the colours are not.
    import matplotlib.cm as cm
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    # Explicit rectangles. fig.colorbar(ax=axes) stacks its bars against each
    # other regardless of `pad`, which puts the two swatches and their titles
    # on top of one another; placing them by hand is the only reliable way to
    # get two scales side by side.
    fig.subplots_adjust(right=0.88)
    bars = [(0.905, "Reds", "text", False), (0.938, "Blues", "audio", True)]
    bars = [b for b in bars if b[2] in HEATMAP_CIRCUITS]
    if len(bars) == 1:                       # single scale: put ticks on it
        x, cmap, label, _ = bars[0]
        bars = [(x, cmap, label, True)]
    for x, cmap, label, ticks in bars:
        cax = fig.add_axes([x, 0.18, 0.013, 0.60])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.outline.set_linewidth(0.6)
        if len(bars) > 1:
            cax.set_title(label, fontsize=7, pad=5)
        if ticks:
            cb.ax.tick_params(labelsize=6.5, length=2)
            cb.set_label("|attribution|, shared limits", fontsize=7)
        else:
            cb.set_ticks([])
    # Names the model and the circuit: with the audio panels gone nothing else
    # on the figure says which circuit this is, and the model matters because
    # the layer axis is only meaningful against a known depth.
    models = " / ".join(dict.fromkeys(m for m, _, _, _, _ in cols))
    circuits = " + ".join(HEATMAP_CIRCUITS)
    fig.suptitle("{} \u2014 {}-circuit attribution by layer and prompt position"
                 .format(models, circuits), y=1.0, fontsize=10.5)
    save(fig, out_dir, "m1c_attribution_grid")


# --------------------------------------------------------------------------- #
# m2 -- faithfulness vs sparsity
# --------------------------------------------------------------------------- #
def m2_faithfulness(out_dir):
    """Does the small set of components we kept reproduce the whole model?

    Standard circuit-validation curve (ACDC, AtP*, EAP). Cho et al. threshold
    at 0.8, which is drawn. Without this a localisation claim is unfalsifiable,
    and our own methods note records a case where faithfulness read 1.00 on a
    version of the code that was broken -- so it is necessary, not sufficient.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp3_circuits_*.json")
              if d.get("faithfulness")]
    if not blocks:
        return skip("m2_faithfulness", "no faithfulness block in exp3 files")

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    markers = {"Qwen2-Audio-7B": "o", "SALMONN-13B": "s"}
    palette = [C_AUDIO, C_NEUTRAL, "#805ad5", "#d69e2e"]
    for i, (model, path, d) in enumerate(blocks):
        cat = cat_of(d, path)
        fr = sorted((float(k), v) for k, v in d["faithfulness"].items())
        ax.plot([x for x, _ in fr], [y for _, y in fr],
                marker=markers.get(model, "o"), ms=4, lw=1.5,
                color=palette[i % len(palette)],
                label="{} -- {}".format(model.split("-")[0],
                                        PRETTY.get(cat, cat)))
    ax.axhline(1.0, ls="--", lw=1, color=C_CHANCE, label="full model")
    ax.axhline(0.8, ls=":", lw=1, color=C_TEXT, label="Cho et al. threshold")
    ax.set_xlabel("fraction of components kept")
    ax.set_ylabel("faithfulness")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.10, 0.20, 0.40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # A log axis keeps its minor labels, which print 6x10^-2 straight through
    # the 0.05 major label.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.78, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("A fifth of the components reproduce the behaviour")
    save(fig, out_dir, "m2_faithfulness")


# --------------------------------------------------------------------------- #
# m3 -- where each modality's mass sits
# --------------------------------------------------------------------------- #
def m3_group_magnitude(out_dir):
    """The lopsidedness, as one panel per category.

    Log axis, because the headline is a ratio (roughly 18:1 at the answer
    position for Qwen prosody) and a linear axis renders the audio bar as an
    invisible sliver next to the text bar.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp3_circuits_*.json")
              if d.get("group_magnitudes")]
    if not blocks:
        return skip("m3_group_magnitude", "no group_magnitudes in exp3 files")

    fig, axes = plt.subplots(1, len(blocks), figsize=(3.5 * len(blocks), 3.2),
                             squeeze=False, sharey=True)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        gm = d["group_magnitudes"]
        rows = [g for g in GROUPS if g in gm]
        y = np.arange(len(rows))
        floor = 1e-4                    # a zero cannot be drawn on a log axis
        t = [max(gm[g].get("text") or 0.0, floor) for g in rows]
        a = [max(gm[g].get("audio") or 0.0, floor) for g in rows]
        ax.barh(y + 0.19, t, height=0.36, color=C_TEXT, label="text")
        ax.barh(y - 0.19, a, height=0.36, color=C_AUDIO, label="audio")
        # A bar sitting on the floor is a substituted zero, not a small value.
        # The text attribution over the audio span is exactly 0.0 by
        # construction -- causal masking makes those activations identical
        # between the clean and corrupt runs -- so it has to be marked as such
        # rather than read as "text contributes a little here".
        for g, val, off in [(g, gm[g].get("text"), 0.19) for g in rows] + \
                           [(g, gm[g].get("audio"), -0.19) for g in rows]:
            if not val:
                ax.annotate("0", xy=(floor, rows.index(g) + off), xytext=(3, 0),
                            textcoords="offset points", va="center",
                            fontsize=6, color=C_CHANCE)
        ax.set_yticks(y)
        ax.set_yticklabels([GROUP_LABEL[g] for g in rows], fontsize=7)
        ax.set_xscale("log")
        # Each bar is the largest single (layer, submodule) component in that
        # group, not the summed mass -- exp3 stores group_magnitudes as a max.
        # Labelling it "|attribution|" reads as a total, which it is not.
        ax.set_xlabel("peak |attribution| in group")
        ax.set_title("{}\n{}".format(model.split("-")[0], PRETTY.get(cat, cat)),
                     fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        # Annotate the ratio at the answer position, which is the sentence the
        # paper actually makes from this figure.
        if "gen" in gm and (gm["gen"].get("audio") or 0) > 0:
            ratio = (gm["gen"].get("text") or 0) / gm["gen"]["audio"]
            ax.annotate("{:.0f}x".format(ratio),
                        xy=(max(t[rows.index("gen")], floor),
                            rows.index("gen") + 0.19),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=C_TEXT,
                        fontweight="bold")
    axes[0][-1].legend(loc="lower right")
    fig.suptitle("At the answer position the text swamps the audio", y=1.03)
    save(fig, out_dir, "m3_group_magnitude")


# --------------------------------------------------------------------------- #
# m4 -- circuit overlap
# --------------------------------------------------------------------------- #
def m4_circuit_overlap(out_dir):
    """Are the two circuits the same machinery, or different machinery?

    IoU with a bootstrap interval and the chance level for a top-k set of this
    size. Several of our intervals contain chance, which is the honest reading
    and is exactly why the chance line has to be drawn rather than described.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp3_circuits_*.json")
              if d.get("overlap_iou")]
    if not blocks:
        return skip("m4_circuit_overlap", "no overlap_iou in exp3 files")

    fig, axes = plt.subplots(1, len(blocks), figsize=(3.4 * len(blocks), 3.2),
                             squeeze=False, sharey=True)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        iou, ci = d["overlap_iou"], d.get("overlap_iou_ci95") or {}
        rows = [g for g in GROUPS + ["ALL"] if iou.get(g) is not None]
        xs = np.arange(len(rows))
        pts = [iou[g] for g in rows]
        lo = [max(0.0, iou[g] - (ci.get(g) or [iou[g], iou[g]])[0]) for g in rows]
        hi = [max(0.0, (ci.get(g) or [iou[g], iou[g]])[1] - iou[g]) for g in rows]
        ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", ms=5, capsize=3,
                    lw=1.4, color=C_NEUTRAL)
        n_layers = (d.get("run") or {}).get("n_layers") or \
            1 + max(int(k.split("|")[0]) for k in d["attr_text"])
        ch = chance_iou(n_layers * 2, (d.get("run") or {}).get("top_frac", 0.2))
        if ch:
            ax.axhline(ch, ls="--", lw=1, color=C_CHANCE,
                       label="chance ({:.3f})".format(ch))
        # Groups whose attribution is identically zero cannot have an overlap;
        # say so on the axis rather than leaving an unexplained gap.
        missing = [g for g in GROUPS if iou.get(g) is None]
        if missing:
            ax.set_xlabel("undefined: {}".format(", ".join(missing)), fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([GROUP_LABEL.get(g, g) for g in rows], rotation=30,
                           ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title("{}\n{}".format(model.split("-")[0], PRETTY.get(cat, cat)),
                     fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=7)
    axes[0][0].set_ylabel("circuit overlap (IoU)")
    fig.suptitle("Do the text and audio circuits share components?", y=1.03)
    save(fig, out_dir, "m4_circuit_overlap")


# --------------------------------------------------------------------------- #
# m5 -- probe depth
# --------------------------------------------------------------------------- #
def m5_probe_depth(out_dir):
    """Layer-wise probe accuracy, both models, with both controls drawn.

    Pasad et al. read a speech encoder as a depth trajectory; the shuffled-label
    line is the selectivity control. The projector point is plotted apart from
    the encoder line because it is a different module, and it is the one that
    decides whether the signal reaches the language model at all.
    """
    blocks = []
    for model, root in roots():
        for path in found(root, "exp2_encoder_probe*.json"):
            d = load(path) or {}
            for r in d.get("results", []):
                if r.get("category") in MECH_CATEGORIES:
                    blocks.append((model, r))
    if not blocks:
        return skip("m5_probe_depth", "no exp2_encoder_probe*.json")

    fig, axes = plt.subplots(1, len(blocks), figsize=(3.5 * len(blocks), 3.2),
                             squeeze=False, sharey=True)
    for ax, (model, r) in zip(axes[0], blocks):
        layers = r["layers"]
        enc, proj = layers[:-1], layers[-1]
        xs = [x["layer"] for x in enc]
        acc = [x["accuracy"] for x in enc]
        shuf = [x["shuffled"] for x in enc]
        ax.fill_between(xs, shuf, acc, color=C_AUDIO, alpha=0.13,
                        label="selectivity")
        ax.plot(xs, acc, marker="o", ms=2.5, lw=1.5, color=C_AUDIO,
                label="linear probe")
        ax.plot(xs, shuf, ls=":", lw=1.2, color=C_CHANCE, label="shuffled labels")
        ax.axhline(r["chance"], ls="--", lw=1, color=C_TEXT, label="chance")
        ax.scatter([xs[-1] + 1], [proj["accuracy"]], marker="D", s=34,
                   color=C_NEUTRAL, zorder=5, label="post-projector")
        ax.set_xlabel("audio encoder layer")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title("{}\n{} -- {} clips, {}".format(
            model.split("-")[0], PRETTY.get(r["category"], r["category"]),
            r["n_clips"], r.get("cv_scheme", "")), fontsize=8)
    axes[0][0].set_ylabel("probe accuracy")
    axes[0][-1].legend(loc="lower right", fontsize=7)
    fig.suptitle("The signal survives the encoder and the projector", y=1.04)
    save(fig, out_dir, "m5_probe_depth")


# --------------------------------------------------------------------------- #
# m6 -- per-item margins
# --------------------------------------------------------------------------- #
def m6_margin_distribution(out_dir):
    """The distribution behind the mean margin.

    A mean margin of -7.73 nats is compatible with two very different worlds:
    every item slightly wrong, or a bimodal split. Reporting the spread is the
    standard ask for a logit-difference metric, and it is the only figure here
    that shows individual items. Computed from exp1's stored per-option
    log-probs, so it costs no GPU time.
    """
    rows = []
    for model, root in roots():
        d = load(os.path.join(root, "exp1_capability_screen.json"))
        for rec in (d or {}).get("records", []):
            if (rec["condition"] == "Mismatched" and rec["modality"] == "text_audio"
                    and rec["category"] in MECH_CATEGORIES
                    and rec.get("claimed") and rec.get("scores")):
                s = rec["scores"]
                if rec["correct_letter"] in s and rec["claimed"] in s:
                    rows.append((model, rec["category"],
                                 s[rec["correct_letter"]] - s[rec["claimed"]]))
    if not rows:
        return skip("m6_margin_distribution", "no exp1 records with scores")

    pairs = sorted({(m, c) for m, c, _ in rows})
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(pairs) + 1.9))
    rng = np.random.default_rng(0)      # jitter only; seeded so the figure is stable
    # x in axes fraction, y in data coordinates -- puts the per-row annotation
    # at a fixed distance from the right spine without pinning every row to the
    # same height, which is what a plain annotate() in axes coords would do.
    right = matplotlib.transforms.blended_transform_factory(ax.transAxes,
                                                            ax.transData)
    fractions = []
    for i, (model, cat) in enumerate(pairs):
        vals = np.array([v for m, c, v in rows if m == model and c == cat])
        ax.scatter(vals, np.full_like(vals, i) + rng.uniform(-0.16, 0.16, vals.size),
                   s=9, alpha=0.45, color=C_TEXT, edgecolors="none")
        ax.boxplot(vals, positions=[i], vert=False, widths=0.5, showfliers=False,
                   medianprops={"color": "black", "lw": 1.4},
                   boxprops={"color": "#4a5568"},
                   whiskerprops={"color": "#4a5568"},
                   capprops={"color": "#4a5568"})
        below = float((vals < 0).mean())
        fractions.append(below)
        ax.text(1.01, i, "{:.0%} favour text\nn = {}".format(below, vals.size),
                transform=right, va="center", ha="left", fontsize=7,
                color=C_CHANCE)
    ax.axvline(0, ls="--", lw=1.2, color="black")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(["{}\n{}".format(m.split("-")[0], PRETTY.get(c, c))
                        for m, c in pairs], fontsize=7)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlabel("decision margin, log P(audio-correct) - log P(claimed)   [nats]")
    ax.grid(axis="x", alpha=0.25)
    # State the range rather than rounding it up to "every item": the fraction
    # below zero runs from 0.82 to 1.00, so a claim of unanimity is false in
    # five of the six rows the figure draws.
    ax.set_title("Text preference shifts the whole distribution "
                 "({:.0%}-{:.0%} of items below zero)".format(
                     min(fractions), max(fractions)))
    save(fig, out_dir, "m6_margin_distribution")


# --------------------------------------------------------------------------- #
# m7 -- back-patching grid
# --------------------------------------------------------------------------- #
def m7_backpatch_grid(out_dir):
    """A null result, drawn so that it reads as a null.

    Two rows, and the second is not optional. On SALMONN the audio row rises
    from a 0.015 baseline to 0.18, which on its own reads as the intervention
    working; the capability row underneath shows the model losing 0.86 -> 0.10
    over the same cells. An audio-only panel would publish a broken model as a
    repaired one.

    The colour scale is pinned to [0, 1] rather than autoscaled, because
    autoscaling a flat field to its own noise range is how a null acquires a
    gradient.
    """
    blocks = per_model("exp4_backpatching_*.json")
    if not blocks:
        return skip("m7_backpatch_grid", "no exp4_backpatching_*.json")

    fig, axes = plt.subplots(2, len(blocks), figsize=(3.5 * len(blocks), 5.2),
                             squeeze=False)
    for col, (model, path, d) in enumerate(blocks):
        cat = cat_of(d, path)
        rows = d.get("sweep") or []
        dsts = sorted({r["dst"] for r in rows})
        wins = sorted({r["window"] for r in rows})
        for row, (key, label, cmap) in enumerate(
                (("audio", "audio accuracy", "Blues"),
                 ("capability", "capability (matched)", "Greens"))):
            ax = axes[row][col]
            grid = np.full((len(wins), len(dsts)), np.nan)
            for r in rows:
                if r.get(key) is not None:
                    grid[wins.index(r["window"]), dsts.index(r["dst"])] = r[key]
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                           vmin=0.0, vmax=1.0, interpolation="nearest")
            for i in range(len(wins)):
                for j in range(len(dsts)):
                    if not np.isnan(grid[i, j]):
                        # Flip to white once the cell is dark enough to swallow
                        # dark text -- the capability row saturates at 1.00.
                        ax.text(j, i, "{:.2f}".format(grid[i, j]), ha="center",
                                va="center", fontsize=7,
                                color="white" if grid[i, j] > 0.6 else "#1a202c")
            ax.set_xticks(range(len(dsts)))
            ax.set_xticklabels(dsts)
            ax.set_yticks(range(len(wins)))
            ax.set_yticklabels(wins)
            ax.set_ylabel("window" if col == 0 else "")
            base = (d.get("baseline") or {}).get(key)
            if row == 0:
                ctrl = (d.get("mid_layer_control") or {}).get("audio")
                bits = []
                if base is not None:
                    bits.append("baseline {:.3f}".format(base))
                if ctrl is not None:
                    bits.append("mid-layer src {:.3f}".format(ctrl))
                ax.set_title("{} -- {}\n{}".format(model.split("-")[0],
                                                   PRETTY.get(cat, cat),
                                                   ";  ".join(bits)), fontsize=8)
            else:
                ax.set_xlabel("destination layer")
                ax.set_title("{}{}".format(
                    label, "" if base is None else
                    "   (baseline {:.3f})".format(base)), fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle("Back-patching, and what it costs", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "m7_backpatch_grid")


# --------------------------------------------------------------------------- #
# m8 -- source sweep: gain against damage
# --------------------------------------------------------------------------- #
def m8_source_sweep(out_dir):
    """Why the null is a null: the only sources that move the margin are the
    ones that break the model.

    Two y-axes on purpose. The claim is about the *relationship* between the
    two curves -- gain arrives exactly where capability leaves -- and separate
    panels would make a reader do that join by eye.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp5_patch_diagnosis_*.json")
              if (d.get("test_b_source_sweep") or {}).get("sweep")]
    if not blocks:
        return skip("m8_source_sweep", "no exp5_patch_diagnosis_*.json")

    fig, axes = plt.subplots(1, len(blocks), figsize=(4.6 * len(blocks), 3.4),
                             squeeze=False)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        tb = d["test_b_source_sweep"]
        rows = sorted(tb["sweep"], key=lambda r: r["src"])
        xs = [r["src"] for r in rows]
        ax.plot(xs, [r.get("margin_delta") for r in rows], marker="o", ms=4,
                lw=1.6, color=C_AUDIO, label="margin gain (nats)")
        ax.axhline(0, ls="--", lw=1, color=C_CHANCE)
        ax.set_xlabel("source layer")
        ax.set_ylabel("margin gain vs. baseline [nats]", color=C_AUDIO)
        ax.tick_params(axis="y", labelcolor=C_AUDIO)
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(xs, [r.get("capability") for r in rows], marker="s", ms=4,
                 lw=1.6, ls="--", color=C_TEXT, label="capability")
        base_cap = (tb.get("baseline") or {}).get("capability")
        if base_cap is not None:
            ax2.axhline(base_cap, ls=":", lw=1.2, color=C_TEXT)
        ax2.set_ylabel("capability (matched items)", color=C_TEXT)
        ax2.tick_params(axis="y", labelcolor=C_TEXT)
        ax2.set_ylim(0, 1.05)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        ax.set_title("{} -- {}".format(model.split("-")[0],
                                       PRETTY.get(cat, cat)), fontsize=9)
    fig.suptitle("Any gain from the audio side is bought by breaking the model",
                 y=1.02)
    save(fig, out_dir, "m8_source_sweep")


# --------------------------------------------------------------------------- #
# m9 -- residual norm profile
# --------------------------------------------------------------------------- #
def m9_norm_profile(out_dir):
    """Residual-stream norm against depth, log scale.

    This is the mechanical reason a late-layer source cannot simply be written
    into an early layer: the two live at different scales, so the patch is an
    amplitude shock before it is an information transfer.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp5_patch_diagnosis_*.json")
              if (d.get("test_a_norms") or {}).get("profile")]
    if not blocks:
        return skip("m9_norm_profile", "no test_a_norms in exp5 files")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for model, path, d in blocks:
        prof = d["test_a_norms"]["profile"]
        xs = sorted(int(k) for k in prof)
        ax.plot(xs, [prof[str(x)] for x in xs], marker="o", ms=2.5, lw=1.5,
                label="{} -- {}".format(model.split("-")[0],
                                        PRETTY.get(cat_of(d, path), "")))
        ratio = d["test_a_norms"].get("late_over_early")
        if ratio:
            ax.annotate("late/early = {:.1f}x".format(ratio),
                        xy=(xs[0], prof[str(xs[-1])]), xytext=(6, 0),
                        textcoords="offset points", ha="left", fontsize=8,
                        color=C_CHANCE)
    ax.set_yscale("log")
    ax.set_xlabel("language model layer")
    ax.set_ylabel("residual-stream norm")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Late and early layers are not on the same scale")
    save(fig, out_dir, "m9_norm_profile")


# --------------------------------------------------------------------------- #
# m10 -- attention knockout by band
# --------------------------------------------------------------------------- #
def m10_knockout_bands(out_dir):
    """Geva et al.'s knockout, restricted to a band of layers.

    The positive control is drawn as a line, not a bar: it is the ceiling the
    method can reach when it certainly works, and every null below it has to be
    read against it.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp6_readout_attenuation_*.json")
              if d.get("sweep")]
    if not blocks:
        return skip("m10_knockout_bands", "no exp6_readout_attenuation_*.json")

    fig, axes = plt.subplots(1, len(blocks), figsize=(4.4 * len(blocks), 3.2),
                             squeeze=False, sharey=True)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        rows = d["sweep"]
        gammas = sorted({r["gamma"] for r in rows})
        bands = list(dict.fromkeys(r["band"] for r in rows))
        width = 0.8 / max(1, len(gammas))
        for i, g in enumerate(gammas):
            xs, ys = [], []
            for j, b in enumerate(bands):
                hit = [r for r in rows if r["band"] == b and r["gamma"] == g]
                if hit:
                    xs.append(j + i * width - 0.4 + width / 2)
                    ys.append(hit[0].get("margin_delta", 0.0))
            ax.bar(xs, ys, width=width * 0.9,
                   label="gamma = {:g}".format(g),
                   color=C_AUDIO if g == 0.0 else C_CHANCE)
            # Every bar here is a fraction of a nat against a positive control
            # of 2.57, so at a scale that shows the control they are invisible.
            # Print the value: an unreadably small bar and a missing bar look
            # identical otherwise.
            for x, y in zip(xs, ys):
                ax.annotate("{:+.3f}".format(y), xy=(x, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=6, color=C_CHANCE, rotation=90)
        pc = (d.get("positive_control") or {}).get("margin_delta")
        if pc is not None:
            ax.axhline(pc, ls="--", lw=1.2, color=C_NEUTRAL,
                       label="positive control")
            ax.set_ylim(top=pc * 1.32)   # headroom, or the legend lands on it
        ax.axhline(0, lw=0.8, color="black")
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands)
        ax.set_xlabel("layer band")
        ax.set_title("{} -- {}".format(model.split("-")[0],
                                       PRETTY.get(cat, cat)), fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("margin gain [nats]")
    axes[0][-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Knocking the claim out of a band of layers moves nothing", y=1.03)
    save(fig, out_dir, "m10_knockout_bands")


# --------------------------------------------------------------------------- #
# m11 -- the routing bisection
# --------------------------------------------------------------------------- #
def m11_routing(out_dir):
    """The central mechanistic result, as three bars.

    Blocking the answer position's own view of the claim changes nothing;
    blocking the positions in between recovers the entire effect. The claim
    reaches the readout by a detour, which is why the obvious intervention is
    null.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp6_readout_attenuation_*.json")
              if d.get("routing")]
    if not blocks:
        return skip("m11_routing", "no routing block in exp6 files")

    labels = {"gen": "answer position\nonly", "post_claim": "positions\nin between",
              "all": "every\nposition"}
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.8 * len(blocks), 3.2),
                             squeeze=False, sharey=True)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        route = d["routing"]
        xs = np.arange(len(route))
        ax.bar(xs, [r.get("margin_delta", 0.0) for r in route],
               color=C_AUDIO, width=0.6)
        ax.axhline(0, lw=0.9, color="black")
        pc = (d.get("positive_control") or {}).get("margin_delta")
        if pc is not None:
            ax.axhline(pc, ls="--", lw=1.2, color=C_NEUTRAL,
                       label="positive control")
        ax.set_xticks(xs)
        ax.set_xticklabels([labels.get(r["rows"], r["rows"]) for r in route],
                           fontsize=8)
        rec = d.get("routing_recovered_fraction")
        ax.set_title("{} -- {}{}".format(
            model.split("-")[0], PRETTY.get(cat, cat),
            "\nrecovered fraction {:.2f}".format(rec) if rec is not None else ""),
            fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="center left", fontsize=7)
    axes[0][0].set_ylabel("margin gain [nats]")
    fig.suptitle("The claim reaches the answer position by a detour", y=1.03)
    save(fig, out_dir, "m11_routing")


# --------------------------------------------------------------------------- #
# m12 -- the generative intervention and its controls
# --------------------------------------------------------------------------- #
def m12_generative_fix(out_dir):
    """The free-form result with the two controls that decide whether it counts.

    gen-TDR excludes descriptions naming several labels or none, so its
    denominator moves between conditions. That denominator is printed under
    each point: the verdict logic refuses to call an effect below 15, and a
    reader has to be able to see the same thing.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp6_readout_attenuation_*.json")
              if d.get("freeform")]
    if not blocks:
        return skip("m12_generative_fix", "no freeform block in exp6 files")

    fig, axes = plt.subplots(1, len(blocks), figsize=(4.8 * len(blocks), 3.6),
                             squeeze=False, sharey=True)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        # gamma=0.25 is inert at every band; interleaving it turns the panel
        # into a sawtooth that hides the comparison. It belongs in the text.
        ff = [r for r in d["freeform"]
              if r.get("band") is None or r.get("gamma") in (None, 0.0)]
        if not ff:
            ax.set_visible(False)
            continue
        xs = np.arange(len(ff))
        names = [r["condition"].replace(" g=0", "") for r in ff]

        gt = [(r.get("mismatched") or {}).get("generative_tdr") for r in ff]
        ci = [(r.get("mismatched") or {}).get("generative_tdr_ci95") for r in ff]
        lo = [(g - c[0]) if (g is not None and c) else 0 for g, c in zip(gt, ci)]
        hi = [(c[1] - g) if (g is not None and c) else 0 for g, c in zip(gt, ci)]
        ax.errorbar(xs, [g if g is not None else np.nan for g in gt],
                    yerr=[lo, hi], marker="o", ms=5, lw=1.6, capsize=3,
                    color=C_TEXT, label="follows the false claim (gen-TDR)")
        ax.plot(xs, [(r.get("matched") or {}).get("names_true_rate") for r in ff],
                marker="s", ms=4, lw=1.4, color=C_NEUTRAL,
                label="capability (claim true)")
        txt = [(r.get("text_only") or {}).get("names_true_rate") for r in ff]
        if any(v is not None for v in txt):
            ax.plot(xs, [v if v is not None else np.nan for v in txt],
                    marker="^", ms=4, lw=1.4, ls=":", color=C_CHANCE,
                    label="cost (no audio)")
        # The denominator, which is what makes an apparent effect readable.
        # Sits above the plot rather than below it: under the axis it collides
        # with the rotated condition labels.
        for x, r in zip(xs, ff):
            n_dec = (r.get("mismatched") or {}).get("decided")
            if n_dec is not None:
                ax.annotate("n={}".format(n_dec), xy=(x, 1.015),
                            xycoords=("data", "axes fraction"), ha="center",
                            fontsize=6.5, fontweight="bold" if n_dec < 15 else "normal",
                            color=C_TEXT if n_dec < 15 else C_CHANCE)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        # pad clears the n= row printed just above the axes
        ax.set_title("{} -- {}".format(model.split("-")[0],
                                       PRETTY.get(cat, cat)), fontsize=9, pad=16)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("rate")
    axes[0][-1].legend(loc="center left", fontsize=7)
    fig.suptitle("Attenuating the claim in open generation "
                 "(n = descriptions naming exactly one label)", y=1.03)
    save(fig, out_dir, "m12_generative_fix")


# --------------------------------------------------------------------------- #
# m13 -- generation quality
# --------------------------------------------------------------------------- #
def m13_generation_quality(out_dir):
    """Does the intervention wreck the text? Cho et al. found circuit ablation
    did, and a multiple-choice capability control cannot see that.

    Log-probability is scored in a claim-free context, which is the load-bearing
    detail: scored against the claim-bearing prompt, text that correctly
    contradicts the claim is penalised as improbable and a working intervention
    reports as degeneration.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp6_readout_attenuation_*.json")
              if d.get("freeform")]
    if not blocks:
        return skip("m13_generation_quality", "no freeform block in exp6 files")

    metrics = [("mean_token_logprob", "mean per-token log-prob"),
               ("repetition_rate", "repetition rate"),
               ("median_tokens", "median length [tokens]")]
    fig, axes = plt.subplots(len(blocks), 3, figsize=(9.6, 2.9 * len(blocks)),
                             squeeze=False)
    for row, (model, path, d) in enumerate(blocks):
        cat = cat_of(d, path)
        ff = [r for r in d["freeform"]
              if r.get("band") is None or r.get("gamma") in (None, 0.0)]
        names = [r["condition"].replace(" g=0", "") for r in ff]
        xs = np.arange(len(ff))
        for col, (key, label) in enumerate(metrics):
            ax = axes[row][col]
            for sub, colour, marker in (("mismatched", C_TEXT, "o"),
                                        ("matched", C_NEUTRAL, "s")):
                ys = [(r.get(sub) or {}).get(key) for r in ff]
                if any(v is not None for v in ys):
                    ax.plot(xs, [v if v is not None else np.nan for v in ys],
                            marker=marker, ms=4, lw=1.4, color=colour, label=sub)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if col == 0:
                ax.text(-0.34, 0.5, "{}\n{}".format(model.split("-")[0],
                                                    PRETTY.get(cat, cat)),
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=8, fontweight="bold")
        axes[row][-1].legend(fontsize=7, loc="best")
    fig.suptitle("Generation quality under the intervention "
                 "(scored by the unmodified model, claim removed)", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "m13_generation_quality")


# --------------------------------------------------------------------------- #
# m14 -- three-way outcome mass
# --------------------------------------------------------------------------- #
def m14_outcome_mass(out_dir):
    """Where the answers actually go, across every model we ran.

    LISTEN's point: a binary text-vs-audio ratio hides the items matching
    neither cue, and on some models that residual is most of the mass. Stacked
    bars keep all three visible.
    """
    sums = {}
    for path in found("results", "*_summary.json"):
        d = load(path)
        if d and "per_category" in d:
            sums[os.path.basename(path).replace("_summary.json", "")] = d
    if not sums:
        return skip("m14_outcome_mass", "no *_summary.json files")

    models = sorted(sums)
    cats = [c for c in CATEGORIES
            if any(c in s["per_category"] for s in sums.values())]
    fig, axes = plt.subplots(1, len(cats), figsize=(3.1 * len(cats), 3.4),
                             squeeze=False, sharey=True)
    for ax, cat in zip(axes[0], cats):
        ys = np.arange(len(models))
        text, audio, neither = [], [], []
        for m in models:
            t = (sums[m]["per_category"].get(cat) or {}).get("tdr") or {}
            n = t.get("n_usable") or 0
            text.append((t.get("follow_text") or 0) / n if n else 0)
            audio.append((t.get("follow_audio") or 0) / n if n else 0)
            neither.append((t.get("follow_neither") or 0) / n if n else 0)
        ax.barh(ys, text, color=C_TEXT, label="follows text")
        ax.barh(ys, audio, left=text, color=C_AUDIO, label="follows audio")
        ax.barh(ys, neither, left=np.add(text, audio), color=C_CHANCE,
                label="neither")
        ax.set_yticks(ys)
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of conflict items")
        ax.set_title(PRETTY.get(cat, cat), fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    axes[0][-1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Under conflict, where does the answer come from?", y=1.02)
    save(fig, out_dir, "m14_outcome_mass")


# --------------------------------------------------------------------------- #
# m15 -- the capability gate
# --------------------------------------------------------------------------- #
def m15_capability_gate(out_dir):
    """The screen that decides which results may be read as suppression.

    This is the figure behind two exclusions in the write-up -- SALMONN and the
    sarcasm category -- so it earns a place in the paper rather than the
    appendix. A bar that fails to clear its own chance line is the whole
    argument for setting that model or category aside.
    """
    rows = []
    for model, root in roots(every=True):      # exempt: see MECH_MODELS
        d = load(os.path.join(root, "exp1_capability_screen.json"))
        if not d:
            continue
        for cat in CATEGORIES:
            sub = [r for r in (d.get("records") or [])
                   if r["category"] == cat and r["condition"] == "Paralinguistic"
                   and r["modality"] == "audio_only"]
            if sub:
                k = sum(bool(r["correct"]) for r in sub)
                rows.append((model, cat, k, len(sub), 1.0 / sub[0]["n_options"],
                             (d.get("screen_verdicts") or {}).get(cat)))
    if not rows:
        return skip("m15_capability_gate", "no exp1_capability_screen.json")

    models = list(dict.fromkeys(r[0] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        mine = [r for r in rows if r[0] == model]
        xs = np.arange(len(mine))
        for x, (_, cat, k, n, ch, passed) in zip(xs, mine):
            p, lo, hi = err_bars(k, n)
            ax.bar([x], [p], width=0.55,
                   color=C_NEUTRAL if passed else C_CHANCE,
                   yerr=[[lo], [hi]], capsize=3, error_kw={"lw": 0.9})
            ax.plot([x - 0.32, x + 0.32], [ch, ch], color=C_TEXT, lw=1.6)
            ax.annotate("{}/{}".format(k, n), xy=(x, p + hi),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=C_CHANCE)
        ax.set_xticks(xs)
        ax.set_xticklabels([PRETTY.get(r[1], r[1]) for r in mine],
                           rotation=25, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(model, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("audio-only accuracy")
    # Proxy artists: an empty bar() produces a container with no patch, so the
    # legend falls back to the colour cycle and both swatches come out blue.
    handles = [matplotlib.lines.Line2D([], [], color=C_TEXT, lw=1.6,
                                       label="chance"),
               matplotlib.patches.Patch(color=C_NEUTRAL, label="passes screen"),
               matplotlib.patches.Patch(color=C_CHANCE, label="fails -- excluded")]
    axes[0][-1].legend(handles=handles, loc="upper right", fontsize=7)
    fig.suptitle("Can the model do the task from audio alone? "
                 "(the gate on every result below)", y=1.03)
    save(fig, out_dir, "m15_capability_gate")


# --------------------------------------------------------------------------- #
FIGURES = {
    "m1c": m1c_attribution_grid, "m2": m2_faithfulness, "m3": m3_group_magnitude,
    "m4": m4_circuit_overlap, "m5": m5_probe_depth, "m6": m6_margin_distribution,
    "m7": m7_backpatch_grid, "m8": m8_source_sweep, "m9": m9_norm_profile,
    "m10": m10_knockout_bands, "m11": m11_routing, "m12": m12_generative_fix,
    "m13": m13_generation_quality, "m14": m14_outcome_mass,
    "m15": m15_capability_gate,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure keys to build, e.g. --only m1 m6")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="categories for the mechanism figures (default: {}). "
                         "m14 and m15 always draw every category."
                         .format(", ".join(MECH_CATEGORIES)))
    ap.add_argument("--models", nargs="*", default=None,
                    help="models for the mechanism figures (default: {}). "
                         "m14 and m15 always draw every model."
                         .format(", ".join(MECH_MODELS)))
    args = ap.parse_args(argv)

    if args.models:
        known = [m for m, _ in MODEL_ROOTS]
        bad = [m for m in args.models if m not in known]
        if bad:
            raise SystemExit("unknown model(s): {}\nknown: {}".format(
                ", ".join(bad), ", ".join(known)))
        globals()["MECH_MODELS"] = list(args.models)

    if args.categories:
        bad = [c for c in args.categories if c not in CATEGORIES]
        if bad:
            raise SystemExit("unknown categor(ies): {}\nknown: {}".format(
                ", ".join(bad), ", ".join(CATEGORIES)))
        globals()["MECH_CATEGORIES"] = list(args.categories)

    keys = args.only or list(FIGURES)
    unknown = [k for k in keys if k not in FIGURES]
    if unknown:
        raise SystemExit("unknown figure(s): {}\nknown: {}".format(
            ", ".join(unknown), ", ".join(FIGURES)))

    print("writing {}/  (mechanism figures: {} | {})".format(
        args.out, ", ".join(MECH_MODELS), ", ".join(MECH_CATEGORIES)))
    for k in keys:
        try:
            FIGURES[k](args.out)
        except Exception as exc:                  # one bad figure must not
            print("  ERROR in {}: {}: {}".format(  # block the rest
                k, type(exc).__name__, exc))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
