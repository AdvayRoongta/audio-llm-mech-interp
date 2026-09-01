#!/usr/bin/env python3
"""The three figures for the mechanistic-interpretability section.

Numbered in the order the argument runs, and named for what they draw. The
number, the function name and the output filename are the same string, so
`--only 2` builds `fig2_attribution_by_layer_and_position.{pdf,png}` and there
is nothing to look up:

  fig1  probe accuracy by encoder layer      the audio signal survives the
        line -- exp2_encoder_probe           encoder, so this is not a
                                             perception failure
        Pasad et al. 2021; Hewitt & Liang 2019 for the shuffled-label control

  fig2  attribution by layer and position    but the decision is driven by the
        heatmap -- exp3_circuits             text claim at the answer position
        ROME (Meng et al. 2022) fig 2; Chen et al. 2026 fig 3, the audio-LALM
        version of the same grid

  fig3  knockout by token position           and the claim reaches the answer
        bars -- exp6_readout_attenuation     through the tokens in between
        Geva et al. 2023 fig 3

Each figure names the convention it follows, because a reviewer reads a figure
type before reading its caption and a familiar shape does half the explaining.

Usage, from the repo root:

    python3 viz/make_mechinterp_figures.py
    python3 viz/make_mechinterp_figures.py --out figures --only 1 3

Every figure is skipped with a printed reason if its inputs are absent, so this
is safe to run against a partial results tree.
"""
import argparse
import glob
import json
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

# Position groups in sequence order, so fig2's y-axis reads as a spatial map of
# the prompt the way Chen et al. lay theirs out.
GROUPS = ["audio", "claim", "query", "options", "gen"]
GROUP_LABEL = {"audio": "audio tokens", "claim": "text claim", "query": "question",
               "options": "answer options", "gen": "answer position"}

# fig3's conditions, named for what stops seeing the claim rather than for the
# JSON key. "positions in between" only parses once you already know the
# mechanism, which is the thing the figure is supposed to teach.
ROW_ORDER = ["gen", "post_claim", "all"]
ROW_LABEL = {"gen": "answer\nposition",
             "post_claim": "every position\nafter the claim",
             "all": "every\nposition"}

# Where each model's results live. Qwen writes to the top level for historical
# reasons; every later model gets its own subdirectory.
MODEL_ROOTS = [("Qwen2-Audio-7B", "results"),
               ("SALMONN-13B", os.path.join("results", "salmonn"))]

# SALMONN is excluded from the mechanistic analysis: it does not clear the
# capability screen, and a model that cannot do the task from audio cannot be
# shown to be suppressing audio. So these figures are Qwen-only by default.
# Override with --models to draw a panel per model again. The evidence for the
# exclusion is the capability screen, which is a different section of the paper
# and not one of these three figures.
MECH_MODELS = ["Qwen2-Audio-7B"]

# sarcasm_pacing is run but not reported: its audio does not settle a single
# correct label (removing the claim lifts audio accuracy only to 0.233 against
# 0.100 chance), and its free-form descriptions name one identifiable label
# only ~10 times in 30, so the intervention's intervals overlap the baseline.
# Underpowered rather than negative -- excluded here for the same reason it is
# excluded from the write-up. --categories overrides.
MECH_CATEGORIES = ["emotional_prosody", "speaker_identity"]

# Circuits drawn in fig2. The audio panels are near-blank on a scale shared
# with the text panels, which is the finding but costs half the figure width to
# show it -- and a blank panel reads as a rendering failure as easily as a
# result. The text-vs-audio comparison is carried numerically in the write-up
# (text 3.29 against audio 0.18), so the heatmap shows the text circuit alone.
# Set to ("text", "audio") to restore.
HEATMAP_CIRCUITS = ("text",)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def roots(every=False):
    """Model roots for a figure. `every=True` opts out of the model filter."""
    if every:
        return MODEL_ROOTS
    return [(m, r) for m, r in MODEL_ROOTS if m in MECH_MODELS]


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


def ci_err(p, ci):
    """(point, lower_err, upper_err) from a stored absolute interval.

    exp6 writes ci95 as bounds on the rate, not as k/n, so a Wilson helper does
    not apply. The p=0 rows store a lower bound of -6.5e-18, so clamp.
    """
    if not ci:
        return p, 0.0, 0.0
    return p, max(0.0, p - ci[0]), max(0.0, ci[1] - p)


def chance_for(root, cat):
    """1/n_options for a category, read off the capability screen.

    exp6 does not record the option count and the categories do not share one:
    prosody is 8-way (0.125), identity 2-way (0.500). Hardcoding it would put a
    wrong reference line on an accuracy figure, which is a wrong claim rather
    than a cosmetic slip.
    """
    d = load(os.path.join(root, "exp1_capability_screen.json"))
    for r in (d or {}).get("records") or []:
        if r.get("category") == cat and r.get("n_options"):
            return 1.0 / r["n_options"]
    return None


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


# --------------------------------------------------------------------------- #
# fig1 -- the signal is in the encoder
# --------------------------------------------------------------------------- #
def fig1_probe_accuracy_by_encoder_layer(out_dir):
    """Probe accuracy by encoder layer.

    One line, one chance rate, and the layer where the line first reaches
    ceiling. Drawn as a line rather than a strip because position is a finer channel than
    lightness, and because the dips are the honest part. The write-up says
    accuracy is "1.000 from layer 13 onward"; it is not. It touches 1.000 at 13,
    falls to 0.975 at 21 and 0.988 at 22 and 27, and only holds ceiling from 28.
    A line plot shows that without anyone having to say it; a heatmap of the same
    numbers hides it in two shades of the same blue.

    Scope: exp2 covers speaker_identity only, so this rules out encoder loss for
    speaker gender and for nothing else. There is no Qwen encoder probe for
    emotional prosody -- the open gap in the argument chain -- and the panel says
    so rather than leaving a reader to assume the result generalises.
    """
    blocks = []
    for model, root in roots():
        for path in found(root, "exp2_encoder_probe*.json"):
            d = load(path) or {}
            for r in d.get("results", []):
                if r.get("category") in MECH_CATEGORIES:
                    blocks.append((model, r))
    if not blocks:
        return skip("fig1_probe_accuracy_by_encoder_layer", "no exp2_encoder_probe*.json")

    fig, axes = plt.subplots(1, len(blocks), figsize=(5.4 * len(blocks), 3.6),
                             squeeze=False, sharey=True)
    for ax, (model, r) in zip(axes[0], blocks):
        layers = r["layers"]
        # the last entry is the projector, a different module on a different
        # x-scale; it gets its own tick rather than being tacked onto the line.
        enc = [x for x in layers if str(x.get("name", "")).startswith("encoder")]
        proj = next((x for x in layers
                     if not str(x.get("name", "")).startswith("encoder")), None)
        xs = [x["layer"] for x in enc]
        acc = [x["accuracy"] for x in enc]
        ch = r.get("chance")

        ax.set_axisbelow(True)
        ax.plot(xs, acc, marker="o", ms=3.2, lw=1.8, color=C_AUDIO, zorder=3)
        if ch is not None:
            ax.axhline(ch, ls=":", lw=1.3, color=C_CHANCE, zorder=1)

        # First layer at ceiling, and the last dip after it. Both are read off
        # the data: "from layer 13 onward" is the claim the figure has to be
        # able to contradict.
        top = max(acc)
        first = next((x for x, a in zip(xs, acc) if a >= top - 1e-9), None)
        dips = [(x, a) for x, a in zip(xs, acc) if x > first and a < top - 1e-9]
        if first is not None:
            ax.axvline(first, ls="--", lw=1.2, color=C_TEXT, zorder=2)
            ax.annotate("first reaches {:.3f}\nat layer {}".format(top, first),
                        xy=(first, top), xytext=(6, -14),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=7, color=C_TEXT, annotation_clip=False)
        if dips:
            lx, la = dips[-1]
            ax.annotate("dips to {:.3f} at {}".format(
                            min(a for _, a in dips),
                            min(dips, key=lambda t: t[1])[0]),
                        xy=(lx, la), xytext=(0, -12),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6.5, color=C_CHANCE)

        if proj is not None:
            px = xs[-1] + 2
            ax.scatter([px], [proj["accuracy"]], marker="D", s=40,
                       color=C_NEUTRAL, zorder=5)
            ax.annotate("post-\nprojector", xy=(px, proj["accuracy"]),
                        xytext=(0, -12), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color=C_NEUTRAL)
            ax.set_xticks(list(range(0, xs[-1] + 1, 4)) + [px])
            ax.set_xticklabels([str(v) for v in range(0, xs[-1] + 1, 4)]
                               + ["proj"], fontsize=7.5)

        ax.set_xlabel("audio encoder layer")
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title("{} -- {}{}".format(
            model.split("-")[0], PRETTY.get(r["category"], r["category"]),
            "   (chance {:.3f})".format(ch) if ch is not None else ""),
            fontsize=9)

    axes[0][0].set_ylabel("linear-probe accuracy")
    cat = blocks[0][1]
    fig.suptitle("The encoder does not discard the signal", y=1.10, fontsize=11)
    fig.text(0.5, 1.03,
             "{} clips, {}, {} actors.  exp2 covers speaker_identity only: this "
             "rules out encoder loss for speaker gender, not for prosody."
             .format(cat.get("n_clips"), cat.get("cv_scheme", ""),
                     cat.get("n_actors")),
             ha="center", fontsize=6.8, color=C_CHANCE)
    save(fig, out_dir, "fig1_probe_accuracy_by_encoder_layer")


# --------------------------------------------------------------------------- #
# fig2 -- but the decision is made from the text
# --------------------------------------------------------------------------- #
def fig2_attribution_by_layer_and_position(out_dir):
    """Attribution by decoder layer and prompt position, one panel per category.

    Attention and MLP summed, one shared vmax so the categories are comparable
    by eye. Chen et al. facet all their attributes into one grid for the same
    reason.

    No suptitle, by request. The bold category labels sit above their panels
    and the colourbar names the quantity. What the title carried and nothing
    else on the figure does is the model and the circuit, so the caption has
    to say "Qwen2-Audio-7B, text circuit" -- without a known depth the layer
    axis means nothing.

    An earlier version split attn and MLP into separate rows. Dropped -- but
    the split carried a result worth keeping in the text: text mass at the
    answer position is 72% attention (15.1 vs 6.0) while audio mass at the
    audio tokens is 71% MLP (1.44 vs 0.58). The claim is routed to the
    readout; the acoustics are processed in place and largely not moved.
    """
    blocks = per_model("exp3_circuits_*.json")
    if not blocks:
        return skip("fig2_attribution_by_layer_and_position", "no exp3_circuits_*.json")

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

    # One row: attention and MLP summed per (layer, group). The question is
    # where the mass sits, not which mechanism carries it.
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
    save(fig, out_dir, "fig2_attribution_by_layer_and_position")


# --------------------------------------------------------------------------- #
# fig3 -- and it gets there through the tokens in between
# --------------------------------------------------------------------------- #
def fig3_knockout_by_position(out_dir):
    """Which token positions have to lose sight of the claim.

    Three conditions, layers held at all 32 and gamma at 0, differing only in
    which query rows stop attending to the claim. Blocking the answer position
    alone changes nothing; blocking every position downstream of the claim
    recovers the whole effect. The claim is not read directly off the claim
    tokens at the readout -- it is relayed through the tokens in between.

    The layer-band sweep is not drawn. Every band is null at the answer
    position, and half a panel of flat caps buys nothing that the footnote does
    not say in one line; the numbers are in exp6.sweep and in the text.

    Audio-consistent accuracy, which is how the write-up states this result and
    what a reader can hold against a chance line. The margin is the more
    sensitive metric and lives in exp6.routing; it is not drawn. Prosody's
    answer-position bar is 0.000 exactly, so the bar outline and the Wilson
    whisker carry the "this ran and measured nothing" job on their own.

    Capability is NOT drawn, by request. It matters and it is not flattering:
    blocking every position after the claim lifts prosody 0.000 -> 0.533 while
    matched-item accuracy falls 1.000 -> 0.367, so most of that gain is the
    model no longer reading text at all. Identity costs less, 1.000 -> 0.800.
    Those numbers are in exp6.routing[].capability and belong in the caption if
    they are not on the axis -- otherwise the panel reads as a repair, and a
    gain bought by damaging the model is not a gain.

    gamma = 0 only. gamma = 0.25 is a null at every band, and interleaving
    inert bars turns the panel into a sawtooth. The null belongs in the text.
    """
    blocks = [(m, p, d) for m, p, d in per_model("exp6_readout_attenuation_*.json")
              if d.get("sweep") and d.get("routing")]
    if not blocks:
        return skip("fig3_knockout_by_position", "no exp6 sweep+routing blocks")

    # Wide panels on purpose: the two-line tick labels need about 0.83in of
    # tick pitch before "every position / after the claim" touches its
    # neighbour.
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.9 * len(blocks), 3.9),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(wspace=0.42)
    for ax, (model, path, d) in zip(axes[0], blocks):
        cat = cat_of(d, path)
        base = d.get("baseline") or {}

        route = {r["rows"]: r for r in d["routing"]}
        rows = [route[k] for k in ROW_ORDER if k in route]
        if not rows:
            continue
        ticks = [ROW_LABEL[k] for k in ROW_ORDER if k in route]
        xs = list(range(len(rows)))
        ys = [r.get("audio") or 0.0 for r in rows]
        errs = [ci_err(y, r.get("ci95")) for y, r in zip(ys, rows)]

        ax.set_axisbelow(True)
        ch = chance_for(os.path.dirname(path), cat)

        # edgecolor is load-bearing, not decoration: prosody's answer-position
        # condition is 0.000 exactly, and a zero-height bar with no edge draws
        # nothing at all -- the panel reads as two bars and one missing one.
        # With an edge, a measured zero renders as a flat cap the width of a
        # bar, which is what it is.
        ax.bar(xs, ys, width=0.62, color=C_AUDIO, edgecolor=C_AUDIO,
               linewidth=1.4, zorder=2,
               yerr=[[e[1] for e in errs], [e[2] for e in errs]],
               capsize=3, error_kw={"lw": 0.9, "zorder": 3})
        # A zero-height bar and a missing bar are the same picture, and
        # prosody's answer-position bar is 0.000. The printed rate, the Wilson
        # whisker and the bar outline are what say the run happened and
        # measured nothing.
        for x, y, e in zip(xs, ys, errs):
            ax.annotate("{:.3f}".format(y), xy=(x, y + e[2]), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=C_CHANCE, zorder=4)

        ax.axhline(0, lw=0.8, color="black", zorder=1)
        # bottom below zero on purpose: prosody's baseline is exactly 0.000 and
        # would otherwise be drawn on the bottom spine and read as the axis.
        ax.set_ylim(-0.04, 1.02)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=6.5, linespacing=1.15)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        # Chance goes in the title, not on the line. The categories do not
        # share a rate (0.125 against 0.500) and every spot beside the line is
        # already occupied.
        ax.set_title("{} -- {}{}".format(
            model.split("-")[0], PRETTY.get(cat, cat),
            "   (chance {:.3f})".format(ch) if ch is not None else ""),
            fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # Reference lines. Dotted against dashed, because prosody's baseline is
        # 0.000 and sits on the axis floor: the two have to be told apart by
        # style, not only by position.
        b_aud = base.get("audio")
        if b_aud is not None:
            ax.axhline(b_aud, ls="--", lw=1.2, color=C_TEXT, zorder=1)
        if ch is not None:
            ax.axhline(ch, ls=":", lw=1.2, color=C_CHANCE, zorder=1)

        # Group headers, x in data and y in axes fraction so the offsets do not
        # move with the y range. The "held" line is load-bearing: without it,
        # "which layers are blocked" invites the reader to assume the positions
        # varied too, and the 2x2 collapses into six unrelated conditions.
        # These three offsets assume two-line 7pt tick labels.
        tr = ax.get_xaxis_transform()
        for x0, x1, head in ((-0.45, len(rows) - 0.55,
                              "which positions are blocked"),):
            ax.plot([x0, x1], [-0.17, -0.17], transform=tr, color=C_CHANCE,
                    lw=0.9, clip_on=False)
            ax.text((x0 + x1) / 2, -0.21, head, transform=tr, ha="center",
                    va="top", fontsize=8, fontweight="bold", color="#1a202c")

    # Proxy artists: a bar container and a twin-axis line do not produce a
    # coherent auto-legend. Upper right is unavailable -- the position bars
    # live there.
    # The subtitle asserts the layer sweep is null. Derive it rather than
    # writing it down: a later run with a live band would otherwise be
    # published under a caption saying it is dead. 1 nat is the threshold the
    # experiment's own verdict logic uses.
    band_peak = max(
        (abs(r.get("margin_delta") or 0.0)
         for _, _, d in blocks
         for r in d.get("sweep") or [] if r.get("gamma") in (None, 0.0)),
        default=None)
    if band_peak is None:
        band_note = ""
    elif band_peak < 1.0:
        band_note = ("Restricting the same knockout to a band of layers is "
                     "null at every band  (exp6.sweep)")
    else:
        band_note = ("Restricting the same knockout to a band of layers moves "
                     "the margin by up to {:.3f} nats  (exp6.sweep)"
                     .format(band_peak))

    n = (blocks[0][2].get("run") or {}).get("n_items")
    handles = [
        matplotlib.patches.Patch(
            color=C_AUDIO,
            label="audio-consistent accuracy"
                  + (" (n={} mismatched)".format(n) if n else "")),
        matplotlib.lines.Line2D([], [], color=C_TEXT, ls="--", lw=1.2,
                                label="no intervention"),
    ]
    # Only the outermost axes carry a unit label; an inner twin label lands on
    # the next panel's y-axis.
    # The denominator belongs to the bars, not to the axis: capability is
    # scored on the Matched items, a disjoint list, and exp6 does not record
    # how many of those there were. An n on the shared axis label would claim
    # both series share it.
    axes[0][0].set_ylabel("accuracy")
    # Figure-level, in the header band. With three bars to a panel there is no
    # in-axes region left that does not sit on a bar, the capability line or a
    # reference line.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=3, fontsize=7)
    fig.suptitle("The claim reaches the answer through the tokens in between",
                 y=1.17, fontsize=11)
    # The layer sweep is the other half of this result. One line under the
    # title is worth more than half a panel of flat caps, and it goes here
    # rather than at the foot, where it lands on the group headers.
    if band_note:
        fig.text(0.5, 1.10, band_note, ha="center", fontsize=6.8,
                 color=C_CHANCE)
    save(fig, out_dir, "fig3_knockout_by_position")

# --------------------------------------------------------------------------- #
# the set, in argument order
# --------------------------------------------------------------------------- #
# A list, not a dict: the order is part of the numbering, and the --only key is
# the position. Each function's name is also its output filename.
FIGURES = [
    fig1_probe_accuracy_by_encoder_layer,
    fig2_attribution_by_layer_and_position,
    fig3_knockout_by_position,
]


def resolve(keys):
    """--only values -> functions. Accepts '1', 'fig1', or 'probe'."""
    chosen, unknown = [], []
    for k in keys:
        want = str(k).lower().lstrip("f").lstrip("ig")
        hit = None
        for i, fn in enumerate(FIGURES, 1):
            if str(k) in (str(i), "fig{}".format(i)) or want == str(i):
                hit = fn
            elif str(k).lower() in fn.__name__:
                hit = fn
            if hit:
                break
        (chosen if hit else unknown).append(hit or k)
    if unknown:
        raise SystemExit("unknown figure(s): {}\nknown: {}".format(
            ", ".join(map(str, unknown)),
            ", ".join("{} = {}".format(i, fn.__name__)
                      for i, fn in enumerate(FIGURES, 1))))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="figures to build by number, e.g. --only 1 3")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="categories to draw (default: {})"
                         .format(", ".join(MECH_CATEGORIES)))
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to draw (default: {})"
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

    print("writing {}/  ({} | {})".format(
        args.out, ", ".join(MECH_MODELS), ", ".join(MECH_CATEGORIES)))
    for fn in resolve(args.only) if args.only else FIGURES:
        try:
            fn(args.out)
        except Exception as exc:                  # one bad figure must not
            print("  ERROR in {}: {}: {}".format(  # block the other two
                fn.__name__, type(exc).__name__, exc))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
