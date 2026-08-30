#!/usr/bin/env python3
"""The encoder probe as a standalone paper figure -- one panel, no chance line.

`make_mechinterp_figures.py:m5_probe_depth` draws the same sweep inside the
mechanism set, with the chance rule and a per-model panel grid. This script is
the version that goes in the paper body: a single wide panel, Qwen only, and no
chance line -- the shuffled-label control is the reference the reader needs, and
a second horizontal rule at 0.5 competes with it for the same job.

House style (rcParams, colours, the pdf+png pair) is copied from
`make_mechinterp_figures.py` on purpose, so this figure sits in the same visual
system as the rest of the plates.

Usage, from the repo root:

    python3 viz/make_probe_depth_figure.py
    python3 viz/make_probe_depth_figure.py --out figures --width 6.5 --height 3.4
"""
import argparse
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("needs matplotlib:  pip3 install matplotlib")

# --------------------------------------------------------------------------- #
# house style -- identical to make_mechinterp_figures.py
# --------------------------------------------------------------------------- #
C_AUDIO, C_CHANCE, C_NEUTRAL = "#2b6cb0", "#718096", "#2f855a"

PRETTY = {"emotional_prosody": "Emotional prosody",
          "speaker_identity": "Speaker identity",
          "sarcasm_pacing": "Sarcasm / pacing"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})

SOURCE = os.path.join("results", "exp2_encoder_probe.json")
CATEGORY = "speaker_identity"


def load_block(path, category):
    """The one `results[]` entry for `category`, or None."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for r in d.get("results", []):
        if r.get("category") == category:
            return r
    return None


def probe_depth(out_dir, name, size):
    r = load_block(SOURCE, CATEGORY)
    if not r:
        raise SystemExit("no {} block in {}".format(CATEGORY, SOURCE))

    layers = r["layers"]
    enc, proj = layers[:-1], layers[-1]
    xs = [p["layer"] for p in enc]
    acc = [p["accuracy"] for p in enc]
    shuf = [p["shuffled"] for p in enc]
    # the projector sits three units out, not one: the gap is what stops it
    # reading as "layer 33", and it keeps its tick label clear of layer 32's
    x_proj = xs[-1] + 3
    x_break = xs[-1] + 1.5

    fig, ax = plt.subplots(figsize=size)

    # selectivity: the probe's real reading over what it gets from noise
    ax.fill_between(xs, shuf, acc, color=C_AUDIO, alpha=0.13, lw=0,
                    label="selectivity")
    ax.plot(xs, acc, marker="o", ms=2.5, lw=1.5, color=C_AUDIO,
            label="linear probe")
    ax.plot(xs, shuf, ls=":", lw=1.2, color=C_CHANCE, label="shuffled labels")

    # the projector is a different module, so it is drawn off the encoder line
    ax.axvline(x_break, lw=0.8, ls=(0, (1, 3)), color="0.65", zorder=1)
    ax.plot([xs[-1], x_proj], [acc[-1], proj["accuracy"]], ls=(0, (1, 2)),
            lw=1.0, color=C_AUDIO, alpha=0.6, zorder=2)
    ax.scatter([x_proj], [proj["accuracy"]], marker="D", s=34,
               color=C_NEUTRAL, zorder=5, label="post-projector")

    ticks = [t for t in range(0, xs[-1] + 1, 4)]
    ax.set_xticks(ticks + [x_proj])
    ax.set_xticklabels([str(t) for t in ticks] + ["proj"])
    ax.set_xlim(-1, x_proj + 1.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("audio encoder layer")
    ax.set_ylabel("probe accuracy")
    ax.grid(alpha=0.25)
    ax.set_title("{} -- {}, {} clips, {}".format(
        "Qwen2-Audio-7B", PRETTY.get(r["category"], r["category"]),
        r["n_clips"], r.get("cv_scheme", "")))
    ax.legend(loc="lower right")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "{}.{}".format(name, ext)))
    plt.close(fig)
    print("  wrote {}/{}.pdf (+.png)".format(out_dir, name))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="probe_depth_encoder")
    ap.add_argument("--width", type=float, default=6.5)
    ap.add_argument("--height", type=float, default=3.4)
    a = ap.parse_args()
    probe_depth(a.out, a.name, (a.width, a.height))
