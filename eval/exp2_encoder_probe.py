"""Experiment 2 -- does the paralinguistic signal survive the Whisper encoder?

Nobody has probed the encoder layers for paralinguistic survival, so this
decides whether "suppression" is even the right word for what we measure.

Whisper was trained to transcribe. Its late layers are optimised for *what was
said*, which raises a concrete worry: emotion and speaker identity may already
be degraded before the projector runs. If so, the decoder is not suppressing
anything -- it never received the signal.

Method is the standard layer-wise linear probe used on self-supervised speech
models (Pasad et al.):

  1. per encoder layer, take the hidden states
  2. mean-pool over the VALID frames only -- the feature extractor pads every
     clip to 30 s, and pooling across that padding washes out short clips
  3. cross-validated logistic regression to predict emotion / gender
  4. a label-shuffled control, so "the probe found structure" is separable from
     "the probe memorised 100 examples"

Then the same probe on the post-projector vectors, which is what the language
model actually receives. Encoder-last vs post-projector is the only direct test
of the assumption that the projector loses nothing.

Reading it:
  accuracy holds through the last layer -> signal reaches the LM; suppression
                                           is decoder-side, go run exp3/exp4
  accuracy peaks mid-stack then collapses -> Whisper discards it; claim encoder
                                             loss, not suppression

No text is involved here, so this uses each clip once, ignoring conditions.

Run from the repo root:
    python eval/exp2_encoder_probe.py
    python eval/exp2_encoder_probe.py --category speaker_identity --folds 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

# What each category's probe is asked to decode.
PROBE_TARGET = {
    "emotional_prosody": "true_emotion",
    "speaker_identity": "ground_truth_label",
    "sarcasm_pacing": "true_emotion",
}


def unique_clips(items, category):
    """One row per audio file. Conditions reuse the same clips, so probing all
    of them would leak duplicates across CV folds and inflate accuracy."""
    seen, rows = set(), []
    for it in C.select(items, category=category):
        # text_only items carry an audio_path for provenance but are never
        # encoded with audio, so they have no input_features to probe.
        if it.get("modality") not in ("audio_only", "text_audio"):
            continue
        path = it.get("_audio")
        if not path or path in seen:
            continue
        label = it.get(PROBE_TARGET[category]) or it.get("ground_truth_label")
        if label is None:
            continue
        seen.add(path)
        rows.append({"path": path, "label": str(label), "item": it})
    return rows


def layer_vectors(model, processor, item, tower, proj):
    """Mean-pooled hidden state per encoder layer, plus the projected output."""
    import torch

    inputs = C.encode(model, processor, item)
    feats = inputs["input_features"].to(model.dtype)

    with torch.no_grad():
        out = tower(feats, output_hidden_states=True)

        # Valid mel frames -> encoder frames. The conv stack halves the length
        # once; pooling to audio tokens happens after the encoder, so the
        # hidden states here are still at the post-conv rate.
        n_mel = int(inputs["feature_attention_mask"].sum())
        n_enc = (n_mel - 1) // 2 + 1

        vecs = []
        for h in out.hidden_states:
            n = max(1, min(n_enc, h.shape[1]))
            vecs.append(h[0, :n].float().mean(0).cpu().numpy())

        projected = proj(out.last_hidden_state)
        n = max(1, min(n_enc, projected.shape[1]))
        vecs.append(projected[0, :n].float().mean(0).cpu().numpy())
    return vecs


def probe_layer(X, y, folds=5, seed=0):
    """CV accuracy for one layer, and the shuffled-label control."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    real = float(cross_val_score(pipe, X, y, cv=cv).mean())
    shuffled = float(cross_val_score(
        pipe, X, np.random.RandomState(seed).permutation(y), cv=cv).mean())
    return real, shuffled


def run_category(model, processor, items, category, args):
    import numpy as np

    rows = unique_clips(items, category)
    if getattr(args, "limit", None):
        rows = rows[:args.limit]
    if len(rows) < args.folds * 2:
        print("{}: only {} usable clips, skipping".format(category, len(rows)))
        return None

    labels = [r["label"] for r in rows]
    classes = sorted(set(labels))
    counts = {c: labels.count(c) for c in classes}
    if min(counts.values()) < args.folds:
        print("{}: smallest class has {} clips < {} folds; reduce --folds"
              .format(category, min(counts.values()), args.folds))
        return None

    print("\n{}: {} clips, {} classes {}".format(category, len(rows), len(classes), counts))

    tower, proj = C.audio_tower(model), C.projector(model)
    feats = []
    for i, r in enumerate(rows, 1):
        feats.append(layer_vectors(model, processor, r["item"], tower, proj))
        if i % 20 == 0:
            print("  extracted {}/{}".format(i, len(rows)))

    n_layers = len(feats[0])
    chance = 1.0 / len(classes)
    out = []
    for l in range(n_layers):
        X = np.stack([f[l] for f in feats])
        real, shuffled = probe_layer(X, np.array(labels), args.folds, args.seed)
        name = "post-projector" if l == n_layers - 1 else "encoder {}".format(l)
        out.append({"layer": l, "name": name, "accuracy": real,
                    "shuffled": shuffled, "chance": chance})
        print("  {:<16} acc {:.3f}   shuffled {:.3f}   chance {:.3f}".format(
            name, real, shuffled, chance))
    return {"category": category, "n_clips": len(rows), "classes": classes,
            "chance": chance, "layers": out}


def verdict(result):
    enc = result["layers"][:-1]
    projd = result["layers"][-1]
    peak = max(enc, key=lambda r: r["accuracy"])
    last = enc[-1]
    chance = result["chance"]
    drop = peak["accuracy"] - last["accuracy"]

    print("\n{}".format(result["category"]))
    print("  peak            {} = {:.3f}".format(peak["name"], peak["accuracy"]))
    print("  encoder output  {:.3f}  (drop from peak {:+.3f})".format(last["accuracy"], -drop))
    print("  post-projector  {:.3f}  (change {:+.3f})".format(
        projd["accuracy"], projd["accuracy"] - last["accuracy"]))

    if last["accuracy"] <= last["shuffled"] + 0.02:
        msg = "encoder output is no better than shuffled labels: nothing linearly decodable. Claim encoder loss."
    elif last["accuracy"] < chance + 0.05:
        msg = "encoder output is at chance: Whisper discards it. Claim encoder loss, not suppression."
    elif drop > 0.15:
        msg = "substantial late-encoder degradation, but signal still present at the projector."
    else:
        msg = "signal survives to the language model. Suppression is decoder-side -> run exp3/exp4."
    print("  -> {}".format(msg))
    return msg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", choices=C.CATEGORIES, help="restrict to one category")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit", type=int,
                    help="cap clips per category (smoke test before a full run)")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    C.set_seed(args.seed)

    items = C.load_items(args.input, require_audio=True)
    model, processor, info = C.load_model(four_bit=args.four_bit)

    cats = [args.category] if args.category else C.CATEGORIES
    results, verdicts = [], {}
    for cat in cats:
        res = run_category(model, processor, items, cat, args)
        if res:
            results.append(res)

    print("\n" + "=" * 74)
    print("VERDICT -- did the signal reach the language model?")
    print("=" * 74)
    for res in results:
        verdicts[res["category"]] = verdict(res)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "exp2_encoder_probe.json")
    with open(out, "w") as f:
        json.dump({"run": dict(info, seed=args.seed, folds=args.folds,
                               dataset=args.input),
                   "verdicts": verdicts, "results": results}, f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
