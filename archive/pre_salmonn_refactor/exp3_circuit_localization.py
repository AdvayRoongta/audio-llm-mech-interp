"""Experiment 3 -- where in the decoder does the suppression live?

Cho et al. (arXiv:2606.18924) localised text and audio circuits for *lexical*
conflicts and named paralinguistic conflicts as future work. This is that
experiment, on this dataset.

Method follows theirs -- attribution patching scored as (layer, submodule,
position-group) tuples -- reimplemented on plain transformers hooks, since
TransformerLens has no Qwen2-Audio support. Cho et al. use integrated
gradients on top of this; that part is NOT implemented here, because doing it
across all components at once breaks cross-position gradient flow and doing it
correctly needs a per-component path. --ig-steps 1 (plain AP) is the only
supported setting, and it is the default.

Two circuits, two counterfactuals. A circuit is defined by what changes the
answer, so each modality needs its own clean/corrupt pair:

    text circuit    same audio, the claim swapped between two options
    audio circuit   same prompt, the audio swapped between two truths

Alignment is the thing that makes or breaks this. Patching position i assumes
position i means the same thing in both runs, so:

  * audio is padded/trimmed to a fixed duration -> identical audio-token counts
  * pairs whose prompts tokenize to different lengths are rejected outright

Both are asserted, loudly, before any attribution is computed. A misaligned
pair produces confident nonsense rather than an error, which is the worst
failure mode available here.

Outputs:
  1. attribution per (layer, submodule, position group)
  2. IoU between the text and audio circuits, per position group. Cho et al.
     found ~0.01 at Data positions and 0.54-0.89 at Query/Generation for
     lexical conflicts; whether paralinguistic behaves the same is the open
     question this answers.
  3. faithfulness -- real activation patching on the top-k components, to check
     the linear approximation actually held

Run from the repo root:
    python eval/exp3_circuit_localization.py --category emotional_prosody
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def submodules(model):
    """(index, name, module) for every attention and MLP block we score."""
    out = []
    for i, layer in enumerate(C.decoder_layers(model)):
        if hasattr(layer, "self_attn"):
            out.append((i, "attn", layer.self_attn))
        if hasattr(layer, "mlp"):
            out.append((i, "mlp", layer.mlp))
    return out


def position_groups(model, processor, inputs, item):
    """Split positions the way Cho et al. do, because the Data-vs-Query
    contrast is the whole comparison.

    Their partition is Data / Query / Generation. Lumping the claim, the
    question and the option list into one "text" group would conflate Data with
    Query and make their numbers incomparable to ours. So:

        audio     the audio token span            (Data, audio side)
        claim     the sentence asserting a label  (Data, text side)
        query     the question between them       (Query)
        options   the "A. ... B. ..." list        (Query, answer surface)
        gen       the final position              (Generation)

    Any piece that cannot be located is omitted rather than guessed at, so a
    prompt-format change shows up as a missing group instead of a wrong one.
    """
    seq = inputs["input_ids"].shape[1]
    groups = {"gen": (seq - 1, seq - 1)}

    span = C.audio_span(inputs, C.audio_token_id(model))
    if span:
        groups["audio"] = span

    prompt = item.get("prompt", "")
    cut = prompt.find("\nA. ")

    # claim: the line mentioning the asserted label
    claim = claim_value(item)
    if claim and cut > 0:
        for line in prompt[:cut].splitlines():
            if claim.lower() in line.lower() and "transcript" not in line.lower():
                got = C.find_span(processor, inputs, line.strip())
                if got:
                    groups["claim"] = got
                break

    # options: the lettered list through the end of the prompt
    if cut > 0:
        got = C.find_span(processor, inputs, prompt[cut:].strip())
        if got:
            groups["options"] = got

    # query: whatever text sits between the claim and the options
    if "claim" in groups and "options" in groups:
        lo, hi = groups["claim"][1] + 1, groups["options"][0] - 1
        if lo <= hi:
            groups["query"] = (lo, hi)
    elif span and cut > 0 and "options" in groups:
        lo, hi = span[1] + 1, groups["options"][0] - 1
        if lo <= hi:
            groups["query"] = (lo, hi)

    # Fallback so the experiment still runs on an unexpected prompt format.
    if not any(k in groups for k in ("claim", "query", "options")):
        lo = (span[1] + 1) if span else 0
        if lo <= seq - 2:
            groups["text"] = (lo, seq - 2)
    return groups


def capture(model, inputs):
    """Activations of every scored submodule for one run."""
    import torch

    acts, handles = {}, []
    for i, name, mod in submodules(model):
        def mk(key):
            def hook(m, inp, out):
                acts[key] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        handles.append(mod.register_forward_hook(mk((i, name))))
    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles:
        h.remove()
    return acts


def metric_fn(model, logits, letter_ids_a, letter_ids_b):
    """log P(answer A) - log P(answer B) at the final position."""
    import torch
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    a = torch.stack([lp[i] for i in letter_ids_a]).max()
    b = torch.stack([lp[i] for i in letter_ids_b]).max()
    return a - b


def attributions(model, clean_inputs, corrupt_inputs, groups, ids_a, ids_b, ig_steps):
    """attr = (a_clean - a_corrupt) . d(metric)/d(a), from one forward+backward.

    The hooks here only *observe*: they call retain_grad() on each submodule
    output and keep a reference. They must not replace the output. An earlier
    version substituted a cached blend at every attention and MLP module, which
    turned each submodule into a constant with respect to its inputs -- that
    severs attention's cross-position information flow, so every position
    except the last had exactly zero gradient and the whole circuit collapsed
    onto the generation position.

    Note that faithfulness did not catch that: patching the final position
    alone does recover the effect, so a degenerate circuit still scored 1.00.
    """
    if ig_steps and ig_steps > 1:
        raise NotImplementedError(
            "ig_steps > 1 is not implemented. Interpolating every submodule's\n"
            "output simultaneously breaks cross-position gradient flow (that was\n"
            "the bug this function used to have), and doing it correctly needs a\n"
            "per-component path, which costs n_components x n_steps forwards.\n"
            "Use --ig-steps 1 for plain attribution patching.")

    import torch  # noqa: F401  (kept for parity with the other helpers)

    a_clean = capture(model, clean_inputs)

    held, handles = {}, []
    for i, name, mod in submodules(model):
        def mk(key):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.requires_grad:
                    h.retain_grad()
                held[key] = h
                return None          # observe only; never modify the output
            return hook
        handles.append(mod.register_forward_hook(mk((i, name))))

    model.zero_grad(set_to_none=True)
    out = model(**corrupt_inputs, use_cache=False)
    metric_fn(model, out.logits, ids_a, ids_b).backward()
    for h in handles:
        h.remove()

    total = {}
    no_grad = []
    for i, name, _ in submodules(model):
        key = (i, name)
        h = held.get(key)
        grad = None if h is None else h.grad
        if grad is None:
            no_grad.append(key)
            for g in groups:
                total[(i, name, g)] = 0.0
            continue
        delta = (a_clean[key] - h.detach()).float()
        contrib = (delta * grad.float())[0]
        for gname, (lo, hi) in groups.items():
            total[(i, name, gname)] = float(contrib[lo:hi + 1].sum())

    if no_grad:
        print("   WARNING: no gradient reached {} of {} submodules".format(
            len(no_grad), len(held) or len(no_grad)))
    model.zero_grad(set_to_none=True)
    return total


def patch_components(model, clean_inputs, corrupt_inputs, components, groups):
    """Real activation patching on a chosen component set (faithfulness check)."""
    import torch

    a_clean = capture(model, clean_inputs)
    want = {}
    for (i, name, g) in components:
        want.setdefault((i, name), []).append(groups[g])

    handles = []
    for i, name, mod in submodules(model):
        def mk(key):
            def hook(m, inp, out):
                if key not in want:
                    return out
                is_tuple = isinstance(out, tuple)
                h = (out[0] if is_tuple else out).clone()
                for lo, hi in want[key]:
                    h[:, lo:hi + 1, :] = a_clean[key][:, lo:hi + 1, :].to(h.dtype)
                return (h,) + out[1:] if is_tuple else h
            return hook
        handles.append(mod.register_forward_hook(mk((i, name))))
    with torch.no_grad():
        out = model(**corrupt_inputs, use_cache=False)
    for h in handles:
        h.remove()
    return out.logits


# Below this, an attribution is indistinguishable from float noise. Causal
# attention makes whole groups exactly zero -- e.g. the audio span cannot see a
# claim that comes after it, so for the text circuit every audio-position
# attribution is 0.0. Ranking zeros with a stable sort returns insertion order,
# so two circuits "agree" perfectly and IoU comes out 1.000 while meaning
# nothing at all.
SIGNAL_FLOOR = 1e-8


def group_magnitude(attr, group=None):
    vals = [abs(v) for k, v in attr.items() if group is None or k[2] == group]
    return max(vals) if vals else 0.0


def has_signal(attr, group=None):
    return group_magnitude(attr, group) > SIGNAL_FLOOR


def top_set(attr, frac, group=None):
    rows = [(k, abs(v)) for k, v in attr.items() if group is None or k[2] == group]
    if not rows or max(v for _, v in rows) <= SIGNAL_FLOOR:
        return set()          # no signal: refuse to rank noise
    rows.sort(key=lambda kv: -kv[1])
    return {k for k, _ in rows[:max(1, int(len(rows) * frac))]}


def iou(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def mean_attr(per_pair):
    """Component-wise mean across pairs."""
    if not per_pair:
        return {}
    keys = set().union(*(d.keys() for d in per_pair))
    return {k: sum(d.get(k, 0.0) for d in per_pair) / len(per_pair) for k in keys}


def bootstrap_iou(per_text, per_audio, group, frac, n_boot=1000, seed=0):
    """Percentile CI for the IoU, resampling pairs with replacement.

    Returns (point_estimate, lo, hi) or None when either side has no signal.
    """
    import random as _random

    point_t, point_a = mean_attr(per_text), mean_attr(per_audio)
    if not has_signal(point_t, group) or not has_signal(point_a, group):
        return None
    point = iou(top_set(point_t, frac, group), top_set(point_a, frac, group))

    rng = _random.Random(seed)
    vals = []
    for _ in range(n_boot):
        bt = [per_text[rng.randrange(len(per_text))] for _ in range(len(per_text))]
        ba = [per_audio[rng.randrange(len(per_audio))] for _ in range(len(per_audio))]
        mt, ma = mean_attr(bt), mean_attr(ba)
        if not has_signal(mt, group) or not has_signal(ma, group):
            continue
        vals.append(iou(top_set(mt, frac, group), top_set(ma, frac, group)))
    if not vals:
        return None
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return point, lo, hi


def claim_value(item):
    """The label the prompt actually asserts, e.g. 'female'.

    claim_swapped() rewrites the prompt but leaves the dataset's claimed_* field
    untouched, so on a swapped item that field names the label that is no longer
    there. _claim_swapped_to records what the prompt now says and takes
    precedence -- otherwise anything locating the claim in the text searches for
    the wrong word.
    """
    if item.get("_claim_swapped_to") is not None:
        return item["_claim_swapped_to"]
    import run_qwen2audio_eval as R
    key = R.first_present_key(item, R.CLAIMED_KEY_CANDIDATES.get(item.get("category"), []))
    return item.get(key) if key else None


def claim_swapped(item, new_value):
    """Rewrite the claim to assert `new_value`, changing nothing else.

    The option list repeats the same words as the claim ("A. male / B. female"),
    so substitution is confined to the text above it -- editing the options
    would change the answer key rather than the claim. Requiring exactly one
    match is the guard: zero means the claim is phrased some other way, more
    than one means we cannot tell which mention is the claim, and in both cases
    the pair is dropped instead of silently mangled.
    """
    import re

    old = claim_value(item)
    if old is None or new_value is None:
        return None
    prompt = item.get("prompt", "")
    cut = prompt.find("\nA. ")
    if cut < 0:
        return None
    head, tail = prompt[:cut], prompt[cut:]
    new_head, n = re.subn(r"\b{}\b".format(re.escape(str(old))), str(new_value),
                          head, flags=re.IGNORECASE)
    if n != 1:
        return None
    out = dict(item)
    out["prompt"] = new_head + tail
    out["_claim_swapped_to"] = new_value
    return out


def audio_swapped(a, b):
    """Item `a`'s prompt with item `b`'s audio.

    The prompt is byte-identical, so tokenization, option order and length all
    match automatically -- only the acoustics differ. That is the cleanest
    audio counterfactual this dataset admits, since every prompt embeds its own
    transcript and so is unique per clip.
    """
    out = dict(a)
    out["_audio"] = b["_audio"]
    out["audio_path"] = b.get("audio_path")
    out["_swapped_from"] = b["_uid"]
    return out


def pick_pairs(items, category, model, processor, fixed_seconds, max_pairs):
    """Aligned (clean, corrupt, letter_a, letter_b) tuples for both circuits.

    text circuit   Matched vs Mismatched for the SAME clip: identical audio,
                   the claim is the only thing that moves.
    audio circuit  one item's prompt with another item's audio: identical
                   tokens, the acoustics are the only thing that moves.
    """
    pool = [i for i in C.select(items, category=category, modality="text_audio")
            if i.get("_audio")]
    text_pairs, audio_pairs, rejected = [], [], 0

    # ---- text circuit -------------------------------------------------- #
    # Each condition uses a disjoint set of clips, so there is no Matched item
    # sharing a clip with a Mismatched one. The counterfactual is therefore
    # built in place: rewrite the claim to agree with the audio and change
    # nothing else.
    mism = [i for i in pool if i.get("condition") == "Mismatched"]
    for corrupt in mism:
        if len(text_pairs) >= max_pairs:
            break
        truth, claim = C.correct_letter(corrupt), C.claimed_letter(corrupt)
        if not truth or not claim or truth == claim:
            continue
        clean = claim_swapped(corrupt, corrupt.get("ground_truth_label"))
        if clean is None:
            rejected += 1              # claim not found exactly once; unsafe to edit
            continue
        ia = C.encode(model, processor, clean, fixed_seconds=fixed_seconds)
        ib = C.encode(model, processor, corrupt, fixed_seconds=fixed_seconds)
        if ia["input_ids"].shape != ib["input_ids"].shape:
            rejected += 1              # the two claim words tokenize differently
            continue
        text_pairs.append((clean, corrupt, truth, claim))

    # ---- audio circuit ------------------------------------------------- #
    mism = [i for i in pool if i.get("condition") == "Mismatched"]
    for a in mism:
        if len(audio_pairs) >= max_pairs:
            break
        a_truth = C.correct_letter(a)
        for b in mism:
            if b["_audio"] == a["_audio"]:
                continue
            b_label = b.get("ground_truth_label")
            b_letter = R_resolve(a, b_label)
            if not b_letter or not a_truth or b_letter == a_truth:
                continue
            audio_pairs.append((a, audio_swapped(a, b), a_truth, b_letter))
            break

    return text_pairs, audio_pairs, rejected


def R_resolve(item, value):
    """Letter for `value` inside `item`'s option map, or None."""
    if value is None:
        return None
    import run_qwen2audio_eval as R
    return R.resolve_value_to_letter(C.option_map(item), value)


def run_pair(model, processor, clean, corrupt, letter_a, letter_b, fixed_seconds, ig_steps):
    """Attributions for one clean/corrupt pair.

    The alignment assertion is not decoration: a misaligned pair yields
    confident nonsense rather than an error, which is the worst failure mode
    available here.
    """
    ia = C.encode(model, processor, clean, fixed_seconds=fixed_seconds)
    ib = C.encode(model, processor, corrupt, fixed_seconds=fixed_seconds)
    if ia["input_ids"].shape != ib["input_ids"].shape:
        raise AssertionError("misaligned pair: {} vs {}".format(
            tuple(ia["input_ids"].shape), tuple(ib["input_ids"].shape)))
    if C.option_map(clean) != C.option_map(corrupt):
        raise AssertionError("option maps differ; letters are not comparable")

    # Letters resolve inside the corrupt prompt, since that is the context the
    # forward pass actually runs in.
    letters = C.letter_token_ids(processor, list(C.option_map(corrupt).keys()))
    ids_a, ids_b = letters[letter_a], letters[letter_b]

    groups = position_groups(model, processor, ia, clean)
    attr = attributions(model, ia, ib, groups, ids_a, ids_b, ig_steps)
    return attr, groups, ia, ib, ids_a, ids_b


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", default="emotional_prosody", choices=C.CATEGORIES)
    ap.add_argument("--fixed-seconds", type=float, default=3.0)
    ap.add_argument("--ig-steps", type=int, default=1,
                    help="only 1 is supported; see attributions() for why")
    ap.add_argument("--top-frac", type=float, default=0.20, help="Cho et al. used 0.20 for Qwen")
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for the IoU confidence interval")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    C.set_seed(args.seed)

    items = C.load_items(args.input, require_audio=True)
    model, processor, info = C.load_model(eager=True, four_bit=args.four_bit)
    if args.four_bit:
        print("NOTE: 4-bit perturbs activations. Fine for ranking components;")
        print("      re-check any headline number in fp16.")

    text_pairs, audio_pairs, rejected = pick_pairs(
        items, args.category, model, processor, args.fixed_seconds, args.max_pairs)
    print("\n{}: {} text pairs, {} audio pairs ({} rejected for misalignment)".format(
        args.category, len(text_pairs), len(audio_pairs), rejected))
    if not text_pairs or not audio_pairs:
        raise SystemExit("not enough aligned pairs to localize a circuit")

    def accumulate(pairs, label):
        """Per-pair attributions, kept individually so IoU can be bootstrapped.

        A single pair's circuit is noisy, so the mean alone gives no way to say
        whether two categories differ or whether either differs from Cho et al.
        """
        per_pair, groups_seen = [], None
        for n, (clean, corrupt, la, lb) in enumerate(pairs, 1):
            attr, groups, ia, ib, ids_a, ids_b = run_pair(
                model, processor, clean, corrupt, la, lb,
                args.fixed_seconds, args.ig_steps)
            per_pair.append(attr)
            groups_seen = groups
            print("  {} pair {}/{}".format(label, n, len(pairs)))
        return per_pair, mean_attr(per_pair), groups_seen

    per_text, attr_text, groups_t = accumulate(text_pairs, "text")
    per_audio, attr_audio, groups_a = accumulate(audio_pairs, "audio")

    print("\n" + "=" * 74)
    print("CIRCUIT OVERLAP (IoU at {:.0%} sparsity)".format(args.top_frac))
    print("=" * 74)
    overlap, magnitudes, cis = {}, {}, {}
    print("  {:<8} {:>8} {:>16}  {:>11} {:>11}".format(
        "group", "IoU", "95% CI", "|attr|text", "|attr|audio"))
    for g in sorted(set(groups_t) & set(groups_a)) + ["ALL"]:
        grp = None if g == "ALL" else g
        mt, ma = group_magnitude(attr_text, grp), group_magnitude(attr_audio, grp)
        magnitudes[g] = {"text": mt, "audio": ma}
        boot = bootstrap_iou(per_text, per_audio, grp, args.top_frac,
                             n_boot=args.n_boot, seed=args.seed)
        if boot is None:
            overlap[g], cis[g] = None, None
            print("  {:<8} {:>8} {:>16}  {:>11.2e} {:>11.2e}   <- no signal, IoU undefined"
                  .format(g, "n/a", "", mt, ma))
        else:
            point, lo, hi = boot
            overlap[g], cis[g] = point, [lo, hi]
            print("  {:<8} {:>8} {:>16}  {:>11.2e} {:>11.2e}".format(
                g, "{:.3f}".format(point), "[{:.2f}, {:.2f}]".format(lo, hi), mt, ma))
    print("\n  CIs are percentile bootstrap over {} resamples of {} text / {} audio"
          .format(args.n_boot, len(per_text), len(per_audio)))
    print("  pairs. With this few pairs they are wide -- that is the point.")

    dead = [g for g, v in overlap.items() if v is None]
    if dead:
        print("\n  Groups with no signal: {}.".format(", ".join(dead)))
        print("  Causal attention means a group that cannot see what the")
        print("  counterfactual changed has identical activations in both runs,")
        print("  so its attributions are exactly zero. That is expected, not a")
        print("  bug -- but it is not overlap, and must not be reported as 1.0.")
    print("\nCho et al. (lexical, Qwen): ~0.01 at Data positions, 0.54-0.89 at")
    print("Query/Generation. Compare -- a different shape here is the finding.")

    # Faithfulness on the text circuit.
    import torch
    clean, corrupt, la, lb = text_pairs[0]
    attr, groups, ia, ib, ids_a, ids_b = run_pair(
        model, processor, clean, corrupt, la, lb, args.fixed_seconds, args.ig_steps)
    with torch.no_grad():
        m_clean = float(metric_fn(model, model(**ia, use_cache=False).logits, ids_a, ids_b))
        m_corrupt = float(metric_fn(model, model(**ib, use_cache=False).logits, ids_a, ids_b))

    print("\n" + "=" * 74)
    print("FAITHFULNESS -- real patching on the top components")
    print("=" * 74)
    print("clean {:+.3f}   corrupt {:+.3f}   gap {:+.3f}\n".format(
        m_clean, m_corrupt, m_clean - m_corrupt))
    faith = {}
    for frac in (0.05, 0.10, 0.20, 0.40):
        logits = patch_components(model, ia, ib, top_set(attr, frac), groups)
        m = float(metric_fn(model, logits, ids_a, ids_b))
        f = (m - m_corrupt) / (m_clean - m_corrupt) if m_clean != m_corrupt else float("nan")
        faith["{:.2f}".format(frac)] = f
        print("  sparsity {:.2f}   metric {:+.3f}   faithfulness {:.2f}".format(frac, m, f))
    print("\nfaithfulness near 1.0 means the selected components explain the effect;")
    print("near 0 means the attribution ranking did not survive real patching.")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "exp3_circuits_{}.json".format(args.category))
    with open(out, "w") as f:
        json.dump({"run": dict(info, category=args.category, ig_steps=args.ig_steps,
                               top_frac=args.top_frac, fixed_seconds=args.fixed_seconds,
                               n_text_pairs=len(text_pairs), n_audio_pairs=len(audio_pairs),
                               seed=args.seed),
                   "overlap_iou": overlap,
                   "overlap_iou_ci95": cis,
                   "group_magnitudes": magnitudes,
                   "faithfulness": faith,
                   "attr_text": {"{}|{}|{}".format(*k): v for k, v in attr_text.items()},
                   "attr_audio": {"{}|{}|{}".format(*k): v for k, v in attr_audio.items()}},
                  f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
