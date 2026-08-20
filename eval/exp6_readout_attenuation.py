"""Experiment 6 -- attenuate the claim at the readout, with a coherence control.

exp5 eliminated every audio-side lever. Relocating the audio representation
buys +0.51 nats at best, amplifying it +0.10, and both against a baseline margin
of -7.32 (log P(audio truth) - log P(text claim)). Six percent of the gap. The
audio is present, it reaches the decision, and it cannot compete.

exp3 located where the gap lives: at the generation position the text circuit
carries ~16x the audio circuit's attribution, peaking around layer 29. This
experiment intervenes there instead.

Method: attention knockout (Geva et al., arXiv:2304.14767) -- block the position
that decides the answer from attending to the claim -- generalised to a graded
form. Adding log(gamma) to the attention scores for the claim keys before
softmax down-weights them by a factor of gamma; gamma=0 is the hard knockout
Geva et al. use, gamma=1 is untouched.

Restricted by layer, and that restriction is the whole point. Blocking the claim
at every layer merely deletes it from the model's view: TDR would collapse and
it would prove nothing, because a model that cannot see the claim is not a model
that resists it. Attenuating only the late layers where exp3 measured the
override is a mechanistic claim; attenuating everywhere is input ablation.

THE CONTROL THAT MATTERS. Cho et al. (arXiv:2606.18924) found circuit ablation
lifts closed-set audio accuracy 0.33 -> 0.82 while destroying free-form
coherence. Our usual capability control is accuracy on Matched items, which is
itself closed-set multiple choice -- the exact regime where their intervention
also looked healthy. It is structurally blind to the failure they documented.

So this experiment generates free-form continuations under the intervention and
scores them with the *unmodified* model. If attenuation produces degenerate
text, the clean model assigns it low probability. Without that, a "success" here
would be indistinguishable from reproducing their known negative result.

Run from the repo root:
    python eval/exp6_readout_attenuation.py --category emotional_prosody
    python eval/exp6_readout_attenuation.py --category emotional_prosody --limit 40
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import exp3_circuit_localization as E3  # noqa: E402  (reuse validated span logic)


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
class ClaimAttenuator:
    """Down-weight the answer position's attention to the claim, per layer.

    Modifies the mask the model is already passing rather than substituting a
    precomputed one. A fixed mask breaks the moment generate() extends the
    sequence -- the mask stays at prompt length while attention grows -- and the
    coherence control depends on generation, so it has to survive that.

    Adding log(gamma) to the claim keys before softmax scales those weights by
    gamma and lets the rest renormalise; gamma=0 is the hard knockout of
    Geva et al. (arXiv:2304.14767). During prefill the affected rows are the
    answer position onward; during decoding there is a single row, which is the
    answer being produced, so the attenuation continues to apply.

    Layer restriction is the point of the experiment: attenuating at every layer
    just removes the claim from the model's view.
    """

    def __init__(self, model, layers_to_change, claim_span, gen_pos, gamma,
                 rows="gen"):
        self.layers = C.decoder_layers(model)
        self.targets = set(layers_to_change)
        self.lo, self.hi = claim_span
        self.gen_pos = gen_pos
        self.gamma = gamma
        # "gen"  -- only the answer position's view of the claim (the experiment)
        # "all"  -- every position's view, i.e. the claim is removed from the
        #           model's input entirely. Only meaningful as a positive
        #           control, never as a proposed fix.
        self.rows = rows
        self._handles = []

    def __enter__(self):
        import torch

        def mk(i):
            def hook(module, args, kwargs):
                if i not in self.targets:
                    return args, kwargs
                mask = kwargs.get("attention_mask")
                if mask is None or mask.dim() != 4:
                    raise RuntimeError(
                        "layer {} received {} instead of a 4D attention mask. "
                        "The graded knockout modifies the mask, so one has to "
                        "exist: load with eager (sdpa passes None and applies "
                        "causality inside the kernel).".format(
                            i, "None" if mask is None else
                            "a {}D tensor".format(mask.dim())))
                m = mask.clone()
                hi = min(self.hi, m.shape[-1] - 1)
                if self.lo > hi:
                    return args, kwargs
                q_len = m.shape[-2]
                if self.rows == "all":
                    # every query position loses sight of the claim
                    rows = slice(0, q_len)
                else:
                    # prefill: the answer row onward. decode: the only row is
                    # the answer position, so row 0.
                    rows = slice(0, 1) if q_len == 1 else slice(
                        min(self.gen_pos, q_len - 1), q_len)
                if self.gamma <= 0:
                    # finfo.min rather than -inf: an actual -inf can propagate to
                    # NaN in fp16 softmax, and min is already a hard block.
                    m[..., rows, self.lo:hi + 1] = torch.finfo(m.dtype).min
                else:
                    m[..., rows, self.lo:hi + 1] += math.log(self.gamma)
                    # rows="all" touches entries causality already masked to
                    # finfo.min; subtracting more underflows to -inf in fp16,
                    # and a row of all -inf gives NaN out of softmax.
                    m.clamp_(min=torch.finfo(m.dtype).min)
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = m
                return args, kwargs
            return hook

        for i, layer in enumerate(self.layers):
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(mk(i), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def margin(scores, item):
    a, c = C.correct_letter(item), C.claimed_letter(item)
    if a not in scores or c not in scores:
        return None
    return scores[a] - scores[c]


def prepare(model, processor, item, fixed_seconds):
    """Inputs plus the claim span and generation position."""
    inputs = C.encode(model, processor, item, fixed_seconds=fixed_seconds)
    groups = E3.position_groups(model, processor, inputs, item)
    claim = groups.get("claim")
    seq = inputs["input_ids"].shape[1]
    return inputs, claim, seq - 1


def predict(model, processor, item, inputs, layers=None,
            claim=None, gen_pos=None, gamma=1.0, rows="gen"):
    letters = list(C.option_map(item).keys())
    if layers is None or claim is None:
        pred, sc = C.predict_letter(model, processor, inputs, letters)
        return pred, margin(sc, item)
    with ClaimAttenuator(model, layers, claim, gen_pos, gamma, rows):
        pred, sc = C.predict_letter(model, processor, inputs, letters)
    return pred, margin(sc, item)


def condition(model, processor, prep_mis, prep_mat, layers, gamma, rows="gen"):
    """Score both item sets under one intervention. Returns (mismatched, matched)."""
    out = []
    for prep in (prep_mis, prep_mat):
        recs = []
        for it, inputs, claim, gen_pos in prep:
            p, mg = predict(model, processor, it, inputs, layers,
                            claim, gen_pos, gamma, rows)
            recs.append(dict(prediction=p, correct_letter=C.correct_letter(it),
                             claimed=C.claimed_letter(it),
                             correct=p == C.correct_letter(it), margin=mg))
        out.append(recs)
    return out[0], out[1]


def summarise(records):
    blk = C.accuracy_block(records)
    vals = [r["margin"] for r in records if r.get("margin") is not None]
    return (blk["accuracy"], blk["ci95"][0], blk["ci95"][1],
            (sum(vals) / len(vals)) if vals else float("nan"))


# --------------------------------------------------------------------------- #
# the free-form coherence control
# --------------------------------------------------------------------------- #
def freeform_fluency(model, processor, prepared, layers, gamma, max_new_tokens=30):
    """Generate under the intervention, score the text with the clean model.

    Closed-set accuracy cannot see the degeneration Cho et al. reported, so this
    measures it directly: mean token log-probability, under the unmodified
    model, of what the intervened model produced. Degenerate or repetitive text
    scores low. Also reports a repetition rate, since loops can score
    deceptively well on per-token probability.
    """
    import torch

    scores, reps, samples = [], [], []
    for item, inputs, claim, gen_pos in prepared:
        ctx = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            if layers is not None:
                with ClaimAttenuator(model, layers, claim, gen_pos, gamma):
                    out = model.generate(**ctx, max_new_tokens=max_new_tokens,
                                         do_sample=False)
            else:
                out = model.generate(**ctx, max_new_tokens=max_new_tokens,
                                     do_sample=False)

        new = out[0, inputs["input_ids"].shape[1]:]
        if new.numel() < 2:
            continue
        text = processor.tokenizer.decode(new, skip_special_tokens=True)

        # Score the produced continuation with the UNMODIFIED model.
        full = torch.cat([inputs["input_ids"], new.unsqueeze(0)], dim=1)
        kw = {k: v for k, v in inputs.items()}
        kw["input_ids"] = full
        if "attention_mask" in kw:
            kw["attention_mask"] = torch.ones_like(full)
        with torch.no_grad():
            logits = model(**kw, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[0, inputs["input_ids"].shape[1] - 1:-1], -1)
        tok_lp = lp.gather(-1, new.unsqueeze(-1)).squeeze(-1)
        scores.append(float(tok_lp.mean()))

        toks = new.tolist()
        reps.append(1.0 - len(set(toks)) / len(toks))
        if len(samples) < 3:
            samples.append(text.replace("\n", " ")[:90])

    return {"mean_token_logprob": (sum(scores) / len(scores)) if scores else float("nan"),
            "repetition_rate": (sum(reps) / len(reps)) if reps else float("nan"),
            "n": len(scores), "samples": samples}


# --------------------------------------------------------------------------- #
def band_for(name, n_layers):
    if name == "all":
        return list(range(n_layers))
    if name == "late":
        return list(range(int(n_layers * 0.69), n_layers))     # ~22-31 of 32
    if name == "mid":
        return list(range(int(n_layers * 0.34), int(n_layers * 0.69)))
    if name == "early":
        return list(range(0, int(n_layers * 0.34)))
    raise ValueError(name)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=C.DEFAULT_DATASET)
    ap.add_argument("--category", default="emotional_prosody", choices=C.CATEGORIES)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--bands", nargs="+", default=["late", "mid", "all"],
                    choices=["early", "mid", "late", "all"])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--fixed-seconds", type=float, default=3.0)
    ap.add_argument("--freeform-items", type=int, default=12,
                    help="items used for the coherence control (generation is slow)")
    ap.add_argument("--max-new-tokens", type=int, default=30)
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

    # eager, not sdpa: sdpa skips building an explicit mask whenever causality
    # is inferable (batch 1, no padding), and hands the layer attention_mask=None
    # while the kernel applies is_causal internally. There is then no mask object
    # to add log(gamma) to. eager always materialises a 4D float mask.
    model, processor, info = C.load_model(eager=True, four_bit=args.four_bit)
    n_layers = len(C.decoder_layers(model))
    print("{}: {} mismatched, {} matched | {} decoder layers".format(
        args.category, len(mism), len(matched), n_layers))

    print("\nlocating claim spans ...")
    prep_mis, prep_mat, skipped = [], [], 0
    for dest, src in ((prep_mis, mism), (prep_mat, matched)):
        for it in src:
            inputs, claim, gen_pos = prepare(model, processor, it, args.fixed_seconds)
            if claim is None:
                skipped += 1
                continue
            dest.append((it, inputs, claim, gen_pos))
    print("  {} usable, {} skipped (claim span not locatable)".format(
        len(prep_mis) + len(prep_mat), skipped))
    if not prep_mis:
        raise SystemExit("no items with a locatable claim span")

    # ---- baseline -------------------------------------------------------- #
    base_mis = [dict(prediction=p, correct_letter=C.correct_letter(it),
                     claimed=C.claimed_letter(it),
                     correct=p == C.correct_letter(it), margin=m)
                for it, i, c, g in prep_mis
                for p, m in [predict(model, processor, it, i)]]
    base_mat = [dict(prediction=p, correct_letter=C.correct_letter(it),
                     claimed=C.claimed_letter(it),
                     correct=p == C.correct_letter(it), margin=m)
                for it, i, c, g in prep_mat
                for p, m in [predict(model, processor, it, i)]]
    b_acc, b_lo, b_hi, b_marg = summarise(base_mis)
    b_cap, _, _, _ = summarise(base_mat)
    b_tdr = C.tdr_block(base_mis)

    print("\n" + "=" * 78)
    print("BASELINE")
    print("=" * 78)
    print("  audio accuracy {:.3f} [{:.2f},{:.2f}]   capability {:.3f}   TDR {:.3f}"
          .format(b_acc, b_lo, b_hi, b_cap, b_tdr.get("tdr", float("nan"))))
    print("  margin {:+.2f}  <- the gap attenuation has to close".format(b_marg))

    # ---- positive control -------------------------------------------------- #
    # Without this the sweep is uninterpretable: a null result is indistinguish-
    # able from a mask edit that never took effect. Removing the claim from
    # EVERY position at EVERY layer has to collapse TDR -- the model cannot
    # follow a claim it cannot read. If this moves nothing, nothing below is
    # evidence about anything.
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL -- claim removed from every position, every layer")
    print("=" * 78)
    print("  Not a candidate fix: a model that cannot see the claim is not a")
    print("  model that resists it. This only certifies the knockout works.\n")

    pc_mis, pc_mat = condition(model, processor, prep_mis, prep_mat,
                               list(range(n_layers)), 0.0, rows="all")
    pc_acc, pc_lo, pc_hi, pc_marg = summarise(pc_mis)
    pc_cap, _, _, _ = summarise(pc_mat)
    pc_tdr = C.tdr_block(pc_mis).get("tdr")
    print("  audio accuracy {:.3f} [{:.2f},{:.2f}]   capability {:.3f}   TDR {:.3f}"
          .format(pc_acc, pc_lo, pc_hi, pc_cap, pc_tdr if pc_tdr is not None else float("nan")))
    print("  margin {:+.2f}   d(margin) {:+.2f}".format(pc_marg, pc_marg - b_marg))

    control_ok = abs(pc_marg - b_marg) > 1.0
    if control_ok:
        print("\n  Knockout verified: blocking the claim moves the margin.")
    else:
        print("\n  *** POSITIVE CONTROL FAILED ***")
        print("  Blocking the claim everywhere changed the margin by {:+.2f} nats."
              .format(pc_marg - b_marg))
        print("  The model cannot be reading the claim through the attention")
        print("  mask at all, so every row below is a null of the method, not")
        print("  a fact about the model. Check the claim span before reporting.")

    # ---- sweep ------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("ATTENUATING THE GENERATION POSITION'S VIEW OF THE CLAIM")
    print("=" * 78)
    print("  Bands are layer ranges; all of them attenuate only the answer")
    print("  position's row. Contrast with the positive control above, which")
    print("  attenuates every row.\n")
    print("  {:<7}{:>7}{:>9}{:>16}{:>9}{:>10}{:>12}".format(
        "band", "gamma", "audio", "95% CI", "margin", "d(marg)", "capability"))

    rows = []
    for band in args.bands:
        layers = band_for(band, n_layers)
        for gamma in args.gammas:
            mis, mat = condition(model, processor, prep_mis, prep_mat,
                                 layers, gamma)
            acc, lo, hi, mg_ = summarise(mis)
            cap, _, _, _ = summarise(mat)
            t = C.tdr_block(mis)
            rows.append({"band": band, "layers": [layers[0], layers[-1]],
                         "gamma": gamma, "audio": acc, "ci95": [lo, hi],
                         "margin": mg_, "margin_delta": mg_ - b_marg,
                         "capability": cap, "tdr": t.get("tdr")})
            print("  {:<7}{:>7.2f}{:>9.3f}   [{:.2f},{:.2f}]{:>9.2f}{:>+10.2f}{:>12.3f}"
                  .format(band, gamma, acc, lo, hi, mg_, mg_ - b_marg, cap))

    # ---- coherence control ------------------------------------------------ #
    print("\n" + "=" * 78)
    print("FREE-FORM COHERENCE CONTROL")
    print("=" * 78)
    print("  Closed-set accuracy is blind to the degeneration Cho et al. report,")
    print("  because their ablation also looked healthy in closed-set choice.")
    print("  Generating under the intervention and scoring with the clean model.\n")

    ff_prep = prep_mis[:args.freeform_items]
    clean_ff = freeform_fluency(model, processor, ff_prep, None, 1.0,
                                args.max_new_tokens)
    print("  {:<22}{:>10}{:>14}".format("condition", "logprob", "repetition"))
    print("  {:<22}{:>10.3f}{:>14.3f}".format(
        "baseline", clean_ff["mean_token_logprob"], clean_ff["repetition_rate"]))

    candidates = list(rows)
    best = max(candidates, key=lambda r: r["margin_delta"]) if candidates else None
    ff_best = None
    if best:
        layers = band_for(best["band"], n_layers)
        ff_best = freeform_fluency(model, processor, ff_prep, layers,
                                   best["gamma"], args.max_new_tokens)
        print("  {:<22}{:>10.3f}{:>14.3f}".format(
            "{} gamma={}".format(best["band"], best["gamma"]),
            ff_best["mean_token_logprob"], ff_best["repetition_rate"]))
        drop = clean_ff["mean_token_logprob"] - ff_best["mean_token_logprob"]
        print("\n  fluency drop: {:+.3f} nats/token, repetition {:+.3f}".format(
            -drop, ff_best["repetition_rate"] - clean_ff["repetition_rate"]))
        for s in ff_best["samples"]:
            print("    sample: {!r}".format(s))

    # ---- verdict ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  reference: removing the claim from every position moves the margin "
          "{:+.2f}".format(pc_marg - b_marg))
    print("  (that is the ceiling any targeted intervention is working toward)")

    if not control_ok:
        text = ("INCONCLUSIVE: the positive control did not move the margin, so "
                "the knockout is not reaching the model's use of the claim. "
                "Nothing here is evidence about where the override lives; fix "
                "the intervention before reading the sweep.")
    elif not best:
        text = "no targeted band tested"
    else:
        closed = best["margin_delta"] / abs(b_marg) if b_marg else 0.0
        broke_closed = best["capability"] < b_cap - 0.05
        fl_drop = (clean_ff["mean_token_logprob"] -
                   ff_best["mean_token_logprob"]) if ff_best else 0.0
        broke_free = fl_drop > 0.5 or (
            ff_best and ff_best["repetition_rate"] -
            clean_ff["repetition_rate"] > 0.2)

        print("  best targeted: {} gamma={}  margin {:+.2f} ({:.0%} of the gap), "
              "audio {:.3f}".format(best["band"], best["gamma"],
                                    best["margin_delta"], closed, best["audio"]))

        if broke_free:
            text = ("REPRODUCES CHO ET AL.'S NEGATIVE: the intervention moves the "
                    "decision but free-form fluency drops {:.2f} nats/token. "
                    "Closed-set accuracy did not show this. Not a usable fix."
                    .format(fl_drop))
        elif broke_closed:
            text = ("REJECT: capability fell {:.3f} -> {:.3f}.".format(
                b_cap, best["capability"]))
        elif closed > 0.5 and best["audio"] > b_acc + 0.05:
            text = ("VIABLE: targeted attenuation closes {:.0%} of the gap, lifts "
                    "audio accuracy {:.3f} -> {:.3f}, and free-form fluency holds. "
                    "This is the intervention to develop."
                    .format(closed, b_acc, best["audio"]))
        elif closed > 0.1:
            text = ("PARTIAL: {:.0%} of the gap closed with fluency intact. Real "
                    "but not sufficient; worth pushing on layer selection."
                    .format(closed))
        else:
            text = ("NO: attenuating the readout closes only {:.0%} of the gap. "
                    "The override is not localised where exp3 pointed, or is "
                    "distributed beyond what this granularity can reach."
                    .format(closed))
    print("\n  " + text)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       "exp6_readout_attenuation_{}.json".format(args.category))
    with open(out, "w") as f:
        json.dump({"run": dict(info, category=args.category, seed=args.seed,
                               limit=args.limit, bands=args.bands,
                               gammas=args.gammas, n_items=len(prep_mis),
                               skipped_no_claim_span=skipped),
                   "positive_control": {"audio": pc_acc, "margin": pc_marg,
                                        "margin_delta": pc_marg - b_marg,
                                        "capability": pc_cap, "tdr": pc_tdr,
                                        "passed": control_ok},
                   "baseline": {"audio": b_acc, "capability": b_cap,
                                "margin": b_marg, "tdr": b_tdr.get("tdr")},
                   "sweep": rows,
                   "freeform_baseline": clean_ff,
                   "freeform_best": ff_best,
                   "verdict": text}, f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
