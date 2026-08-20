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
import re
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
        # "gen"        -- only the answer position's view of the claim
        # "post_claim" -- every position downstream of the claim. Tests whether
        #                 the claim reaches the readout via relay positions
        #                 (query/option tokens) rather than by direct attention.
        # "all"        -- every position, i.e. the claim is removed from the
        #                 model's input entirely. A positive control, never a
        #                 proposed fix.
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
                elif self.rows == "post_claim":
                    # decode: the single row is downstream by construction
                    rows = slice(0, 1) if q_len == 1 else slice(
                        min(hi + 1, q_len - 1), q_len)
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


def prepare_freeform(model, processor, item, fixed_seconds):
    """Encode an open-ended item and locate its claim.

    exp3's position_groups anchors on the option list (it looks for "\nA. ") to
    bound the claim search, and the open-ended prompt has no option list -- so
    it finds no claim and every item would be silently dropped. The claim line
    itself is unchanged, so search the whole prompt for it directly, matching
    exp3's rule: the line mentioning the asserted label, excluding the
    transcript, which quotes the words rather than asserting the label.
    """
    inputs = C.encode(model, processor, item, fixed_seconds=fixed_seconds)
    claim = E3.claim_value(item)
    span = None
    if claim:
        for line in item.get("prompt", "").splitlines():
            if claim.lower() in line.lower() and "transcript" not in line.lower():
                span = C.find_span(processor, inputs, line.strip())
                break
    return inputs, span, inputs["input_ids"].shape[1] - 1


def predict(model, processor, item, inputs, layers=None,
            claim=None, gen_pos=None, gamma=1.0, rows="gen"):
    letters = list(C.option_map(item).keys())
    if layers is None or claim is None:
        pred, sc = C.predict_letter(model, processor, inputs, letters)
        return pred, margin(sc, item)
    with ClaimAttenuator(model, layers, claim, gen_pos, gamma, rows):
        pred, sc = C.predict_letter(model, processor, inputs, letters)
    return pred, margin(sc, item)


FREEFORM_INSTRUCTION = ("Given the information above, describe the speaker in "
                        "one or two sentences.")


def freeform_item(item):
    """The same item with the option list removed, so generation runs on.

    The closed-set prompt ends in "Answer with the letter", so the model emits
    one letter and stops -- roughly two tokens, on which a fluency measure means
    nothing. The claim sentence is kept verbatim, so the claim span is still
    locatable and attenuating it is still the same intervention.
    """
    lines, keep = item["prompt"].split("\n"), []
    for ln in lines:
        if re.match(r"^[A-Z]\.\s", ln.strip()):
            break
        keep.append(ln)
    head = "\n".join(keep).rstrip()
    # the closing question presupposes the option list that was just dropped
    head = re.sub(r"\n[^\n]*\?\s*$", "", head).rstrip()
    out = dict(item)
    out["prompt"] = head + "\n\n" + FREEFORM_INSTRUCTION
    return out


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
def strip_claim(item):
    """The item with the claim line removed, for neutral scoring."""
    claim = E3.claim_value(item)
    if not claim:
        return item
    keep = [ln for ln in item["prompt"].splitlines()
            if not (claim.lower() in ln.lower()
                    and "transcript" not in ln.lower())]
    out = dict(item)
    out["prompt"] = "\n".join(keep)
    return out


def label_texts(item):
    """(claimed label, true label) as they would appear in prose."""
    claimed = E3.claim_value(item)
    opts = C.option_map(item)
    true = opts.get(C.correct_letter(item))
    return (str(claimed) if claimed else None, str(true) if true else None)


def mentions(text, label):
    """Does the description name this label? Prefix match, so 'disgust'
    catches 'disgusted' and 'surprise' catches 'surprised', while the leading
    word boundary keeps 'male' from matching inside 'female'."""
    if not label:
        return False
    return re.search(r"\b" + re.escape(label.lower()), text.lower()) is not None


def freeform_fluency(model, processor, prepared, layers, gamma,
                     max_new_tokens=30, rows="gen"):
    """Generate under the intervention, score the text with the clean model.

    Closed-set accuracy cannot see the degeneration Cho et al. reported, so this
    measures it directly: mean token log-probability, under the unmodified
    model, of what the intervened model produced. Degenerate or repetitive text
    scores low. Also reports a repetition rate, since loops can score
    deceptively well on per-token probability.
    """
    import torch

    scores, reps, samples, lens = [], [], [], []
    said_claim = said_true = named_true = 0
    for item, inputs, claim, gen_pos, score_inputs in prepared:
        ctx = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            if layers is not None:
                with ClaimAttenuator(model, layers, claim, gen_pos, gamma, rows):
                    out = model.generate(**ctx, max_new_tokens=max_new_tokens,
                                         do_sample=False)
            else:
                out = model.generate(**ctx, max_new_tokens=max_new_tokens,
                                     do_sample=False)

        new = out[0, inputs["input_ids"].shape[1]:]
        if new.numel() < 1:
            continue
        lens.append(int(new.numel()))
        text = processor.tokenizer.decode(new, skip_special_tokens=True)

        # Score with the UNMODIFIED model, in a context with the claim REMOVED.
        # Scoring against the claim-bearing prompt conflates two different
        # things: text that is incoherent, and text that simply contradicts the
        # claim. The second is exactly what a working intervention produces, so
        # scoring there would penalise success as if it were degeneration.
        ctx_ids = score_inputs["input_ids"]
        full = torch.cat([ctx_ids, new.unsqueeze(0)], dim=1)
        kw = {k: v for k, v in score_inputs.items()}
        kw["input_ids"] = full
        if "attention_mask" in kw:
            kw["attention_mask"] = torch.ones_like(full)
        with torch.no_grad():
            logits = model(**kw, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[0, ctx_ids.shape[1] - 1:-1], -1)
        tok_lp = lp.gather(-1, new.unsqueeze(-1)).squeeze(-1)
        scores.append(float(tok_lp.mean()))

        claimed_lab, true_lab = label_texts(item)
        m_claim, m_true = mentions(text, claimed_lab), mentions(text, true_lab)
        named_true += int(m_true)
        if m_claim != m_true:                 # ignore "both" and "neither"
            said_claim += int(m_claim)
            said_true += int(m_true)

        toks = new.tolist()
        reps.append(1.0 - len(set(toks)) / len(toks))
        if len(samples) < 3:
            samples.append(text.replace("\n", " ")[:90])

    med = sorted(lens)[len(lens) // 2] if lens else 0
    decided = said_claim + said_true
    n_gen = len(scores)
    return {"generative_tdr": (said_claim / decided) if decided else None,
            "generative_tdr_ci95": C.wilson_ci(said_claim, decided) if decided else None,
            "names_true_ci95": C.wilson_ci(named_true, n_gen) if n_gen else None,
            "n_items": n_gen,
            # names the true label; on matched items the claim agrees with the
            # audio, so this is the generative capability control
            "names_true_rate": (named_true / n_gen) if n_gen else None,
            "said_claim": said_claim, "said_true": said_true, "decided": decided,
            "mean_token_logprob": (sum(scores) / len(scores)) if scores else float("nan"),
            "repetition_rate": (sum(reps) / len(reps)) if reps else float("nan"),
            "n": len(scores), "samples": samples, "median_tokens": med,
            # a fluency number computed on one or two tokens says nothing
            "informative": med >= 5}


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
    ap.add_argument("--freeform-items", type=int, default=30,
                    help="items used for the coherence control (generation is slow)")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--freeform-gammas", type=float, nargs="+", default=[0.0, 0.25],
                    help="attenuation strengths for the generative sweep; the "
                         "graded knockout only shows its value here")
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

    # ---- routing ----------------------------------------------------------- #
    # The gen-row sweep and the positive control differ only in WHICH positions
    # lose sight of the claim, so the gap between them localises the route.
    # post_claim blocks every downstream position from reading the claim; if
    # that recovers most of the control's effect, the claim reaches the readout
    # through relay positions rather than by direct attention.
    print("\n" + "=" * 78)
    print("ROUTING -- which positions carry the claim to the readout")
    print("=" * 78)
    print("  All at every layer, gamma=0. 'gen' and 'all' repeat the two")
    print("  extremes above so the bisection reads in one table.\n")
    print("  {:<12}{:>9}{:>16}{:>9}{:>10}{:>12}".format(
        "rows", "audio", "95% CI", "margin", "d(marg)", "capability"))

    route = []
    for mode in ["gen", "post_claim", "all"]:
        r_mis, r_mat = condition(model, processor, prep_mis, prep_mat,
                                 list(range(n_layers)), 0.0, rows=mode)
        acc, lo, hi_, mg_ = summarise(r_mis)
        cap, _, _, _ = summarise(r_mat)
        t = C.tdr_block(r_mis)
        route.append({"rows": mode, "audio": acc, "ci95": [lo, hi_], "margin": mg_,
                      "margin_delta": mg_ - b_marg, "capability": cap,
                      "tdr": t.get("tdr")})
        print("  {:<12}{:>9.3f}   [{:.2f},{:.2f}]{:>9.2f}{:>+10.2f}{:>12.3f}"
              .format(mode, acc, lo, hi_, mg_, mg_ - b_marg, cap))

    span = route[-1]["margin_delta"] - route[0]["margin_delta"]
    if abs(span) > 1e-9:
        recovered = (route[1]["margin_delta"] - route[0]["margin_delta"]) / span
        print("\n  blocking downstream positions recovers {:.0%} of the effect of"
              .format(recovered))
        print("  removing the claim outright.")
    else:
        recovered = float("nan")

    # ---- coherence control ------------------------------------------------ #
    print("\n" + "=" * 78)
    print("FREE-FORM COHERENCE CONTROL")
    print("=" * 78)
    print("  Closed-set accuracy is blind to the degeneration Cho et al. report,")
    print("  because their ablation also looked healthy in closed-set choice.")
    print("  Generating under the intervention and scoring with the clean model.\n")

    # rebuilt without the option list, or the model answers with one letter and
    # a fluency measure over two tokens means nothing
    def build_ff(prep):
        out = []
        for it, _, _, _ in prep[:args.freeform_items]:
            fi = freeform_item(it)
            f_inputs, f_claim, f_gen = prepare_freeform(
                model, processor, fi, args.fixed_seconds)
            if f_claim is None:
                continue
            # neutral context for scoring: same audio and question, no claim
            score_inputs = C.encode(model, processor, strip_claim(fi),
                                    fixed_seconds=args.fixed_seconds)
            out.append((fi, f_inputs, f_claim, f_gen, score_inputs))
        return out

    ff_mis, ff_mat = build_ff(prep_mis), build_ff(prep_mat)
    print("  open-ended prompt: {} mismatched, {} matched items\n".format(
        len(ff_mis), len(ff_mat)))
    if not ff_mis:
        print("  *** no claim span located in the open-ended prompt; the")
        print("  coherence control did not run ***\n")

    def fmt(v, spec="{:.3f}"):
        return "n/a" if v is None else spec.format(v)

    print("  {:<14}{:>8}{:>15}{:>10}{:>10}{:>9}{:>7}".format(
        "condition", "gen-TDR", "95% CI", "mis:true", "mat:true", "logprob", "rep"))
    print("  {:<14}{:>8}{:>15}{:>10}{:>10}{:>9}{:>7}".format(
        "", "lower=fix", "", "higher=fix", "= capab.", "", ""))

    ff_rows = []
    # gamma is swept here, not fixed at a hard block. The knockout is graded on
    # purpose, and generation is the only place that choice has ever mattered --
    # the closed-set sweep is flat at every gamma. A partial attenuation that
    # keeps more of the claim while still yielding to the audio is the operating
    # point worth finding.
    conditions = [("baseline", None, 1.0)]
    for b in args.bands:
        for g in args.freeform_gammas:
            conditions.append(("{} g={:g}".format(b, g), band_for(b, n_layers), g))

    for name, layers, gamma in conditions:
        f_mis = freeform_fluency(model, processor, ff_mis, layers, gamma,
                                 args.max_new_tokens)
        f_mat = freeform_fluency(model, processor, ff_mat, layers, gamma,
                                 args.max_new_tokens)
        ff_rows.append({"condition": name, "band": None if layers is None else name.split()[0],
                        "gamma": gamma, "mismatched": f_mis, "matched": f_mat})
        ci = f_mis.get("generative_tdr_ci95")
        ci_s = "[{:.2f},{:.2f}] n={}".format(ci[0], ci[1], f_mis.get("decided")) \
            if ci else ""
        print("  {:<14}{:>8}{:>15}{:>10}{:>10}{:>9}{:>7}".format(
            name, fmt(f_mis.get("generative_tdr")), ci_s,
            fmt(f_mis.get("names_true_rate")), fmt(f_mat.get("names_true_rate")),
            fmt(f_mis.get("mean_token_logprob"), "{:+.2f}"),
            fmt(f_mis.get("repetition_rate"), "{:.2f}")))

    clean_ff = ff_rows[0]["mismatched"]
    clean_mat = ff_rows[0]["matched"]
    print("\n  gen-TDR: of descriptions naming exactly one label, the fraction")
    print("  naming the CLAIMED one -- the free-form analogue of TDR. Scored in")
    print("  a claim-free context, so contradicting the claim is not penalised.")
    print("  mat:true is the capability control: on matched items the claim")
    print("  agrees with the audio, so a drop there is the cost of the fix.")

    # pick the best free-form condition on gen-TDR, not on the closed-set sweep
    base_cap = clean_mat.get("names_true_rate")
    cand = [r for r in ff_rows[1:]
            if r["mismatched"].get("generative_tdr") is not None]
    # a condition that buys gen-TDR by losing matched-item accuracy is not a
    # fix, so it cannot win the selection either
    keeps_cap = [r for r in cand
                 if base_cap is None
                 or (r["matched"].get("names_true_rate") is not None
                     and r["matched"]["names_true_rate"] >= base_cap - 0.05)]
    pool = keeps_cap or cand
    ff_best_row = min(pool, key=lambda r: r["mismatched"]["generative_tdr"]) \
        if pool else None
    if cand and not keeps_cap:
        print("\n  (no condition held matched-item accuracy; selecting on "
              "gen-TDR alone)")
    ff_best = ff_best_row["mismatched"] if ff_best_row else None

    if ff_best_row:
        print("\n  best free-form condition: {}".format(ff_best_row["condition"]))
        print("  median generated length: {} tokens (baseline), {} (intervened)"
              .format(clean_ff.get("median_tokens"), ff_best.get("median_tokens")))
        if not (clean_ff.get("informative") and ff_best.get("informative")):
            print("  *** too few tokens generated for this to mean anything ***")
        for smp in clean_ff.get("samples", []):
            print("    baseline: {!r}".format(smp))
        for smp in ff_best.get("samples", []):
            print("    {:<8}: {!r}".format(ff_best_row["condition"], smp))

    # ---- verdict ---------------------------------------------------------- #
    # best closed-set row, for the closed-vs-free-form contrast below
    best = max(rows, key=lambda r: r["margin_delta"]) if rows else None

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
        ff_informative = bool(ff_best and ff_best["informative"]
                              and clean_ff["informative"])
        broke_free = ff_informative and (fl_drop > 0.5 or (
            ff_best["repetition_rate"] - clean_ff["repetition_rate"] > 0.2))

        print("  best targeted: {} gamma={}  margin {:+.2f} ({:.0%} of the gap), "
              "audio {:.3f}".format(best["band"], best["gamma"],
                                    best["margin_delta"], closed, best["audio"]))

        g0 = clean_ff.get("generative_tdr")
        g1 = ff_best.get("generative_tdr") if ff_best else None
        gen_gain = (g0 - g1) if (g0 is not None and g1 is not None) else None
        # the capability control: matched items, where the claim is TRUE.
        # An intervention that merely stops the model reading text drops this
        # as hard as it drops gen-TDR, and is not a fix.
        c0 = clean_mat.get("names_true_rate")
        c1 = (ff_best_row["matched"].get("names_true_rate")
              if ff_best_row else None)
        cap_cost = (c0 - c1) if (c0 is not None and c1 is not None) else None

        if gen_gain is not None and gen_gain > 0.2 and not broke_free:
            if cap_cost is not None and cap_cost > 0.5 * gen_gain:
                text = ("NOT A FIX -- INDISCRIMINATE: gen-TDR falls {:.3f} -> "
                        "{:.3f} on mismatched items, but matched-item accuracy "
                        "falls {:.3f} -> {:.3f} alongside it. The intervention "
                        "suppresses the text claim whether or not it is true, "
                        "so it buys resistance to false claims by discarding "
                        "true ones."
                        .format(g0, g1, c0, c1))
            else:
                text = ("FREE-FORM EFFECT: closed-set choice is unmoved ({:.0%} "
                        "of the gap) but in open generation the description "
                        "flips toward the audio, gen-TDR {:.3f} -> {:.3f}, "
                        "with fluency intact and matched-item accuracy holding "
                        "at {}. This is the strongest patch candidate so far."
                        .format(closed, g0, g1,
                                "n/a" if c1 is None else "{:.3f}".format(c1)))
        elif broke_free:
            text = ("REPRODUCES CHO ET AL.'S NEGATIVE: the intervention moves the "
                    "decision but free-form fluency drops {:.2f} nats/token in a "
                    "claim-free scoring context, so this is degeneration rather "
                    "than disagreement. Not a usable fix.".format(fl_drop))
        elif broke_closed:
            text = ("REJECT: capability fell {:.3f} -> {:.3f}.".format(
                b_cap, best["capability"]))
        elif closed > 0.5 and best["audio"] > b_acc + 0.05:
            text = ("VIABLE: targeted attenuation closes {:.0%} of the gap and "
                    "lifts audio accuracy {:.3f} -> {:.3f}{}."
                    .format(closed, b_acc, best["audio"],
                            ", with free-form fluency intact" if ff_informative
                            else "; FREE-FORM FLUENCY UNTESTED, verify before "
                                 "claiming this is usable"))
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
                   "routing": route,
                   "routing_recovered_fraction": recovered,
                   "freeform": ff_rows,
                   "freeform_baseline": clean_ff,
                   "generative_tdr_baseline": clean_ff.get("generative_tdr"),
                   "freeform_best": ff_best,
                   "verdict": text}, f, indent=2)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
