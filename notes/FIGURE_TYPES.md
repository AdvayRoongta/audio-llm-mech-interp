# Figure types in mechanistic interpretability, and which ones we can fill

A companion to `METHODS_PROVENANCE.md`. That file records where each *method*
came from; this one records what each *figure* plots, how the plotted quantity
was computed, and what a reader should not conclude from it.

Built by `viz/make_mechinterp_figures.py`. The older `viz/make_figures.py`
covers the behavioural story and four of the headline plots.

**Scope: Qwen2-Audio, `emotional_prosody` and `speaker_identity`.** SALMONN-13B
is excluded because it does not clear the capability screen; `sarcasm_pacing` is
excluded as underpowered. Neither is drawn. Two figures deliberately ignore both
filters — `m15`, which *is* part of the evidence for the exclusions, and `m14`,
which is behavioural rather than mechanistic. `--models` and `--categories`
override.

All 17 figures render as of the 2026-08-27 pull, which brought in the Qwen
`exp5` and `exp6` results that six of them had been waiting on.

Section 1 is the methodology behind the three figures that carry the argument.
Section 2 is the shape catalogue. Sections 3–5 are the full figure list, the
gaps, and sources.

---

## 1. Methodology for the main figures

Three figures carry the paper's argument: **m1** (with its aggregation **m3**),
**m11**, and **m15**, which gates both. Everything else either validates these
or rules out an alternative.

### 1.1 m1 / m3 — attribution by layer and prompt position

**What each cell is.** One `(layer, submodule, position-group)` attribution,
plotted as `|attr|`. Submodules are `attn` and `mlp`; position groups are
`audio`, `claim`, `query`, `options`, `gen`. That is 32 × 2 × 5 = 320 cells for
Qwen2-Audio (40 × 2 × 5 = 400 for a 40-layer model, if a second one is added).

**m3 plots a peak, not a sum.** Each bar is the single largest
`(layer, submodule)` component within that position group — that is what exp3
stores in `group_magnitudes`. So the headline 18:1 at the answer position for
Qwen prosody is a ratio of peaks (3.29 against 0.18), not of total mass. The
summed mass tells the same story and is worth quoting alongside it: 21.1 against
1.3 nats at the answer position, 16:1. They are different quantities and the
axis now says which.

**The quantity.** Attribution patching. For a component with activation `a`,

```
attr = (a_clean - a_corrupt) · ∂M/∂a
```

with the gradient taken on the **corrupt** run and `a_clean` cached from a
clean forward. One forward and one backward score every component at once;
exact activation patching would need one run per component. Contributions are
summed over the token positions belonging to each group
(`contrib[lo:hi+1].sum()`).

**The metric `M`.** `log P(audio-correct option) − log P(claimed option)` at the
final sequence position, over option-letter token ids. This is the same decision
margin used in exp5 and exp6, so the figures are commensurable.

**Two counterfactuals, one per circuit.** A circuit is defined by what changes
the answer, so each modality needs its own clean/corrupt pair:

| circuit | clean | corrupt |
|---|---|---|
| text | claim rewritten to agree with the audio | the original conflicting claim |
| audio | the item's own audio | another item's audio, different truth |

The audio counterfactual keeps the prompt byte-identical, so tokenization,
option order and length match automatically and only the acoustics differ. The
text counterfactual has to be built by rewriting the claim in place, because the
dataset's conditions use disjoint clip sets — no Matched item shares a clip with
a Mismatched one. Substitution is confined to the text above the option list and
requires exactly one match, or the pair is dropped.

**Alignment, which is what makes or breaks this.** Patching position *i* assumes
position *i* means the same thing in both runs. So audio is padded/trimmed to a
fixed 3.0 s, giving identical audio-token counts, and any pair whose two prompts
tokenize to different lengths is rejected outright. Both are asserted before any
attribution is computed — a misaligned pair produces confident nonsense rather
than an error, which is the worst failure mode available here.

**n = 8 pairs per circuit**, averaged component-wise.

**Why the two panels share a colour scale.** Scaled independently, the audio
circuit looks as busy as the text circuit while carrying a fraction of the mass.
The shared scale is the finding, not a rendering preference.

**Validation, plotted separately.**

- **m2, faithfulness.** Real activation patching restricted to the top-k
  components, checking the linear approximation held. 0.999–1.003 at top-20%
  on both categories.
- **m4, circuit overlap.** IoU between the two top-k sets, percentile bootstrap
  over pairs, 1,000 resamples. Chance is computed per model rather than
  hardcoded: for two independent top-k sets from n components,
  `IoU = k / (2n − k)` with `k = floor(0.2n)` — 0.103 for Qwen's 64 components.
  It is computed rather than hardcoded so a model with a different layer count
  gets its own chance level.
- **Signal floor, 1e-8.** A group with no signal returns an empty top-set rather
  than a stably-sorted all-zero ranking. Without it, two groups that both
  contribute nothing "agree" perfectly and IoU reads 1.000.

**What a reviewer should hold against these figures.**

1. **Plain AtP, not AP-IG.** Cho et al. run integrated gradients with K=5 on top
   of attribution patching. That is not implemented here: interpolating every
   submodule's output simultaneously breaks cross-position gradient flow, and
   doing it correctly needs a per-component path costing
   `n_components × n_steps` forwards. `--ig-steps > 1` raises rather than
   silently doing the broken thing. **This is a deliberate, stated divergence
   from the source method and belongs in the paper.**
2. **AtP has documented false-negative modes** (AtP*), so faithfulness is
   necessary but not sufficient. We have direct evidence of that: an earlier
   version replaced each submodule's output with a cached blend, which severed
   attention's cross-position flow and collapsed the entire circuit onto the
   generation position — and **still scored faithfulness 1.00**, because
   patching the final position alone does recover the effect. The current hooks
   observe only; they call `retain_grad()` and never modify an output.
3. **n = 8.** The IoU intervals are wide and several contain chance. The
   attribution means are correspondingly soft.
4. **These ran on the 518-item dataset**, before it was scaled to 3,594. The
   methods are current; the numbers describe a smaller sample and should be
   regenerated before publication.
5. **Exact zeros are real, not small.** For the text circuit, every
   audio-position attribution is exactly 0.0 — causal masking means the audio
   span cannot see a claim that comes after it, so those activations are
   bit-identical between the clean and corrupt runs. `m3` marks them `0` rather
   than letting a log-axis floor read as a small contribution.

### 1.2 m11 — the routing bisection

**The intervention.** Attention knockout after Geva et al., in graded form.
`log(gamma)` is added to the claim's key positions in the additive attention
mask before softmax, which scales those weights by gamma and lets the rest
renormalise; `gamma = 0` recovers their hard knockout.

Two implementation details are load-bearing. The hard block writes
`finfo.min`, not `-inf`, because a true `-inf` propagates to NaN through fp16
softmax. And the attenuator modifies the mask the model already passes rather
than substituting a precomputed one, so the mask stays valid as `generate()`
extends the sequence. The run **requires eager attention** and raises if it does
not get a 4D float mask — sdpa passes `None` and applies causality inside the
kernel, leaving nothing to modify.

**The three conditions differ only in which query rows lose sight of the
claim**, holding layers, gamma and items fixed:

| condition | rows blocked |
|---|---|
| `gen` | the answer position onward |
| `post_claim` | every position downstream of the claim |
| `all` | every position |

**Metric.** The same margin as above, reported as `margin_delta` against the
un-intervened baseline.

**The positive control is what makes the nulls readable.** Blocking the claim at
every position and every layer must move the decision. If it does not, the
intervention is not reaching the model, and every null below is a null of the
method rather than a fact about the model. The verdict returns INCONCLUSIVE when
the margin moves less than 1 nat. It is not a candidate fix — a model that
cannot see the claim is not a model that resists it.

**Result.** `gen` is null (−0.057 nats); `post_claim` and `all` both recover the
full positive-control effect (+2.571), recovered fraction 1.00. The claim is
absorbed into the question and option tokens before the answer position reads
it, so cutting the direct edge changes nothing.

### 1.3 m15 — the capability screen

Paralinguistic / `audio_only` items only. Accuracy against a chance rate of
`1/n_options`, with a binomial test; **passing requires both** a point estimate
above chance and p < 0.05, since a point estimate above chance on a small sample
is not a pass. Wilson intervals throughout, because several rates sit at exactly
0.0 or 1.0 where the normal approximation misbehaves.

Scoring is by option log-probability rather than generate-and-parse, so
abstention cannot masquerade as a parse failure — the failure mode that made an
earlier sarcasm run uninterpretable at 47% discards.

This figure is the gate. A model or category that does not clear its own chance
line cannot be shown to be suppressing anything, and is excluded from everything
downstream. It is the argument behind both exclusions in the write-up.

---

## 2. The grammar

Mechanistic interpretability figures divide the way the methods do — into
**observational** (what is encoded, and where) and **interventional** (what
happens when we change something). The survey framing is Lin et al.,
*A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models*
(arXiv:2502.17516), which sorts the methods as linear probing, logit lens,
causal tracing and circuit analysis, representation decomposition, task vectors,
SAEs, and neuron-level description.

### Observational

| shape | what it plots | canonical source |
|---|---|---|
| **Layer-wise probe trajectory** | probe accuracy vs. layer index, with chance and a shuffled-label control | Pasad et al. 2021 (speech); Hewitt & Liang 2019 for selectivity |
| **Logit lens / tuned lens** | layer x position, cell = the token the model would emit there | nostalgebraist; Belrose et al. 2023 |
| **Attention pattern heatmap** | source x destination positions, per head | ubiquitous — and per Jain & Wallace 2019 a hypothesis, never evidence |
| **Representation-similarity curve** | cross-modal cosine similarity vs. depth | Cho et al. 2026's convergence result |

### Interventional

| shape | what it plots | canonical source |
|---|---|---|
| **Causal-tracing heatmap** | (layer, token position) grid, colour = effect of restoring/patching that cell | ROME (Meng et al. 2022) fig 2 — *the* iconic mech-interp figure |
| **Position-group tracing** | the same, with the sequence bucketed into prompt roles rather than words | Chen et al. 2026 for audio LALMs; needed the moment a span has no word to label |
| **Layer sweep** | one metric vs. the layer intervened on, per attribute, per model | Chen et al. 2026 fig 2; Meng et al. 2022 |
| **Faithfulness vs. sparsity** | fraction of components kept vs. fraction of behaviour reproduced | ACDC (Conmy et al. 2023), AtP* (Kramár et al. 2024), EAP |
| **Attention-knockout curve** | effect of severing an edge, restricted to a band of layers | Geva et al. 2023 fig 3 |
| **Circuit-overlap (IoU)** | set overlap between two discovered circuits, against a chance level | Cho et al. 2026 |
| **Intervention sweep grid** | source x destination (or window x destination), colour = post-intervention metric | Meng et al. 2022 |
| **Logit-difference distribution** | per-item spread of the decision metric, not just its mean | Zhang & Nanda 2024, on patching metrics |
| **Before/after with controls** | the intervention beside the controls that decide whether it counts | no single source; it is what separates a result from an artifact |

### The two that are not plots

Every strong paper in this area also carries a **methodology schematic** as
Figure 1 (clean run, corrupted run, where the patch lands) and a **qualitative
example table** showing real outputs before and after. Chen et al. 2026 open
with the first; Cho et al. close with the second. We have material for both and
have built neither — see the gaps below.

---

## 2. What we built, and what feeds it

Seventeen files, from `python3 viz/make_mechinterp_figures.py`.

| figure | type | source data | says |
|---|---|---|---|
| `m1_attribution_map` (x3) | causal-tracing heatmap, position-grouped | `exp3.attr_text` / `attr_audio` | text mass concentrates at the answer position in late layers; the audio panel is nearly empty on the same colour scale |
| `m2_faithfulness` | faithfulness vs. sparsity | `exp3.faithfulness` | the top 20% of components reproduce the full model (≥0.99) |
| `m3_group_magnitude` | position-group modality bars | `exp3.group_magnitudes` | peak ratio 18x text:audio at the answer position for prosody, 13x for identity |
| `m4_circuit_overlap` | IoU with bootstrap CI vs. chance | `exp3.overlap_iou` + `_ci95` | overlap is above chance but several intervals are wide |
| `m5_probe_depth` | layer-wise probe + selectivity | `exp2.results[].layers` | the signal survives the encoder and the projector — **for gender only**; there is no Qwen prosody probe |
| `m6_margin_distribution` | per-item logit difference | `exp1.records[].scores` | 82–100% of items sit below zero; it is a distribution shift, not a few outliers |
| `m7_backpatch_grid` | intervention sweep grid, 2 rows | `exp4.sweep` | every cell at the floor, capability intact: nothing moved |
| `m8_source_sweep` | effect vs. collateral damage | `exp5.test_b_source_sweep` | margin gain arrives exactly where capability collapses |
| `m9_norm_profile` | residual-norm growth | `exp5.test_a_norms` | late/early = 7.5x, so a late→early patch is an amplitude shock first |
| `m10_knockout_bands` | attention knockout by band | `exp6.sweep` | every band is a null against a positive control of +2.57 nats |
| `m11_routing` | routing bisection | `exp6.routing` | answer position alone: nothing. Positions in between: the whole effect |
| `m12_generative_fix` | intervention + controls | `exp6.freeform` | with the decided-description count printed under each point |
| `m13_generation_quality` | fluency / repetition / length | `exp6.freeform` | whether the intervention wrecks the text |
| `m14_outcome_mass` | three-way follow-rate, 7 models | `*_summary.json` `tdr` | the "neither" residual, which a binary TDR hides |
| `m15_capability_gate` | screen vs. chance, Wilson CIs | `exp1` Paralinguistic items | the gate behind the SALMONN and sarcasm exclusions (draws every model) |

### Three choices in there that are not cosmetic

- **`m1` shares one colour scale across the text and audio panels.** Scaled
  independently, the audio circuit looks as busy as the text circuit while
  carrying a fifth of the mass. The shared scale *is* the finding.
- **`m7` pins its colour scale to [0, 1] and carries a capability row.** A null
  autoscaled to its own noise acquires a gradient. The capability row is kept
  even though Qwen's is flat at 1.00, because a gain bought by damaging the
  model is not a gain, and the figure has to be able to show that. It did for
  SALMONN before the exclusion: audio rose 0.015 → 0.18 while capability fell
  0.865 → 0.10 over the same cells.
- **`m12` prints the gen-TDR denominator under every point.** The metric drops
  descriptions naming several labels or none, so the denominator moves between
  conditions. The verdict logic already refuses to call an effect below 15; a
  reader needs to be able to see the same thing. (Currently unrendered — see
  section 4.)

---

## 3. Gaps

**Everything now renders.** The 2026-08-27 pull added
`results/exp5_patch_diagnosis_emotional_prosody.json` and
`results/exp6_readout_attenuation_{emotional_prosody,speaker_identity,sarcasm_pacing}.json`,
all Qwen2-Audio/fp16. Every figure's numbers were checked against
`mech_interp_section.txt` and agree: baseline margin −7.323, TDR 0.983 / 0.933,
routing recovered fraction 1.00 in both categories, gen-TDR 1.000 → 0.375 and
1.000 → 0.433, matched 0.767 → 0.900 and 0.933 → 1.000, text-only flat at 0.567
and 1.000, best source layer 16 at +0.511 nats, late sources to −0.381,
amplitude saturating at +0.097 before capability collapses to 0.100 at α = 5.

Two coverage notes:

- **exp5 covers `emotional_prosody` only**, so `m8` and `m9` are single-panel.
- **exp2 covers `speaker_identity` only**, so there is still no encoder probe
  for emotional prosody. Per §11 of `METHODOLOGY_FULL.md`, exp2 is what
  separates encoder loss from decoder suppression — for prosody that separation
  currently rests on no measurement. This is the one remaining gap in the
  argument chain, and it is a cheap run.

**One inconsistency to resolve before this is shown outside the project:** `m15`
marks SALMONN's prosody bar green, "passes screen", because the stored verdict
is a pass (0.188 vs 0.125 chance on 133 items). The write-up says it "does not
clear it in substance", citing 0.133 — chance — on the 60-item subset the
intervention ran on. Both are true of different samples, but a figure captioned
*passes* beside prose saying *excluded* invites the question. Either tighten the
screen criterion to the subset the mechanism runs on, or show both numbers.

**Not built, and worth building:**

- **Figure 1, the methodology schematic.** Clean run / corrupt run / where the
  patch lands, in this project's own terms. Chen et al. and Cho et al. both open
  with one. Ours is a hand-drawn figure, not a plot script.
- **The qualitative example table.** `exp6.freeform[].samples` holds real
  generations per condition — "The speaker's emotion is happy" (the false claim)
  against what the intervened model says instead. This is the figure that makes
  the result legible to someone who reads no other panel.
- **Recency / claim-order.** `make_figures.py` fig3 covers it, but only for
  summaries carrying `tdr.by_order` — which the Qwen, DeSTA and SALMONN
  behavioural summaries predate. Those three are silently absent from that plot.
- **Cross-modal similarity vs. depth.** Cho et al.'s convergence result is the
  natural companion to `m1`, and we have never computed it.
- **A logit-lens panel.** AudioLens does this for audio models. Nothing in our
  pipeline currently emits per-layer vocabulary projections.

---

## 4. Reading list for the shapes

- Meng et al. 2022, *Locating and Editing Factual Associations in GPT*
  (arXiv:2202.05262) — the causal-tracing heatmap.
- Chen, Huang & Lee 2026, *Causal Tracing of Audio-Text Fusion in Large Audio
  Language Models* (arXiv:2603.13768) — the closest prior art to this project;
  layer-wise recovery-rate curves faceted by attribute, and token-wise tracing
  bucketed into prompt roles. Also reports Qwen fusing abruptly in the final
  third of the network, which is the same depth range our `m1` lights up.
- Geva et al. 2023 (arXiv:2304.14767) — attention knockout, restricted by band.
- Conmy et al. 2023 (ACDC) and Kramár et al. 2024 (AtP*) — faithfulness curves,
  and why an approximation needs one.
- Zhang & Nanda 2024 (arXiv:2309.16042), *Towards Best Practices of Activation
  Patching* — metric choice, and reporting distributions over means.
- Lin et al. 2025 (arXiv:2502.17516) — the multimodal survey and its taxonomy.
