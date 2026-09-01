# Figure types in mechanistic interpretability, and which ones we can fill

A companion to `METHODS_PROVENANCE.md`. That file records where each *method*
came from; this one records what each *figure* plots, how the plotted quantity
was computed, and what a reader should not conclude from it.

Built by `viz/make_mechinterp_figures.py`. The older `viz/make_figures.py`
covers the behavioural story and four of the headline plots.

**Scope: Qwen2-Audio, `emotional_prosody` and `speaker_identity`.** SALMONN-13B
is excluded because it does not clear the capability screen; `sarcasm_pacing` is
excluded as underpowered. Neither is drawn. `--models` and `--categories`
override. The capability screen itself is not one of these figures any more —
it is behavioural, it lives in the results table, and it is still what licenses
both exclusions.

**Three figures, numbered in the order the argument runs.** The number, the
function name and the output filename are one string, so there is nothing to
look up:

| # | file | shape | says |
|---|---|---|---|
| 1 | `fig1_probe_accuracy_by_encoder_layer` | layer-wise probe trajectory | the signal survives the encoder, so this is not a perception failure |
| 2 | `fig2_attribution_by_layer_and_position` | position-grouped causal map | but the decision is made from the text claim, at the answer position, late |
| 3 | `fig3_knockout_by_position` | attention-knockout bars | and the claim reaches the answer through the tokens in between |

Section 1 is the methodology behind each. Section 2 is the shape catalogue this
project draws from. Sections 3 and 4 are the gaps and the sources.

---

## 1. Methodology

### 1.1 fig1 — probe accuracy by encoder layer

**What each point is.** A linear probe fitted on mean-pooled activations from
one audio-encoder layer, scored by accuracy. 33 encoder layers plus the
post-projector representation, which gets its own tick because it is a different
module and not encoder layer 33.

**The data.** `exp2_encoder_probe.json`, `speaker_identity` only: 80 clips, 24
actors, classes `female` / `male`, chance 0.500, 5-fold **speaker-independent**
CV, seed 0, Qwen2-Audio-7B-Instruct at fp16. Speaker-independent is the part
that matters — with speaker-dependent folds a probe can memorise voices and
report a ceiling that means nothing.

**The control, drawn.** Each layer also carries a shuffled-label probe (Hewitt &
Liang 2019), plotted as the dotted line. It runs 0.245–0.513 across the layers,
i.e. at chance, and the shaded band between it and the probe is *selectivity* —
what the probe reads that it does not get from noise. Without it a ceiling could
as easily be probe capacity as signal.

**No chance rule at 0.500.** A second horizontal line competes with the shuffled
control for the same job, and the control is the one that answers the question:
it lands where a probe of this capacity lands on labels that mean nothing. The
projector sits three units out behind a break rule, because it is a different
module and not encoder layer 33.

**Drawn as a line, not a strip.** Position is a finer visual channel than
lightness, and the dips are the honest part: the write-up's "1.000 from layer 13
onward" is not literally true. Accuracy touches 1.000 at layer 13, falls to
0.975 at 21 and 0.988 at 22 and 27, and only holds ceiling from 28. A line shows
that without anyone having to say it; a heatmap of the same numbers hides it in
two shades of the same blue. The dip is visible on the line but not annotated,
so this paragraph is the only thing that points at it.

**What a reader should not conclude.** That the encoder preserves paralinguistic
information *in general*. exp2 covers speaker gender and nothing else. There is
no Qwen encoder probe for emotional prosody, so for prosody the separation
between encoder loss and decoder suppression currently rests on no measurement —
see the gaps. The panel says so on the figure rather than leaving a reader to
assume it generalises.

### 1.2 fig2 — attribution by layer and prompt position

**What each cell is.** One `(layer, submodule, position-group)` attribution,
plotted as `|attr|`, with `attn` and `mlp` summed into one row per group.
Position groups are `audio`, `claim`, `query`, `options`, `gen`. That is
32 × 2 × 5 = 320 components for Qwen2-Audio.

**The quantity.** Attribution patching. For a component with activation `a`,

```
attr = (a_clean - a_corrupt) · ∂M/∂a
```

with the gradient taken on the **corrupt** run and `a_clean` cached from a clean
forward. One forward and one backward score every component at once; exact
activation patching would need one run per component. Contributions are summed
over the token positions belonging to each group (`contrib[lo:hi+1].sum()`).

**The metric `M`.** `log P(audio-correct option) − log P(claimed option)` at the
final sequence position, over option-letter token ids. The same decision margin
used in exp5 and exp6, so the figures are commensurable.

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

**Only the text circuit is drawn.** On a scale shared with the text panels the
audio panels are near-blank, which is the finding but costs half the figure
width to show, and a blank panel reads as a rendering failure as easily as a
result. The comparison is carried numerically instead: peak ratio 18:1
text:audio at the answer position for prosody, 13:1 for identity (3.29 against
0.18), and 21.1 against 1.3 nats of summed mass, 16:1. `HEATMAP_CIRCUITS =
("text", "audio")` restores both panels on one shared scale, which is the
comparison drawn rather than quoted.

**No suptitle**, by request. The bold category labels sit above their panels and
the colourbar names the quantity. What the title carried and nothing else on the
figure does is the model and the circuit, so the caption has to say
"Qwen2-Audio-7B, text circuit" — without a known depth the layer axis means
nothing.

**Exact zeros are real, not small.** For the text circuit every audio-position
attribution is exactly 0.0: causal masking means the audio span cannot see a
claim that comes after it, so those activations are bit-identical between the
clean and corrupt runs.

**What a reviewer should hold against this figure.**

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
   Faithfulness was 0.999–1.003 at top-20% on both categories when it was
   plotted; it is no longer drawn, and that number belongs in the text.
3. **n = 8.** The attribution means are correspondingly soft.
4. **These ran on the 518-item dataset**, before it was scaled to 3,594. The
   methods are current; the numbers describe a smaller sample and should be
   regenerated before publication.

### 1.3 fig3 — attention knockout by token position

**The intervention.** Attention knockout after Geva et al., in graded form.
`log(gamma)` is added to the claim's key positions in the additive attention
mask before softmax, which scales those weights by gamma and lets the rest
renormalise; `gamma = 0` recovers their hard knockout.

Two implementation details are load-bearing. The hard block writes `finfo.min`,
not `-inf`, because a true `-inf` propagates to NaN through fp16 softmax. And
the attenuator modifies the mask the model already passes rather than
substituting a precomputed one, so the mask stays valid as `generate()` extends
the sequence. The run **requires eager attention** and raises if it does not get
a 4D float mask — sdpa passes `None` and applies causality inside the kernel,
leaving nothing to modify.

**The three bars differ only in which query rows lose sight of the claim**,
holding layers at all 32, gamma at 0, and items fixed:

| bar | rows blocked |
|---|---|
| answer position | the answer position onward |
| every position after the claim | every position downstream of the claim |
| every position | every position |

**Nothing on the panel names the panel.** The titles and the group headers are
off, so the figure does not say which category is which, nor what the dotted rule
is. Left is emotional prosody against a chance rate of 0.125, right is speaker
identity against 0.500, the dotted rule is that rate and the dashed red rule is
the un-intervened baseline. All four facts live in the caption and nowhere else.

**The metric is audio-consistent accuracy on the 60 Mismatched items**, with
Wilson intervals, against `1/n_options` read off the capability screen — 0.125
for prosody's 8 options, 0.500 for identity's 2. This is how the write-up states
the result and what a reader can hold against a chance line. The margin is the
more sensitive metric and lives in `exp6.routing[].margin_delta`; it is not
drawn.

**Result.** Blocking the answer position alone is null: prosody 0.000 → 0.000
(margin −0.001 nats), identity 0.067 → 0.083 (+0.015). Blocking every position
after the claim recovers the whole effect: prosody 0.000 → 0.533 (+10.538
nats), identity 0.067 → 0.717 (+5.802). `all` is identical to `post_claim` in
both categories. The claim is absorbed into the question and option tokens
before the answer position reads it, so cutting the direct edge changes nothing.

**The layer sweep is not drawn.** `exp6.sweep` varies the layer band — late
22–31, middle 10–21, all 0–31 — with positions held at the answer position, and
every band is null: the largest `|margin_delta|` across both categories is 0.054
nats. Half a panel of flat caps buys nothing. An earlier version carried a
subtitle derived from `exp6.sweep` at render time, which meant a later run with
a live band would have contradicted its own figure; that subtitle was removed,
so this paragraph is now the only thing asserting the null and it has to be
re-checked by hand after any re-run.

**`sweep["all", gamma=0]` and `routing["gen"]` are the same run**, identical to
the last float bit. The 2×2 has five measurements, not six.

**The positive control is what makes the nulls readable.** Blocking the claim at
every position and every layer must move the decision. If it does not, the
intervention is not reaching the model, and every null is a null of the method
rather than a fact about the model. The verdict returns INCONCLUSIVE when the
margin moves less than 1 nat.

**Capability is not drawn, by request, and it is not flattering.** The
`post_claim` condition is byte-identical to that positive control, which
`METHODS_PROVENANCE.md` describes as *not a candidate fix — a model that cannot
see the claim is not a model that resists it*. Prosody's matched-item accuracy
falls 1.000 → 0.367 under it, so most of the accuracy gain is the model no
longer reading text at all; identity costs less, 1.000 → 0.800. Those numbers
are in `exp6.routing[].capability` and **belong in the caption**, because
otherwise the panel reads as a repair, and a gain bought by damaging the model
is not a gain.

**gamma = 0 only.** gamma = 0.25 is a null at every band, and interleaving inert
bars turns the panel into a sawtooth. The null belongs in the text.

**One number to settle before publication.** The draft quotes 0.633 / 0.783 for
`post_claim` and 0.183 for identity's answer position. The committed JSON and
`mech_interp_section.txt` both say 0.533 / 0.717 and 0.083. Either the draft
quotes an uncommitted run or it needs correcting.

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
| **Layer-wise probe trajectory** | probe accuracy vs. layer index, with chance and a shuffled-label control | Pasad et al. 2021 (speech); Hewitt & Liang 2019 for selectivity — **fig1** |
| **Logit lens / tuned lens** | layer x position, cell = the token the model would emit there | nostalgebraist; Belrose et al. 2023 |
| **Attention pattern heatmap** | source x destination positions, per head | ubiquitous — and per Jain & Wallace 2019 a hypothesis, never evidence |
| **Representation-similarity curve** | cross-modal cosine similarity vs. depth | Cho et al. 2026's convergence result |

### Interventional

| shape | what it plots | canonical source |
|---|---|---|
| **Causal-tracing heatmap** | (layer, token position) grid, colour = effect of restoring/patching that cell | ROME (Meng et al. 2022) fig 2 — *the* iconic mech-interp figure |
| **Position-group tracing** | the same, with the sequence bucketed into prompt roles rather than words | Chen et al. 2026 for audio LALMs; needed the moment a span has no word to label — **fig2** |
| **Layer sweep** | one metric vs. the layer intervened on, per attribute, per model | Chen et al. 2026 fig 2; Meng et al. 2022 |
| **Faithfulness vs. sparsity** | fraction of components kept vs. fraction of behaviour reproduced | ACDC (Conmy et al. 2023), AtP* (Kramár et al. 2024), EAP |
| **Attention-knockout curve** | effect of severing an edge, restricted to a band of layers or a set of positions | Geva et al. 2023 fig 3 — **fig3** |
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

### Three choices in there that are not cosmetic

- **`fig2` shares one colour scale across its panels.** Scaled independently,
  each category looks as busy as the other regardless of how much mass it
  carries. The shared scale *is* the finding. The same argument applied to the
  text/audio split when both circuits were drawn.
- **`fig3` draws its bars with an edge.** Prosody's answer-position bar is
  0.000 exactly, and a zero-height bar with no edge draws nothing at all — the
  panel would read as two bars and one missing one. With an edge, a measured
  zero renders as a flat cap the width of a bar, which is what it is. The
  printed rate and the Wilson whisker do the same job.
- **`fig1` is a line and not a strip.** A null or a dip autoscaled into
  lightness disappears; on a position axis it does not. This is the same reason
  a null must never be autoscaled to its own noise — it acquires a gradient and
  reads as structure.

---

## 3. Gaps

**All three figures render** against the 2026-08-27 pull, which brought in the
Qwen `exp5` and `exp6` results. Their numbers were checked against
`mech_interp_section.txt` and agree, with the one exception noted in §1.3:
baseline margin −7.329 prosody / −4.843 identity, routing recovered fraction
1.00 in both categories.

- **exp2 covers `speaker_identity` only**, so there is still no encoder probe
  for emotional prosody. Per §11 of `METHODOLOGY_FULL.md`, exp2 is what
  separates encoder loss from decoder suppression — for prosody that separation
  currently rests on no measurement. This is the one remaining gap in the
  argument chain, it is now load-bearing because fig1 opens the argument, and it
  is a cheap run.
- **The capability screen is no longer plotted here.** It still gates
  everything, and it carries an unresolved inconsistency worth settling before
  the exclusions are defended in print: the stored verdict passes SALMONN's
  prosody (0.188 against 0.125 chance on 133 items) while the write-up says it
  "does not clear it in substance", citing 0.133 — chance — on the 60-item
  subset the intervention ran on. Both are true of different samples. Either
  tighten the screen criterion to the subset the mechanism runs on, or show
  both numbers.
- **exp6 records no Matched denominator.** `n_items` is the mismatched list
  only, so the capability rates quoted in §1.3 cannot be turned back into k/n:
  prosody's 0.367 fits 11/30 and 22/60 equally. Adding `n_matched` to the run
  header is a one-line fix in `eval/exp6_readout_attenuation.py`.

**Not built, and worth building:**

- **Figure 1, the methodology schematic.** Clean run / corrupt run / where the
  patch lands, in this project's own terms. Chen et al. and Cho et al. both open
  with one. Ours is a hand-drawn figure, not a plot script.
- **The qualitative example table.** `exp6.freeform[].samples` holds real
  generations per condition — "The speaker's emotion is happy" (the false claim)
  against what the intervened model says instead. This is the figure that makes
  the result legible to someone who reads no other panel.
- **Cross-modal similarity vs. depth.** Cho et al.'s convergence result is the
  natural companion to fig2, and we have never computed it.
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
  third of the network, which is the same depth range fig2 lights up.
- Geva et al. 2023 (arXiv:2304.14767) — attention knockout, restricted by band.
- Pasad et al. 2021 (arXiv:2107.04734) — layer-wise probing of a speech encoder.
- Hewitt & Liang 2019 (arXiv:1909.03368) — control tasks, and why a probe needs
  a shuffled-label baseline.
- Conmy et al. 2023 (ACDC) and Kramár et al. 2024 (AtP*) — faithfulness curves,
  and why an approximation needs one.
- Zhang & Nanda 2024 (arXiv:2309.16042), *Towards Best Practices of Activation
  Patching* — metric choice, and reporting distributions over means.
- Lin et al. 2025 (arXiv:2502.17516) — the multimodal survey and its taxonomy.
