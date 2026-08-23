# Full methodology

What we actually did, step by step, with the parameters. This is the companion
to `METHODS_PROVENANCE.md`, which records where each method came from and what
we changed; this file records the procedure itself.

---

## 1. Dataset

3,594 items across three conflict categories. A fourth (non-speech events) was
built and set aside before the mechanistic work.

| category | items | source | options | chance |
|---|---|---|---|---|
| emotional_prosody | 1,197 | RAVDESS | 8 emotions | 0.125 |
| speaker_identity | 1,200 | RAVDESS | 2 genders | 0.500 |
| sarcasm_pacing | 1,197 | MUStARD++ | 10 tones | 0.100 |

### Conditions

Every category is built in four conditions, adapted from LISTEN:

- **Paralinguistic** (`audio_only`) — audio, no text claim. The capability
  ceiling: can the model do this at all from the voice?
- **Neutral-Text** — audio plus text that carries no claim about the voice.
  Isolates the effect of adding text at all, separately from adding a *false*
  claim.
- **Matched** — audio plus a text claim that agrees with the voice. Correct
  answers here are ambiguous between using audio and using text, which is why
  this is a capability control and not evidence of grounding.
- **Mismatched** — audio plus a text claim that contradicts the voice. This is
  the conflict condition and the only one where dominance is measurable.

Each of the text-bearing conditions exists in two modalities, `text_audio` and
`text_only`. The `text_only` copies matter: for a Mismatched `text_only` item
the claim is the only evidence, so the correct answer *is* the claim by
construction (verified: correct letter equals the claimed label on 133/133
prosody and 150/150 identity items, and equals the true label on 266/266 and
300/300 of the `text_audio` items). Pooling the two modalities therefore
inflates dominance, and all reported TDR is computed on `text_audio` only.

### Counterbalancing

- **Order.** Each conflict item exists with the claim before and after the
  audio reference (`claim_first` / `claim_last`), balanced within the
  `text_audio` conflict items (133/133 for prosody, 150/150 for identity), so
  recency is measurable rather than assumed away.
- **Option position.** The correct letter is shuffled per item; a constant-letter
  strategy scores 0.158 against the observed 0.92 to 1.00, and accuracy is flat
  across all eight answer positions, so dominance is not letter bias.

### Known gaps

- `sarcasm_pacing` audio is incomplete (463 of 798 clips; MUStARD++ identifies
  the rest by episode ids the public release does not distribute), and the loss
  is uneven, falling hardest on the Mismatched condition (140/266 against
  168/266 for Matched).

---

## 2. Models

Behavioural sweep: Qwen2-Audio-7B-Instruct, SALMONN-13B, Phi-4-multimodal,
AZeroS, DeSTA2.5. Mechanistic work: Qwen2-Audio-7B-Instruct, with SALMONN-13B
attempted and excluded (see §9).

Qwen2-Audio specifics that the mechanistic code depends on: a Whisper-style
audio encoder (1,280-d) feeding a projector into a 32-layer language model
(4,096-d), 25 audio tokens per second at 40 ms per token, audio token id 151646.
Loaded fp16, `device_map="auto"`.

**Attention implementation.** All mechanistic experiments load with
`attn_implementation="eager"`. Under `sdpa`, transformers returns an empty
attentions tuple with no error, and skips materialising an attention mask
whenever causality is inferable — so gradients through attention and any
mask-editing intervention silently do nothing.

---

## 3. Scoring protocol

**Option log-probability, not generate-then-parse.** Every option letter is a
single token, so one forward pass scores all of them and the prediction is the
argmax. This removes the parse step entirely.

This is a deliberate change from the source protocol, made because an early
sarcasm run discarded 47% of outputs when the model answered in words the option
list did not contain, and the surviving 53% had been rescued by substring
matching — a non-random sample. Log-probability scoring means abstention cannot
masquerade as a parse failure.

**One question per context.** Each item is a fresh forward pass. No conversation
history carries between items, so an answer cannot be contaminated by earlier
ones.

**Fixed audio duration** (`--fixed-seconds 3.0`) wherever positions are compared
across runs. Patching requires that position *i* means the same thing in the
clean and corrupt runs, which fails if clips yield different numbers of audio
tokens.

---

## 4. Metrics

- **TDR (text-dominance rate)** — of Mismatched `text_audio` items, the fraction
  where the prediction equals the claimed label. Computed over parsed items.
- **Audio accuracy** — fraction where the prediction equals the true label.
- **Capability** — accuracy on Matched items. Detects a broken model; cannot
  distinguish text-following from audio-following, since both are correct there.
- **Neither** — the residual: neither the claim nor the truth.
- **Margin** — `log P(audio-correct option) - log P(claimed option)`, in nats.
  Introduced because audio accuracy saturated at 0.000 and could not show
  partial movement. Negative means the model favours the text.
- **gen-TDR** — free-form analogue of TDR: of descriptions naming exactly one
  label, the fraction naming the claimed one. Descriptions naming both or
  neither are excluded, so its denominator is reported.
- **names_true_rate** — fraction of all descriptions naming the true label.
  Exclusion-free, so it is the measure to prefer when the two disagree.

Interval estimates are Wilson intervals throughout (the normal approximation
misbehaves at the 0.000 and 1.000 rates we frequently hit). Circuit overlap uses
a percentile bootstrap, 1,000 resamples.

---

## 5. exp1 — capability screen

**Question.** Can the model do the task from audio alone? If not, no conflict
result from it can be read as suppression.

**Procedure.** Score the `Paralinguistic` (audio-only) items per category.
Compare accuracy against chance (1/n_options) with a binomial test.

**Pass criterion.** Accuracy above chance *and* p < 0.05. Both are required —
a point estimate above chance on a small sample is not a pass.

**Outcome.** This is a gate, not a preliminary. Categories and models that fail
are excluded from everything downstream, and the exclusions are reported.

---

## 6. exp2 — encoder probe

**Question.** Does the audio information survive into the language model, or is
it lost in the encoder?

**Procedure.**

1. Run the audio through the encoder, capturing hidden states at every encoder
   layer plus the post-projector output.
2. Mean-pool over **valid frames only**. The feature extractor pads every clip
   to 30 s; pooling across the padding washes out short clips.
3. Fit logistic regression (standardised features, `max_iter=2000`) to predict
   the label from each layer's pooled representation.
4. Report accuracy per layer, so the depth profile is visible rather than a
   single number.

**Controls.**

- **Speaker-independent splits** — `StratifiedGroupKFold`, 5 folds, grouped by
  `actor_id`, so no speaker appears in both train and test. Without this the
  probe can score well by memorising voices.
- **Shuffled labels** — refit on permuted labels. A different control from the
  above: it checks the probe is not fitting noise in high-dimensional features.
- **Post-projector layer included** — the encoder carrying the signal is not
  enough; it has to survive the projection into the language model's space.

---

## 7. exp3 — circuit localization

**Question.** Which components drive the decision, and how does text compare to
audio at each position?

**Procedure.** Attribution patching: a gradient-based approximation to
activation patching that needs one forward and one backward pass instead of one
run per component. Attribution is computed for both a text-side and an
audio-side corruption, so the two modalities are directly comparable.

**Position groups.** Five, not one lumped text group: `audio`, `claim`, `query`,
`options`, `gen`. Merging claim, query and options conflates what the source
methodology treats as separate roles.

**Parameters.** `--top-frac 0.20`, `--max-pairs 8`, `--n-boot 1000`,
`--ig-steps 1` (plain attribution patching; integrated gradients is a stated
divergence and the code raises rather than silently approximating it).

**Controls.**

- **Faithfulness** — keep only the top 20% of components and check they
  reproduce the full model's behaviour. Reported at 0.05/0.10/0.20/0.40.
- **Signal floor** (1e-8) — groups with no signal return an empty top-set rather
  than a stably-sorted all-zero ranking, which otherwise produces a spurious
  IoU of 1.000 between two groups that both contribute nothing.
- **Hooks observe only.** An earlier version replaced each submodule's output
  with a cached blend, which stopped attention carrying information between
  positions and zeroed the gradient at every non-final position. It scored
  faithfulness 1.00 while broken.

---

## 8. exp4 and exp5 — repair attempts

### exp4, back-patching

Copy activations from a late layer back into early layers and re-run.
`--dst 0 2 4`, `--windows 1 3 5`.

**Controls.** Capability on Matched items, and a **mid-layer source control** —
if a mid-layer source does as well as a late one, there is no late-to-early
mechanism being exploited.

### exp5, diagnosing the null

Four tests, run only because exp4 returned a flat zero and the reason mattered:

- **A, norm profile** — are the patched activations even on the right scale?
- **B, source sweep** — vary which layer supplies the activations
  (`--sources` across depth), with an **identity control**: patch a span with
  itself and require the baseline back. The run aborts if it fails, because a
  patching harness that cannot reproduce its own baseline produces numbers that
  look fine.
- **C, norm-matched** — rescale the patch to rule out a pure magnitude effect.
- **D, scaling** — window widths `1 3 5 7` and amplitudes `1.5 2.0 3.0 5.0`.

---

## 9. exp6 — readout attenuation

The main experiment. Five parts, run in this order.

### 9.1 Baseline

Closed-set TDR, audio accuracy, capability and margin on 60 Mismatched and 60
Matched `text_audio` items.

### 9.2 Positive control

Block the claim from **every position at every layer** — the input ablation.
This must move the decision; if it does not, the intervention is not reaching
the model and every null below is a null of the method rather than a fact about
the model. The verdict returns INCONCLUSIVE when the margin moves less than
1 nat.

This is not a candidate fix. A model that cannot see the claim is not a model
that resists it.

### 9.3 Layer sweep (attention knockout)

Graded attention knockout: adding `log(gamma)` to the claim's keys before
softmax scales those attention weights by gamma, with gamma = 0 the hard
knockout. Bands are `early` / `mid` / `late` / `all`; gammas `0.0` and `0.25`.

The band restriction *is* the experiment. Attenuating at every layer merely
deletes the claim.

Implementation note: the attenuator modifies the mask the model already passes
rather than substituting a precomputed one, which is required for the mask to
stay valid as `generate()` extends the sequence. It clamps to `finfo.min` after
the add, since applying `log(gamma)` to entries causality already masked
underflows to `-inf` in fp16, and a fully `-inf` row gives NaN out of softmax.

### 9.4 Routing bisection

Vary **which positions** lose sight of the claim, at every layer, gamma = 0:

- `gen` — the answer position only
- `post_claim` — every position downstream of the claim
- `all` — every position

The gap between `gen` and `all` localises the route; `post_claim` bisects it.

### 9.5 Free-form generation

Closed-set accuracy is structurally blind to the degeneration reported for
circuit ablation, because the capability control is itself multiple choice.

- **Prompt** — the item rewritten with the option list and the
  option-presupposing question removed, claim sentence kept verbatim. Otherwise
  the model answers with one letter and a fluency measure over two tokens is
  meaningless.
- **Decoding** — greedy (`do_sample=False`), so no sampling noise between
  conditions. Same items across all conditions: a paired comparison.
- **Scoring** — mean per-token log-probability under the *unmodified* model, in
  a context with **the claim removed**. Scoring against the claim-bearing prompt
  penalises correct-but-disagreeing text as improbable, which reports a working
  intervention as degeneration.
- **Also recorded** — repetition rate (loops can score well per token), median
  generated length, and a refusal to treat fluency as passing below 5 tokens.

### 9.6 Controls on the fix

- **Capability** — Matched items, where the claim is true.
- **Cost** — `Matched` + `text_only` items: no audio, claim is the only
  evidence. This is the control that separates "resists false claims" from
  "ignores text", which the Matched control cannot do.
- **Late band** — same perturbation, different depth. Rules out generic
  attention damage.
- **All layers** — rules out equivalence with input deletion.

### 9.7 Verdict guards

The verdict refuses to report an effect when:

- the positive control did not move (method not connected);
- fewer than 15 descriptions named exactly one label, or the intervened and
  baseline intervals overlap (underpowered);
- matched-item accuracy fell by more than half the gen-TDR gain
  (indiscriminate suppression rather than a fix);
- free-form fluency dropped in the claim-free context (degeneration).

---

## 10. Infrastructure

Modal, A100. Two volumes: `audio-corpus` (`/corpus`, holding `dataset_audio/`
and `results/`) and `hf-cache`. Results are pulled locally with
`modal run modal/modal_app.py --fetch-only`. The per-model behavioural wrappers
write to a separate volume (`qwen2audio-results`), fetched by
`modal/sweep_models.py --stage fetch`.

The behavioural sweep runs in three stages — `audit` (CPU, free, catches
dataset/prompt mismatches), `smoke` (`--limit 8`, GPU, confirms the model loads
and scores), then `full` — because the expensive mistake is discovering a
per-model mismatch two hours into a billed run.

---

## 11. What gates what

The order matters and is not arbitrary:

1. exp1 gates everything. A model or category at chance from audio cannot show
   suppression, so no downstream result from it is interpretable.
2. exp2 separates encoder loss from decoder suppression. If the signal never
   arrives, the question is a perception question, not a conflict question.
3. exp3 locates the decision before anything tries to change it.
4. exp4 and exp5 rule out the audio side.
5. exp6's positive control must fire before its nulls mean anything, and its
   routing result explains why its own layer sweep is null.
6. The free-form measurement is where the effect lives; the closed-set
   measurement is where it does not. These are separate regimes and are never
   compared to each other.
