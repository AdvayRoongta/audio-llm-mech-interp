# Where each method came from

Nothing in this pipeline is a new method. Every experiment adapts a published
procedure to audio–text conflict, and this file records which one, what we kept,
and what we changed. Read it as the provenance appendix for the paper's methods
section.

Where a deviation from the source is listed, it is a deviation we know about and
can defend. Anything else should be treated as an unintended divergence.

---

## Prior work this project builds on

| short name | reference | what we take from it |
|---|---|---|
| **LISTEN** | Chen et al., *Do Audio LLMs Really LISTEN, or Just Transcribe?*, arXiv:2510.10444 | the condition × modality design, the audio-only capability screen, the "neither" outcome |
| **Cho et al.** | *Who Wins the Conflict?*, arXiv:2606.18924 | circuit localization by attribution patching, position grouping, back-patching, the suppression-vs-erasure framing, modality ordering |
| **Pasad et al.** | [*Layer-wise Analysis of a Self-supervised Speech Representation Model*, arXiv:2107.04734](https://arxiv.org/abs/2107.04734) | layer-wise linear probing of a speech encoder |
| **AtP\*** | [Kramár et al., arXiv:2403.00745](https://arxiv.org/abs/2403.00745) | attribution patching as an O(1)-pass approximation to activation patching, and its known failure modes |
| **Geva et al.** | [*Dissecting Recall of Factual Associations*, arXiv:2304.14767](https://arxiv.org/abs/2304.14767) | attention knockout as a causal test |
| **StreamingLLM** | [Xiao et al., arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | attention sinks: early positions absorb attention mass as a no-op |
| **Jain & Wallace** | [*Attention is not Explanation*, arXiv:1902.10186](https://arxiv.org/abs/1902.10186) | why an attention map is a hypothesis, not evidence |
| **PARSA-Bench** | (via project notes) | speaker-gender recognition is near-solved for Qwen models, which is what licenses reading identity TDR as dominance rather than incapacity |

---

## exp1 — capability screen and TDR

**Source: LISTEN.** The four conditions (`Paralinguistic` / `Neutral-Text` /
`Matched` / `Mismatched`) and the audio-only screen are theirs. The screen exists
because of their finding that audio-only emotion accuracy sits at 16.5–34.0%
against a 12.5% uniform baseline: capability is low enough that a conflict
failure cannot be read as suppression unless the screen is passed first.

Also from LISTEN: reporting **"neither"** alongside text- and audio-following.
They found most conflict predictions match neither cue, which a binary TDR
cannot represent.

**Kept:** condition structure, the screen as a gate, the three-way outcome.

**Changed, deliberately:**

1. **Scoring by option log-prob instead of generate-then-parse.** Option letters
   are single tokens, so one forward scores every option. This is not from
   LISTEN — it is a response to a failure we observed: an earlier sarcasm run
   discarded 47% of outputs because the model answered in words the option list
   did not contain, and the surviving 53% had been rescued by substring matching,
   making the sample non-random. Log-prob scoring removes the parse step, so
   abstention cannot masquerade as a parse failure.
2. **TDR computed on `text_audio` items only.** For `text_only` Mismatched items
   the correct answer *is* the claim by construction (verified: 20/20 in every
   category), so pooling them inflates TDR. The pooled figure ran ~0.09 high on
   speaker identity.

**Known limitation:** TDR is computed over parsed items, so it is conditioned on
the model producing an answer. Harmless where invalid rates are 0–4%; it is what
made the earlier sarcasm numbers uninterpretable at 47%.

---

## exp2 — encoder probe

**Source: Pasad et al.** Their method is a layer-wise probe of a speech encoder's
intermediate representations to trace how information evolves with depth. We
apply the same procedure to Whisper inside Qwen2-Audio, asking whether emotion
and speaker identity survive to the projector.

**Kept:** per-layer extraction, mean pooling, a linear probe, reading the result
as a depth trajectory rather than a single number.

**Changed / added:**

1. **Pool over valid frames only.** The Whisper feature extractor pads every clip
   to 30 s. Pooling across the padding would wash out short clips, so the valid
   frame count is taken from `feature_attention_mask` and converted through the
   conv stride.
2. **Post-projector probe.** Not in Pasad et al., which studies an encoder alone.
   We add it because the projector output is literally what the language model
   receives, and "the projector loses nothing" was an untested assumption in our
   own plan.
3. **Speaker-independent folds** (`StratifiedGroupKFold` grouped by `actor_id`).
   RAVDESS has 24 actors each contributing many clips; with random folds the same
   voice appears in train and test and the probe can lean on speaker identity.
   This is standard practice in speech, not an invention.
4. **Label-shuffled control**, so "the probe found structure" is separable from
   "the probe memorised 79 examples".

**Audit performed:** clip duration differs significantly by emotion in RAVDESS
(one-way ANOVA F = 18.1, p = 4×10⁻¹⁹), which would be a confound. A
duration-only probe with the same folds reaches 0.190 against 0.143 chance; the
encoder probe reaches 0.730. Duration contributes but does not explain the
result.

---

## exp3 — circuit localization

**Source: Cho et al.**, who localised text and audio circuits for *lexical*
conflicts and named paralinguistic conflicts as future work. Their method is
attribution patching with integrated gradients, scored as
(layer, position, submodule) tuples, thresholded by faithfulness at a sparsity
level (20% for Qwen), built on TransformerLens.

**Method itself: AtP\*** — attribution patching as a gradient-based O(1)-pass
approximation to exhaustive activation patching.

**Kept:** two counterfactuals (one per modality), position grouping, the sparsity
threshold, faithfulness verification by real activation patching, and IoU between
the two circuits as the comparison statistic.

**Changed, and why:**

1. **Plain transformers hooks, not TransformerLens** — it has no Qwen2-Audio
   support.
2. **Integrated gradients is NOT implemented.** This is a real divergence from
   Cho et al. and must be stated in the paper. Interpolating every submodule's
   output simultaneously breaks cross-position gradient flow (see the bug below);
   doing IG correctly needs a per-component path costing
   `n_components × n_steps` forwards. `--ig-steps > 1` raises rather than
   silently doing the broken thing. We run plain AtP.
3. **Position groups are audio / claim / query / options / gen**, mirroring their
   Data / Query / Generation split. An earlier version lumped claim, query and
   options into one "text" group, which merged their Data with their Query and
   made the comparison invalid.
4. **Bootstrap CIs on the IoU**, resampling pairs. Cho et al. report point
   values; with 8 pairs a bare IoU cannot support a claim of difference. Chance
   IoU for top-20% of 64 components is 0.103, and several of our intervals
   contain it.

**Two bugs found by running it, both of which produced confident nonsense:**

- *Attribution replacement.* The first version **replaced** every attention and
  MLP output with a cached blend. That makes each submodule a constant with
  respect to its inputs, so attention stops carrying information between
  positions and every non-final position has identically zero gradient. The
  entire circuit collapsed onto the generation position. Correct attribution
  patching observes activations and gradients from an ordinary forward/backward;
  it never substitutes. **Faithfulness scored 1.00 on the broken version**,
  because patching the final position alone does recover the effect — so
  faithfulness is necessary but not sufficient validation. This is an instance of
  the false-negative failure modes AtP\* documents.
- *Zero-signal groups reported as perfect overlap.* Causal masking makes the
  audio span's activations bit-identical between clean and corrupt runs for the
  text circuit (the claim follows the audio), so those attributions are exactly
  0.0. Sorting an all-zero list with a stable sort returns insertion order, so
  both circuits produced the same top-k and IoU came out **1.000** — reading as
  "perfect overlap" while meaning "no signal". Now reported as undefined, with
  per-group magnitudes printed alongside.

**Attention-sink correction (StreamingLLM):** where raw attention mass is
measured, the first few positions are excluded from the denominator, since they
absorb attention as a no-op. Without it the measurement largely reflects how idle
a head is. Attention spans are also length-normalised, because the audio span is
longer than the text span and would otherwise look attention-heavy for free.

**Framing (Jain & Wallace):** attention maps are treated as hypotheses. Every
attention-based claim is paired with a causal test.

---

## exp4 — back-patching

**Source: Cho et al.** Cache the audio span's hidden states at a late layer,
write them into an early layer, run once. Their reported result: lexical audio
accuracy 0.35 → 0.46 with text capability unharmed. Their hyperparameters are
settled — late→early, ~31 → 0–4, window ∈ {1,3,5}, single iteration, iterating
hurts — so **we do not redo the grid**. We test transfer to paralinguistic
conflicts with a confirmatory sweep around their optimum.

**Kept:** direct replacement with no scaling, their layer/window ranges, single
iteration, and the capability control (accuracy on non-conflicting items must not
drop).

**Added:** a **mid-layer source control**. If sourcing from a middle layer works
as well as from a late one, then "late→early" is not the mechanism and the
matured-representation story is wrong. This is our addition, and it is what makes
a null result interpretable.

**Also from Cho et al., as a negative constraint:** they showed circuit ablation
raises audio accuracy 0.33 → 0.82 but destroys free-form coherence. We therefore
do **not** propose ablation as an intervention — that avenue is already
settled negative.

**Reporting fix:** when every configuration sits at the floor, the mid-layer
control can beat the late-layer one by coincidence, and the verdict logic
reported "late→early is not the mechanism" — implying some other layer worked.
Nothing worked. "No effect" is now tested first.

---

## exp5 — patch diagnosis

Not from a paper. exp4 returned a flat null, and this asks which of its
assumptions failed before abandoning the approach. The sweep structure (vary
the source layer, the width of the destination window, the amplitude) is the
standard way activation-patching work isolates a null, e.g. the layer sweeps in
Meng et al. (arXiv:2202.05262).

The **margin**, log P(audio option) - log P(claimed option), replaces accuracy
because accuracy was saturated at 0.000 and could not show partial movement. A
continuous read-out of a saturated binary is ours, and should be labelled so.

The **identity control** -- patch a span with itself and require the baseline
back -- is the standard patching sanity check, and it aborts the run rather
than warning, because a patching harness that fails it produces numbers that
look fine.

## exp6 — readout attenuation

**Attention knockout**, Geva et al. (arXiv:2304.14767). They set attention
weights from a target position to a source span to zero to test whether a
specific edge carries information. Ours is the graded form: adding log(gamma)
to the claim's keys before softmax scales those weights by gamma and lets the
rest renormalise, with gamma=0 recovering their hard knockout.

The **layer bands** follow their practice of restricting the knockout to a
contiguous range rather than applying it everywhere; the restriction is the
experiment, since knocking out at every layer merely deletes the claim.

The **positive control** (remove the claim from every position, every layer) is
ours, added after a six-condition sweep of nulls proved uninterpretable: a null
from "this edge does not carry the claim" and a null from "the intervention is
not reaching the model" are indistinguishable without a condition that must
work. It is not a proposed fix.

The **free-form coherence control** answers Cho et al. (arXiv:2606.18924), who
found circuit ablation lifting closed-set accuracy while degrading open
generation. Since our capability control is itself multiple-choice, it is
structurally blind to that failure. Generating under the intervention and
scoring with the unmodified model is the natural test.

Two details are ours and load-bearing:

- Scoring happens in a **claim-free context**. Scoring against the prompt that
  contains the claim conflates incoherent text with text that merely
  contradicts the claim -- and the second is what success looks like. Scored
  the wrong way, a working intervention is reported as degeneration; that
  happened before the fix.
- **gen-TDR**, the free-form analogue of TDR: of descriptions naming exactly
  one label, the fraction naming the claimed one. Descriptions naming both or
  neither are excluded, matching how TDR treats undecided cases, so its
  denominator is below the item count and is reported. `names_true_rate` is
  computed over all items and is the exclusion-free version.

The **matched-item generative control** is ours and decides the result. An
intervention that stops the model reading text at all produces the same
gen-TDR drop as one that resists false claims. Matched items, where the claim
is true, separate them.

## Statistics

| choice | source | why |
|---|---|---|
| Wilson score interval | Wilson (1927) | several proportions sit at exactly 0.0 or 1.0, where the normal approximation fails |
| Percentile bootstrap for IoU | Efron | no distributional assumption available for a set-overlap statistic |
| `StratifiedGroupKFold` | standard speech practice | speaker-independent evaluation |
| Binomial test vs. `1/k` | — | the screen's chance level is the option count, not 0.5 |

---

## Design decisions that are ours, and should be labelled as such

These are not adapted from prior work. They are small, but a reader should know
they are not inherited:

1. **Option log-prob scoring** in the mechanistic experiments (exp1's rationale
   above). The behavioural evals still use generate-then-parse, so running both
   on Qwen is a genuine robustness check on whether TDR depends on output
   parsing.
2. **The mid-layer source control** in exp4.
3. **Fixed-duration padding** for patching, so every clip yields the same number
   of audio tokens. Patching position *i* assumes position *i* means the same
   thing in both runs; without this the alignment is silently wrong.
4. **Claim rewriting in place** for the text counterfactual. The dataset's
   conditions use disjoint clip sets, so no Matched item shares a clip with a
   Mismatched one — the counterfactual has to be constructed. Substitution is
   confined to the text above the option list (the options repeat the same
   words) and requires exactly one match, or the pair is dropped.

---

## Things prior work already settled — do not re-run

From the project's literature review, recorded here so nobody spends GPU time
rediscovering them:

- **Lexical text dominance** is established across models and languages
  (Cho et al.). Our contribution is paralinguistic, not lexical.
- **Circuit ablation as a fix** is a settled negative (Cho et al.): it works in
  closed-set choice and destroys free-form coherence.
- **Back-patching hyperparameters** for lexical conflicts are settled
  (Cho et al.), including the sweep.
- **Sarcasm collapse** — models detect that a conflict exists but collapse
  diverse sarcastic emotions into "ridicule"/"frustration" (LISTEN). Our own
  sarcasm result reproduced this before the category was set aside for
  incomplete audio.
- **Speaker gender recognition** is near-solved for Qwen models (PARSA-Bench),
  which is why identity TDR reads as dominance rather than incapacity.

## State of the committed result files

`results/exp*.json` were produced on the **518-item** dataset, before it was
scaled to 3,594. The methods they used are the ones described above, but the
numbers describe a smaller sample and should be regenerated before publication.

One file is internally stale in a way worth naming: the `verdict` string in
`results/exp4_backpatching_*.json` reads *"a mid-layer source does as well, so
late->early is not the mechanism"*. That text was written by the pre-fix branch
ordering. Every configuration sat at the floor (audio accuracy 0.000), and the
mid-layer control's 0.025 beat it by coincidence. The correct reading is **no
effect**, and the current code reports that. The underlying numbers in the file
are unaffected -- only the summary sentence is wrong.

## Open at the time of writing

- No integrated gradients (plain AtP only) — a stated divergence from Cho et al.
- exp6's fix is demonstrated in open generation, not closed-set choice, which
  is unmoved. The routing table is the reason the two differ, but the
  free-form result is one category on one model so far.
- exp6 assumes the claim's token span is known. That holds by construction here
  and would need a way to identify the span in deployment.
- exp2/exp3/exp4 have run on Qwen2-Audio only; the accessors are architecture-specific.
- `sarcasm_pacing` is excluded for incomplete MUStARD++ audio, not for any
  scientific reason.
- Phi-4-multimodal fails the capability screen on both categories, so its TDR
  cannot be read as suppression.
