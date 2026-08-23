# Mechanistic analysis — draft section

*Draft for the 6-page workshop paper. Prose is deliberately plain; the point is
that every claim below is one we can support, with the number attached. Editorial
notes are marked `>> NOTE`. Target length ~2 to 2.5 pages including one figure.*

*All results in this section are Qwen2-Audio-7B-Instruct (fp16, eager attention,
32 decoder layers), 60 mismatched and 60 matched items per category for the
closed-set measurements and 30 items per condition for the generative ones.*

---

## 4. Why the model ignores the audio

The behavioural results show that when a text claim contradicts the voice, the
model follows the text almost every time (TDR 0.983 for emotional prosody, 0.933
for speaker identity). This section asks why. We work through four questions in
order: is the audio information present, where is the decision made, can the
usual repairs fix it, and what does work.

---

### 4.1 The audio information is present and usable

The first possibility to rule out is that the model simply cannot hear the
difference — that the encoder discards prosody before the language model ever
sees it.

It does not. A linear probe recovers speaker gender from the encoder at 1.000
accuracy (chance 0.5) from encoder layer 13 onward, and still at 1.000 after the
projector, meaning the signal survives into the language model. A shuffled-label
control on the same features sits at 0.43. Splits are speaker-independent across
24 actors, so the probe cannot be memorising individual voices.

A probe only shows the information is *decodable*, not that the model can *use*
it. So we also test this causally: when we remove the text claim from the input,
the model answers from audio at 0.533 (prosody) and 0.717 (identity), up from
0.000 and 0.067 with the claim present. Chance is 0.125 and 0.5 respectively.

The model has the information, can act on it, and does not.

>> NOTE: this is the strongest single pair of numbers we have — 0.000 vs 0.533
>> for the same items, same model, one input removed. Consider leading the whole
>> paper with it.

---

### 4.2 The decision is made at the answer position, and text dominates it

We localise the responsible components with attribution patching, grouping
positions into audio, claim, query, options, and the answer position.

Keeping only the top 20% of components by attribution reproduces the full
model's behaviour (faithfulness 1.00), so the located circuit is the one that
matters. Within it, the answer position is where the two modalities compete, and
the competition is lopsided: text attribution 3.29 against audio attribution
0.18, a ratio of roughly 18 to 1.

This tells us where to intervene. It does not yet tell us how.

---

### 4.3 The standard repairs do not work

We first tried the repair used in prior work on lexical conflicts: back-patching,
copying late-layer activations back to early layers so the audio evidence gets a
second pass.

It had no effect at all. Audio accuracy stayed at 0.000 across every source
layer, destination layer, and window width we tried. A mid-layer source did as
well as a late one, which means there was no late-to-early mechanism being
exploited in the first place.

We then asked whether the audio signal could simply be made louder. Relocating
which layer we patch from, widening the destination window, and amplifying the
patched activations all move the decision margin by very little: the best case is
+0.51 nats against a gap of −7.33 nats, about 6%. Amplification saturates at
+0.10 and then breaks the model outright, producing NaNs at high gain.

The conclusion is that this cannot be fixed from the audio side. The text is not
winning because it is louder.

>> NOTE: keep this subsection short — three or four sentences in the final
>> version. It matters for honesty and it motivates 4.4, but it is not the
>> contribution.

---

### 4.4 The claim reaches the answer indirectly

Since the audio side is a dead end, we intervene on the text side, using
attention knockout: we block the answer position from attending to the claim
tokens.

This does nothing. Blocking that connection at every one of the 32 layers leaves
the decision unchanged (0% of the gap closed).

The reason is that by the time the model produces an answer, the claim is no
longer only in the claim tokens. We test this by varying *which* positions lose
sight of the claim, holding everything else fixed:

| positions blocked | effect on the decision |
|---|---|
| the answer position only | 0% |
| every position after the claim | 100% |
| every position | 100% |

Blocking the positions *between* the claim and the answer reproduces the entire
effect of deleting the claim from the input, and does so in both categories. The
claim is copied into the intervening query and option tokens early, and the
answer position reads it from there. Cutting the direct edge changes nothing
because the information takes a detour.

This is why the obvious intervention fails, and it is the main mechanistic
finding of the paper.

>> NOTE: left panel of Fig. X is exactly this table. Lead the figure with it.

---

### 4.5 Blocking the claim mid-network restores the audio in generation

The closed-set result above is a null at every layer. But closed-set choice
turned out to be the wrong place to look.

When the model writes a free-form description rather than picking a letter, the
same intervention has a large effect — as long as it is applied in the middle
layers. We prompt the model to describe the speaker in one or two sentences, with
the claim still present, and measure which label the description actually names.

| | prosody | identity |
|---|---|---|
| names the claimed (false) label — baseline | 1.000 | 1.000 |
| names the claimed label — mid-layer block | **0.375** [0.21, 0.57] | **0.433** [0.27, 0.61] |
| names the true label — baseline | 0.000 | 0.000 |
| names the true label — mid-layer block | **0.500** | **0.567** |

Baseline intervals are [0.86, 1.00] and [0.89, 1.00], so they do not overlap the
intervened condition in either category.

The generated text does not degrade. Scored by the unmodified model in a
claim-free context, mean token log-probability *improves* (−1.28 to −0.64 for
prosody, −1.45 to −0.78 for identity), repetition falls, and median length is
essentially unchanged. Typical output goes from "The speaker is an
English-speaking female who sounds happy" to "The speaker sounds surprised while
stating that kids are talking by the door" — where *surprised* is the true label
and *happy* was the claim.

>> NOTE: scoring in a claim-free context is not a detail. Scored against the
>> prompt that contains the claim, correct-but-disagreeing text is penalised as
>> improbable, and the working intervention reads as degeneration. State this in
>> one sentence in the methods so reviewers know we thought about it.

Two differences explain why generation moves and multiple choice does not. In
closed-set scoring a single position is attenuated; during generation every token
produced is blocked from the claim. And the open-ended prompt has no option list,
so there is far less intervening text to relay the claim — which §4.4 identified
as the actual route.

---

### 4.6 Controls

Four controls, each ruling out a specific alternative reading.

**It did not break the model.** On matched items, where the claim agrees with the
audio, accuracy *rises*: 0.767 → 0.900 (prosody) and 0.933 → 1.000 (identity).
The intervention makes the model better at the honest cases, not worse.

**It is not "ignore all text."** On text-only items, where there is no audio and
the claim is the only evidence available, accuracy is unchanged: 0.567 → 0.567
and 1.000 → 1.000. The intervention downweights the claim rather than deleting
it, so it only changes the outcome when something else is competing.

**It is not generic attention damage.** The same intervention applied to late
layers has no effect at all (claimed-label rate stays at 1.000 in both
categories), despite perturbing attention just as much.

**It is not equivalent to hiding the input.** Applying the block at every layer
suppresses more but costs capability (0.667 in prosody, below the 0.767
baseline). More suppression is not more benefit; the effect is specific to a
depth range.

>> NOTE: all four belong in the paper. The second is the one a reviewer will ask
>> for, and we have it — do not cut it for space.

---

### 4.7 What this does and does not show

The intervention is unconditional. We block the claim whether or not it is true;
the model does not learn to judge the conflict. It works because downweighting
only flips an outcome when competing evidence exists, and we measure that cost
directly at zero on the items where the claim is the sole evidence. We have not
tested the harder case of ambiguous audio paired with a reliable text claim.

The effect appears in open generation and not in closed-set choice (0% and 1% of
the decision gap for the two categories). Since these models are typically used
by asking them to describe what they hear, we consider generation the setting of
interest, but the limit should be read literally.

Further limits: all mechanistic results are on a single model, Qwen2-Audio;
the method assumes the claim's token span is known, which holds by construction
here; and a partial block (γ = 0.25) had no effect at any depth, so the
intervention behaves as a switch rather than a dial.

>> NOTE: report the γ = 0.25 null explicitly. It is a clean negative and it stops
>> us describing the method as "graded", which the data does not support.

---

### Methods provenance

No method here is new. Attention knockout follows Geva et al.; attribution
patching follows AtP*; layer-wise probing of the encoder follows Pasad et al.;
circuit localisation and back-patching follow Cho et al., whose ablation result
we reproduce as a negative. What is ours is the set of controls: the positive
control that certifies the knockout is connected, claim-free scoring of generated
text, the generative claim-following measure, and the matched and text-only
controls in §4.6.

>> NOTE: full mapping, including every deviation, is in
>> `notes/METHODS_PROVENANCE.md`. Cite it or fold two sentences of it into the
>> methods section.
