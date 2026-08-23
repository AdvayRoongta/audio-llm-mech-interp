# Mechanistic analysis — draft section

*Draft for the 6-page workshop paper. Written long and explained fully, on the
assumption the PI will cut the explanations that the venue's audience does not
need. Every number is from `results/`; `>> NOTE` marks a decision still open or a
place where the wording is load-bearing.*

*All results are Qwen2-Audio-7B-Instruct (fp16, eager attention, 32 decoder
layers). Closed-set measurements use 60 mismatched and 60 matched items per
category; generative measurements use 30 items per condition.*

---

## 4. Why the model ignores the voice

The behavioural results show that when a text claim contradicts the voice, the
model follows the text almost every time — a text-dominance rate (TDR) of 0.983
for emotional prosody and 0.933 for speaker identity, where TDR is the fraction
of conflict items on which the model's answer matches the false text claim rather
than what the audio actually contains. This section asks why that suppression of
paralinguistic information happens, and whether it can be undone.

We work through four questions in order. Is the audio information reaching the
model at all? Where in the network does the decision get made? Do the repairs
that worked in prior studies work here? And if not, what does?

>> NOTE: a sentence early on should say that we study one model in depth here,
>> and that the behavioural section already established the effect is not unique
>> to it. Otherwise a reader wonders why a single-model analysis licenses a
>> general claim.

---

### 4.1 The model does hear the voice

The simplest explanation would be that the audio encoder throws prosody away
before the language model ever sees it — that the model follows the text because
the text is all it has. We can rule that out.

To check what information is present inside the network, we use a *probe*: a
small linear classifier trained on the model's own internal activations while the
model itself stays frozen. If a simple classifier can read a property off those
activations, the property is present in a form the model could use. Probing
layer by layer shows where in the network it appears and whether it survives.

Speaker gender is recoverable at 1.000 accuracy from encoder layer 13 onward,
against a chance rate of 0.5. Crucially it is still at 1.000 after the
*projector* — the module that converts the audio encoder's output into the
representation the language model actually reads. The signal is not lost in
transit. Whatever goes wrong happens after this point, inside the language model.

Two controls make that number trustworthy. First, we split the data by actor, so
no speaker appears in both training and test. Without this, a probe could score
well by memorising individual voices rather than learning anything about gender.
Second, we retrain the probe on randomly shuffled labels, which gives 0.43 —
near chance, as it should be. This is a different check: it confirms the probe is
not simply fitting noise in high-dimensional activations.

A probe shows the information is *readable*. It does not show the model *uses*
it. For that we need a causal test, so we simply delete the text claim from the
input and ask the same questions again. With the claim present the model answers
from the audio 0.000 of the time for prosody and 0.067 for identity. With the
claim removed, those become 0.533 and 0.717, against chance rates of 0.125 (eight
emotion options) and 0.5 (two gender options).

Same model, same audio, same questions. The only difference is one sentence of
text. The model can do this task from audio and does not when text is available.

>> NOTE: this pair — 0.000 versus 0.533 — is the strongest single result in the
>> paper. Consider opening with it rather than burying it in the analysis.

---

### 4.2 Where the decision is made, and what wins there

To find the components responsible, we use *attribution patching*: a method that
estimates how much each part of the network contributes to the final decision.
The honest version of this measurement would re-run the model once per component
with that component replaced, which is far too slow for a network this size.
Attribution patching approximates the same quantity using gradients, needing only
one forward and one backward pass.

Because it is an approximation, it needs verifying. We do this with a
*faithfulness* check: take only the top 20% of components the method ranked
highest, keep those, and see whether they reproduce what the full model does.
They do, at 1.00. So the small set we identified really is the machinery driving
the behaviour, not an artifact of the approximation.

We group positions in the input by their role — the audio tokens, the text claim,
the question, the answer options, and the position where the model produces its
answer. Here "position" means a slot in the input sequence; the model reads text
and audio as a sequence of tokens, and the answer is computed at the final one.

The competition between the two modalities happens at that answer position, and
it is badly lopsided: text contributes 3.29 against audio's 0.18, roughly 18 to
1. This tells us where to intervene. It does not yet tell us how.

---

### 4.3 Strengthening the audio does not help

Our first attempt followed Cho et al., who studied conflicts between two text
sources and found that *back-patching* reduced the dominance of one over the
other. Back-patching means taking the activations a late layer has computed and
copying them back into an early layer, then re-running the rest of the network —
in effect giving the evidence a second pass through the model's reasoning, with
more opportunity to influence the outcome.

In our setting it did nothing at all. Audio accuracy stayed at 0.000 across every
source layer, destination layer, and window width we tried. A mid-layer source
performed no worse than a late one, and every configuration sat at the floor,
which means there was no late-to-early mechanism being exploited in the first
place — the method was not being applied wrongly, there was simply nothing there
for it to move.

We then asked a blunter question: can the audio evidence just be made stronger?
We swept which layer we copy from, how many layers we write into, and how much we
amplify the copied activations.

To measure small effects we need something finer than accuracy, which was pinned
at zero and could not show partial movement. We use the *decision margin*: the
difference in log-probability the model assigns to the audio-correct answer
versus the text-claimed answer, in nats. Negative means the model favours the
text. At baseline this margin is −7.32, a very large preference.

Everything we tried moved it slightly and never enough. The best single source
layer (layer 16) gained +0.51 nats — under a tenth of the gap. Sources in the
late layers actually made it worse, down to −0.38. Widening the destination
window helped less rather than more. Amplification saturated around +0.10, and at
the highest gain the model broke outright: its output became numerically unstable
and its ability to do the task at all collapsed to 0.100.

The conclusion is that this cannot be fixed from the audio side. The audio is not
losing because it is too quiet, so making it louder does not change the outcome.

>> NOTE: in the final version this can be compressed to a short paragraph. It
>> matters because it rules out the obvious fix and motivates §4.4, but it is not
>> the contribution and should not eat space.

---

### 4.4 The claim reaches the answer by an indirect route

Since the audio side was a dead end, we intervened on the text side using
*attention knockout*: the model's attention mechanism lets each position read
from other positions, and we can sever a specific connection by masking it, then
see whether the behaviour changes. If severing a connection changes nothing, that
connection was not carrying the information.

We blocked the answer position from attending to the claim tokens. This also did
nothing — blocking that connection at all 32 layers left the decision unchanged,
closing 0% of the gap.

The reason turns out to be informative. By the time the model produces an answer,
the claim is no longer only in the claim tokens. It has already been copied into
the words that sit between the claim and the answer. We tested this by varying
*which* positions lose sight of the claim, holding everything else fixed:

| positions blocked from seeing the claim | effect on the decision |
|---|---|
| the answer position only | 0% |
| every position after the claim | 100% |
| every position | 100% |

Blocking the intervening positions reproduces the entire effect of deleting the
claim from the input, and does so identically in both categories. So the claim is
absorbed early into the question and option tokens, and the answer position reads
it from them. Cutting the direct connection changes nothing because the
information travels by a detour.

This explains why the obvious intervention fails, and it is the central
mechanistic finding of the paper.

>> NOTE: the left panel of Fig. X is this table. It is the clearest thing we
>> have — 0%, 100%, 100% — and should lead the figure.

---

### 4.5 Blocking the claim mid-network restores the audio in generation

Everything above measures multiple-choice answers, where the model picks a
letter. That turned out to be the wrong place to look.

When the model writes a description instead of picking an option, the same
intervention has a large effect — provided it is applied in the middle layers. We
prompt it to describe the speaker in one or two sentences, leave the false claim
in the prompt, and record which label the description actually names.

| | prosody | identity |
|---|---|---|
| names the false claimed label — no intervention | 1.000 | 1.000 |
| names the false claimed label — mid-layer block | **0.375** [0.21, 0.57] | **0.433** [0.27, 0.61] |
| names the true label — no intervention | 0.000 | 0.000 |
| names the true label — mid-layer block | **0.500** | **0.567** |

Without intervention the model repeats the false claim every single time. With
it, it names the true label about half the time. The baseline intervals are
[0.86, 1.00] and [0.89, 1.00], so they do not overlap the intervened condition in
either category.

The generated text does not degrade — it improves. Scored by the unmodified model,
mean per-token log-probability rises from −1.28 to −0.64 for prosody and −1.45 to
−0.78 for identity, repetition falls, and length is essentially unchanged. In
practice the output moves from "The speaker is an English-speaking female who
sounds happy" to "The speaker sounds surprised while stating that kids are
talking by the door" — where *surprised* is what the voice actually conveys and
*happy* was the false claim.

One methodological point matters here. We score the generated text in a context
with the claim removed. If we score it against the prompt that still contains the
claim, then text which correctly contradicts the claim is judged improbable, and
a working intervention looks like a model producing nonsense. The measurement has
to be blind to agreement with the claim, or it measures the wrong thing.

Two differences explain why generation moves while multiple choice does not. In
multiple-choice scoring only one position is attenuated; during generation every
token the model produces is blocked from the claim. And the description prompt
has no list of options in it, so there is far less intervening text available to
carry the claim forward — which §4.4 identified as the actual route.

>> NOTE: this ordering states the negative before the positive. It is the honest
>> order but it means the first sentence about our contribution is a null. Lead
>> with the effect and explain the null afterwards if the PI prefers.

---

### 4.6 Controls

Four controls, each ruling out a specific alternative explanation. All four
should stay in the paper.

**It did not damage the model.** On matched items, where the claim agrees with
the audio, accuracy goes *up*: 0.767 to 0.900 for prosody, 0.933 to 1.000 for
identity. The intervention makes the model better on the honest cases, which is
the opposite of what a broken model would do.

**It is not simply "ignore all text."** This is the important one. On text-only
items, where there is no audio at all and the claim is the only evidence
available, accuracy is unchanged: 0.567 to 0.567 for prosody and 1.000 to 1.000
for identity. We reduce the claim's influence rather than removing it, so the
outcome only changes when there is competing evidence to change it to. When the
claim is all the model has, it still uses it.

**It is not generic damage from perturbing attention.** Applying the identical
block to the late layers has no effect whatsoever — the model still repeats the
false claim 1.000 of the time in both categories, despite the same amount of
interference.

**It is not equivalent to hiding the claim from the model.** Applying the block
at every layer suppresses the claim more but costs capability, dropping to 0.667
on matched prosody items, below the 0.767 baseline. More suppression is not more
benefit. The effect is specific to a range of depths, which is not what
input-deletion-by-degrees would produce.

---

### 4.7 What this shows and what it does not

The intervention is unconditional. We reduce attention to the claim whether or
not the claim is true; the model has not learned to judge the conflict. It works
because reducing a signal only changes an outcome when something else is
competing, and we measure that cost directly: zero, on the items where the claim
is the only evidence. We have not tested the harder case of genuinely ambiguous
audio paired with a reliable text claim, and we do not claim to have fixed
multimodal conflict resolution.

The effect appears in open generation and not in multiple-choice answering, where
it closes 0% and 1% of the decision gap. Since these models are typically used by
asking them to describe what they hear, we take generation to be the setting of
interest, but the limit should be read literally rather than generalised.

Three further limits. All mechanistic results are on a single model, so the
mechanism is demonstrated rather than shown to be universal. The method assumes
the claim's token span is known, which is true by construction in our data but
would need solving in deployment. And a partial block, reducing attention to a
quarter rather than removing it, had no effect at any depth — so the intervention
behaves as a switch rather than a dial, and we do not describe it as graded.

>> NOTE: report the partial-block null explicitly. It is a clean negative, and
>> stating it stops us claiming a tunable method the data does not support.

---

### Methods provenance

No method in this section is new. Probing follows Pasad et al., attribution
patching follows AtP\*, attention knockout follows Geva et al., and circuit
localisation and back-patching follow Cho et al., whose repair result we
reproduce here as a negative in a new setting. What is ours is the set of
controls: verifying the knockout is connected before trusting a null result,
scoring generated text in a claim-free context, measuring which label a free-form
description names, and the matched and text-only controls in §4.6.

>> NOTE: the full mapping, including every deviation from the source method, is
>> in `notes/METHODS_PROVENANCE.md`. Either cite it or fold two sentences into
>> the methods section.
