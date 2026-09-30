# Text Dominance in Paralinguistic Conflict: Tracing Suppression Through Audio LLM Circuits

Official code, data and analysis for the paper.

**[Dataset](data/)** · **[Experiments](eval/)** · **[Results](results/)** · **[Running on an A100](modal/MODAL.md)**

---

## Abstract

Audio LLMs deployed in settings such as crisis hotlines and customer support are only
useful beyond speech-to-text pipelines if they use acoustic cues rather than the
transcript alone. We test whether they prioritise transcript content over paralinguistic
cues (emotion, speaker identity, sarcasm) and trace the mechanism when they do. Across
six audio LLMs under LISTEN-style conflict conditions (Chen et al., 2026), the mean
text-dominance rate on mismatched text+audio items is 0.78 (0.86 excluding sarcasm). In
the most text-dominant screened model, probing shows speaker identity remains linearly
decodable at ceiling at the language model's input, so the failure is not perceptual.
Back-patching, reported to repair lexical text dominance, has no measurable effect here;
but it also fails on its own source task in our replication, so we treat its null as
uninformative rather than as evidence of non-transfer. Attention knockout partially
restores audio-grounded output by blocking the claim from propagating through
intermediate tokens during open-ended generation, whereas blocking the answer position's
direct connection to the conflicting claim has no effect. The claim's influence therefore
travels through an intermediate relay rather than a single severable edge. The effect
appears only in free-form generation and not in closed-set multiple choice, suggesting
that benchmarks relying on that format may be systematically blind to interventions of
this kind.

---

## What is here

Three things, in the order the paper uses them:

1. **A paralinguistic conflict dataset** (3,594 items). Every item pairs audio carrying a
   non-verbal cue — emotional prosody, speaker gender, sarcastic delivery — with a text
   claim that either agrees with it, says nothing about it, or contradicts it. Built from
   RAVDESS and MUStARD++ under the LISTEN condition × modality design.
2. **A behavioural evaluation** of six audio LLMs on that dataset, reporting the
   text-dominance rate (TDR): the fraction of contradicted items answered with the false
   text claim rather than with what the voice actually contains.
3. **Six mechanistic experiments** on the most text-dominant screened model
   (Qwen2-Audio-7B-Instruct), asking in order whether the audio information arrives,
   where the decision is made, whether the published repair transfers, and — since it
   does not — what does.

## Key findings

**The information arrives.** A linear probe recovers speaker gender at 1.000 accuracy
(chance 0.5) from audio-encoder layer 13 onward, and still at 1.000 after the projector,
i.e. in exactly the representation the language model reads. Splits are actor-disjoint,
and a shuffled-label control lands at 0.43. The failure is not perceptual.

**The model can do the task; it declines to when text is present.** Deleting the single
sentence of false claim, holding model, audio and question fixed, raises audio-grounded
accuracy from 0.000 to 0.533 on emotional prosody (chance 0.125) and from 0.067 to 0.717
on speaker identity (chance 0.5).

**The competition is lopsided and late.** Attribution patching, verified by a faithfulness
check at 1.00 on the top 20% of components, puts text at 3.29 against audio's 0.18 at the
answer position — roughly 18:1 — peaking around layer 29.

**Back-patching does not move it, and neither does anything on the audio side.** Cho et
al.'s late→early back-patch leaves audio accuracy at 0.000 across every source layer,
destination layer and window width. Against a baseline decision margin of −7.32 nats, the
best source layer buys +0.51 and amplification saturates at +0.10; at the highest gain the
model breaks outright. We also fail to reproduce the repair on its own lexical source
task, so we read this null as uninformative rather than as evidence of non-transfer.

**The claim travels by a detour.** Blocking the answer position from attending to the
claim tokens, at all 32 layers, closes none of the gap. Blocking every position after the
claim closes all of it; blocking every position gives an identical result. The claim is
absorbed early into the question and option tokens and re-read from them. This is the
central mechanistic finding: there is no single severable edge.

**The effect is invisible to multiple choice.** Under mid-layer attenuation the model
names the false label in 0.375 of free-form prosody descriptions (95% CI 0.21–0.57) and
0.433 of identity descriptions (0.27–0.61), against 1.000 at baseline in both; it names
the true label — never named at baseline — in 0.500 and 0.567. The same intervention
closes none of the closed-set gap for prosody and 1% for identity. Generation quality
improves rather than degrades: mean per-token log-probability rises from −1.28 to −0.64
(prosody) and −1.45 to −0.78 (identity), scored by the unmodified model in a context with
the claim removed.

**Four controls.** Matched items improve (0.767→0.900 prosody, 0.933→1.000 identity), so
the model is not damaged. Text-only items are unchanged (0.567 and 1.000 before and
after), so the intervention is not "ignore all text" — it reduces the claim's influence,
which only changes an outcome when something competes with it. The identical block applied
to late layers does nothing, so the effect is not generic attention damage. Blocking at
every layer suppresses more but costs capability (0.667 on matched prosody, below the
0.767 baseline), so it is not equivalent to deleting the claim from the input.

---

## Results

### Text-dominance rate, Mismatched + text+audio items

TDR = answers matching the false text claim / usable items. Chance is not 0.5 here; it is
the probability of naming the claimed label by accident (1/8 for prosody, 1/2 for
identity, 1/10 for sarcasm). Wilson intervals are drawn in `fig2_tdr_by_model`.

| Model | Emotional prosody | Speaker identity | Sarcasm / pacing |
|---|---|---|---|
| Qwen2-Audio-7B-Instruct | **0.989** (263/266) | 0.930 (279/300) | — |
| Nemotron-Labs-Audex-30B-A3B | 0.966 (257/266) | 0.940 (282/300) | 0.436 (116/266) |
| Phi-4-multimodal-instruct | 0.921 (245/266) | 0.940 (282/300) | 0.861 (229/266) |
| SALMONN-13B | 0.786 (209/266) | 0.820 (246/300) | 0.643 (90/140) |
| Baichuan-Omni-1.5 | 0.633 (114/180) | 0.806 (150/186) | 0.285 (73/256) |
| DeSTA2.5-Audio-Llama-3.1-8B | 0.475 (123/259) | **0.983** (295/300) | — |

Mean across models and categories: **0.78**. Every model in the table follows the text on
the majority of contradicted prosody and identity items. Per-category weighted accuracy,
UAR, macro/micro F1 and the claim-first/claim-last order breakdown are in
[`results/*_summary.json`](results/).

### Mechanistic experiments (Qwen2-Audio-7B-Instruct)

| Experiment | Question | Result |
|---|---|---|
| `exp1_capability_screen` | Can the model read the cue with no text claim? | 0.000→0.533 prosody, 0.067→0.717 identity when the claim is removed |
| `exp2_encoder_probe` | Does the signal survive Whisper and the projector? | Gender at 1.000 from encoder layer 13 and after the projector; shuffled-label 0.43 |
| `exp3_circuit_localization` | Where is the decision made? | Answer position, text 3.29 vs audio 0.18; faithfulness 1.00 at 20% sparsity |
| `exp4_backpatching` | Does Cho et al.'s repair transfer? | 0.000→0.000 everywhere; harmful on identity |
| `exp5_patch_diagnosis` | Why not — depth, floor, or norm mismatch? | Floor. Best relocation +0.51 nats, amplification +0.10, against −7.32 |
| `exp6_readout_attenuation` | Intervene on the text side instead | Routing bisection + the generative effect above |

---

## Repository layout

| Path | Contents |
|---|---|
| `build/` | Dataset generation (`build_*.py`, `merge_datasets.py`) |
| `data/` | The generated datasets (`*_examples.json`, `combined_examples.json`) |
| `raw/` | Source CSVs consumed by `build/`. See [`raw/README.md`](raw/README.md) |
| `eval/` | Behavioural inference (`run_*_eval.py`) and the mechanistic experiments (`exp1`–`exp6`) |
| `modal/` | A100 wrappers for every script. See [`modal/MODAL.md`](modal/MODAL.md) |
| `viz/` | Figure generation (`make_figures.py`, `make_mechinterp_figures.py`) |
| `results/` | Run outputs: per-item JSON, summaries, logs |
| `figures/` | Paper figures, PDF and PNG — not committed; rebuild with `viz/` |
| `notes/` | Methodology, provenance and findings write-ups |
| `third_party/` | Vendored upstream MUStARD++ code, unmodified |
| `archive/` | Superseded files, kept rather than deleted |
| `dataset_audio/` | Audio corpus (gitignored; fetched with `scripts/fetch_audio.py`) |

---

## The dataset

3,594 items, three conflict categories × four conditions, following the LISTEN design
(Chen et al., 2026) with one addition: because our items carry both the acoustic label and
the claimed label, a follow-rate can be computed on them, which LISTEN's design does not
support.

| Category | Source | Cue | Options | Matched | Mismatched | Neutral-Text | Paralinguistic |
|---|---|---|---|---|---|---|---|
| `emotional_prosody` | RAVDESS | Emotion in the voice | 8 | 399 | 399 | 266 | 133 |
| `speaker_identity` | RAVDESS | Speaker gender | 2 | 450 | 450 | 150 | 150 |
| `sarcasm_pacing` | MUStARD++ | Sarcastic delivery | 10 | 399 | 399 | 266 | 133 |

**Conditions.** *Matched*: the claim agrees with the voice. *Mismatched*: the claim
contradicts it — this is where TDR is measured. *Neutral-Text*: the text is present but
says nothing about the cue. *Paralinguistic*: audio only, no claim. Each item is also
rendered in three modalities (`text_only`, `audio_only`, `text_audio`), and the
Mismatched items are order-counterbalanced into `claim_first` and `claim_last` so recency
can be separated from dominance.

### Regenerating

From the repo root; every build is seeded and deterministic.

```bash
python build/build_emotional_prosody.py  --n-per-condition 133 --seed 0
python build/build_speaker_identity.py   --n-per-condition 150 --seed 0
python build/build_sarcasm_pacing_v2.py  --input-csv raw/mustard_pp_utterances_only.csv \
                                         --n-per-condition 133 --seed 0
python build/merge_datasets.py           # -> data/combined_examples.json
```

The two RAVDESS builders need no input file: the RAVDESS naming scheme is enumerated
synthetically. The sarcasm builder needs `raw/mustard_pp_utterances_only.csv`; see
[`raw/README.md`](raw/README.md) for how that is derived from `raw/mustard++_text.csv`.

Audio is not committed. Fetch it with `scripts/fetch_audio.py`, or let the Modal wrappers
fetch it for you.

---

## Reproducing the results

Everything below needs a GPU and the audio corpus. [`modal/MODAL.md`](modal/MODAL.md)
documents running it on a rented A100, which is how every number in the paper was
produced; the corpus and the model weights are fetched for you.

### Behavioural evaluation

```bash
python eval/run_qwen2audio_eval.py --input data/combined_examples.json --output-dir results/
python eval/run_desta25_eval.py
python eval/run_salmonn_eval.py
python eval/run_phi4mm_eval.py
python eval/run_baichuan_eval.py
python eval/run_audex_eval.py
```

Each writes per-item JSON plus a `*_summary.json` carrying accuracy, UAR, macro/micro F1,
and the TDR broken down by modality and by claim order. Runs checkpoint and resume;
`--limit 8` is a smoke test and `--dry-run` audits the dataset without loading the model.

### Mechanistic experiments

```bash
python eval/exp1_capability_screen.py
python eval/exp2_encoder_probe.py        --category speaker_identity
python eval/exp3_circuit_localization.py --category emotional_prosody
python eval/exp4_backpatching.py         --category emotional_prosody
python eval/exp5_patch_diagnosis.py      --category emotional_prosody
python eval/exp6_readout_attenuation.py  --category emotional_prosody
```

On Modal:

```bash
python3 -m modal run modal/modal_app.py::experiment --name exp6_readout_attenuation \
        --args "--category emotional_prosody"
```

`exp1` gates the rest, and the gate is load-bearing rather than decorative. If a model's
audio-only accuracy on a category is at chance, then following the text under conflict
does not demonstrate suppression — the signal never arrived — and `exp2`–`exp6` would be
localising something that is not there. This is the criterion that excludes SALMONN from
the mechanistic analysis (see below).

The experiments score by option log-probability rather than by generating a letter and
parsing it: option letters are single tokens, so one forward pass scores every option,
nothing can fail to parse, and the score is graded rather than a hard label — which is
what the patching experiments need. They share `eval/_common.py`, which imports the
dataset, prompt and option conventions from the behavioural runner so the two cannot
drift apart. Two backends are selected with `MECH_BACKEND` (`qwen`, `salmonn`).

Every script records dtype, quantisation, attention implementation and seed in its output.
4-bit perturbs activations; it is acceptable for locating components and not for a
headline Δ, and the recorded fields are there so the two are never compared by accident.

### Figures

```bash
python3 viz/make_figures.py              # fig1-fig7, the paper figures
python3 viz/make_mechinterp_figures.py   # m1-m16, the mechanistic appendix
```

Both read whatever is in `results/` and skip any figure whose inputs are missing, so they
are safe to run mid-sweep. Vector PDF for the paper, PNG for slides.

---

## Scope and negative results

Reported here because they bound what the paper claims.

**SALMONN-13B is excluded from the mechanistic analysis**, by the same capability
criterion applied everywhere else and fixed before the outcome was known. It answers
prosody from audio at 0.188 against 0.125 chance — thin enough to rest on the sample — and
at 0.133 on the intervention subset; on identity it is below chance, 0.453 against 0.500.
The caveat belongs with the exclusion: the screen measures answering in our prompt format,
not perception. SALMONN returns an off-option response on 0.203 of items where Qwen never
does, and its encoder demonstrably carries the signal (0.671 after the projector against
0.143 chance). We report the attempt rather than omitting it.

**Sarcasm / pacing is run but not reported as a result.** A substantial share of
MUStARD++ clips are keyed to episode ids the public release does not distribute, leaving
463 of 798 audio-bearing items — enough for the analysis, but the category cannot supply a
clean answer. Removing the claim raises audio accuracy only to 0.233 against 0.100 chance;
on matched items the claim-free accuracy is 0.167. Several of the ten options — ridicule
and frustration among them — are social judgments about what was said and to whom, not
properties recoverable from the voice. We treat the category as underpowered, not as a
negative result. The missing clips are also not a random subset: they fall most heavily on
the Mismatched condition that carries the phenomenon (140 of 266 retained, against 168 of
266 for Matched).

**Other limits.** The intervention is unconditional — it reduces attention to the claim
whether or not the claim is true — and we measure that cost directly rather than assuming
it away. We have not tested ambiguous audio paired with a reliable claim. The positive
result comes from one model. The method assumes the claim's token span is known, which
holds by construction here and would not in deployment. A partial block (γ=0.25 rather
than 0) has no effect at any depth, so the intervention behaves as a switch, not a dial,
and we do not describe it as graded.

---

## Prior work this builds on

No method here is new; the contribution is the setting, the dataset and the controls.
Probing follows Pasad et al.; attribution patching follows AtP*; attention knockout
follows Geva et al.; circuit localisation and back-patching follow Cho et al., whose
repair result we reproduce here as a negative in a new setting. The condition × modality
design and the prediction-marginal chance baseline follow LISTEN (Chen et al., 2026).

What is ours is the set of controls: verifying that the knockout is connected before
trusting a null result, scoring generated text in a context free of the claim, measuring
which label a free-form description actually names, and the matched and text-only
controls above.

---

## Citation

```bibtex
@inproceedings{paralinguistic-text-dominance-2026,
  title     = {Text Dominance in Paralinguistic Conflict: Tracing Suppression
               Through Audio {LLM} Circuits},
  booktitle = {NeurIPS 2026 Workshop on Interpretability as a Science},
  year      = {2026},
}
```

## Data and licensing

No audio is redistributed here. The corpora remain under their own terms: RAVDESS is
CC BY-NC-SA 4.0, and the upstream MUStARD++ clone ships no licence file at all — see
[`third_party/README.md`](third_party/README.md) and check the upstream repository
directly before redistributing anything derived from it. This repository carries no
licence file yet.

---

<div align="center">

**Accepted to *Interpretability as a Science*, NeurIPS 2026.**

</div>
