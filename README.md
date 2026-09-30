# Text Dominance in Paralinguistic Conflict: Tracing Suppression Through Audio LLM Circuits

Official code and data for the paper.

**[Dataset](data/)** · **[Experiments](eval/)** · **[Results](results/)**

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
| `notes/` | Methodology and provenance write-ups |
| `third_party/` | Vendored upstream MUStARD++ code, unmodified |
| `archive/` | Superseded files, kept rather than deleted |
| `dataset_audio/` | Audio corpus — not committed; fetched with `scripts/fetch_audio.py` |

---

## The dataset

`data/combined_examples.json`, 3,594 items across three conflict categories.

| Category | Source corpus | Cue | Options | Matched | Mismatched | Neutral-Text | Paralinguistic |
|---|---|---|---|---|---|---|---|
| `emotional_prosody` | RAVDESS | Emotion in the voice | 8 | 399 | 399 | 266 | 133 |
| `speaker_identity` | RAVDESS | Speaker gender | 2 | 450 | 450 | 150 | 150 |
| `sarcasm_pacing` | MUStARD++ | Sarcastic delivery | 10 | 399 | 399 | 266 | 133 |

Each item carries `condition` (`Matched`, `Mismatched`, `Neutral-Text`, `Paralinguistic`),
`modality` (`text_only`, `audio_only`, `text_audio`), `order` (`claim_first` /
`claim_last`, counterbalanced on the Mismatched items), the rendered `prompt` and
`options`, the `correct_letter`, and the acoustic and claimed labels.

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

## Running the experiments

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

They share `eval/_common.py`, which imports the dataset, prompt and option conventions
from the behavioural runner so the two cannot drift apart. Two backends are selected with `MECH_BACKEND` (`qwen`,
`salmonn`). Every script records dtype, quantisation, attention implementation and seed
in its output.

### Figures

```bash
python3 viz/make_figures.py              # fig1-fig7, the paper figures
python3 viz/make_mechinterp_figures.py   # m1-m16, the mechanistic appendix
```

---

<div align="center">

**Accepted to *Interpretability as a Science*, NeurIPS 2026.**

</div>
