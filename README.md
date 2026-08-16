# MUStARD++ pacing/prosody eval builds

Dataset-generation scripts for LISTEN-style condition x modality evaluation sets
("Do Audio LLMs Really LISTEN, or Just Transcribe?", EACL 2026) — sarcasm/emotion
conflict pacing, emotional prosody, and speaker-identity eval sets, built from
MUStARD++ labels and the RAVDESS filename convention.

## Layout

| Dir | Contents |
|---|---|
| `build/` | Dataset generation scripts (`build_*.py`) |
| `data/` | Generated JSON datasets (`*_examples.json`), output of the `build/` scripts |
| `raw/` | Source/derived dataset CSVs consumed by `build/` scripts. See `raw/README.md`. |
| `eval/` | Inference + scoring (`run_*_eval.py`) and the mechanistic experiments (`exp*.py`) |
| `results/` | xlsx, output json, run logs — none yet |
| `notes/` | Markdown findings notes — none yet |
| `dataset_audio/` | Audio corpus (gitignored, not committed). Not present yet. |
| `third_party/` | Vendored upstream code, unmodified. See `third_party/README.md`. |
| `archive/` | Superseded or unidentified files, kept rather than deleted. See `archive/README.md`. |

`results/` and `notes/` currently hold only a `.gitkeep` placeholder.

## Regenerating the datasets

Run from the repo root:

```bash
python build/build_sarcasm_pacing_dataset.py \
    --input-csv raw/mustard_pp_utterances_only.csv \
    --n-per-condition 20 --seed 42 --out data/sarcasm_pacing_examples.json

python build/build_emotional_prosody.py \
    --n-per-condition 20 --seed 0 --out data/emotional_prosody_examples.json

python build/build_speaker_identity.py \
    --n-per-condition 20 --seed 0 --out data/speaker_identity_examples.json
```

`build_emotional_prosody.py` and `build_speaker_identity.py` need no input files — the
RAVDESS naming scheme is enumerated synthetically. `build_sarcasm_pacing_dataset.py`
needs `raw/mustard_pp_utterances_only.csv`; see `raw/README.md` for how that file itself
is derived from `raw/mustard++_text.csv`.

## Data sources

See `raw/README.md` for dataset provenance and how the raw CSVs relate to each other,
and `third_party/README.md` for the vendored upstream MUStARD++ code this project
builds on.

## Mechanistic experiments

`run_*_eval.py` measures *what* the models do under conflict. `eval/exp*.py`
ask *why*, and share `eval/_common.py`, which imports the dataset, prompt and
option conventions from `run_qwen2audio_eval.py` so the two cannot drift apart.
They score by option log-prob rather than generating and parsing a letter:
option letters are single tokens, so one forward scores every option and
nothing can fail to parse.

| Script | Question | Needs |
|---|---|---|
| `exp1_capability_screen.py` | Can the model read the cue at all with no text claim? Also splits TDR by `claim_first`/`claim_last`. | GPU |
| `exp2_encoder_probe.py` | Does the signal survive Whisper, or is it gone before the projector? | GPU + scikit-learn |
| `exp3_circuit_localization.py` | Which (layer, submodule, position) carry the claim vs the audio, and do those circuits overlap? | GPU, eager attention |
| `exp4_backpatching.py` | Does Cho et al.'s late→early back-patch fix transfer to paralinguistic conflicts? | GPU |

These need a GPU and the audio corpus. See **[MODAL.md](MODAL.md)** for
running them on Modal, which fetches the audio and the model for you.

Run them from the repo root, after putting the audio corpus in `dataset_audio/`:

```bash
python eval/exp1_capability_screen.py
python eval/exp2_encoder_probe.py --category speaker_identity
python eval/exp3_circuit_localization.py --category emotional_prosody
python eval/exp4_backpatching.py --category emotional_prosody
```

**exp1 gates the rest.** If a category's `Paralinguistic`/`audio_only` accuracy
is at chance, then a failure under conflict does not demonstrate suppression —
the signal never arrived — and exp2-4 would be localizing something that isn't
there. Each script writes JSON to `results/` recording dtype, quantization and
seed, because 4-bit perturbs activations and any headline number should be
re-checked in fp16.
