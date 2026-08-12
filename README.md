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
| `eval/` | Inference + scoring scripts (`test_*.py`) — none yet |
| `results/` | xlsx, output json, run logs — none yet |
| `notes/` | Markdown findings notes — none yet |
| `dataset_audio/` | Audio corpus (gitignored, not committed). Not present yet. |
| `third_party/` | Vendored upstream code, unmodified. See `third_party/README.md`. |
| `archive/` | Superseded or unidentified files, kept rather than deleted. See `archive/README.md`. |

`eval/`, `results/`, `notes/` currently hold only a `.gitkeep` placeholder.

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
