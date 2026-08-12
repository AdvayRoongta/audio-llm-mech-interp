# Data gaps

## 2026-08-12 — RAVDESS download short 10 files

The RAVDESS speech corpus pulled into `dataset_audio/ravdess/` has 1430 `.wav`
files, not the full 1440 (24 actors x 60 files). Ten files are missing, all
either the "neutral" or one "happy" take for actors 1, 2, 3, 4, 5, 7, 9 (see
prior audit conversation for the full list of 10 missing filenames).

Of those 10, only one — `03-01-03-01-01-01-09.wav` — was actually referenced
by a generated dataset. It was used by two items in
`data/emotional_prosody_examples.json`:

- `ep_neutral_text_text_only_016`
- `ep_neutral_text_text_audio_016`

Both were removed. `emotional_prosody_examples.json` now has **138 items**
(was 140) — the `Neutral-Text` condition's `text_only` and `text_audio` cells
each dropped from 20 to 19; every other condition/modality cell is unaffected.

`speaker_identity_examples.json` and `sarcasm_pacing_examples_v2.json`
reference none of the 10 missing files and needed no changes.

`data/combined_examples.json` was regenerated via `build/merge_datasets.py`
to reflect the corrected 138-item file (new total: 398, was 400).

## Prompt-order counterbalancing added — 2026-08-12
Added `order` field (claim_first / claim_last) to all Matched and Mismatched condition items across all 3 categories, to control for recency bias (text claim was always positioned immediately before the question, confounding modality dominance with token-position dominance). Neutral-Text and Paralinguistic conditions unaffected (no competing claim to reorder). Item count: 398 → 518. Original dataset backed up to archive/ before regeneration.
