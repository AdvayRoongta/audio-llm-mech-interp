# listen/

This is the current, active pipeline for the audio-text dominance study
(LISTEN-structured benchmark). It supersedes the repo root's old `examples.json`
and `scripts/` notebooks, which were built before the project was restructured
around LISTEN's four-condition x three-modality design.

## What's here

```
listen/
  build/
    build_emotional_prosody.py     # RAVDESS -> emotional prosody conflict items
    build_speaker_identity.py      # RAVDESS -> speaker gender conflict items
    build_sarcasm_pacing_v2.py     # MUStARD++ -> sarcasm/sincerity conflict items
    merge_datasets.py              # merges the three category files into one
  data/
    combined_examples.json         # the merged dataset, 398 items, ready for inference
  eval/
    run_qwen2audio_eval.py         # Qwen2-Audio inference + scoring (fp16 only)
```

Audio files themselves are **not** in this repo (too large for git). They live
in `dataset_audio/mustard/` and `dataset_audio/ravdess/`, distributed via
Google Drive — ask Vishnu for the current link.

## Categories

- **emotional_prosody** — RAVDESS, 8-way emotion MCQ
- **speaker_identity** — RAVDESS, 2-way gender MCQ
- **sarcasm_pacing** — MUStARD++, 10-way emotion MCQ

Each category uses the same four-condition structure (Neutral-Text, Matched,
Mismatched, Paralinguistic) across three modalities (text_only, audio_only,
text_audio). See `notes/data_gaps.md` for known gaps in source audio coverage.

## Running the pipeline

1. Build each category (only needed if you're regenerating from scratch):
   ```
   python build/build_emotional_prosody.py
   python build/build_speaker_identity.py
   python build/build_sarcasm_pacing_v2.py
   ```
2. Merge into the combined dataset:
   ```
   python build/merge_datasets.py --prosody data/emotional_prosody_examples.json \
       --speaker-identity data/speaker_identity_examples.json \
       --sarcasm data/sarcasm_pacing_examples_v2.json \
       --out data/combined_examples.json
   ```
3. Validate before running inference (no GPU needed, no model downloaded):
   ```
   python eval/run_qwen2audio_eval.py --dry-run
   ```
4. Smoke test on a small subset:
   ```
   python eval/run_qwen2audio_eval.py --experiments emotional_prosody --limit 5
   ```
5. Full run:
   ```
   python eval/run_qwen2audio_eval.py --experiments all
   ```

All inference runs at fp16 only, no quantization. Results write to `results/`
with per-item checkpoint files (`.jsonl`) so an interrupted run can resume
without losing progress — rerun the same command and it picks up where it
left off.

## Known gotchas

- Dataset condition strings are `"Neutral-Text"`, `"Matched"`, `"Mismatched"`,
  `"Paralinguistic"` (capitalized) — not lowercase snake_case. The eval script
  handles this, but if you're writing new tooling against this dataset, match
  the exact casing.
- The MCQ option count differs by category (2-way / 8-way / 10-way) — never
  assume a fixed number of choices.
