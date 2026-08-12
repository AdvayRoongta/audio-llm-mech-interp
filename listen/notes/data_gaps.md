# Data gaps

## RAVDESS download short 10 files (2026-08-12)

The RAVDESS download used to build `emotional_prosody_examples.json` and
`speaker_identity_examples.json` is missing 10 files out of the full 1440-file
corpus. Of those 10, only 2 were actually referenced by generated dataset items.

Missing file: `03-01-03-01-01-01-09.wav` (Actor 9, happy, statement 1, repetition 1)

Affected items (removed from `emotional_prosody_examples.json`):
- `ep_neutral_text_text_only_016`
- `ep_neutral_text_text_audio_016`

**Fix applied:** both items removed. `emotional_prosody_examples.json` went from
140 items to 138. `data/combined_examples.json` regenerated afterward, confirmed
at 398 total items (138 emotional_prosody + 120 speaker_identity + 140
sarcasm_pacing), no other items affected.

The remaining 8 missing RAVDESS files were not referenced by any generated item,
so no further action was needed.

## MUStARD++ audio coverage (2026-08-12)

Initial MUStARD++ audio conversion had a 34-file gap against
`sarcasm_pacing_examples_v2.json`. Fully resolved — re-converted from the
correct full MUStARD++ source (the original download had been the base
690-clip MUStARD corpus, not the extended 1205-clip MUStARD++ set). All 1205
clips converted to uniform 16kHz mono WAV. Final coverage check: 80/80
referenced audio files present, 0 missing, 0 zero-byte. No items dropped.
