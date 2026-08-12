# archive/

Superseded or unidentified files, kept rather than deleted.

## files.zip

Found at repo root during the 2026-08-12 reorg. Contents:

```
build_speaker_identity.py           22502 bytes
speaker_identity_examples.json     101648 bytes
```

Diffed byte-for-byte identical to the versions that were in the root-level `files/`
directory at the time (now `build/build_speaker_identity.py` and
`data/speaker_identity_examples.json`). Appears to be a redundant zipped backup copy,
not a distinct or older version. Archived here rather than deleted since its origin
(who made it, why) is unconfirmed.

## combined_examples_pre_order_counterbalancing.json

Backup of `data/combined_examples.json` (398 items) taken on 2026-08-12, immediately
before regenerating it to add prompt-order counterbalancing (`order`: `claim_first` /
`claim_last`) to the Matched/Mismatched text_audio cells across all three categories.
Kept so the pre-counterbalancing dataset can be diffed against or restored if needed.
See `notes/data_gaps.md` for the count breakdown of that change.

## build_sarcasm_pacing_dataset.py / sarcasm_pacing_examples.json

v1 of the sarcasm-pacing eval set, superseded by `build/build_sarcasm_pacing_v2.py` /
`data/sarcasm_pacing_examples_v2.json`. The v1 design's Mismatched condition claimed
"the speaker is being sarcastic/sincere" — a meta-label, not a genuine content claim the
audio could contradict — so TDR collapsed into plain accuracy. v2 fixes this by reusing
MUStARD++'s own `Explicit_Emotion` (what the words/transcript convey, the text claim) vs
`Implicit_Emotion` (what the delivery actually conveys, the audio truth) columns, giving
Mismatched a real false-claim-vs-audio-truth conflict.
