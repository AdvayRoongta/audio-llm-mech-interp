# Audio-Text Dominance Results (2026-08-14 full runs)

Full 518-item runs, fp16/bf16, no quantization, all three models on Modal (A100-40GB).
See `qwen2audio_summary.json`, `desta25_summary.json`, `salmonn_summary.json` in this folder
for the complete per-category metrics (weighted accuracy, UAR, macro/micro F1, by_condition
and TDR breakdowns).

Post-fix results only (sarcasm_pacing prompt-rendering bug fixed in `526ab8f` / `526aa8d`,
SALMONN RAVDESS 48kHz->16kHz resample bug fixed in `05d91fb`). Anything timestamped before
2026-08-14 predates these fixes and should not be used.

## Weighted Accuracy / UAR (all items)

| Category | Qwen2-Audio-7B-Instruct | DeSTA2.5-Audio-Llama-3.1-8B | SALMONN-13B |
|---|---|---|---|
| emotional_prosody | WA 0.610 / UAR 0.607 | WA 0.686 / UAR 0.689 | WA 0.478 / UAR 0.474 |
| speaker_identity | WA 0.788 / UAR 0.788 | WA 0.730 / UAR 0.730 | WA 0.575 / UAR 0.575 |
| sarcasm_pacing | WA 0.335 / UAR 0.368 | WA 0.217 / UAR 0.231 | WA 0.317 / UAR 0.363 |

## Text Dominance Ratio (TDR), Mismatched-condition items only

follow_text / (follow_text + follow_audio + follow_neither), invalid/unresolved excluded from denominator.

| Category | Qwen2-Audio-7B-Instruct | DeSTA2.5-Audio-Llama-3.1-8B | SALMONN-13B |
|---|---|---|---|
| emotional_prosody | 1.000 | 0.667 | 0.850 |
| speaker_identity | 0.883 | 1.000 | 0.533 |
| sarcasm_pacing | 0.596 | 0.217 | 0.417 |

## n_scored / n_total (invalid responses excluded from scoring)

| Category | Qwen2-Audio-7B-Instruct | DeSTA2.5-Audio-Llama-3.1-8B | SALMONN-13B |
|---|---|---|---|
| emotional_prosody | 177/178 | 175/178 | 178/178 |
| speaker_identity | 160/160 | 152/160 | 160/160 |
| sarcasm_pacing | 173/180 | 180/180 | 180/180 |
