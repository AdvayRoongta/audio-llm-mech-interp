# audio-initial-dataset

Testing **text-dominance vs. paralinguistic signals** in Audio LLMs: does the
model believe the words, or the way they're said?

Each example pairs an audio clip with a text prompt that **contradicts** it
(e.g. prompt says "happy", audio is spoken in anger), then checks whether
Qwen2-Audio follows the text or the audio.

## Quickstart

[`scripts/load_datasets.ipynb`](scripts/load_datasets.ipynb) has an "Open in
Colab" badge at the top (or open it in local Jupyter). It's self-contained:
opened fresh from the badge, the first cell clones this repo and installs
`requirements.txt` automatically, then pulls the 4 datasets below.

## The four categories

One public Hugging Face dataset backs each category. `load_datasets.ipynb`
pulls the first 100 rows of each into `data/raw/<category>/` (audio + a
`manifest.json` to browse) — git-ignored, not the final dataset size.

| category | person | source | listen for |
|---|---|---|---|
| `emotional_prosody` | Advay | [CREMA-D](https://huggingface.co/datasets/confit/cremad-parquet) | same actor, 6 acted emotions |
| `non_speech_event` | Ankur | [ESC-50](https://huggingface.co/datasets/renumics/esc50) | sirens, dogs, crying, glass breaking, ... |
| `speaker_identity` | Danity | [RAVDESS](https://huggingface.co/datasets/xbgoose/ravdess) | known actor + gender per clip |
| `sarcasm_pacing` | Vishnu | [MUStARD](https://huggingface.co/datasets/macabdul9/SarcasmDetection_Mustard) | TV lines labeled sarcastic/sincere |

Need to trim or splice a clip? Use [`pydub`](https://github.com/jiaaro/pydub)
(needs `ffmpeg`) or any free online cutter, and save the result next to the
originals in `data/raw/<category>/audio/`.

## Your 20 examples

Pick ~20 clips from your category's manifest and append one entry per clip to
[`examples.json`](examples.json) — 4 worked examples (one per category) are
already in there as a model to copy:

```json
{
  "id": "prosody_002",
  "audio_path": "data/raw/emotional_prosody/audio/emotional_prosody_000.wav",
  "conflict_type": "emotional_prosody",
  "text_prompt": "The speaker sounds calm and happy. What is the speaker's emotional state?",
  "ground_truth_audio": "anger",
  "ground_truth_text": "happy"
}
```

| field | meaning |
|---|---|
| `id` | short descriptor + number, unique in the file (e.g. `sarcasm_014`) |
| `audio_path` | path to the clip, relative to repo root |
| `conflict_type` | one of the four category names above |
| `text_prompt` | asserts something that contradicts the audio, then asks a direct question |
| `ground_truth_audio` | what's actually true, based on the audio |
| `ground_truth_text` | what the (false) text prompt claims |
