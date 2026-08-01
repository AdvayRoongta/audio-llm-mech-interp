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
`requirements.txt` automatically, then pulls the 3 datasets below.

**On Colab the clips are cached in your own Google Drive**, under
`MyDrive/audio-initial-dataset/`. Colab wipes its local disk when the runtime
recycles, so without this you'd re-download every session — instead, later
runs find the cached copy and skip the download entirely. Each person has
their own copy in their own Drive; nothing is shared between accounts, and
nothing needs to be, since the loader is deterministic (same datasets, same
split, first 100 rows) and everyone ends up with identical clips at identical
paths. The one exception is hand-trimmed clips — those aren't reproducible,
so commit them to git if you make any.

Because the data outlives the session, a separate notebook (e.g. the
mechanistic-interpretability one) never has to re-run this download. It mounts
Drive, points `DATA_ROOT` at the same folder, and resolves clips with
`DATA_ROOT / example["audio_path"]`.

## The categories

One public Hugging Face dataset backs each category. `load_datasets.ipynb`
pulls the first 100 rows of each into `data/raw/<category>/` (audio + a
`manifest.json` to browse) — git-ignored, not the final dataset size.

| category | person | source | listen for |
|---|---|---|---|
| `emotional_prosody` | Advay | [CREMA-D](https://huggingface.co/datasets/confit/cremad-parquet) | same actor, 6 acted emotions |
| `speaker_identity` | Danity | [RAVDESS](https://huggingface.co/datasets/xbgoose/ravdess) | known actor + gender per clip |
| `sarcasm_pacing` | Vishnu | [MUStARD](https://huggingface.co/datasets/macabdul9/SarcasmDetection_Mustard) | TV lines labeled sarcastic/sincere |

<!-- non_speech_event is on hold for the paper for now (ESC-50 clips weren't a good fit) -- re-enable by uncommenting this row and the matching CATEGORIES entry in scripts/load_datasets.ipynb.
| `non_speech_event` | Ankur | [ESC-50](https://huggingface.co/datasets/renumics/esc50) | sirens, dogs, crying, glass breaking, ... |
-->


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
| `conflict_type` | one of the category names above |
| `text_prompt` | asserts something that contradicts the audio, then asks a direct question |
| `ground_truth_audio` | what's actually true, based on the audio |
| `ground_truth_text` | what the (false) text prompt claims |
