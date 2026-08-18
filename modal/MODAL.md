# Running the experiments on Modal

Qwen2-Audio-7B is 16.8 GB and the experiments need CUDA, so they can't run on a
laptop. `modal_app.py` packages the repo, fetches the audio corpus, and runs
`eval/*.py` on a rented A100.

## One-time setup

```bash
pip3 install modal
python3 -m modal setup          # opens a browser to authenticate
```

Use `python3 -m modal` rather than a bare `modal` if the CLI isn't on your PATH
(common with Homebrew/pyenv).

Then, from the repo root, in order:

```bash
python3 -m modal run modal/modal_app.py::fetch_ravdess     # ~10 min, once
python3 -m modal run modal/modal_app.py::fetch_mustard     # ~20 min, once (1.4 GB)
python3 -m modal run modal/modal_app.py::warm_model        # ~10 min, once (16.8 GB)
python3 -m modal run modal/modal_app.py::corpus_status     # check coverage first
```

`corpus_status` should report roughly `119/119`, `120/120`, `76/120`. If it
reports zeros, the fetches haven't run.

## Running

```bash
python3 -m modal run modal/modal_app.py                    # everything, ~80 min
```

Results are copied to `./results/` automatically when it finishes. To pull them
again without rerunning:

```bash
python3 -m modal run modal/modal_app.py --fetch-only
```

Individual experiments:

```bash
python3 -m modal run modal/modal_app.py::experiment --name exp1_capability_screen
python3 -m modal run modal/modal_app.py::experiment --name exp2_encoder_probe --args "--category speaker_identity"
python3 -m modal run modal/modal_app.py::experiment --name exp3_circuit_localization --args "--category emotional_prosody"
python3 -m modal run modal/modal_app.py::experiment --name exp4_backpatching --args "--category emotional_prosody"
python3 -m modal run modal/modal_app.py::main_eval         # the behavioural eval
```

## Smoke test before spending GPU time

Roughly two minutes, and it catches infrastructure problems before a long run
queues behind them:

```bash
python3 -m modal run modal/modal_app.py::experiment --name exp1_capability_screen --args "--category speaker_identity --limit 8"
python3 -m modal run modal/modal_app.py::experiment --name exp3_circuit_localization --args "--category emotional_prosody --max-pairs 1"
```

exp3 is the one to watch: it is the only experiment that holds a backward graph
through the whole 7B model, so if anything runs out of memory it will be that.
If it does, change `GPU = "A100-40GB"` to `"H100"` in `modal_app.py`.

## How it's put together

| piece | what it does |
|---|---|
| `image` | debian + ffmpeg + torch/transformers/sklearn; mounts `eval/`, `data/`, `build/` |
| volume `audio-corpus` | `dataset_audio/` and `results/` — survives between runs |
| volume `hf-cache` | model weights, so only the first run downloads 16.8 GB |
| `GPU = "A100-40GB"` | fits the model with headroom; H100 buys little on minute-long jobs |

`add_local_dir` re-mounts `eval/` on **every** run, so code edits take effect
with no rebuild. If output ever looks stale, check rather than guess:

```bash
python3 -m modal run modal/modal_app.py::code_version
```

It hashes the `eval/*.py` files as Modal sees them and reports whether the
current fixes are present.

## Audio coverage

RAVDESS is fetched in full, so `emotional_prosody` and `speaker_identity` are
complete. MUStARD is partial: the items were built from **MUStARD++**, and 29 of
the 80 clips use episode-style ids (`1_S09E03_056`) that exist only in the
MUStARD++ release, not in the public v1 raw-clip zip. `sarcasm_pacing` therefore
runs on 113 of its 180 items.

Items whose audio is missing are dropped with a per-category warning, never
silently. To complete the set, put the MUStARD++ clips in the volume yourself:

```bash
python3 -m modal volume put audio-corpus ./mustard_plus_plus /dataset_audio/mustard
```

## Gotchas

- **Run from the repo root.** `add_local_dir` resolves relative to
  `modal_app.py`, so a second checkout elsewhere could get mounted instead.
- **`exp2` writes a fixed filename** (`exp2_encoder_probe.json`) regardless of
  `--category`, unlike exp3/exp4 which include it. Running exp2 per category
  overwrites the previous result — save or rename between runs.
- The `HF_TOKEN` warning during model download is harmless; it only affects
  rate limits.
