# Running the SALMONN / DeSTA2.5 evals on Modal

`modal_salmonn_eval.py` and `modal_desta25_eval.py` are thin infra wrappers
around the existing `eval/run_salmonn_eval.py` and `eval/run_desta25_eval.py`.
All dataset/scoring/model-loading logic lives in those two scripts (see their
own docstrings); the Modal files only handle image build, GPU selection,
secrets, and persistent storage.

Both wrappers split into two Modal functions:

- `audit_only` -- CPU only, no GPU, lightweight image (stdlib + numpy). Runs
  the underlying script's `--dry-run` path: loads `data/combined_examples.json`,
  validates every category's field-name resolution, option counts, and audio
  file presence. This is what `--dry-run` invokes. Costs cents, not dollars.
- `run_eval` -- the real GPU run, on the heavy pinned image, billed A100 time.

## 1. Install and authenticate

```bash
pip install modal
python3 -m modal setup      # opens a browser to link this machine to your Modal workspace
```

(On Windows, if you hit a `charmap codec` error from the setup command, set
`PYTHONUTF8=1` first -- it's a console-encoding issue, not an auth failure.)

## 2. Create secrets

```bash
modal secret create github-token GITHUB_TOKEN=<your GitHub PAT>
modal secret create hf-token HF_TOKEN=<your Hugging Face token>
```

- `hf-token` is load-bearing: SALMONN downloads `lmsys/vicuna-13b-v1.1` and
  `tsinghua-ee/SALMONN` from the Hub at run time, and DeSTA2.5 downloads
  `DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B` (which wraps `DeSTA-ntu/Llama-3.1-8B-Instruct`).
  If either underlying repo is gated for your account, the run 401s without it.
- `github-token` is **not currently consumed** by either wrapper -- both
  `bytedance/SALMONN` (salmonn branch) and `kehanlu/DeSTA2.5-Audio` are public
  repos, cloned anonymously during the image build. It's created for
  convenience/future use (e.g. a private fork), not because these scripts need it today.

## 3. SALMONN's BEATs checkpoint (manual download required)

SALMONN needs 4 artifacts. Three come down automatically at run time (Whisper
and Vicuna via `transformers.from_pretrained`, the `salmonn_v1.pth` checkpoint
via `huggingface_hub`). The fourth -- the fine-tuned `BEATs_iter3+ (AS2M)`
encoder -- is hosted on OneDrive, not HuggingFace or git, and cannot be
fetched inside the container. Download it yourself from the link in
`eval/run_salmonn_eval.py`'s own docstring, then upload it once to the shared
Modal volume:

```bash
modal volume put qwen2audio-results <local-path-to-beats.pt> salmonn_artifacts/beats.pt
```

This step is **only needed before a real SALMONN GPU run** -- `--dry-run`
never calls `load_model()`, so it doesn't touch this file. The GPU function
checks for it at startup and raises a clear error (not a confusing downstream
traceback) if it's missing.

## 4. Dry-run both (no GPU, cheap)

```bash
modal run modal/modal_desta25_eval.py --dry-run
modal run modal/modal_salmonn_eval.py --dry-run
```

Each prints the same `=== dataset audit ===` block the local scripts print:
per-category item counts, resolved field names, option counts, condition/
modality breakdowns, and any unresolved claims or missing audio files. Look
for `=== audit PASSED ===` and `dry run complete, model not loaded.` at the
end. A non-zero exit means the audit flagged something -- read the per-line
counts above it to see which category/check failed.

Last verified run (2026-08-14): both passed cleanly -- 518 items across
`emotional_prosody` (178), `speaker_identity` (160), `sarcasm_pacing` (180),
zero unresolved claims, zero missing audio files. The SALMONN dry-run also
exercised (as a side effect of image build) the full pinned legacy
environment -- `torch==2.0.1` / `transformers==4.28.0` / `peft==0.3.0` -- and
the `-b salmonn` clone of `bytedance/SALMONN`, both of which built cleanly on
Modal's infrastructure.

## 5. A real run

```bash
modal run modal/modal_desta25_eval.py --experiments all --limit 5     # smoke test, ~minutes
modal run modal/modal_desta25_eval.py --experiments all               # full run

modal run modal/modal_salmonn_eval.py --experiments all --limit 5
modal run modal/modal_salmonn_eval.py --experiments all
```

Useful flags on both (see each file's `local_entrypoint` for the full list):
`--limit N` caps items per category (smoke test before a full paid run),
`--seed`, `--max-new-tokens`. DeSTA2.5 additionally takes `--model-path`.

Results land in the `qwen2audio-results` volume under `/vol/results/`
(`desta25_<category>_<timestamp>.json` / `salmonn_<category>_<timestamp>.json`,
plus a `*_run_<timestamp>_summary.json`), alongside `/vol/hf_cache` (shared
HF download cache, so repeat runs don't re-download Vicuna/Whisper/etc.) and
`/vol/salmonn_artifacts/beats.pt` from step 3. Pull results down with:

```bash
modal volume get qwen2audio-results results/ ./results_from_modal/
```

## Changelog: bugs found during first real (non-dry-run) smoke tests

Both dry-runs passing only validates the dataset audit -- it never loads a
model or runs inference. The first real `--limit 5` GPU runs on 2026-08-14
surfaced four real bugs, all now fixed. **Any results generated before this
date predate these fixes and should not be trusted** -- in particular, an
earlier partial SALMONN run had every RAVDESS-sourced item (all of
`emotional_prosody` and `speaker_identity`'s audio-bearing items) silently
fail while only `sarcasm_pacing` (MUStARD++-sourced audio) produced real
results.

1. **DeSTA2.5's own `pip install -e .` stripped `hf_transfer`.**
   `modal_desta25_eval.py`'s image installed `huggingface_hub[hf_transfer]`,
   but the later `pip install -e /opt/DeSTA2.5-Audio` step re-pins
   `huggingface_hub` to an exact version with no `[hf_transfer]` extra,
   silently dropping the `hf_transfer` package while
   `HF_HUB_ENABLE_HF_TRANSFER=1` stayed set -- every HF download crashed
   with `ModuleNotFoundError: No module named 'hf_transfer'`. Fixed by
   installing `hf_transfer` as the last image layer, after desta's install
   settles its own `huggingface_hub` pin.

2. **Unpinned NumPy 2.x broke both models' audio-loading chains.** Neither
   `modal_desta25_eval.py` nor `modal_salmonn_eval.py` originally pinned
   NumPy, so pip resolved to the latest 2.x. For DeSTA2.5 this crashed every
   audio-bearing item with `AttributeError: np.sctypes was removed in the
   NumPy 2.0 release` (text_only items were unaffected since they skip audio
   loading). For SALMONN it broke `torch==2.0.1`'s C-API binding entirely
   (`Failed to initialize NumPy: _ARRAY_API not found`), which manifested as
   confusing downstream errors (`Unable to create tensor...`, spurious
   sampling-rate mismatches) on every item, audio or not. Fixed by pinning
   `numpy<2` in both images.

3. **SALMONN never resampled RAVDESS audio to 16kHz (real bug, not
   Modal-specific -- fixed in `eval/run_salmonn_eval.py` itself, commit
   `05d91fb` on `main`).** This dataset mixes RAVDESS audio (48kHz,
   `emotional_prosody`/`speaker_identity`) and MUStARD++ audio (already
   16kHz, `sarcasm_pacing`). SALMONN's own `prepare_one_sample()` hands
   `raw_speech` straight to `WhisperFeatureExtractor` with no resampling of
   its own, assuming the file on disk already matches its 16kHz rate. Every
   RAVDESS-sourced item errored (`WhisperFeatureExtractor was trained using
   a sampling rate of 16000... not 48000`) while MUStARD++-sourced items
   silently worked -- easy to miss since the run still exits 0 (per-item
   errors are caught and logged, not raised). Fixed by resampling on the fly
   in `inference_with_audio_clean()` whenever the source file's rate doesn't
   match, mirroring the script's existing silence-WAV temp-file pattern.

4. **Unpinned `transformers` upper bound broke `modal_qwen2audio_eval.py`
   against pinned `torch==2.4.1`.** `transformers>=4.45.0` (no upper bound)
   let pip resolve a `5.x` release, which raised its own floor to
   `torch>=2.5` and silently disabled transformers' entire PyTorch backend
   rather than erroring at install time -- surfaced at model-load time as
   `Qwen2AudioForConditionalGeneration requires the PyTorch library but it
   was not found`, even though torch was installed. DeSTA2.5's image dodged
   the same trap by accident (its editable install re-pins transformers down
   to a torch==2.4.1-compatible `4.57.6`); SALMONN pins transformers exactly
   (`==4.28.0`) so it was never exposed to begin with. Fixed by capping
   Qwen2-Audio's pin to `transformers>=4.45.0,<5`.

Also worth knowing: the checkpoint/resume logic (`*_progress.jsonl` in
`/vol/results/`) records error records as "done," so re-running after a fix
without clearing the relevant `*_progress.jsonl` files just replays the old
errors instead of re-attempting them. Delete them first:
`modal volume rm qwen2audio-results results/<file>_progress.jsonl`.

## GPU sizing

- DeSTA2.5-Audio-Llama-3.1-8B runs on `A100-40GB` (bf16, ~16GB of weights;
  see `run_desta25_eval.py`'s docstring for why bf16, not fp16, is correct
  here).
- SALMONN-13B runs on `A100-80GB` (fp16 Vicuna-13B + Whisper + BEATs; the
  script's own docstring notes it targets "a single A100, 40GB or 80GB" and
  errors clearly rather than silently quantizing if it doesn't fit -- 80GB is
  the safer default).

Both are hard-coded in each `modal_*.py` file's `gpu=` argument on the
`run_eval` function -- edit that decorator directly to change them.

## Full-run results (2026-08-14)

Full 518-item runs (`--experiments all`, no `--limit`) for all three models,
run one at a time after each passed its `--limit 5` smoke test clean (zero
inference errors) with every bug in the changelog above already fixed.
**These are the first trustworthy full-dataset numbers** -- everything
before this point was either a dry-run (no model load) or a broken/partial
smoke test.

| Model | Category | Scored/Total | Invalid | WA | UAR | TDR | GPU time |
|---|---|---|---|---|---|---|---|
| **Qwen2-Audio-7B-Instruct** | emotional_prosody | 177/178 | 1 | 0.6102 | 0.6068 | 1.0000 (60/60 mism.) | 66.0s |
| | speaker_identity | 160/160 | 0 | 0.7875 | 0.7875 | 0.8833 | 24.2s |
| | sarcasm_pacing | 173/180 | 7 | 0.3353 | 0.3678 | 0.5965 | 108.2s |
| | **total** | **510/518** | **8** | | | | **~198s (~3.3 min)** |
| **DeSTA2.5-Audio-Llama-3.1-8B** | emotional_prosody | 175/178 | 3 | 0.6857 | 0.6888 | 0.6667 | 330.6s |
| | speaker_identity | 152/160 | 8 | 0.7303 | 0.7299 | 1.0000 | 160.6s |
| | sarcasm_pacing | 180/180 | 0 | 0.2167 | 0.2308 | 0.2167 | 493.2s |
| | **total** | **507/518** | **11** | | | | **~984s (~16.4 min)** |
| **SALMONN-13B** | emotional_prosody | 178/178 | 0 | 0.4775 | 0.4744 | 0.8500 | 212.0s |
| | speaker_identity | 160/160 | 0 | 0.5750 | 0.5750 | 0.5333 | 117.5s |
| | sarcasm_pacing | 180/180 | 0 | 0.3167 | 0.3627 | 0.4167 | 143.6s |
| | **total** | **518/518** | **0** | | | | **~473s (~7.9 min)** |

Zero inference *errors* across all 1,554 items (3 models x 518 items) --
"invalid" above means the model's output didn't parse to a valid MCQ letter,
not a crash. TDR is Text Dominance Ratio (fraction of Mismatched-condition
items where the model's answer followed the text claim over the audio
truth), `n/a` entries excluded since every category had usable mismatched
items this run.

GPU time is the script's own `elapsed_seconds` per category (model-load time
and image-build time are not included, since those are one-off/cached costs
rather than per-item inference cost). Result files:

```
/vol/results/qwen2audio_run_20260814_204408_summary.json
/vol/results/desta25_run_20260814_204904_summary.json
/vol/results/salmonn_run_20260814_211310_summary.json
```

plus each category's own `<model>_<category>_<timestamp>.json` (full
per-item records) and `<model>_<category>_progress.jsonl` (checkpoint) in
the same `/vol/results/` directory. Pulled to a local
`results_from_modal/results/` folder via:

```bash
modal volume get qwen2audio-results results ./results_from_modal
```

Note that folder also still holds earlier `--limit 5` smoke-test files
(their timestamps predate the ones listed above) -- those were never
deleted from the volume, only their checkpoints were cleared/superseded
before each retest. The three summary files above are the ones to use.
