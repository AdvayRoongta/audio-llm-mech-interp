# SALMONN mech-interp port — work log

Chronological account of the 2026-08-22 session: what was run, what broke, what
was decided and why. `notes/SALMONN_PORT.md` is the reference doc for *what the
code does now*; this is the record of *how it got there*, including the dead
ends, because several of them are things the next person will hit too.

---

## Starting state

- mech interp (`exp1`–`exp6`) had only ever run on **Qwen2-Audio-7B-Instruct**.
  SALMONN and DeSTA2.5 had behavioural evals only.
- Ankur's committed Qwen results (`results/exp*.json`, 2026-08-15) covered
  **exp1–exp4 only** — no exp5/exp6 results exist for Qwen.
- Those results were produced on the **518-item** dataset. `754b081`
  (2026-08-17) scaled it to **3,594**. So the committed Qwen numbers and
  anything run today are *not* like-for-like.
- The Modal workspace `roongtaadvay` was **empty** — no volumes, apps or
  secrets. Everything was rebuilt.

## Infrastructure rebuilt

1. `hf-token` secret created.
2. `audio-corpus` repopulated: RAVDESS **905/905**; MUStARD **309/532** (223
   missing are MUStARD++ episode ids absent from the public v1 zip).
3. `beats.pt` uploaded to a new `salmonn-artifacts` volume. Validated before
   upload: `cfg`/`model` keys present, `encoder_embed_dim: 768`, 12 layers,
   90.8M params — the `cpt2` variant SALMONN specifies. OneDrive-only, so this
   is the one artifact that cannot be fetched inside a container.
4. `hf-cache-salmonn`: Vicuna-13B + Whisper-large-v2 + `salmonn_v1.pth` (~34 GB).
   Checkpoint snapshot hash `e0e903b8…76666` — **identical to the one in
   `results/salmonn_summary.json`**, so the mech-interp runs are on the same
   weights as the 2026-08-14 behavioural baseline.

## Failures hit, in order

Each of these produced a *wrong or absent* result, not a clean error, until
caught. Worth reading before touching the SALMONN path.

**1. `load_check` — all letter log-probs `nan`.** Shapes were all correct
(40 layers, span width 88, single-token letters), only the values were garbage.
A GPU probe isolated it: Whisper output clean (0/1,920,000 non-finite), BEATs
output **129024/129024 NaN**. Cause: `BEATs.preprocess` scales the waveform by
`2**15` before `ta_kaldi.fbank`; under `maybe_autocast(float16)` a float32
input is downcast and the power spectrum overflows fp16. SALMONN's own
`prepare_one_sample` escapes this only because `sf.read` returns float64.
Fixed by casting the BEATs waveform to float64.
*Autocast was my first hypothesis and it was wrong* — the probe tested it
directly (logits NaN both with and without) rather than assuming.

**2. exp2 — `Input type HalfTensor, weight type FloatTensor`.** `.to(model.dtype)`
assumed the tower and the LM share a dtype. SALMONN loads Whisper/BEATs/Q-Former
in **fp32** and only the LLM in fp16.

**3. exp2 — the projector is structurally different.** `proj(tower_out)` is a
shape error on SALMONN (`1496x1280` vs `2048x768`) because `speech_llama_proj`
sits *after* a Q-Former fed by Whisper+BEATs concatenated. Not a plumbing bug;
required defining `projected_audio()` per backend.

**4. exp3 — `SalmonnShim has no attribute zero_grad`.** Fixed with `zero_grad`
plus a `__getattr__` fallthrough, so the next missing `nn.Module` method is not
another crash on a loaded 13B.

**5. A leaked kwarg my own test suite missed.** Adding `_speech_embeds` to
`encode()` leaked it into `llama_model(**inputs)`. The harness passed anyway
because the fake LM had a `**kw` catch-all. The fake is now strict, and removing
the fix was verified to reproduce the `TypeError`.

**6. Modal `App create rate limit exceeded`** — three `modal run` launches
within two seconds. Stagger them.

## Decisions made, and why

- **`speaker_identity` and `sarcasm_pacing` were NOT run through exp2–exp6.**
  Both fail exp1's capability screen. `speaker_identity` is the trap: TDR 0.820
  reads as strong text dominance, but audio-only accuracy is 0.453 against
  0.500 binary chance — incapacity, not suppression. Running them would have
  produced clean-looking, uninterpretable numbers. Reversible if wanted.
- **exp3 was run alone**, after exp4/5/6 finished, because its backward pass
  through a 13B in fp16 had unknown memory behaviour. It fit on A100-80GB.
- **4-bit refused on SALMONN**, matching `run_salmonn_eval.py`'s repo-wide ban.
- **Committed to a branch, not `main`**, because the `_common.py` rewrite is
  shared infrastructure and Ankur was actively pushing. Results were later
  committed to `main` separately at his request.

## A correction I made mid-session

I initially told the user that `emotional_prosody` passing exp1 at only 0.188
meant "there may not have been much signal to restore," weakening the
backpatching null. **exp2 showed that reasoning was wrong.** Emotion is
linearly decodable at 0.671 in the exact vectors the LM receives. The weak exp1
number measures the model's *behaviour*, not the information's *availability* —
which makes the negative intervention results stronger, not weaker.

## Verification performed

- `eval/test_backend_salmonn.py` — 20 CPU checks on the adapter, no weights.
- `eval/test_refactor_equivalence.py` — differential test of the pre- vs
  post-refactor `_common.py`: the real 3,594-item dataset, stats edge cases,
  hooks and spans against a fake Qwen, spans and scoring against a fake
  processor. **0 failures**, and verified to *bite*: perturbing `audio_span` in
  `_backend_qwen.py` makes it fail.
- Cross-validation: Mismatched/text_audio accuracy came out **0.015** in exp1,
  exp4 and exp5 independently, against **0.017** from the behavioural run via
  free-text generation. Four code paths, same number.
- exp3's controls: identity patch reproduces baseline exactly; mid-layer source
  does nothing; free-form generation stays fluent under intervention.

## Still outstanding

1. **The Qwen GPU regression.** CPU equivalence is proven; identical GPU
   numerics are not. Must be compared **per-item on the 474 shared `uid`s**
   (matching `prediction` and the `scores` dict), not by file diff — the
   committed file predates the dataset scale-up. Currently on hold by request.
2. **Branch `salmonn-mech-interp` is unmerged.** `main` has the results but not
   the code that produces them; the commit message says so explicitly.
3. **exp2 probes Whisper only.** On SALMONN the BEATs branch is arguably more
   interesting — it was never trained to transcribe. New measurement, not a fix.
4. **exp5/exp6 have never been run on Qwen**, so SALMONN currently has more
   mech-interp coverage than the primary model.
