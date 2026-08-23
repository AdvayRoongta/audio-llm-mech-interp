# Running exp1-exp6 on SALMONN-13B

Summary of what changed to make the mechanistic experiments run against
SALMONN-13B as well as Qwen2-Audio, what was deliberately left alone, and the
two bugs found on the way.

Written 2026-08-22. Qwen results are unaffected; see "Regression status" for
what has and has not been re-verified.

---

## What changed, at a glance

| file | status | why |
|---|---|---|
| `eval/exp1,3,4,5,6*.py` | **untouched** | the port was designed so they would not need changing |
| `eval/exp2_encoder_probe.py` | **2 edits** | its post-projector probe and its dtype cast both assumed Qwen's architecture — see below |
| `eval/_common.py` | restructured | now a backend dispatcher; public API unchanged |
| `eval/_backend_qwen.py` | new | the Qwen bodies, moved verbatim out of `_common.py` |
| `eval/_backend_salmonn.py` | new | the SALMONN adapter |
| `eval/test_backend_salmonn.py` | new | 18 CPU-only checks on the adapter |
| `modal/modal_app_salmonn.py` | new | SALMONN's Modal app (its own pinned image) |
| `.gitignore` | `*.pt`, `*.pth` | the BEATs checkpoint is 363 MB and sat in the repo root |

The pre-refactor originals of every file touched here are preserved in
**`archive/pre_salmonn_refactor/`** (verbatim `cb0e4cf` copies, sha-verified),
and in git history at `cb0e4cf` — which is the backup of record. See that
folder's README, or `notes/SALMONN_PORT_LOG.md`, for how to view or restore them.

Select a backend with `MECH_BACKEND=qwen` (default) or `MECH_BACKEND=salmonn`.
The Modal app sets it; nothing else has to know.

---

## The only change to an experiment: exp2

`exp2_encoder_probe.py` probes each encoder layer, then probes "the vectors the
language model actually receives" — the encoder-last vs post-projector
comparison is the whole point of the experiment. It did that inline:

```python
projected = proj(out.last_hidden_state)      # was
vecs.append(C.projected_audio(model, processor, inputs, n_enc))   # now
```

**This is not plumbing, it is a real architectural difference.** Qwen2-Audio has
one encoder feeding one projector, so `projector(tower_out)` is exactly "what
the LM receives". SALMONN does not:

```
Whisper-large-v2 (1280) ─┐
                         ├─ concat 2048 → window-level Q-Former → speech_llama_proj → 88 x 5120 → LM
BEATs (768) ─────────────┘
```

`speech_llama_proj` sits *after* the Q-Former and expects its output, not raw
Whisper states. Calling `proj(tower_out)` on SALMONN is a shape error
(`1496x1280` vs `2048x768`, confirmed on GPU), and would not mean the same thing
even if the dimensions matched.

So each backend now defines `projected_audio()`:

- **Qwen** — `projector(audio_tower(feats))`, pooled over valid frames.
  Numerically identical to the line it replaced.
- **SALMONN** — the output of the whole audio front-end (Whisper + BEATs →
  Q-Former → proj), i.e. the 88x5120 block spliced into the prompt.

Both answer the same question. `n_valid_frames` is ignored on SALMONN: the
Q-Former compresses the clip to a fixed 88 tokens, so there is no padding to
pool across, and truncating a prefix there would silently discard part of the
clip.

`layer_vectors()`'s `proj` argument is now unused but still passed — left in
place to keep the diff on this file as small as possible.

### Second exp2 edit: the tower's dtype, not the model's

`feats = inputs["input_features"].to(model.dtype)` is correct on Qwen, where the
audio tower and the LM share a dtype. SALMONN loads **Whisper, BEATs and the
Q-Former in fp32 and only the LLM in fp16**, so `model.dtype` (fp16) handed the
fp32 encoder half-precision features:

```
RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor)
```

Now casts to `next(tower.parameters()).dtype`, which is right for both — and is
arguably more correct than the original even on Qwen, since it stops assuming
the encoder and the LM are always loaded alike.

### Still open on exp2

It probes **Whisper only**. `audio_tower()` returns the Whisper encoder and
`beats_tower()` is exposed beside it, but the BEATs branch and the
post-Q-Former stack are not probed layer-by-layer. On SALMONN that is arguably
the more interesting measurement — BEATs was never trained to transcribe, so it
is the branch most likely to carry paralinguistic signal Qwen structurally
cannot. Worth doing as a follow-up; it is a new measurement, not a fix.

---

## How the adapter works

exp1-exp6 call `C.load_model / encode / decoder_layers / audio_span /
find_span / score_letters` and `model(**inputs)`. Three things had to be
reconciled.

**1. No audio placeholder token.** Qwen splices audio into `input_ids` at
`audio_token_index`. SALMONN concatenates *embeddings*: `[BOS][p_before]
[speech xN][p_after]`, built by splitting the prompt on `<SpeechHere>`. But
exp3 (`:80, :405, :446`) and exp6 (`:163, :185, :304, :318`) read
`inputs["input_ids"]` for length and span arithmetic.

`encode()` therefore returns a **shadow `input_ids`**: real Vicuna ids at text
positions, sentinel `-1` across the audio block. `find_span`, `audio_span` and
exp3's `position_groups` work unchanged, and `SalmonnShim` never embeds the
shadow. A sentinel appearing past the prompt raises rather than being embedded.

**2. Fixed 88-token audio span.** `window_level_Qformer=True`,
`second_per_window = second_stride = 0.333333`, Whisper always padded to 1500
frames → `kernel = stride = 17`, `L = (1500-17)//17 + 1 = 88` windows x
`num_speech_query_token=1`. **88 audio tokens for every clip, regardless of
duration.** Computed from the model's own config and asserted against a real
forward at load time, never hard-coded. Qwen needs `fixed_seconds` to align
clean/corrupt pairs; here alignment is free (the flag is still honoured).

**3. Geometry differs.** Vicuna-13B: **40 decoder layers, hidden 5120** (Qwen:
32 / 4096), and the LLM is PEFT-wrapped, so layers live at
`llama_model.model.model.layers`.

Checked: exp4/5/6 do **not** need their layer arguments rescaled.
`exp4_backpatching.py:109` does `src = n_layers - 1`, and exp5/exp6 derive
bands from `n_layers` too, so they adapt automatically.

**exp6's generation path** needed care: driven from `inputs_embeds` with no
`input_ids`, transformers returns only the newly generated tokens, so exp6's
`out[0, inputs["input_ids"].shape[1]:]` would slice away everything. The shim
prepends the prompt columns back as sentinels so that slice lands correctly.
It also handles exp6 re-scoring `cat([ctx_ids, generated])` by embedding only
the real tail.

**Quantization is refused.** `--4bit` raises on SALMONN, matching
`run_salmonn_eval.py`'s repo-wide ban and its hard-coded `low_resource=False`.

---

## Two bugs found

### 1. float32 waveform → 100% NaN audio embeddings (ours, fixed)

`load_check` returned correct shapes but every letter log-prob was `nan`.
Isolated on GPU:

```
whisper_out   (1, 1500, 1280)   nan/inf = 0/1,920,000        clean
beats_out     (1, 168, 768)     nan/inf = 129024/129024      100% NaN
```

`BEATs.preprocess` scales the raw waveform by **2^15** before
`ta_kaldi.fbank`. Under SALMONN's `maybe_autocast(float16)` a float32 input is
downcast to fp16, and the power spectrum of a ~7.5e3 signal (~5.7e7) overflows
fp16's 65504 ceiling → `inf` → NaN, which BEATs' self-attention then spreads
across every output position.

SALMONN's own path avoids this **by accident**: `prepare_one_sample` uses
`sf.read`, which returns float64, and autocast only downcasts float32. We load
with `librosa` (float32) so the waveform can be trimmed in memory for
`fixed_seconds`.

Fix — one line in `_backend_salmonn.encode()`, with the reasoning in a comment:

```python
raw_wav = torch.from_numpy(wav.astype("float64")).unsqueeze(0).to(device)
```

The spectrogram stays float32; only BEATs' raw waveform becomes float64.

> **Hazard worth knowing:** any code path handing SALMONN a float32 waveform
> under autocast gets silently-NaN audio embeddings. Nothing raises. This is
> the same species as the 48kHz resample bug (`05d91fb`) — SALMONN's audio
> front-end has undocumented input assumptions that fail quietly.

### 2. A leaked kwarg the test suite missed (ours, fixed)

Adding `_speech_embeds` to `encode()`'s output leaked it into
`llama_model(**inputs)` as an unexpected kwarg. The local harness passed anyway
because the fake LM had a `**kw` catch-all.

The fake is now **strict** (no `**kw`), and removing the `pop` was verified to
reproduce `TypeError: forward() got an unexpected keyword argument
'_speech_embeds'`. A test that cannot fail is not a test.

---

## Regression status

- `_common.py`'s public API is unchanged: nothing removed, only `BACKEND` added.
- 9 of the 10 moved Qwen functions are **byte-identical**; the tenth differs only
  by a parameter rename (`audio_token_id` → `audio_token_id_`, to stop it
  shadowing the module-level function). All three call sites pass positionally.
- All six experiments import and parse args under both backends.
- `eval/test_backend_salmonn.py`: 18/18, CPU-only, no weights needed.

**Not yet done: the Qwen regression run.** Reproducing
`results/exp1_capability_screen.json` needs a GPU and has not been run. Do it
before trusting any comparison between the two models:

```bash
modal run modal/modal_app.py::warm_model
modal run modal/modal_app.py::experiment --name exp1_capability_screen
```

---

## exp1's gate result changes what is worth running (2026-08-22)

The capability screen decides whether a category's downstream numbers mean
"suppression" or "the model never had the signal". On SALMONN, **two of three
categories fail**:

```
emotional_prosody    acc 0.188  CI [0.131, 0.263]  chance 0.125  p 0.035   PASS (weak)
speaker_identity     acc 0.453  CI [0.376, 0.533]  chance 0.500  p 0.289   FAIL -- at chance
sarcasm_pacing       acc 0.127  CI [0.070, 0.218]  chance 0.100  p 0.450   FAIL -- at chance
```

exp1's own verdict: *"Passed the screen: emotional_prosody. Localize
suppression in those only."*

**Do not run exp2-exp6 on speaker_identity or sarcasm_pacing.** They produce
clean-looking numbers about a signal the model never extracted. `speaker_identity`
is the trap: its conflict TDR is high (0.820) which reads as strong text
dominance, but audio-only accuracy is 0.453 against 0.500 binary chance -- the
model cannot identify the speaker from audio at all, so that is **incapacity,
not suppression**, and no intervention can restore it.

**`emotional_prosody` passes weakly** -- 6.3 points above chance, p=0.035, on
the Paralinguistic condition. LISTEN reports 15.7-22.7% for every model it
tested on that condition, so SALMONN is unremarkable. Any suppression claim
here should be stated with that caveat: there was not much signal available to
begin with.

### Cross-validation of the port

The Mismatched/text_audio accuracy for `emotional_prosody` came out at **0.015**
in exp1, exp4 and exp5 independently, and the 2026-08-14 behavioural run got
0.017 through a completely different path (`generate()` + letter parsing vs.
option log-probs). Four measurements, four code paths, same number. That is the
strongest evidence the adapter is faithful.

### Intervention results so far (emotional_prosody)

| experiment | result |
|---|---|
| exp6 readout attenuation | **0% of the -2.29 nat gap**, capability intact -- clean null |
| exp5 patch diagnosis | `AMBIGUOUS`: margin only improves as capability collapses (0.933 -> 0.100 at the late source). Best capability-preserving intervention: **+0.01 nats**. |
| exp4 backpatching | `src=39 -> dst=0`, w=1/3/5: audio +0.12 to +0.17 but **capability ~0.10** in every row |

Two findings worth keeping:

- **Amplification does nothing.** Scaling the audio span in place by alpha up to
  5.0 moves the margin from -2.28 to -2.32 with capability untouched. SALMONN's
  text dominance is not a magnitude problem.
- **The extrapolation is damning.** +0.096 nats per extra destination layer means
  closing 2.29 nats needs ~24 of 40 layers -- and w=7 already costs capability
  0.383. exp5's words: *"an upper bound on plausibility, not a plan."*

Preliminary read: **Cho et al.'s back-patching does not transfer from lexical to
paralinguistic conflict on SALMONN.** Their non-destructiveness claim was
capability delta ~0.00; here it collapses by 0.76. n=60 mismatched items, one
category, one model.

---

## Running it

SALMONN results go to `results/salmonn/` — same filenames, same `run` block
shape as the Qwen results, so nothing is overwritten.

```bash
modal run modal/modal_app_salmonn.py::smoke        # seconds, no GPU
modal run modal/modal_app_salmonn.py::warm_model   # ~34 GB, once
modal run modal/modal_app_salmonn.py::load_check   # ~15 min GPU, validates the adapter
modal run modal/modal_app_salmonn.py::experiment --name exp1_capability_screen
```

`nan_probe` / `nan_probe2` in the Modal app are the diagnostics from bug 1.
They are kept because that class of bug is likely to recur; `nan_probe2`'s
"without BEATs" test is expected to raise (the Q-Former requires the 2048-dim
concatenation, so BEATs is not optional on the released checkpoint).

**Setup gotchas**

- `beats.pt` must be uploaded by hand — OneDrive only, no HF, no git:
  `modal volume put salmonn-artifacts <file> beats.pt`
- Stagger `modal run` launches; three within two seconds hits
  `App create rate limit exceeded`.
- SALMONN cannot share Qwen's environment (`transformers==4.28.0` vs `>=4.45`),
  hence the separate Modal app. `numpy<2` must stay pinned ahead of
  librosa/sklearn.
- A100-**80GB**, not 40GB: Vicuna-13B fp16 is ~26 GB before activations.
