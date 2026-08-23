"""Run the mechanistic experiments (exp1-exp6) against SALMONN-13B on Modal.

The SALMONN counterpart to modal_app.py. It exists as a separate app rather
than a flag on that one because the two models cannot share a Python
environment: modal_app.py's image is transformers>=4.45, and SALMONN pins
transformers==4.28.0 / torch==2.0.1 / peft==0.3.0 and vendors modeling_llama.py
and modeling_whisper.py from that era. The image below is the one already
proven by modal_salmonn_eval.py, plus scikit-learn for exp2's probe.

Nothing about the experiments changes. eval/_common.py dispatches on
MECH_BACKEND, which is set to "salmonn" in ENV here, and eval/_backend_salmonn.py
makes SALMONN answer to the same calls Qwen does. Output files therefore have
the same names and the same "run" block as the Qwen results -- they are written
to results/salmonn/ so the two never overwrite each other.

Volumes (all created on first use):

    audio-corpus       dataset_audio/  -- SHARED with modal_app.py; populate it
                       once with `modal run modal/modal_app.py::fetch_ravdess`
                       and `::fetch_mustard`
    hf-cache-salmonn   Vicuna-13B + Whisper-large-v2 + salmonn_v1.pth (~34 GB).
                       Separate from modal_app.py's hf-cache because these are
                       different weights downloaded by a different, older
                       huggingface_hub -- sharing one cache dir across two
                       hub versions is asking for a corrupt blob.
    salmonn-artifacts  beats.pt -- YOU upload this one, see below.

One-time setup:

    modal secret create hf-token HF_TOKEN=hf_xxxxxxxx

    # BEATs is distributed on OneDrive only -- no HF, no git, nothing in this
    # container can fetch it. Download it by hand from the link in
    # eval/run_salmonn_eval.py's docstring, then:
    modal volume create salmonn-artifacts
    modal volume put salmonn-artifacts <local>/BEATs_iter3_plus_AS2M.pt beats.pt

    modal run modal/modal_app_salmonn.py::smoke        # no GPU, seconds
    modal run modal/modal_app_salmonn.py::warm_model   # ~34 GB, once

Running:

    modal run modal/modal_app_salmonn.py::experiment --name exp1_capability_screen
    modal run modal/modal_app_salmonn.py::experiment --name exp4_backpatching --args "--category emotional_prosody"
    modal run modal/modal_app_salmonn.py                  # exp1, then the rest
    modal run modal/modal_app_salmonn.py --fetch-only     # pull results down

Layer indices are NOT the same as Qwen's. Vicuna-13B has 40 decoder layers, not
32, so exp4/5/6 arguments tuned for Qwen (l_src=31) must be rescaled by you --
nothing here remaps them silently.
"""
import pathlib
import subprocess
import sys

import modal

REPO = pathlib.Path(__file__).parent.parent

# A100-80GB, not modal_app.py's 40GB: Vicuna-13B in fp16 is ~26 GB of weights
# before activations, and exp3 backprops through the whole model. Quantization
# is not an escape hatch here -- eval/_backend_salmonn.py refuses 4-bit and
# run_salmonn_eval.py hard-codes low_resource=False -- so the card has to fit it.
GPU = "A100-80GB"

SALMONN_REPO_DIR = "/opt/SALMONN"
WHISPER_REPO_ID = "openai/whisper-large-v2"
VICUNA_REPO_ID = "lmsys/vicuna-13b-v1.1"
SALMONN_CKPT_REPO_ID = "tsinghua-ee/SALMONN"
SALMONN_CKPT_FILENAME = "salmonn_v1.pth"
BEATS_PATH = "/vol/salmonn_artifacts/beats.pt"

app = modal.App("salmonn-mech-interp")

# Pins copied from modal_salmonn_eval.py, which has already been through a real
# run. The numpy<2 pin is load-bearing and must stay ahead of librosa/sklearn:
# torch==2.0.1 was compiled against NumPy's old C ABI, and NumPy 2.x breaks
# every tensor<->array conversion downstream with a misleading error.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "numpy<2",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "transformers==4.28.0",
        "sentencepiece==0.1.97",
        "accelerate==0.20.3",
        "peft==0.3.0",
        "soundfile",
        "librosa",
        # exp2's probe needs sklearn and exp1's CIs need scipy.
        # modal_salmonn_eval.py has neither (the behavioural eval doesn't use
        # them). Pinned to the last releases built against numpy<2.
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "git clone -b salmonn https://github.com/bytedance/SALMONN.git {}".format(
            SALMONN_REPO_DIR),
    )
    # Local mounts stay the last layers so editing eval/ or data/ never
    # invalidates the slow pip/git layers above.
    .add_local_dir(REPO / "eval", "/repo/eval")
    .add_local_dir(REPO / "data", "/repo/data")
)

corpus = modal.Volume.from_name("audio-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache-salmonn", create_if_missing=True)
artifacts = modal.Volume.from_name("salmonn-artifacts", create_if_missing=True)

VOLUMES = {
    "/corpus": corpus,
    "/root/.cache/huggingface": hf_cache,
    "/vol/salmonn_artifacts": artifacts,
}

ENV = {
    "HF_HOME": "/root/.cache/huggingface",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    # This is what makes eval/_common.py load the SALMONN backend.
    "MECH_BACKEND": "salmonn",
    "SALMONN_REPO": SALMONN_REPO_DIR,
    "SALMONN_WHISPER_PATH": WHISPER_REPO_ID,
    "SALMONN_LLAMA_PATH": VICUNA_REPO_ID,
    "SALMONN_BEATS_PATH": BEATS_PATH,
    # SALMONN_CKPT is resolved per-run by _ckpt(), since hf_hub_download
    # returns a snapshot-hashed path that isn't known until it has run.
    "PYTHONPATH": "{}:/repo/eval".format(SALMONN_REPO_DIR),
}

SECRETS = [modal.Secret.from_name("hf-token")]


def _require_beats():
    import os
    if not os.path.isfile(BEATS_PATH):
        raise SystemExit(
            "Missing {}.\n"
            "SALMONN's fine-tuned BEATs checkpoint is on OneDrive only (link in "
            "eval/run_salmonn_eval.py's docstring). Download it, then:\n"
            "  modal volume put salmonn-artifacts <local-path> beats.pt"
            .format(BEATS_PATH))


def _ckpt():
    """Fetch salmonn_v1.pth into the cache volume and return its path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=SALMONN_CKPT_REPO_ID,
                           filename=SALMONN_CKPT_FILENAME)


def _prepare():
    """Put the container in the state the eval scripts expect.

    They resolve audio relative to the repo root, so the corpus volume is
    symlinked in rather than copied -- same as modal_app.py._prepare().
    """
    import os

    os.chdir("/repo")
    link = "/repo/dataset_audio"
    if not os.path.exists(link):
        os.makedirs("/corpus/dataset_audio", exist_ok=True)
        os.symlink("/corpus/dataset_audio", link)
    os.makedirs("/corpus/results/salmonn", exist_ok=True)


def _run(script, args):
    """Run one eval script, streaming its output, and fail loudly if it fails."""
    import os

    _require_beats()
    _prepare()
    env = dict(os.environ, SALMONN_CKPT=_ckpt())
    cmd = [sys.executable, "-u", script] + list(args)
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/repo", env=env).returncode
    corpus.commit()
    if rc != 0:
        raise RuntimeError("{} exited {}".format(script, rc))
    return rc


# --------------------------------------------------------------------------- #
# setup / checks
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, timeout=60 * 10, env=ENV)
def smoke():
    """No GPU. Catches the mistakes that are expensive to find later.

    Checks, in the order they would otherwise bite: the SALMONN repo is
    importable, beats.pt is present, the corpus has audio, the dataset loads,
    and _common dispatches to the SALMONN backend.
    """
    import os

    _prepare()
    print("SALMONN repo: {}".format(
        "present" if os.path.isdir(SALMONN_REPO_DIR + "/models") else "MISSING"))
    print("beats.pt: {}".format(
        "present" if os.path.isfile(BEATS_PATH) else "MISSING -- see docstring"))

    clips = 0
    for root, _dirs, files in os.walk("/corpus/dataset_audio"):
        clips += sum(1 for f in files if f.endswith((".wav", ".flac", ".mp3")))
    print("audio clips in corpus: {}".format(clips))
    if not clips:
        print("  -> run  modal run modal/modal_app.py::fetch_ravdess  first")

    sys.path.insert(0, "/repo/eval")
    import _common as C
    print("MECH_BACKEND -> {} ({})".format(C.BACKEND, C.MODEL_PATH))
    items = C.load_items("data/combined_examples.json")
    print("dataset: {} items, categories {}".format(
        len(items), sorted({i.get("category") for i in items})))
    return {"clips": clips, "backend": C.BACKEND, "items": len(items),
            "beats": os.path.isfile(BEATS_PATH)}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 90, env=ENV, secrets=SECRETS)
def warm_model():
    """Download Vicuna-13B + Whisper-large-v2 + salmonn_v1.pth once (~34 GB)."""
    from huggingface_hub import snapshot_download

    for repo in (WHISPER_REPO_ID, VICUNA_REPO_ID):
        print("downloading {} ...".format(repo), flush=True)
        snapshot_download(repo_id=repo)
    print("downloading {} ...".format(SALMONN_CKPT_FILENAME), flush=True)
    path = _ckpt()
    hf_cache.commit()
    print("cached. ckpt at {}".format(path))
    return path


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60,
              env=ENV, secrets=SECRETS)
def load_check():
    """Load SALMONN on the GPU and run the backend's self-check, nothing more.

    ~15 minutes of GPU time to prove the shim is right before committing to a
    sweep: layer count, audio-span width, single-token option letters, and a
    real forward through the shim.
    """
    import os

    _require_beats()
    _prepare()
    os.environ["SALMONN_CKPT"] = _ckpt()
    sys.path.insert(0, "/repo/eval")

    import _common as C
    model, processor, info = C.load_model()
    print("\nrun block: {}".format(info))

    items = C.load_items("data/combined_examples.json", require_audio=True)
    item = next(i for i in items if i.get("modality") == "text_audio")
    inputs = C.encode(model, processor, item)
    span = C.audio_span(inputs, C.audio_token_id(model))
    seq = inputs["input_ids"].shape[1]
    letters = sorted(C.option_map(item))
    scores = C.score_letters(model, processor, inputs, letters)

    print("sequence length : {}".format(seq))
    print("audio span      : {} (width {})".format(span, span[1] - span[0] + 1))
    print("decoder layers  : {}".format(len(C.decoder_layers(model))))
    print("letter logprobs : {}".format({k: round(v, 3) for k, v in scores.items()}))
    assert span[1] - span[0] + 1 == info["n_audio_tokens"], "audio span width wrong"
    assert all(v < 0 for v in scores.values()), "log-probs should be negative"
    print("\nload_check OK")
    return {"seq_len": seq, "span": span, "n_layers": info["n_layers"]}


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60,
              env=ENV, secrets=SECRETS)
def nan_probe():
    """Locate where the NaN in the letter log-probs first appears.

    load_check showed correct shapes but nan scores, so the forward runs and
    something inside it is not finite. This walks the pipeline stage by stage
    and reports the first stage that goes bad, then tests the same forward with
    and without autocast -- SALMONN's own generate path wraps the LLM call in
    torch.cuda.amp.autocast(float16) while the shim does not, and the encoders
    load in fp32 while the LLM is fp16.
    """
    import os
    import torch

    _require_beats()
    _prepare()
    os.environ["SALMONN_CKPT"] = _ckpt()
    sys.path.insert(0, "/repo/eval")

    import _common as C
    import _backend_salmonn as B

    model, processor, _info = C.load_model()
    salmonn = model.salmonn

    def stat(name, t):
        t = t.float()
        bad = int((~torch.isfinite(t)).sum())
        print("  {:<26} shape={!s:<22} finite={:<6} min={:<12.4g} max={:<12.4g}".format(
            name, tuple(t.shape), bad == 0, float(t[torch.isfinite(t)].min()) if bad < t.numel() else float("nan"),
            float(t[torch.isfinite(t)].max()) if bad < t.numel() else float("nan")))
        return bad == 0

    items = C.load_items("data/combined_examples.json", require_audio=True)
    item = next(i for i in items if i.get("modality") == "text_audio")
    print("\nitem: {} [{}]".format(item.get("_uid"), item.get("category")))

    print("\n-- dtypes --")
    print("  llama      :", next(salmonn.llama_model.parameters()).dtype)
    print("  whisper enc:", next(salmonn.speech_encoder.parameters()).dtype)
    if salmonn.beats is not None:
        print("  beats      :", next(salmonn.beats.parameters()).dtype)
    print("  qformer    :", next(salmonn.speech_Qformer.parameters()).dtype)
    print("  proj       :", next(salmonn.speech_llama_proj.parameters()).dtype)

    print("\n-- stage by stage --")
    sr = processor.feature_extractor.sampling_rate
    wav = B._waveform(item, sr, None)
    spec = processor.feature_extractor(wav, sampling_rate=sr,
                                       return_tensors="pt")["input_features"].to(model.device)
    raw = torch.from_numpy(wav).unsqueeze(0).to(model.device)
    pad = torch.zeros(len(wav), dtype=torch.bool).unsqueeze(0).to(model.device)
    stat("spectrogram", spec)
    with torch.no_grad():
        se, _ = salmonn.encode_speech(spec, raw_wav=raw, audio_padding_mask=pad)
    stat("speech_embeds", se)

    inputs = B.encode(model, processor, item)
    stat("inputs_embeds", inputs["inputs_embeds"])

    print("\n-- forward, no autocast (what the shim does now) --")
    with torch.no_grad():
        lg = model(**inputs, use_cache=False).logits
    ok_plain = stat("logits", lg)

    print("\n-- forward, WITH autocast fp16 (what SALMONN's own path does) --")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        lg2 = model(**inputs, use_cache=False).logits
    ok_auto = stat("logits", lg2)

    print("\n-- per-layer scan (no autocast) --")
    layers = C.decoder_layers(model)
    first_bad = None
    acts = {}
    hs = []
    def mk(i):
        def hook(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            acts[i] = bool(torch.isfinite(h).all())
        return hook
    hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in hs:
        h.remove()
    for i in range(len(layers)):
        if not acts.get(i, True):
            first_bad = i
            break
    print("  first non-finite layer output: {}".format(
        first_bad if first_bad is not None else "none -- all 40 layers finite"))

    print("\nVERDICT: plain={} autocast={}".format(ok_plain, ok_auto))
    return {"plain_ok": ok_plain, "autocast_ok": ok_auto, "first_bad_layer": first_bad}


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60,
              env=ENV, secrets=SECRETS)
def nan_probe2():
    """The NaN is inside encode_speech. Find which branch produces it.

    nan_probe established: spectrogram finite, speech_embeds NOT finite, and
    autocast makes no difference. So it is Whisper, BEATs, or the Q-Former.
    Five tests, one GPU trip:

      1. Whisper encoder output on its own
      2. BEATs output on its own
      3. encode_speech WITHOUT BEATs (raw_wav=None) -- isolates the BEATs branch
      4. SALMONN's own prepare_one_sample() on the real file -- if this is
         finite and ours is not, the bug is in _waveform(), not the model
      5. float64 raw_wav (what sf.read gives prepare_one_sample) vs our float32
    """
    import os
    import numpy as np
    import torch

    _require_beats()
    _prepare()
    os.environ["SALMONN_CKPT"] = _ckpt()
    sys.path.insert(0, "/repo/eval")
    sys.path.insert(0, SALMONN_REPO_DIR)

    import _common as C
    import _backend_salmonn as B

    model, processor, _ = C.load_model()
    salmonn = model.salmonn
    fe = processor.feature_extractor
    sr = fe.sampling_rate

    def stat(name, t):
        if t is None:
            print("  {:<30} None".format(name)); return None
        f = t.float()
        n_bad = int((~torch.isfinite(f)).sum())
        tot = f.numel()
        good = f[torch.isfinite(f)]
        print("  {:<30} shape={!s:<20} nan/inf={}/{}  range=[{:.4g},{:.4g}]".format(
            name, tuple(f.shape), n_bad, tot,
            float(good.min()) if good.numel() else float("nan"),
            float(good.max()) if good.numel() else float("nan")))
        return n_bad == 0

    items = C.load_items("data/combined_examples.json", require_audio=True)
    item = next(i for i in items if i.get("modality") == "text_audio")
    path = item.get("_audio") or item.get("audio_path")
    print("\nitem {} [{}]\npath {}".format(item.get("_uid"), item.get("category"), path))

    # ---- our waveform -------------------------------------------------- #
    wav = B._waveform(item, sr, None)
    print("\nour _waveform: dtype={} len={} range=[{:.4g},{:.4g}]".format(
        wav.dtype, len(wav), wav.min(), wav.max()))
    spec = fe(wav, sampling_rate=sr, return_tensors="pt")["input_features"].to(model.device)
    raw32 = torch.from_numpy(wav).unsqueeze(0).to(model.device)
    pad = torch.zeros(len(wav), dtype=torch.bool).unsqueeze(0).to(model.device)

    print("\n== 1/2. encoder branches, ours (float32 raw_wav) ==")
    with torch.no_grad(), salmonn.maybe_autocast():
        w = salmonn.speech_encoder(spec, return_dict=True).last_hidden_state
        stat("whisper_out", w)
        b = salmonn.beats.extract_features(raw32, padding_mask=pad, feature_only=True)[0]
        stat("beats_out", b)

    print("\n== 3. encode_speech WITHOUT beats ==")
    with torch.no_grad():
        se_nb, _ = salmonn.encode_speech(spec, raw_wav=None, audio_padding_mask=None)
    stat("speech_embeds (no beats)", se_nb)

    print("\n== 3b. encode_speech WITH beats (ours) ==")
    with torch.no_grad():
        se, _ = salmonn.encode_speech(spec, raw_wav=raw32, audio_padding_mask=pad)
    stat("speech_embeds (ours)", se)

    print("\n== 4. SALMONN's own prepare_one_sample on the real file ==")
    from utils import prepare_one_sample
    rpath, is_tmp = B.R._resample_if_needed(path, sr)
    samples = prepare_one_sample(rpath, fe)
    print("  raw_wav dtype={} len={}".format(samples["raw_wav"].dtype,
                                             samples["raw_wav"].shape))
    stat("their spectrogram", samples["spectrogram"])
    with torch.no_grad():
        se_theirs, _ = salmonn.encode_speech(
            samples["spectrogram"], raw_wav=samples["raw_wav"],
            audio_padding_mask=samples.get("padding_mask"))
    ok_theirs = stat("speech_embeds (theirs)", se_theirs)
    if is_tmp:
        try: os.remove(rpath)
        except OSError: pass

    print("\n== 5. float64 raw_wav, our spectrogram ==")
    raw64 = torch.from_numpy(wav.astype(np.float64)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        b64 = salmonn.beats.extract_features(raw64, padding_mask=pad, feature_only=True)[0]
    stat("beats_out (float64 in)", b64)
    with torch.no_grad():
        se64, _ = salmonn.encode_speech(spec, raw_wav=raw64, audio_padding_mask=pad)
    ok64 = stat("speech_embeds (float64)", se64)

    print("\nVERDICT: theirs_ok={} ours_f64_ok={}".format(ok_theirs, ok64))
    return {"theirs_ok": ok_theirs, "ours_float64_ok": ok64}


# --------------------------------------------------------------------------- #
# the experiments
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60 * 4,
              env=ENV, secrets=SECRETS)
def experiment(name: str, args: str = ""):
    """One mechanistic experiment, e.g. name='exp1_capability_screen'."""
    extra = args.split() if args else []
    return _run("eval/{}.py".format(name),
                ["--input", "data/combined_examples.json",
                 "--output-dir", "/corpus/results/salmonn"] + extra)


@app.local_entrypoint()
def main(fetch_only: bool = False, categories: str = "emotional_prosody speaker_identity"):
    """exp1 first, then the rest for each category.

    exp1 is the gate: it runs across every category, and its verdict is what
    says whether exp2-exp6 are measuring suppression or just incapacity. It is
    also the cheapest way to find out the shim is broken, so it goes first
    regardless.
    """
    if fetch_only:
        save_results_locally()
        return

    cats = categories.split()

    print("=== exp1: capability screen (the gate, all categories) ===")
    experiment.remote("exp1_capability_screen")

    for cat in cats:
        print("\n=== exp2: encoder probe [{}] ===".format(cat))
        experiment.remote("exp2_encoder_probe", "--category {}".format(cat))

    for cat in cats:
        for exp in ("exp3_circuit_localization", "exp4_backpatching",
                    "exp5_patch_diagnosis", "exp6_readout_attenuation"):
            print("\n=== {} [{}] ===".format(exp, cat))
            experiment.remote(exp, "--category {}".format(cat))

    print("\nsaving results locally ...")
    save_results_locally()


def save_results_locally(dest="results/salmonn"):
    """Copy /results/salmonn out of the volume onto this machine."""
    import os

    os.makedirs(dest, exist_ok=True)
    n = 0
    for entry in corpus.iterdir("/results/salmonn"):
        if getattr(entry, "type", None) and str(entry.type).endswith("DIRECTORY"):
            continue
        name = os.path.basename(entry.path)
        with open(os.path.join(dest, name), "wb") as f:
            for chunk in corpus.read_file(entry.path):
                f.write(chunk)
        n += 1
        print("  saved {}/{}".format(dest, name))
    print("\n{} result files written to ./{}/".format(n, dest))
    return n
