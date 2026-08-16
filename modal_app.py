"""Run the eval + mechanistic experiments on Modal GPUs.

Why this exists: Qwen2-Audio-7B is 16.8 GB and the experiments need CUDA, so
they can't run on a laptop. This packages the repo, fetches the audio corpus,
and runs eval/*.py on a rented H100.

Two persistent volumes, so the expensive things happen once:

    audio-corpus   dataset_audio/  (RAVDESS fetched automatically)
    hf-cache       model weights   (16.8 GB, downloaded on first run only)

One-time setup on your machine:

    pip install modal
    modal setup            # opens a browser to authenticate

Then, in order:

    modal run modal_app.py::fetch_ravdess         # ~15 min, once
    modal run modal_app.py::warm_model            # ~10 min, once
    modal run modal_app.py                        # runs the experiments
    modal volume get audio-corpus /results ./results   # pull results down

Individual pieces:

    modal run modal_app.py::experiment --name exp1_capability_screen
    modal run modal_app.py::experiment --name exp2_encoder_probe --args "--category speaker_identity"
    modal run modal_app.py::main_eval --args "--experiments emotional_prosody"

MUStARD audio is not redistributable, so `fetch_ravdess` only covers the two
RAVDESS categories (emotional_prosody, speaker_identity). sarcasm_pacing items
are dropped with a warning unless you upload that audio yourself:

    modal volume put audio-corpus ./dataset_audio/mustard /dataset_audio/mustard
"""
import pathlib
import subprocess
import sys

import modal

REPO = pathlib.Path(__file__).parent
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
GPU = "H100"          # the model is 16.8 GB in fp16; A100-40GB also works

app = modal.App("audio-conflict-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "torch",
        "transformers>=4.45",
        "accelerate",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "scikit-learn",
        "datasets",
        "huggingface_hub",
        "bitsandbytes",
    )
    # Ship the code and data, but never the audio corpus -- that lives in a
    # volume so it survives between runs and isn't re-uploaded every time.
    .add_local_dir(REPO / "eval", "/repo/eval")
    .add_local_dir(REPO / "data", "/repo/data")
    .add_local_dir(REPO / "build", "/repo/build")
)

corpus = modal.Volume.from_name("audio-corpus", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

VOLUMES = {"/corpus": corpus, "/root/.cache/huggingface": hf_cache}
ENV = {"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"}


def _prepare():
    """Put the container in the state the eval scripts expect.

    They resolve audio relative to the repo root, so the corpus volume is
    symlinked in rather than copied.
    """
    import os

    os.chdir("/repo")
    link = "/repo/dataset_audio"
    if not os.path.exists(link):
        os.makedirs("/corpus/dataset_audio", exist_ok=True)
        os.symlink("/corpus/dataset_audio", link)
    os.makedirs("/corpus/results", exist_ok=True)


def _run(script, args):
    """Run one eval script, streaming its output, and fail loudly if it fails."""
    _prepare()
    cmd = [sys.executable, "-u", script] + list(args)
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/repo").returncode
    corpus.commit()
    if rc != 0:
        raise RuntimeError("{} exited {}".format(script, rc))
    return rc


# --------------------------------------------------------------------------- #
# one-time setup
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, env=ENV)
def fetch_ravdess():
    """Populate dataset_audio/ravdess from the Hugging Face copy.

    The dataset's audio_path values are original RAVDESS filenames
    (03-01-05-02-01-02-15.wav). The HF copy carries that exact name in the
    audio dict's `path`, so files are written under it verbatim -- no
    reconstruction, nothing to get subtly wrong.
    """
    import json
    import os

    import datasets
    from datasets import load_dataset

    _prepare()
    out_dir = "/corpus/dataset_audio/ravdess"
    os.makedirs(out_dir, exist_ok=True)

    wanted = set()
    for item in json.load(open("/repo/data/combined_examples.json")):
        p = item.get("audio_path") or ""
        if "ravdess" in p:
            wanted.add(os.path.basename(p))
    print("dataset references {} distinct RAVDESS files".format(len(wanted)))

    ds = load_dataset("xbgoose/ravdess", split="train")
    ds = ds.cast_column("audio", datasets.Audio(decode=False))  # raw bytes, no decode

    written = 0
    for row in ds:
        audio = row["audio"]
        name = os.path.basename(audio.get("path") or "")
        if not name or name not in wanted:
            continue
        dest = os.path.join(out_dir, name)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(audio["bytes"])
        written += 1

    have = {f for f in os.listdir(out_dir) if f.endswith(".wav")}
    missing = sorted(wanted - have)
    corpus.commit()

    print("wrote/verified {} files; {} of {} wanted are present".format(
        written, len(wanted & have), len(wanted)))
    if missing:
        print("MISSING {} (e.g. {})".format(len(missing), missing[:3]))
    return {"wanted": len(wanted), "present": len(wanted & have), "missing": len(missing)}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, gpu=GPU, env=ENV)
def warm_model():
    """Download the 16.8 GB checkpoint into the cache volume once."""
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    import torch

    AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    hf_cache.commit()
    print("cached. dtype={} device={}".format(model.dtype, model.device))
    return str(model.dtype)


@app.function(image=image, volumes=VOLUMES, timeout=60 * 10, env=ENV)
def corpus_status():
    """What audio is actually present, per category."""
    import json
    import os

    _prepare()
    items = json.load(open("/repo/data/combined_examples.json"))
    by_cat = {}
    for it in items:
        p = it.get("audio_path")
        if not p or it.get("modality") not in ("audio_only", "text_audio"):
            continue
        cat = it.get("category")
        present = os.path.exists(os.path.join("/repo", p))
        got, tot = by_cat.get(cat, (0, 0))
        by_cat[cat] = (got + bool(present), tot + 1)

    print("{:<22}{:>10}".format("category", "audio on disk"))
    for cat, (got, tot) in sorted(by_cat.items()):
        print("{:<22}{:>10}".format(cat, "{}/{}".format(got, tot)))
    return {k: {"present": v[0], "total": v[1]} for k, v in by_cat.items()}


# --------------------------------------------------------------------------- #
# the experiments
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60 * 4, env=ENV)
def main_eval(args: str = ""):
    """The behavioural eval: eval/run_qwen2audio_eval.py."""
    extra = args.split() if args else []
    return _run("eval/run_qwen2audio_eval.py",
                ["--input", "data/combined_examples.json",
                 "--output-dir", "/corpus/results"] + extra)


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60 * 4, env=ENV)
def experiment(name: str, args: str = ""):
    """One mechanistic experiment, e.g. name='exp1_capability_screen'."""
    extra = args.split() if args else []
    return _run("eval/{}.py".format(name),
                ["--input", "data/combined_examples.json",
                 "--output-dir", "/corpus/results"] + extra)


@app.local_entrypoint()
def main():
    """Run the experiments that the available audio supports.

    sarcasm_pacing is skipped: its audio is not redistributable, and its
    prompts were missing the option list in the run we analysed, so its
    numbers would not be interpretable regardless.
    """
    status = corpus_status.remote()
    ready = [c for c, v in (status or {}).items() if v["present"] > 0]
    print("\ncategories with audio: {}\n".format(ready or "NONE"))
    if not ready:
        print("Run `modal run modal_app.py::fetch_ravdess` first.")
        return

    print("=== exp1: capability screen (the gate) ===")
    experiment.remote("exp1_capability_screen")

    for cat in ("emotional_prosody", "speaker_identity"):
        if cat not in ready:
            continue
        print("\n=== exp2: encoder probe [{}] ===".format(cat))
        experiment.remote("exp2_encoder_probe", "--category {}".format(cat))

    if "emotional_prosody" in ready:
        print("\n=== exp3: circuit localization [emotional_prosody] ===")
        experiment.remote("exp3_circuit_localization", "--category emotional_prosody")
        print("\n=== exp4: back-patching [emotional_prosody] ===")
        experiment.remote("exp4_backpatching", "--category emotional_prosody")

    print("\nResults are in the audio-corpus volume under /results.")
    print("Pull them down with:")
    print("    modal volume get audio-corpus /results ./results")
