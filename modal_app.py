"""Run the eval + mechanistic experiments on Modal GPUs.

Why this exists: Qwen2-Audio-7B is 16.8 GB and the experiments need CUDA, so
they can't run on a laptop. This packages the repo, fetches the audio corpus,
and runs eval/*.py on a rented A100.

Two persistent volumes, so the expensive things happen once:

    audio-corpus   dataset_audio/  (RAVDESS + MUStARD fetched automatically)
    hf-cache       model weights   (16.8 GB, downloaded on first run only)

One-time setup on your machine:

    pip install modal
    modal setup            # opens a browser to authenticate

Then, in order:

    modal run modal_app.py::fetch_ravdess         # ~10 min, once
    modal run modal_app.py::fetch_mustard         # ~20 min, once (1.4 GB zip)
    modal run modal_app.py::warm_model            # ~10 min, once (16.8 GB)
    modal run modal_app.py                        # runs everything, saves to ./results

Results are copied to ./results automatically at the end. To pull them again
without rerunning:

    modal run modal_app.py --fetch-only

Individual pieces:

    modal run modal_app.py::experiment --name exp1_capability_screen
    modal run modal_app.py::experiment --name exp2_encoder_probe --args "--category speaker_identity"
    modal run modal_app.py::main_eval --args "--experiments emotional_prosody"

Audio coverage. RAVDESS is fetched in full, so emotional_prosody and
speaker_identity are complete. MUStARD is only partly available: our items were
built from MUStARD++, and 29 of the 80 clips use episode-style ids
(1_S09E03_056) that exist only in the MUStARD++ release, not in the public v1
raw-clip zip. So sarcasm_pacing runs on 113 of its 180 items. Items whose audio
is missing are dropped with a per-category warning, never silently.

To complete it, put the MUStARD++ clips in the volume yourself:

    modal volume put audio-corpus ./mustard_plus_plus /dataset_audio/mustard
"""
import pathlib
import subprocess
import sys

import modal

REPO = pathlib.Path(__file__).parent
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
# The model is 16.8 GB in fp16. exp1/2/4 are forward-only and memory-light;
# exp3 backprops through the whole model, which needs the headroom. A100-40GB
# fits all of it with room to spare and costs well under an H100 -- and since
# these jobs are minutes, not hours, the H100's speed edge buys little.
GPU = "A100-40GB"

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


@app.function(image=image, volumes=VOLUMES, timeout=60 * 90, env=ENV)
def fetch_mustard():
    """Populate dataset_audio/mustard from the MichiganNLP raw-clip release.

    The MUStARD GitHub repo ships only features and transcripts -- no media.
    The raw clips are a separate 1.4 GB zip on Hugging Face. Utterance media is
    converted to 16 kHz mono wav named <id>_u.wav, matching what the dataset
    references.

    Caveat: our items were built from MUStARD++, whose episode-style ids
    (1_S09E03_056) do not exist in this v1 release. Those are reported as
    missing rather than silently skipped.
    """
    import json
    import os
    import re
    import subprocess
    import urllib.request
    import zipfile

    _prepare()
    out_dir = "/corpus/dataset_audio/mustard"
    os.makedirs(out_dir, exist_ok=True)

    wanted = set()
    for item in json.load(open("/repo/data/combined_examples.json")):
        p = item.get("audio_path") or ""
        if "mustard" in p:
            wanted.add(os.path.basename(p))

    def stem(name):
        return name[:-6] if name.endswith("_u.wav") else os.path.splitext(name)[0]

    want_ids = {stem(n): n for n in wanted}
    episode_style = {i for i in want_ids if re.search(r"S\d+E\d+", i)}
    print("{} clips wanted; {} are MUStARD++ episode ids not in the v1 release".format(
        len(want_ids), len(episode_style)))

    zip_path = "/corpus/mmsd_raw_data.zip"
    if not os.path.exists(zip_path):
        url = ("https://huggingface.co/datasets/MichiganNLP/MUStARD/"
               "resolve/main/mmsd_raw_data.zip")
        print("downloading 1.4 GB from {} ...".format(url))
        urllib.request.urlretrieve(url, zip_path)
        corpus.commit()

    extract_dir = "/corpus/mmsd_raw"
    if not os.path.isdir(extract_dir):
        print("extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        corpus.commit()

    # Index every media file by stem, so layout changes in the zip don't matter.
    by_stem = {}
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in (".mp4", ".mkv", ".avi", ".aac", ".wav", ".m4a"):
                by_stem.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    print("indexed {} media files in the archive".format(len(by_stem)))

    written, missing = 0, []
    for cid, target in sorted(want_ids.items()):
        dest = os.path.join(out_dir, target)
        if os.path.exists(dest):
            written += 1
            continue
        src = by_stem.get(cid)
        if not src:
            missing.append(cid)
            continue
        # 16 kHz mono wav is what the processor resamples to anyway.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vn", "-ac", "1", "-ar", "16000", dest], check=True)
        written += 1
    corpus.commit()

    print("wrote/verified {}/{} clips".format(written, len(want_ids)))
    if missing:
        print("MISSING {} ({} of them are MUStARD++ episode ids): {}".format(
            len(missing), len(set(missing) & episode_style), sorted(missing)[:5]))
    return {"wanted": len(want_ids), "present": written, "missing": len(missing),
            "mustard_plus_plus_ids": len(episode_style)}


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


@app.function(image=image, timeout=60 * 5)
def code_version():
    """Confirm which copy of the code Modal is actually running.

    add_local_dir re-mounts eval/ on every run, so edits should propagate with
    no rebuild -- but when output looks unchanged after an edit, guessing is
    worse than checking.
    """
    import hashlib
    import os

    print("{:<34} {:>8}  {}".format("file", "bytes", "sha256[:12]"))
    for name in sorted(os.listdir("/repo/eval")):
        if not name.endswith(".py"):
            continue
        path = os.path.join("/repo/eval", name)
        data = open(path, "rb").read()
        print("{:<34} {:>8}  {}".format(
            name, len(data), hashlib.sha256(data).hexdigest()[:12]))

    src = open("/repo/eval/exp3_circuit_localization.py").read()
    fixed = "SIGNAL_FLOOR" in src and "no signal, IoU undefined" in src
    print("\nexp3 zero-signal fix present: {}".format(fixed))
    if not fixed:
        print("Modal is running an OLD exp3. Re-run from the repo root, or")
        print("force a rebuild with:  modal run --force-build modal_app.py::code_version")
    return {"exp3_fix_present": fixed}


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
def main(fetch_only: bool = False):
    """Run every experiment the available audio supports.

    fetch_only just pulls existing results down without rerunning anything.
    It is a flag rather than a second local_entrypoint because Modal only
    auto-selects an entrypoint when there is exactly one, and a second one
    breaks plain `modal run modal_app.py`.

    exp1 runs across all categories with audio -- it is the gate, and its
    verdict decides whether exp2-4 mean anything for that category. exp3 and
    exp4 are the expensive ones, so they run only where exp1 has previously
    passed (emotional_prosody, speaker_identity in the Colab pass).

    sarcasm_pacing is included now: its prompts were fixed in 526aa8d, and the
    v1 raw clips cover 113 of its 180 items. Expect the per-category warning
    about the 29 MUStARD++ ids that are still missing.
    """
    if fetch_only:
        save_results_locally()
        return

    status = corpus_status.remote()
    ready = [c for c, v in (status or {}).items() if v["present"] > 0]
    partial = {c: v for c, v in (status or {}).items() if 0 < v["present"] < v["total"]}
    print("\ncategories with audio: {}".format(ready or "NONE"))
    for c, v in partial.items():
        print("   {} is PARTIAL: {}/{} clips".format(c, v["present"], v["total"]))
    print()
    if not ready:
        print("No audio. Run fetch_ravdess and fetch_mustard first.")
        return

    print("=== exp1: capability screen (the gate, all categories) ===")
    experiment.remote("exp1_capability_screen")

    for cat in ready:
        print("\n=== exp2: encoder probe [{}] ===".format(cat))
        experiment.remote("exp2_encoder_probe", "--category {}".format(cat))

    # Only where the screen has already passed -- these are the costly ones.
    for cat in ("emotional_prosody", "speaker_identity"):
        if cat not in ready:
            continue
        print("\n=== exp3: circuit localization [{}] ===".format(cat))
        experiment.remote("exp3_circuit_localization", "--category {}".format(cat))
        print("\n=== exp4: back-patching [{}] ===".format(cat))
        experiment.remote("exp4_backpatching", "--category {}".format(cat))

    print("\nsaving results locally ...")
    save_results_locally()


def save_results_locally(dest="results"):
    """Copy /results out of the volume onto this machine."""
    import os

    os.makedirs(dest, exist_ok=True)
    n = 0
    for entry in corpus.iterdir("/results"):
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


