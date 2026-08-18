#!/usr/bin/env python3
"""
Modal wrapper for eval/run_qwen2audio_eval.py (Qwen2-Audio-7B-Instruct).

Runs the existing, unmodified eval/run_qwen2audio_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage. All dataset/scoring/model-loading
logic lives in eval/run_qwen2audio_eval.py.

Simpler than the DeSTA2.5 and SALMONN wrappers: Qwen2-Audio-7B-Instruct is a
single public HF model (Qwen/Qwen2-Audio-7B-Instruct, no gating), loaded
straight via transformers.Qwen2AudioForConditionalGeneration -- no repo
clone, no editable pip install, no legacy dependency pins, no manually
downloaded artifacts.

Still applies the same two fixes both other wrappers needed the hard way
(see MODAL_SETUP.md's changelog section):
  * numpy<2 pinned explicitly -- an unpinned/latest NumPy 2.x broke the
    librosa/audio-loading chain for both DeSTA2.5 and SALMONN.
  * hf_transfer installed as the LAST image layer, after everything else --
    for DeSTA2.5 this mattered because its editable install silently
    re-pinned huggingface_hub without the [hf_transfer] extra. This image
    has no editable install step, so that specific failure mode doesn't
    apply here, but the ordering is kept anyway as a cheap defensive
    consistency measure rather than assuming this image is exempt.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the full walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your HF token>

  Qwen/Qwen2-Audio-7B-Instruct is a public, ungated model -- HF_TOKEN isn't
  required for it specifically, but the secret is reused here (same as the
  other two wrappers) for consistency and to avoid anonymous-download rate
  limits. No github-token is needed at all: this image clones nothing.

Usage:
  modal run modal/modal_qwen2audio_eval.py --dry-run
  modal run modal/modal_qwen2audio_eval.py --experiments all --limit 5
  modal run modal/modal_qwen2audio_eval.py --experiments emotional_prosody --limit 20
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with modal_desta25_eval.py / modal_salmonn_eval.py
SCRIPT_NAME = "run_qwen2audio_eval.py"

app = modal.App("qwen2audio-eval")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _with_repo_data(image: modal.Image) -> modal.Image:
    """Mount the eval script + dataset from the local repo. Kept as the last
    layer on every image per Modal's guidance: local-file mounts should not
    sit underneath layers that rarely change, so editing data/ or the eval
    script never invalidates the (slow) pip layers above it."""
    return (
        image
        .add_local_dir(str(LOCAL_ROOT / "data"), remote_path=f"{CONTAINER_ROOT}/data")
        .add_local_dir(str(LOCAL_ROOT / "dataset_audio"), remote_path=f"{CONTAINER_ROOT}/dataset_audio")
        .add_local_file(
            str(LOCAL_ROOT / "eval" / SCRIPT_NAME),
            remote_path=f"{CONTAINER_ROOT}/eval/{SCRIPT_NAME}",
        )
    )


# Lightweight image for --dry-run: the script's dry-run path only touches
# stdlib + (optionally) numpy -- see run_qwen2audio_eval.py's main(), the
# `import torch` / model-loading calls happen after the dry-run early-return.
# Keeping this separate means a dataset problem surfaces in seconds on CPU,
# not after a multi-minute GPU container boots.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run.
gpu_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        # >=4.45 for Qwen2-Audio support, <5 because transformers 5.x raised
        # its floor to torch>=2.5 -- left unpinned above that, pip resolves
        # a 5.x release that silently disables transformers' whole PyTorch
        # backend against torch==2.4.1 ("PyTorch was not found") instead of
        # erroring at install time. DeSTA2.5's image dodged this by accident
        # (its editable install re-pins transformers down to 4.57.6); capped
        # explicitly here since there's no such step to do it for us.
        "transformers>=4.45.0,<5",
        "accelerate",
        "soundfile",
        "librosa",
        # <2, not bare -- see module docstring / MODAL_SETUP.md changelog:
        # this exact "numpy 2.x breaks the audio-loading chain" bug already
        # hit both DeSTA2.5 (np.sctypes removed) and SALMONN (torch's numpy
        # C-API binding). Pinned early so later steps' `numpy>=...` deps are
        # satisfied by this version rather than upgraded.
        "numpy<2",
    )
    # Installed LAST, after everything else that could touch huggingface_hub
    # -- see module docstring for why, even though this image (unlike
    # DeSTA2.5's) has no editable install step that's known to strip it.
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_qwen2audio_eval  # noqa: E402

    return run_qwen2audio_eval


@app.function(image=dry_run_image, timeout=600)
def audit_only(argv: list) -> int:
    script = _import_script()
    return script.main(argv)


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/vol": volume},
)
def run_eval(argv: list) -> int:
    os.makedirs("/vol/hf_cache", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")

    script = _import_script()
    code = script.main(argv)
    volume.commit()
    return code


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    experiments: str = "all",
    limit: int = None,
    seed: int = 0,
    max_new_tokens: int = 64,
    model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
):
    argv = [
        "--experiments", *experiments.split(),
        "--input", f"{CONTAINER_ROOT}/data/combined_examples.json",
        "--output-dir", "/vol/results",
        "--model-path", model_path,
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]

    if dry_run:
        print(">>> Qwen2-Audio dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    else:
        print(">>> Qwen2-Audio full run: launching on A100-40GB. This is billed GPU time.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
