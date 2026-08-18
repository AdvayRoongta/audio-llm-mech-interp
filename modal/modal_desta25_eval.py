#!/usr/bin/env python3
"""
Modal wrapper for eval/run_desta25_eval.py (DeSTA2.5-Audio-Llama-3.1-8B).

Runs the existing, unmodified eval/run_desta25_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage. All dataset/scoring/model-loading
logic lives in eval/run_desta25_eval.py -- see that file's docstring for
the "why" behind its precision (bfloat16, not fp16) and text_only handling.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the full walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your HF token>

  DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B wraps DeSTA-ntu/Llama-3.1-8B-Instruct
  (a Llama-3.1 redistribution); HF_TOKEN is required if that repo is gated
  for your account. A github-token secret is NOT needed here -- both repos
  this script touches (kehanlu/DeSTA2.5-Audio on GitHub, the two HF repos
  above) are public.

Usage:
  modal run modal/modal_desta25_eval.py --dry-run
  modal run modal/modal_desta25_eval.py --experiments all --limit 5
  modal run modal/modal_desta25_eval.py --experiments emotional_prosody --limit 20 --gpu A100-40GB
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with modal_salmonn_eval.py: results/ + HF cache
SCRIPT_NAME = "run_desta25_eval.py"

app = modal.App("desta25-eval")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _with_repo_data(image: modal.Image) -> modal.Image:
    """Mount the eval script + dataset from the local repo. Kept as the last
    layer on every image per Modal's guidance: local-file mounts should not
    sit underneath layers that rarely change, so editing data/ or the eval
    script never invalidates the (slow) pip/apt layers above it."""
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
# stdlib + (optionally) numpy -- see run_desta25_eval.py's main(), the
# `import desta` / `import torch` calls happen after the dry-run early-return.
# Keeping this separate means a dataset problem surfaces in seconds on CPU,
# not after a multi-minute GPU container boots.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run: desta package (pip-installable, not a sys.path
# hack -- see the script's docstring) plus its transformers>=4.49 floor.
gpu_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers>=4.49.0",
        "accelerate",
        "soundfile",
        "librosa",
        # <2, not bare: something in the librosa/resampy/numba audio-loading
        # chain still calls the pre-2.0 np.sctypes API (removed in NumPy 2.0),
        # which crashed every audio-bearing item ("np.sctypes was removed")
        # while text_only items (no audio load) were unaffected. Pinned here,
        # before desta's install, so desta's own `numpy>=1.17` requirement is
        # satisfied by this version and left alone (same as the other already-
        # satisfied packages in its install log) rather than silently upgraded.
        "numpy<2",
    )
    .run_commands(
        "git clone https://github.com/kehanlu/DeSTA2.5-Audio.git /opt/DeSTA2.5-Audio",
        "pip install -e /opt/DeSTA2.5-Audio",
    )
    # Installed AFTER desta's own editable install, not before: desta's
    # setup.py re-pins huggingface_hub to an exact version with no [hf_transfer]
    # extra, which silently drops the hf_transfer package if it was only
    # installed earlier as that extra's dependency -- HF_HUB_ENABLE_HF_TRANSFER=1
    # would then crash on a ModuleNotFoundError at download time. Installing it
    # last guarantees it survives whatever huggingface_hub version desta wants.
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_desta25_eval  # noqa: E402

    return run_desta25_eval


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
    model_path: str = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
    gpu: str = "A100-40GB",
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
        print(">>> DeSTA2.5 dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    else:
        if gpu != "A100-40GB":
            print(">>> NOTE: --gpu is currently fixed to A100-40GB in the deployed function; "
                  "edit modal_desta25_eval.py's gpu_image function decorator to change it.")
        print(">>> DeSTA2.5 full run: launching on A100. This is billed GPU time.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
