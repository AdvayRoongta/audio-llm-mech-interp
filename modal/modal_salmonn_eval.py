#!/usr/bin/env python3
"""
Modal wrapper for eval/run_salmonn_eval.py (SALMONN-13B).

Runs the existing, unmodified eval/run_salmonn_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage, and locating SALMONN's 4 artifacts.
All dataset/scoring/model-loading logic lives in eval/run_salmonn_eval.py --
see that file's docstring for the "why" behind fp16/low_resource=False and
the silent-WAV text_only handling.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the full walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your HF token>

  Vicuna-13B-v1.1 (lmsys/vicuna-13b-v1.1) and the SALMONN checkpoint
  (tsinghua-ee/SALMONN) are downloaded from HF at run time; HF_TOKEN must be
  set if either is gated for your account. A github-token secret is NOT
  needed here -- bytedance/SALMONN (salmonn branch) is a public repo.

  SALMONN's fine-tuned BEATs_iter3+ (AS2M) checkpoint is hosted on OneDrive,
  not HuggingFace or git (see the OneDrive link inside
  eval/run_salmonn_eval.py's own docstring), so it cannot be fetched inside
  the container automatically. Download it yourself, then upload it once to
  the shared Modal volume:

    modal volume put qwen2audio-results <local-path-to-beats.pt> salmonn_artifacts/beats.pt

  The GPU function checks for that file at startup and fails with a clear
  message (not a confusing downstream traceback) if it's missing.

Usage:
  modal run modal/modal_salmonn_eval.py --dry-run
  modal run modal/modal_salmonn_eval.py --experiments all --limit 5
  modal run modal/modal_salmonn_eval.py --experiments emotional_prosody --limit 20
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with modal_desta25_eval.py: results/ + HF cache + salmonn_artifacts/
SCRIPT_NAME = "run_salmonn_eval.py"

SALMONN_REPO_DIR = "/opt/SALMONN"
WHISPER_REPO_ID = "openai/whisper-large-v2"
VICUNA_REPO_ID = "lmsys/vicuna-13b-v1.1"
SALMONN_CKPT_REPO_ID = "tsinghua-ee/SALMONN"
SALMONN_CKPT_FILENAME = "salmonn_v1.pth"
BEATS_VOLUME_PATH = "/vol/salmonn_artifacts/beats.pt"

app = modal.App("salmonn-eval")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _with_repo_data(image: modal.Image) -> modal.Image:
    """Mount the eval script + dataset from the local repo. Kept as the last
    layer on every image per Modal's guidance: local-file mounts should not
    sit underneath layers that rarely change, so editing data/ or the eval
    script never invalidates the (slow) pip/apt/git layers above it."""
    return (
        image
        .add_local_dir(str(LOCAL_ROOT / "data"), remote_path=f"{CONTAINER_ROOT}/data")
        .add_local_dir(str(LOCAL_ROOT / "dataset_audio"), remote_path=f"{CONTAINER_ROOT}/dataset_audio")
        .add_local_file(
            str(LOCAL_ROOT / "eval" / SCRIPT_NAME),
            remote_path=f"{CONTAINER_ROOT}/eval/{SCRIPT_NAME}",
        )
    )


# Lightweight image for --dry-run: the script's dry-run path is pure stdlib
# (plus optional numpy) -- see run_salmonn_eval.py's main(), `import torch`
# and the SALMONN/transformers/peft pins only get touched inside load_model(),
# which dry-run never calls. Deliberately NOT the pinned legacy image below:
# a dataset problem should surface in seconds on CPU, without first waiting
# on (and risking a break in) SALMONN's fragile torch==2.0.1 build.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.10").pip_install("numpy")
)

# Full image for a real run. Pins mirror run_salmonn_eval.py's own docstring
# exactly (torch==2.0.1 / transformers==4.28.0 / peft==0.3.0 / etc.) -- this
# script does not run in the same environment as the DeSTA2.5/Qwen2-Audio
# scripts, by SALMONN's own requirement (vendored modeling_llama.py /
# modeling_whisper.py from that transformers era).
gpu_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.0.1",
        "torchaudio==2.0.2",
        # <2, not bare: torch==2.0.1 predates NumPy 2.0 and was compiled
        # against its old C ABI. Left unpinned, later steps (librosa) pull
        # in the newest NumPy 2.x, which breaks that ABI ("Failed to
        # initialize NumPy: _ARRAY_API not found") and corrupts every
        # downstream numpy<->torch tensor conversion -- this is what actually
        # caused the "padding=True" and "sampling rate 16000...not 48000"
        # errors seen on every item, not a real audio-pipeline bug. Pinned
        # here (before librosa/etc.) so their `numpy>=...` requirements are
        # satisfied by this version and left alone. Same class of bug as
        # modal_desta25_eval.py's numpy<2 pin -- see that file's comment.
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
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        f"git clone -b salmonn https://github.com/bytedance/SALMONN.git {SALMONN_REPO_DIR}",
    )
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_salmonn_eval  # noqa: E402

    return run_salmonn_eval


@app.function(image=dry_run_image, timeout=600)
def audit_only(argv: list) -> int:
    script = _import_script()
    return script.main(argv)


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/vol": volume},
)
def run_eval(argv: list) -> int:
    from huggingface_hub import hf_hub_download

    os.makedirs("/vol/hf_cache", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")

    if not os.path.isfile(BEATS_VOLUME_PATH):
        raise SystemExit(
            "Missing {}. Download SALMONN's fine-tuned BEATs checkpoint from the "
            "OneDrive link in eval/run_salmonn_eval.py's docstring, then run:\n"
            "  modal volume put {} <local-path-to-beats.pt> salmonn_artifacts/beats.pt".format(
                BEATS_VOLUME_PATH, VOLUME_NAME
            )
        )

    ckpt_path = hf_hub_download(
        repo_id=SALMONN_CKPT_REPO_ID,
        filename=SALMONN_CKPT_FILENAME,
        cache_dir="/vol/hf_cache",
    )

    argv = argv + [
        "--salmonn-repo", SALMONN_REPO_DIR,
        "--whisper-path", WHISPER_REPO_ID,
        "--llama-path", VICUNA_REPO_ID,
        "--beats-path", BEATS_VOLUME_PATH,
        "--ckpt", ckpt_path,
    ]

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
):
    argv = [
        "--experiments", *experiments.split(),
        "--input", f"{CONTAINER_ROOT}/data/combined_examples.json",
        "--output-dir", "/vol/results",
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]

    if dry_run:
        print(">>> SALMONN dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    else:
        print(">>> SALMONN full run: launching on A100-80GB. This is billed GPU time, "
              "and will download Vicuna-13B + Whisper + the SALMONN checkpoint on first run.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
