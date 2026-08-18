#!/usr/bin/env python3
"""
Modal wrapper for eval/run_phi4mm_eval.py (microsoft/Phi-4-multimodal-instruct).

Runs the existing, unmodified eval/run_phi4mm_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage. All dataset/scoring/model-loading
logic lives in eval/run_phi4mm_eval.py -- see that file's docstring for the
"why" behind its dependency pins and the audio-vs-text_only prompt format.

Phi-4 gets its own dedicated pinned image, NOT the shared environment used by
modal_qwen2audio_eval.py/modal_desta25_eval.py: transformers==4.48.2 +
peft==0.13.2 (required below) would silently break DeSTA2.5's
transformers>=4.49.0 floor if installed into that shared image. Same reason
modal_salmonn_eval.py has its own image.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the full walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your HF token>

  microsoft/Phi-4-multimodal-instruct is a public, ungated model -- HF_TOKEN
  isn't required for it specifically, but the secret is reused here (same as
  the Qwen2-Audio wrapper) for consistency and to avoid anonymous-download
  rate limits. No github-token secret is needed: this image clones nothing.

Usage:
  modal run modal/modal_phi4mm_eval.py --dry-run
  modal run modal/modal_phi4mm_eval.py --experiments all --limit 5
  modal run modal/modal_phi4mm_eval.py --experiments emotional_prosody --limit 20
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with modal_qwen2audio_eval.py / modal_desta25_eval.py / modal_salmonn_eval.py
SCRIPT_NAME = "run_phi4mm_eval.py"

app = modal.App("phi4mm-eval")

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
# stdlib + (optionally) numpy -- see run_phi4mm_eval.py's main(), the
# `import torch` / model-loading calls happen after the dry-run early-return.
# Keeping this separate means a dataset problem surfaces in seconds on CPU,
# not after a multi-minute GPU container boots with Phi-4's pinned-legacy
# dependency set below.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run. Pins mirror run_phi4mm_eval.py's own docstring
# exactly (transformers==4.48.2 / peft==0.13.2 / backoff) -- these are NOT
# guesses, they were confirmed necessary by a real local repro: Phi-4's
# custom modeling_phi4mm.py wires up vision/speech LoRA adapters via
# peft.get_peft_model() at __init__, and that breaks under transformers>=4.50
# (PreTrainedModel stopped inheriting GenerationMixin by default) with
#   AttributeError: 'Phi4MMModel' object has no attribute 'prepare_inputs_for_generation'
# torch==2.6.0 / transformers==4.48.2 / peft==0.13.2 / accelerate==1.3.0 /
# soundfile==0.13.1 / pillow==11.1.0 / scipy==1.15.2 / torchvision==0.21.0 /
# backoff==2.2.1 is a community-confirmed working combination for this exact
# model, not an arbitrary pin set.
gpu_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        # Phi-4's vision LoRA branch is constructed unconditionally at model
        # __init__ (even for audio-only inference) -- torchvision satisfies
        # its image-processing import path so that constructor doesn't fail
        # before generation-side code is ever reached.
        "torchvision==0.21.0",
        # NOT >=4.49 like DeSTA2.5, NOT unpinned like Qwen2-Audio -- this is
        # the one version confirmed locally to keep PreTrainedModel inheriting
        # GenerationMixin, which Phi-4's remote-code LoRA wiring depends on.
        # See eval/run_phi4mm_eval.py's docstring for the exact traceback this
        # avoids.
        "transformers==4.48.2",
        # NOT the newer default (0.19.1 broke this model locally, see above).
        "peft==0.13.2",
        "accelerate==1.3.0",
        "soundfile==0.13.1",
        "pillow==11.1.0",
        "scipy==1.15.2",
        # Required by Phi-4's custom modeling code (imports backoff directly)
        # -- missing this raises ImportError before the model ever loads, and
        # it's easy to miss because none of the other three eval scripts in
        # this repo need it.
        "backoff==2.2.1",
        # <2, not bare -- same class of bug documented in the other three
        # wrappers' images (numpy 2.x breaking the audio-loading chain).
        # Phi-4's inference path here uses soundfile directly, not librosa,
        # so this is cheap defensive insurance rather than a confirmed local
        # repro for this specific model, unlike the pins above it.
        "numpy<2",
    )
    # Deliberately NOT installing flash-attn: no prebuilt wheel was available
    # for local testing (RTX 5070 Ti, sm_120), so eager attention is what was
    # actually validated end-to-end, and that's this image's default (see
    # --attn-implementation in run_phi4mm_eval.py, defaults to "eager").
    # flash-attn's pip install falls back to a from-source build when no
    # prebuilt wheel matches the exact torch/cuda/python combination, which
    # is slow and not something to trigger sight-unseen -- if you want to try
    # flash_attention_2 on Modal's A100 image, confirm a matching wheel
    # exists first, then add it here explicitly.
    #
    # Installed LAST, after everything else that could touch huggingface_hub
    # -- same defensive ordering as the other three wrappers' images.
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_phi4mm_eval  # noqa: E402

    return run_phi4mm_eval


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
    model_path: str = "microsoft/Phi-4-multimodal-instruct",
    attn_implementation: str = "eager",
):
    argv = [
        "--experiments", *experiments.split(),
        "--input", f"{CONTAINER_ROOT}/data/combined_examples.json",
        "--output-dir", "/vol/results",
        "--model-path", model_path,
        "--attn-implementation", attn_implementation,
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]

    if dry_run:
        print(">>> Phi-4-multimodal dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    else:
        print(">>> Phi-4-multimodal full run: launching on A100-40GB. This is billed GPU time.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
