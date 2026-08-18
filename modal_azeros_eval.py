#!/usr/bin/env python3
"""
Modal wrapper for eval/run_azeros_eval.py (AudenAI/azeros).

Runs the existing, unmodified eval/run_azeros_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage, and the auden/k2 install this model
needs. All dataset/scoring/model-loading logic lives in eval/run_azeros_eval.py
-- see that file's docstring for the "why" behind its dependency pins and the
text_only limitation.

AZeroS gets its own dedicated pinned image, NOT the shared environment used
by modal_qwen2audio_eval.py/modal_desta25_eval.py, and NOT the isolated venv
pattern used for Phi-4 either: this is the ONLY model in the lineup that
can't run locally at all, on any OS this project has access to -- its
Zipformer encoder hard-imports k2 (auden/models/zipformer/modules/scaling.py),
and k2 ships no Windows wheels, only "Pre-compiled CUDA wheels (Linux)". Linux
is exactly what a Modal container is, so this is where AZeroS gets its first
real test run.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the full walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your HF token>

  AudenAI/azeros is a public, ungated model -- HF_TOKEN isn't required for it
  specifically, but the secret is reused here for consistency and to avoid
  anonymous-download rate limits, same as the other wrappers.

Usage:
  modal run modal_azeros_eval.py --dry-run          # CPU dataset audit, no GPU
  modal run modal_azeros_eval.py --sanity-check      # 5-example smoke test on GPU, stop there
  modal run modal_azeros_eval.py --experiments all --limit 5
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with the other four wrappers: results/ + HF cache
SCRIPT_NAME = "run_azeros_eval.py"

AUDEN_REPO_DIR = "/opt/Auden"
AUDEN_EXAMPLES_DIR = f"{AUDEN_REPO_DIR}/examples/azeros"

app = modal.App("azeros-eval")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _with_repo_data(image: modal.Image) -> modal.Image:
    """Mount the eval script + dataset from the local repo. Kept as the last
    layer on every image per Modal's guidance: local-file mounts should not
    sit underneath layers that rarely change, so editing data/ or the eval
    script never invalidates the (slow) pip/git layers above it."""
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
# stdlib + (optionally) numpy -- see run_azeros_eval.py's main(), the
# `import torch` / `from model import AzerosModel` calls happen after the
# dry-run early-return. Keeping this separate means a dataset problem
# surfaces in seconds on CPU, not after a multi-minute GPU container boots
# with the auden/k2 install below.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run (sanity-check or full). Pins mirror
# run_azeros_eval.py's own docstring exactly -- torch==2.4.1 chosen
# specifically because it has a verified k2 wheel (see below), not picked
# arbitrarily.
gpu_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        # <2, not bare -- same class of numpy/audio-chain bug documented in
        # every other wrapper's image in this repo.
        "numpy<2",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        # >=4.40 per auden's requirements.txt floor, <5 because transformers
        # 5.x requires torch>=2.5 -- left unpinned above, pip resolved
        # transformers==5.15.0, which silently disabled its own PyTorch
        # backend against torch==2.4.1 ("PyTorch was not found") and left
        # LRScheduler undefined in trainer_pt_utils.py (NameError at import
        # time, inside examples/azeros/model.py's own `from
        # transformers.trainer_pt_utils import LabelSmoother`). Same failure
        # class already documented in modal_qwen2audio_eval.py's image --
        # should have applied the same cap here from the start.
        "transformers>=4.40,<5",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4",
        "PyYAML>=6.0",
        "huggingface_hub>=0.20",
        "six>=1.16.0",
        "tensorboard>=2.12",
        "kaldialign>=0.7",
        "lhotse>=1.20",
        "tqdm>=4.65",
        "soundfile>=0.12",
    )
    # k2 is a HARD, unconditional import-time dependency of the Zipformer
    # encoder (see run_azeros_eval.py's docstring for the exact traceback
    # this avoids) -- not optional, not just for training. This exact
    # version+torch+cuda combination is verified to exist in k2's own wheel
    # index (confirmed against pip's own resolver error listing every real
    # published version, after an initial off-by-one-day guess -- 0715 vs
    # the real 0714 -- failed to resolve): k2-fsa publishes "Pre-compiled
    # CUDA wheels (Linux)" only (no Windows wheels at all, which is why this
    # model never ran locally), and 1.24.4.dev20250714+cuda12.1.torch2.4.1
    # is a real, listed build for torch==2.4.1 + cuda12.1 (PyPI's default
    # torch==2.4.1 wheel bundles cu121).
    .pip_install(
        "k2==1.24.4.dev20250714+cuda12.1.torch2.4.1",
        find_links="https://k2-fsa.github.io/k2/cuda.html",
    )
    # k2's compiled CUDA extension (_k2*.so) needs CUDA runtime .so files --
    # libnvrtc.so.12 in particular -- discoverable via the dynamic linker at
    # import time. torch's own pip-installed nvidia-*-cu12 dependencies
    # bundle those exact files, but only under site-packages, not the system
    # linker path, so `import k2` fails with "ImportError: libnvrtc.so.12:
    # cannot open shared object file" even though the library is right there
    # on disk. Setting this via os.environ at script runtime (before `import
    # k2`) did NOT fix it -- confirmed by testing -- so it's baked into the
    # image itself instead, present in the environment from container start.
    # This is also what was silently masking the real error as an
    # "Unrecognized model identifier: zipformer" ValueError: auden's
    # model_config.py catches ANY exception from its own AutoConfig.for_model
    # and falls back to HF's AutoConfig (which has never heard of
    # "zipformer"), so the missing-library ImportError never surfaced until
    # traced by hand.
    .env({
        "LD_LIBRARY_PATH": ":".join([
            "/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cuda_cupti/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/curand/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib",
            "/usr/local/lib/python3.11/site-packages/nvidia/nvtx/lib",
        ])
    })
    # The `auden` package itself -- not on PyPI, must be installed from
    # source. examples/azeros/{model.py,model_config.py,utils.py} (the
    # actual AzerosModel driver code run_azeros_eval.py imports) live
    # alongside the package, not inside it -- AUDEN_EXAMPLES_DIR below is
    # added to sys.path at runtime by load_model(), not installed as part
    # of this pip step.
    .run_commands(
        f"git clone https://github.com/AudenAI/Auden.git {AUDEN_REPO_DIR}",
        f"pip install -e {AUDEN_REPO_DIR}",
    )
    # Installed LAST, after everything else that could touch huggingface_hub
    # -- same defensive ordering as every other wrapper's image in this repo.
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_azeros_eval  # noqa: E402

    return run_azeros_eval


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
    sanity_check: bool = False,
    experiments: str = "all",
    limit: int = None,
    seed: int = 0,
    max_new_tokens: int = 64,
    model_path: str = "AudenAI/azeros",
):
    argv = [
        "--experiments", *experiments.split(),
        "--input", f"{CONTAINER_ROOT}/data/combined_examples.json",
        "--output-dir", "/vol/results",
        "--model-path", model_path,
        "--examples-dir", AUDEN_EXAMPLES_DIR,
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]

    if dry_run:
        print(">>> AZeroS dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    elif sanity_check:
        print(">>> AZeroS sanity check: launching on A100-40GB for a 5-example smoke "
              "test only (no checkpointing, no full run). This is billed GPU time.\n")
        code = run_eval.remote(argv + ["--sanity-check"])
    else:
        print(">>> AZeroS full run: launching on A100-40GB. This is billed GPU time.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
