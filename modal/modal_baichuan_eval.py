#!/usr/bin/env python3
"""
Modal wrapper for eval/run_baichuan_eval.py (baichuan-inc/Baichuan-Omni-1d5).

Runs the existing, unmodified eval/run_baichuan_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage. All dataset/scoring/model-loading
logic lives in eval/run_baichuan_eval.py -- see that file's docstring for
the "why" behind trust_remote_code, the bind_processor() call, and the
inline-JSON audio reference format.

Initially expected to need no isolated venv and no compiled CUDA kernels
(plain debian_slim + pip, like modal_qwen2audio_eval.py / modal_desta25_eval.py)
-- that assumption did NOT survive real smoke-testing. Baichuan-Omni-1.5's
own trust_remote_code modeling file unconditionally reaches `import
flash_attn` just to load the model CLASS (not only on the TTS/
speech-generation path this project doesn't use), and no prebuilt flash_attn
wheel tried matched this environment's actual torch ABI (two variants
failed with different real `undefined symbol` errors, confirmed
deterministic, not flaky). This image therefore compiles flash_attn from
source against the exact installed torch, same CUDA-devel-base approach
modal_audex_eval.py uses for mamba-ssm/causal-conv1d -- see the gpu_image
comment below for the full story. Its own reference requirements.txt pins
transformers==4.45.0 / numpy==1.26.3 / librosa==0.10.2.post1 /
soundfile==0.12.1; those are reused here verbatim rather than guessed.
deepspeed/diffusers/easydict/einops/omegaconf are also required just to
import the model class (confirmed one at a time via real failures, not
preemptive guesses) -- deepspeed's own CUDA_HOME compatibility check is
disabled via its documented inference-only env vars rather than requiring a
full CUDA toolkit just to satisfy an import this project never exercises
functionally.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the general walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your Hugging Face token>

  baichuan-inc/Baichuan-Omni-1d5 is public on HF, so hf-token isn't
  strictly required for this model specifically, but it's reused here for
  consistency and to avoid anonymous-download rate limits, same as every
  other wrapper in this repo.

Usage:
  modal run modal/modal_baichuan_eval.py --dry-run
  modal run modal/modal_baichuan_eval.py --experiments all --limit 5
  modal run modal/modal_baichuan_eval.py --experiments emotional_prosody --limit 20
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with every other wrapper: results/ + HF cache
SCRIPT_NAME = "run_baichuan_eval.py"

app = modal.App("baichuan-eval")

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
# stdlib + numpy -- see run_baichuan_eval.py's main(), the `import torch` /
# model-loading calls happen after the dry-run early-return.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run.
#
# NOT the simple debian_slim + pip pattern this file started with. Real
# smoke-test attempts established that Baichuan-Omni-1.5's own
# trust_remote_code modeling file (modeling_omni.py) unconditionally reaches
# `import flash_attn` just to load the model CLASS -- not only on the
# TTS/audio-generation path this project doesn't use -- contradicting the
# initial "no compiled CUDA kernels needed" assumption. Two prebuilt
# flash_attn wheels (cu12torch2.6, cxx11abiFALSE and cxx11abiTRUE) were tried
# against torch==2.6.0+cu124 and BOTH failed with different real ABI
# mismatches at import time (`undefined symbol`), confirmed deterministic
# across repeated identical runs, not a fluke. Rather than keep guessing
# prebuilt wheel variants, this image compiles flash_attn from source
# against the exact installed torch -- same proven approach
# modal_audex_eval.py uses for mamba-ssm/causal-conv1d -- which guarantees
# ABI correctness at the cost of the "no special CUDA build" simplicity this
# file originally aimed for. torch pinned to the same version already used
# elsewhere in this repo's Modal environment (modal_phi4mm_eval.py).
gpu_image = _with_repo_data(
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "llvm", "git", "build-essential", "ninja-build")
    # torch's cpp_extension build probes for clang++ on this base image
    # instead of falling back to g++/build-essential -- confirmed by a real
    # build failure here, same issue modal_audex_eval.py hit on the same
    # nvidia/cuda base image family.
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # flash_attn's setup.py needs these at metadata-generation/build
        # time -- confirmed by real build failures, not a preemptive guess
        # (`ModuleNotFoundError: No module named 'packaging'`, and
        # wheel/setuptools for --no-build-isolation below, same class of
        # issue modal_audex_eval.py hit with causal-conv1d/mamba-ssm).
        "packaging", "wheel", "setuptools", "ninja",
    )
    # flash_attn==2.6.3, matching baichuan_omni_requirements.txt's own pin
    # exactly (unlike the abandoned prebuilt-wheel attempt, source builds
    # against whatever version is requested, so there's no reason to deviate
    # from the upstream pin here). --no-build-isolation so it links against
    # the torch already installed above, not a fresh isolated-venv torch.
    # MAX_JOBS caps parallel compile jobs to a value Modal's build workers
    # can actually deliver; flash_attn's own setup.py otherwise infers a very
    # high job count from host CPU count and can OOM the build.
    #
    # Deliberately its OWN layer, immediately after torch and before every
    # other pip_install below: this is by far the slowest step in this image
    # (a real from-source CUDA kernel compile), and several more ordinary
    # Python packages were discovered missing one at a time while
    # smoke-testing (see below) -- each one invalidated every layer after it,
    # forcing a full flash_attn recompile for what should have been a
    # 10-second fix. Keeping flash_attn early and everything else after it
    # means later additions only rebuild cheap layers.
    .env({"MAX_JOBS": "4"})
    .pip_install(
        "flash_attn==2.6.3",
        extra_options="--no-build-isolation",
    )
    # A later, separate apt_install layer (not merged into the one at the top
    # of this image) -- deliberately so a native dep discovered late in
    # smoke-testing (libcairo2, for the cairosvg pip package below) doesn't
    # invalidate and force a recompile of the expensive flash_attn layer
    # directly above.
    .apt_install("libcairo2")
    .pip_install(
        # exact pins from baichuan-inc/Baichuan-Omni-1.5's own
        # baichuan_omni_requirements.txt
        "transformers==4.45.0",
        "numpy==1.26.3",
        "librosa==0.10.2.post1",
        "soundfile==0.12.1",
        # required for device_map -- not in the upstream requirements.txt,
        # but the reference demo's own load_model_tokenizer() uses
        # device_map='cuda', which needs this.
        "accelerate",
        # required by the model's own trust_remote_code modeling file --
        # confirmed by a real from_pretrained() failure here, one at a time,
        # not a preemptive guess (transformers' check_imports() walks the
        # dynamic-module import graph and fails on the first missing package
        # it finds, so this list grew incrementally across several retries):
        "easydict",
        "einops",
        "omegaconf",
        # deepspeed: also confirmed required just to IMPORT the model class
        # (not only for the TTS/audio-generation path this project doesn't
        # use) -- modeling_omni.py's import graph reaches it unconditionally.
        "deepspeed==0.14.4",
        "diffusers",
        # required by audio_modeling_omni.py's own `import torch, fire` --
        # confirmed the same way as the packages above.
        "fire",
        "ujson",
        # required by visual_modeling_omni.py's video/image-decoding path --
        # this project only uses text/audio modalities, but
        # modeling_omni.py's import graph reaches these unconditionally
        # regardless.
        "decord",
        "opencv-python-headless",
        "av",
        "imagesize",
        "cairosvg",
    )
    # huggingface_hub[hf_transfer] extras syntax did not reliably pull in
    # the hf_transfer package here -- confirmed by a real download failure
    # ("Fast download using 'hf_transfer' is enabled ... but 'hf_transfer'
    # package is not available"). Installed explicitly instead.
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # deepspeed's own __init__ eagerly checks CUDA_HOME at import time to
    # decide whether it can JIT-compile its custom ops -- confirmed by a
    # real import failure here (`MissingCUDAException: CUDA_HOME does not
    # exist`). This project only needs `import deepspeed` to succeed (it's
    # pulled in by modeling_omni.py's import graph, not actually used for
    # training/distributed ops here), so DeepSpeed's own documented
    # inference-only env vars skip that compatibility check rather than
    # requiring a full CUDA-devel toolkit just to satisfy an import.
    .env({"DS_BUILD_OPS": "0", "DS_SKIP_CUDA_CHECK": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_baichuan_eval  # noqa: E402

    return run_baichuan_eval


@app.function(image=dry_run_image, timeout=600)
def audit_only(argv: list) -> int:
    script = _import_script()
    return script.main(argv)


def _resolve_baichuan_local_path(model_repo_id: str) -> str:
    """Download the full model repo to a local directory and return that
    path, instead of handing from_pretrained() a bare HF repo id.

    Confirmed necessary by a real, repeatable failure: loading directly from
    the bare repo id, transformers' dynamic-module check_imports() reports
    the repo's own co-located vector_quantize.py (a real file in this repo,
    not a PyPI package -- confirmed via a PyPI 404) as a "missing package",
    apparently because that sibling file hasn't been fetched yet at the
    point check_imports() runs for whichever file references it. Same root
    cause class, same fix, as modal_audex_eval.py's checkpoint-folder
    download: when every file already exists on local disk in one
    directory, transformers' dynamic-module resolution finds sibling
    imports directly rather than depending on hub-fetch ordering.
    """
    from huggingface_hub import snapshot_download

    print(">>> resolving Baichuan-Omni-1.5 from {} (cached under /vol/hf_cache "
          "after the first run; ~22.5GB on a cold cache)".format(model_repo_id))
    local_path = snapshot_download(repo_id=model_repo_id, cache_dir="/vol/hf_cache")
    print(">>> local path: {}".format(local_path))
    return local_path


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    # 6h (the original value here) was not generous enough -- a real full
    # 3-category/3594-item run measured at ~10.6s/item (emotional_prosody:
    # 1197 records in 12736.6s) needs ~10.6h end to end, so the 6h cap killed
    # a real in-progress run via FunctionTimeoutError partway through
    # speaker_identity, not any code bug (0 correctness=="error" records at
    # the point of the kill). Checkpoint/resume already made that recoverable
    # with no lost work, but the timeout itself needed raising so a full run
    # (or a full rerun from an empty checkpoint) can actually complete in one
    # function call. 14h gives real margin above the ~10.6h measured cost.
    timeout=14 * 60 * 60,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/vol": volume},
)
def run_eval(argv: list, model_repo_id: str) -> int:
    os.makedirs("/vol/hf_cache", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")

    local_model_path = _resolve_baichuan_local_path(model_repo_id)
    argv = argv + ["--model-path", local_model_path]

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
    max_new_tokens: int = 200,
    model_path: str = "baichuan-inc/Baichuan-Omni-1d5",
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
        print(">>> Baichuan-Omni-1.5 dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--model-path", model_path, "--dry-run"])
    else:
        print(">>> Baichuan-Omni-1.5 full run: launching on A100-40GB. This is billed GPU time.\n")
        code = run_eval.remote(argv, model_path)

    if code:
        raise SystemExit(code)
