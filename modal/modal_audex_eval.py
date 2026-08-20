#!/usr/bin/env python3
"""
Modal wrapper for eval/run_audex_eval.py (nvidia/Nemotron-Labs-Audex-30B-A3B).

Runs the existing, unmodified eval/run_audex_eval.py inside a Modal
container. This file only handles environment/infra: image build, GPU
selection, secrets, persistent storage, and resolving Audex's two
HF-repo-hosted artifacts (the checkpoint folder and NVIDIA's own inference
helper scripts). All dataset/scoring/model-loading logic lives in
eval/run_audex_eval.py -- see that file's docstring for the "why" behind the
native-backend/bf16 choice and the text_only limitation.

Audex gets its own dedicated image, isolated from every other wrapper in
this repo, same reasoning as modal_phi4mm_eval.py's and modal_salmonn_eval.py's
isolation: transformers==5.14.0 here would break Phi-4-multimodal's
transformers==4.48.2 pin (and vice versa) if they ever shared an
environment. Audex additionally needs a CUDA *devel* base image (not the
debian_slim used by every other wrapper) because causal-conv1d/mamba-ssm
compile CUDA extensions at install time via --no-build-isolation -- that
requires nvcc and CUDA headers to be present in the image, which a runtime-only
CUDA image does not provide.

--------------------------------------------------------------------------- #
Setup (see MODAL_SETUP.md for the general walkthrough)
--------------------------------------------------------------------------- #

  pip install modal
  python3 -m modal setup
  modal secret create hf-token HF_TOKEN=<your Hugging Face token>

  nvidia/Nemotron-Labs-Audex-30B-A3B is public on HF (NVIDIA OneWay
  Noncommercial License, not a gated repo), so hf-token isn't strictly
  required for this model specifically, but it's reused here for consistency
  and to avoid anonymous-download rate limits on a ~60GB checkpoint pull,
  same as every other wrapper in this repo.

Two artifacts are pulled from the model's HF repo at run time (both cached
on the shared volume under /vol/hf_cache, so this only happens once):

  1. checkpoint_folder_full/ -- the actual model (config, tokenizer,
     modeling_nemotron_h_audio_native.py, ~13 safetensors shards + audio
     weights, audio_preprocessor/). This is the *audio QA/understanding*
     checkpoint; checkpoint_folder_audiogen/ (TTS) and checkpoint_folder_textonly/
     are deliberately NOT downloaded -- this project only needs audio-in/
     text-out inference. ~60GB, confirmed self-contained (no references
     outside its own folder) via the repo's own file listing.
  2. inference_scripts_hf/{inference_hf.py,audio_utils.py} -- NVIDIA's own
     reference inference code, imported directly by run_audex_eval.py
     rather than reimplemented. See that script's module docstring for why.

Usage:
  modal run modal/modal_audex_eval.py --dry-run
  modal run modal/modal_audex_eval.py --experiments all --limit 5
  modal run modal/modal_audex_eval.py --experiments emotional_prosody --limit 20
  # rerun only text_only items (e.g. after a fix to the text-only path):
  modal run modal/modal_audex_eval.py --experiments emotional_prosody --modality-filter text_only --limit 5
"""

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).parent.parent
CONTAINER_ROOT = "/repo"
VOLUME_NAME = "qwen2audio-results"  # shared with every other wrapper: results/ + HF cache
SCRIPT_NAME = "run_audex_eval.py"

MODEL_REPO_ID = "nvidia/Nemotron-Labs-Audex-30B-A3B"
CHECKPOINT_SUBDIR = "checkpoint_folder_full"
INFERENCE_UTILS_SUBDIR = "inference_scripts_hf"
INFERENCE_UTILS_FILES = ["inference_hf.py", "audio_utils.py", "__init__.py"]

# text_only items use Audex's dedicated text-only entry point -- a genuinely
# separate model/code path (Nemotron-Cascade-2's plain text backbone,
# NemotronHForCausalLM), not the audio-conditioned checkpoint above with
# audio omitted. Its own checkpoint_folder_textonly in the HF repo does NOT
# contain the actual shard weight files though -- only its own small
# config/tokenizer/index. Confirmed by diffing its model.safetensors.index.json
# against checkpoint_folder_full's: every one of its 6243 tensor keys maps to
# the identical shard filename in both, so those shards are the same bytes
# already being downloaded above for checkpoint_folder_full (the 490 extra
# tensors only in checkpoint_folder_full -- audio_encoder.*, etc. -- live in
# the separate model-audio.safetensors, which checkpoint_folder_textonly's
# index never references). So this only needs a small extra metadata
# download plus symlinks into the shards already resolved by
# _resolve_audex_artifacts() -- no second ~60GB download.
TEXTONLY_SUBDIR = "checkpoint_folder_textonly"
TEXTONLY_META_FILES = [
    "config.json",
    "configuration_nemotron_h.py",
    "modeling_nemotron_h.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

app = modal.App("audex-eval")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _with_repo_data(image: modal.Image) -> modal.Image:
    """Mount the eval script + dataset from the local repo. Kept as the last
    layer on every image per Modal's guidance: local-file mounts should not
    sit underneath layers that rarely change, so editing data/ or the eval
    script never invalidates the (slow) apt/pip/compile layers above it."""
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
# stdlib + numpy -- see run_audex_eval.py's main(), the `import torch` /
# model-loading calls happen after the dry-run early-return. Keeping this
# separate means a dataset problem surfaces in seconds on CPU, not after a
# multi-minute GPU container boots with the compiled-CUDA-extension image
# below.
dry_run_image = _with_repo_data(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
)

# Full image for a real run. Base is a CUDA *devel* image (not debian_slim
# like the rest of this repo's wrappers) because causal-conv1d/mamba-ssm
# below compile CUDA extensions against the installed torch/CUDA at pip-install
# time via --no-build-isolation, which needs nvcc + CUDA headers present in
# the image, not just the CUDA runtime.
#
# Install order matches this project's exact hard requirement:
#   1. torch (a version confirmed to work elsewhere in this repo, on this
#      same CUDA 12.4 Modal image family -- modal_phi4mm_eval.py already
#      pins torch==2.6.0 successfully here. Not independently reconfirmed
#      for Audex specifically since there is no local GPU to test the
#      compiled-kernel build against before this runs on Modal -- flag any
#      torch/CUDA ABI mismatch surfaced by the smoke test rather than
#      treating this pin as settled.)
#   2. transformers==5.14.0, safetensors==0.8.0
#   3. causal-conv1d==1.6.2.post1, mamba-ssm==2.3.1, --no-build-isolation
#      (must come after transformers/safetensors and after torch, since
#      both compile against the already-installed torch)
gpu_image = _with_repo_data(
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "ffmpeg", "ninja-build")
    # torch's cpp_extension build probed for clang++ (not installed in this
    # image) instead of falling back to g++/build-essential, confirmed by a
    # real build failure here (`which clang++` -> exit 1, uncaught). Pinning
    # CC/CXX explicitly avoids depending on whatever compiler this base image
    # would otherwise auto-detect.
    .env({"CC": "gcc", "CXX": "g++"})
    # torch==2.5.1 -- landed here after two real build/runtime failures on
    # this exact image, not a first guess:
    #   torch==2.6.0: causal-conv1d/mamba-ssm import with
    #     `undefined symbol: ...torchCheckFail...` (confirmed matching a
    #     currently-open upstream bug, Dao-AILab/causal-conv1d#77 -- a C++ ABI
    #     break for torch 2.6+cu124).
    #   torch==2.4.1: causal-conv1d/mamba-ssm import fine (ABI break avoided),
    #     but transformers==5.14.0's own loading path
    #     (transformers.distributed.sharding_utils) does
    #     `from torch.distributed.tensor import DTensor`, which 2.4.1 doesn't
    #     export at that public path yet (PyTorch's own 2.5 docs are the
    #     first version showing that public API).
    # 2.5.1 is the version actually sitting between both constraints: newer
    # than 2.4.1 (has the DTensor export transformers needs) and older than
    # the confirmed-broken 2.6.0. Not yet independently reconfirmed by a
    # maintainer for this exact causal-conv1d==1.6.2.post1/mamba-ssm==2.3.1/
    # transformers==5.14.0 triple -- flag any further ABI or import failure
    # rather than assuming this pin is the final answer.
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==5.14.0",
        "safetensors==0.8.0",
        # required by NVIDIA's own inference_hf.py (`from packaging.version import Version`)
        "packaging",
    )
    # --no-build-isolation means pip does NOT create an isolated build venv
    # for the next step, so build-time deps (wheel, setuptools, ninja) must
    # already be present in this environment -- confirmed necessary by a
    # real build failure here (`ModuleNotFoundError: No module named 'wheel'`)
    # the first time this image was built, not a preemptive guess.
    .pip_install("wheel", "setuptools", "ninja")
    .pip_install(
        "causal-conv1d==1.6.2.post1",
        "mamba-ssm==2.3.1",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "librosa",
        "soundfile",
        # <2, not bare -- same class of numpy/torch ABI bug documented in
        # every other wrapper's image in this repo.
        "numpy<2",
        "huggingface_hub[hf_transfer]",
        # required for device_map="single" -- confirmed by a real
        # from_pretrained() failure here, not a preemptive guess
        # ("requires `accelerate`. You can install it with `pip install accelerate`").
        "accelerate",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _import_script():
    import sys

    sys.path.insert(0, f"{CONTAINER_ROOT}/eval")
    import run_audex_eval  # noqa: E402

    return run_audex_eval


@app.function(image=dry_run_image, timeout=600)
def audit_only(argv: list) -> int:
    script = _import_script()
    return script.main(argv)


def _resolve_audex_artifacts() -> tuple[str, str]:
    """Download checkpoint_folder_full/ and NVIDIA's inference_scripts_hf/
    from the model's HF repo, cached on the shared volume. Returns
    (local checkpoint dir, local inference-utils dir). Only downloads once --
    subsequent runs hit the volume-cached copy.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    print(">>> resolving Audex artifacts from {} (cached under /vol/hf_cache "
          "after the first run; ~60GB checkpoint download on a cold cache)".format(MODEL_REPO_ID))

    checkpoint_dir = snapshot_download(
        repo_id=MODEL_REPO_ID,
        allow_patterns=[f"{CHECKPOINT_SUBDIR}/*", f"{CHECKPOINT_SUBDIR}/**"],
        cache_dir="/vol/hf_cache",
    )
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_SUBDIR)
    if not os.path.isdir(checkpoint_path):
        raise SystemExit(
            "Expected {} inside the downloaded snapshot but it's not there -- "
            "the repo's folder layout may have changed since this wrapper was "
            "written; inspect the snapshot at {} before retrying.".format(
                CHECKPOINT_SUBDIR, checkpoint_dir)
        )

    utils_dir = os.path.join("/vol/hf_cache", "audex_inference_scripts_hf")
    os.makedirs(utils_dir, exist_ok=True)
    for fname in INFERENCE_UTILS_FILES:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=f"{INFERENCE_UTILS_SUBDIR}/{fname}",
            local_dir="/vol/hf_cache/_audex_utils_download",
        )
    downloaded_dir = os.path.join("/vol/hf_cache/_audex_utils_download", INFERENCE_UTILS_SUBDIR)
    for fname in INFERENCE_UTILS_FILES:
        src = os.path.join(downloaded_dir, fname)
        dst = os.path.join(utils_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            import shutil
            shutil.copy(src, dst)

    print(">>> checkpoint: {}".format(checkpoint_path))
    print(">>> inference utils: {}".format(utils_dir))
    return checkpoint_path, utils_dir


def _resolve_textonly_artifacts(checkpoint_path: str) -> str:
    """Assemble a local directory Audex's dedicated text-only entry point
    (NemotronHForCausalLM, per checkpoint_folder_textonly/config.json's
    auto_map) can load directly with plain AutoModelForCausalLM.from_pretrained.

    checkpoint_path is checkpoint_folder_full's already-downloaded local dir
    (from _resolve_audex_artifacts() above). See the TEXTONLY_SUBDIR comment
    near the top of this file for why this only needs a small metadata
    download plus symlinks into those already-downloaded shards, not a
    second full checkpoint.
    """
    import json

    from huggingface_hub import hf_hub_download

    print(">>> resolving Audex text-only artifacts (small metadata download; "
          "shard weights reused from checkpoint_folder_full, already downloaded above)")

    textonly_dir = os.path.join("/vol/hf_cache", "audex_textonly_merged")
    os.makedirs(textonly_dir, exist_ok=True)

    meta_download_dir = "/vol/hf_cache/_textonly_meta_download"
    for fname in TEXTONLY_META_FILES:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=f"{TEXTONLY_SUBDIR}/{fname}",
            local_dir=meta_download_dir,
        )
    downloaded_dir = os.path.join(meta_download_dir, TEXTONLY_SUBDIR)
    for fname in TEXTONLY_META_FILES:
        src = os.path.join(downloaded_dir, fname)
        dst = os.path.join(textonly_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)

    index_path = os.path.join(textonly_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))
    for shard in shard_names:
        src = os.path.join(checkpoint_path, shard)
        dst = os.path.join(textonly_dir, shard)
        if not os.path.isfile(src):
            raise SystemExit(
                "checkpoint_folder_textonly's index expects shard {} but it is not "
                "present in the downloaded checkpoint_folder_full at {} -- the repo's "
                "shard layout may have changed since this wrapper was written; inspect "
                "both checkpoints' model.safetensors.index.json before retrying.".format(
                    shard, checkpoint_path)
            )
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(">>> text-only checkpoint assembled at: {} ({} shards symlinked)".format(
        textonly_dir, len(shard_names)))
    return textonly_dir


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=10 * 60 * 60,  # generous: ~60GB cold download + CUDA kernel compile + 30B bf16 model
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/vol": volume},
)
def run_eval(argv: list) -> int:
    os.makedirs("/vol/hf_cache", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")

    checkpoint_path, utils_dir = _resolve_audex_artifacts()
    # Always resolved, harmless if unused: it's a small metadata download plus
    # symlinks into shards already pulled above, and run_audex_eval.py's main()
    # only actually requires/loads it when the selected items need it
    # (i.e. --modality-filter text_only or a mixed run touching text_only items).
    textonly_path = _resolve_textonly_artifacts(checkpoint_path)

    argv = argv + [
        "--model-path", checkpoint_path,
        "--inference-utils-dir", utils_dir,
        "--textonly-model-path", textonly_path,
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
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    modality_filter: str = None,
):
    argv = [
        "--experiments", *experiments.split(),
        "--input", f"{CONTAINER_ROOT}/data/combined_examples.json",
        "--output-dir", "/vol/results",
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if modality_filter is not None:
        argv += ["--modality-filter", modality_filter]

    if dry_run:
        print(">>> Audex dry-run: CPU-only dataset audit, no GPU, no model load.\n")
        code = audit_only.remote(argv + ["--dry-run"])
    else:
        print(">>> Audex full run: launching on a single A100-80GB, bfloat16, native "
              "Nemotron-H backend. This is billed GPU time, and will download a ~60GB "
              "checkpoint on first run.\n")
        code = run_eval.remote(argv)

    if code:
        raise SystemExit(code)
