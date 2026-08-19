#!/usr/bin/env python3
"""Populate ./dataset_audio/ on this machine.

The per-model Modal wrappers (modal/modal_*_eval.py) mount the corpus with
add_local_dir(LOCAL_ROOT / "dataset_audio"), so the audio has to exist locally
before any of them can run -- a Modal volume is not enough for those. This
script builds that directory.

    dataset_audio/ravdess/03-01-05-02-01-02-15.wav
    dataset_audio/mustard/1_10496_u.wav

Only files the dataset actually references are written, so re-running after the
dataset grows fetches just the difference.

    pip3 install datasets soundfile
    python3 scripts/fetch_audio.py                 # both corpora
    python3 scripts/fetch_audio.py --only ravdess
    python3 scripts/fetch_audio.py --check         # report coverage, fetch nothing

RAVDESS comes from the Hugging Face mirror. MUStARD needs the 1.4 GB raw-clip
zip and ffmpeg to convert. MUStARD++ episode-style ids (1_S09E03_056) are not
in the public v1 release and are reported as missing -- their videos are behind
a Google Drive link that has to be downloaded by hand.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "data", "combined_examples.json")
AUDIO_ROOT = os.path.join(REPO, "dataset_audio")
MUSTARD_ZIP_URL = ("https://huggingface.co/datasets/MichiganNLP/MUStARD/"
                   "resolve/main/mmsd_raw_data.zip")


def wanted(kind):
    """Basenames of the clips the dataset references for one corpus."""
    with open(DATASET) as f:
        items = json.load(f)
    out = set()
    for item in items:
        path = item.get("audio_path") or ""
        if "/{}/".format(kind) in path:
            out.add(os.path.basename(path))
    return out


def present(kind):
    d = os.path.join(AUDIO_ROOT, kind)
    if not os.path.isdir(d):
        return set()
    return {f for f in os.listdir(d) if f.endswith(".wav")}


def report(kind):
    w, p = wanted(kind), present(kind)
    have = len(w & p)
    print("  {:<10} {:>4}/{:<4} present".format(kind, have, len(w)), end="")
    print("   ({} missing)".format(len(w - p)) if w - p else "   complete")
    return w, p


# --------------------------------------------------------------------------- #
def fetch_ravdess():
    """Write the referenced RAVDESS clips from the Hugging Face mirror.

    The HF rows carry the original RAVDESS filename in the audio dict's `path`,
    so files land under that name verbatim rather than being reconstructed from
    emotion/intensity/actor codes -- one less thing to get subtly wrong.
    """
    try:
        import datasets
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("needs `pip3 install datasets soundfile`")

    want = wanted("ravdess")
    missing = want - present("ravdess")
    if not missing:
        print("  ravdess already complete ({} clips)".format(len(want)))
        return
    print("  ravdess: need {} of {} clips; loading the HF mirror "
          "(~325 MB the first time)".format(len(missing), len(want)))

    out_dir = os.path.join(AUDIO_ROOT, "ravdess")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("xbgoose/ravdess", split="train")
    # decode=False keeps the raw bytes, which skips torchcodec entirely.
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    written = 0
    for row in ds:
        audio = row["audio"]
        name = os.path.basename(audio.get("path") or "")
        if name in missing:
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(audio["bytes"])
            written += 1
    print("  ravdess: wrote {} clips".format(written))


def fetch_mustard(keep_zip=False):
    """Download the raw-clip zip and convert the referenced utterances to wav."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found -- `brew install ffmpeg`")

    want = wanted("mustard")
    missing = want - present("mustard")
    if not missing:
        print("  mustard already complete ({} clips)".format(len(want)))
        return

    def stem(name):
        return name[:-6] if name.endswith("_u.wav") else os.path.splitext(name)[0]

    want_ids = {stem(n): n for n in missing}
    episode = {i for i in want_ids if re.search(r"S\d+E\d+", i)}
    print("  mustard: need {} of {} clips ({} are MUStARD++ episode ids not in "
          "the public v1 release)".format(len(missing), len(want), len(episode)))

    out_dir = os.path.join(AUDIO_ROOT, "mustard")
    os.makedirs(out_dir, exist_ok=True)

    work = os.path.join(REPO, ".mustard_raw")
    os.makedirs(work, exist_ok=True)
    zip_path = os.path.join(work, "mmsd_raw_data.zip")
    if not os.path.exists(zip_path):
        print("  downloading 1.4 GB ...")
        urllib.request.urlretrieve(MUSTARD_ZIP_URL, zip_path)

    extract_dir = os.path.join(work, "extracted")
    if not os.path.isdir(extract_dir):
        print("  extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    # Index by file stem so a layout change in the archive doesn't break this.
    by_stem = {}
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in (".mp4", ".mkv", ".avi", ".aac",
                                                  ".wav", ".m4a"):
                by_stem.setdefault(os.path.splitext(f)[0], os.path.join(root, f))

    written, absent = 0, []
    for cid, target in sorted(want_ids.items()):
        src = by_stem.get(cid)
        if not src:
            absent.append(cid)
            continue
        # 16 kHz mono is what every processor resamples to anyway.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vn", "-ac", "1", "-ar", "16000",
                        os.path.join(out_dir, target)], check=True)
        written += 1
    print("  mustard: wrote {} clips".format(written))
    if absent:
        n_ep = len(set(absent) & episode)
        print("  mustard: {} still missing ({} of them MUStARD++ episode ids). "
              "Those need the Drive download named in the MUStARD++ README."
              .format(len(absent), n_ep))

    if not keep_zip:
        shutil.rmtree(work, ignore_errors=True)
        print("  cleaned up {} (pass --keep-zip to reuse it)".format(work))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["ravdess", "mustard"])
    ap.add_argument("--check", action="store_true", help="report coverage, fetch nothing")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the 1.4 GB MUStARD download for reuse")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(DATASET):
        raise SystemExit("no dataset at {} -- run from the repo root".format(DATASET))

    print("dataset: {}".format(os.path.relpath(DATASET, REPO)))
    print("target:  {}/\n".format(os.path.relpath(AUDIO_ROOT, REPO)))
    print("current coverage:")
    for kind in ("ravdess", "mustard"):
        report(kind)

    if args.check:
        return 0

    print()
    if args.only in (None, "ravdess"):
        fetch_ravdess()
    if args.only in (None, "mustard"):
        fetch_mustard(keep_zip=args.keep_zip)

    print("\nfinal coverage:")
    for kind in ("ravdess", "mustard"):
        report(kind)
    print("\ndataset_audio/ is ready; the Modal wrappers mount it from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
