#!/usr/bin/env python3
"""Drive every model's Modal eval through the same three stages.

Each model has its own modal.App (modal_qwen2audio_eval.py, modal_salmonn_eval.py,
...), so this is a plain local orchestrator that shells out to `modal run` rather
than a Modal app itself. Nothing here touches model or scoring logic.

Three stages, cheapest first, because the expensive mistake is discovering a
per-model dataset mismatch two hours into a billed run:

    audit   CPU-only dataset audit per model. Free. Catches prompt/field
            mismatches after a dataset change.
    smoke   --limit 8, GPU. Confirms the model actually loads and scores.
    full    the real run.

Usage, from the repo root:

    python3 modal/sweep_models.py --stage audit
    python3 modal/sweep_models.py --stage smoke
    python3 modal/sweep_models.py --stage full
    python3 modal/sweep_models.py --stage full --models qwen2audio phi4mm

A stage stops nothing on failure -- every model is attempted and the summary
table at the end reports which passed, so one broken wrapper doesn't hide the
state of the other four.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Model key -> its Modal wrapper. Order is cheapest/most-likely-to-work first,
# so a systemic problem surfaces on the fast one.
MODELS = {
    "qwen2audio": "modal_qwen2audio_eval.py",
    "phi4mm": "modal_phi4mm_eval.py",
    "azeros": "modal_azeros_eval.py",
    "desta25": "modal_desta25_eval.py",
    "salmonn": "modal_salmonn_eval.py",
}


def run_one(model, script, stage, experiments, seed, extra):
    """One `modal run` invocation. Returns (ok, seconds, tail_of_output)."""
    cmd = [sys.executable, "-m", "modal", "run",
           os.path.join("modal", script),
           "--experiments", experiments,
           "--seed", str(seed)]
    if stage == "audit":
        cmd.append("--dry-run")
    elif stage == "smoke":
        cmd += ["--limit", "8"]
    cmd += extra

    print("\n" + "=" * 74)
    print("[{}] {}".format(stage.upper(), model))
    print("=" * 74)
    print("+ " + " ".join(cmd), flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = time.time() - t0

    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:] if len(out) > 4000 else out, flush=True)
    return proc.returncode == 0, elapsed, summarise_failure(out)


import re

# Lines that carry no diagnostic value: apt/pip chatter dominates the tail of a
# failed build, and Modal's "Created objects" tree is drawn with the same box
# characters as its error box.
_NOISE = ("Unpacking ", "Setting up ", "Selecting ", "Preparing to unpack",
          "Downloading ", "Collecting ", "Reading database", "Get:", "Fetched ",
          "Installing collected", "Successfully installed", "[notice]",
          "Processing triggers", "✓ ", "├──", "└──", "🔨")

# Audit counters that mean trouble when non-zero.
_AUDIT_FLAGS = ("unresolved claimed->letter:",
                "claimed letter == correct letter (data bug if >0):",
                "audio modality items with no audio_path:",
                "audio_path set but file not found on disk:")


def summarise_failure(out):
    """Pull the actual reason out of a failed run.

    Three sources, most specific first. The dataset audit is checked before
    Modal's error box because an audit failure exits via SystemExit and prints
    no box at all -- and because "335 files not found" is far more useful than
    "SystemExit(1)".
    """
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]

    # 1. our own dataset audit
    if any("FLAGGED ISSUES" in l for l in lines):
        bad, cat = [], "?"
        for line in lines:
            m = re.match(r"\[(\w+)\] n=", line.strip())
            if m:
                cat = m.group(1)
            for flag in _AUDIT_FLAGS:
                if flag in line:
                    value = line.rsplit(":", 1)[-1].strip()
                    if value.isdigit() and int(value) > 0:
                        bad.append("{} {} {}".format(cat, flag.rstrip(":"), value))
        if bad:
            return "audit flagged: " + "; ".join(bad[:2])
        return "dataset audit flagged issues"

    # 2. Modal's error box, which is drawn as ╭─ Error ... ╰
    for i, line in enumerate(lines):
        if line.lstrip().startswith("╭") and "Error" in line:
            body = []
            for nxt in lines[i + 1:]:
                if nxt.lstrip().startswith("╰"):
                    break
                body.append(nxt.strip("│ ").strip())
            body = [b for b in body if b and not set(b) <= set("─ ")]
            if body:
                return " ".join(body)[:300]

    # 3. last line that looks like a real error
    for line in reversed(lines):
        if any(line.lstrip().startswith(k) for k in _NOISE):
            continue
        if any(tok in line for tok in ("Error", "error:", "Exception", "Traceback",
                                       "SystemExit", "Usage:", "not exist")):
            return line.strip()[:300]

    for line in reversed(lines):
        if not any(line.lstrip().startswith(k) for k in _NOISE):
            return line.strip()[:300]
    return "(no output)"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["audit", "smoke", "full", "fetch"])
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--experiments", default=None,
                    help='"all", or a quoted space-separated subset. Defaults to '
                         'the categories whose audio is actually complete.')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=int, default=20,
                    help="seconds between models; Modal rate-limits app creation")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="anything after this is passed through to modal run")
    return ap.parse_args(argv)


def complete_categories():
    """Categories whose referenced audio is fully present locally.

    The per-model audit fails outright when any audio_path is missing, so
    defaulting to "all" means one incomplete corpus blocks the categories that
    are ready. MUStARD++ episode ids are unobtainable from the public release,
    so sarcasm_pacing is normally the one left out.
    """
    import json

    dataset = os.path.join(REPO, "data", "combined_examples.json")
    # audio_path values already begin with "dataset_audio/", so they resolve
    # against the repo root, not against dataset_audio/ itself.
    if not os.path.exists(dataset):
        return None

    seen, missing = set(), set()
    with open(dataset) as f:
        for item in json.load(f):
            path = item.get("audio_path")
            if not path or item.get("modality") not in ("audio_only", "text_audio"):
                continue
            cat = item.get("category")
            seen.add(cat)
            if not os.path.exists(os.path.join(REPO, path)):
                missing.add(cat)
    return sorted(seen - missing), sorted(missing)


MODEL_VOLUME = "qwen2audio-results"   # every wrapper shares this one


def fetch_results(dest="results"):
    """Pull the per-model summaries out of the shared Modal volume.

    The five wrappers write to volume "qwen2audio-results" at /results, which is
    a different volume from the one modal_app.py uses for the mechanistic
    experiments -- so `modal_app.py --fetch-only` does not bring these down.
    """
    os.makedirs(dest, exist_ok=True)
    listing = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "ls", MODEL_VOLUME, "/results"],
        cwd=REPO, capture_output=True, text=True)
    if listing.returncode:
        print(listing.stderr.strip()[:400])
        return 1

    names = [l.strip() for l in listing.stdout.splitlines()
             if l.strip().endswith("_summary.json")]
    if not names:
        print("no *_summary.json in {}:/results yet".format(MODEL_VOLUME))
        return 0

    got = 0
    for name in names:
        remote = "/" + name.lstrip("/")
        r = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get",
             MODEL_VOLUME, remote, dest],
            cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            print("  pulled {}".format(os.path.basename(name)))
            got += 1
        else:
            print("  FAILED {}: {}".format(os.path.basename(name),
                                           r.stderr.strip()[:120]))
    print("\n{} summary file(s) in ./{}/".format(got, dest))
    print("Now run: python viz/make_figures.py")
    return 0


def main(argv=None):
    args = parse_args(argv)

    if args.stage == "fetch":
        return fetch_results()

    if args.experiments is None:
        found = complete_categories()
        if not found or not found[0]:
            args.experiments = "all"
            print("Could not determine audio coverage; using --experiments all.\n")
        else:
            ready, incomplete = found
            args.experiments = " ".join(ready)
            print("Categories with complete audio: {}".format(", ".join(ready)))
            if incomplete:
                print("Skipping (audio incomplete): {}".format(", ".join(incomplete)))
                print("The per-model audit fails outright on a missing file, so")
                print("including these would block the ones that are ready.")
                print("Override with --experiments all once the audio is in place.")
            print()

    if args.stage == "full":
        print("Full runs are billed GPU time across {} model(s).".format(len(args.models)))
        print("Run --stage audit and --stage smoke first if you have not.\n")

    results = []
    for n, model in enumerate(args.models):
        if n and args.delay:
            # Modal rate-limits app creation; five back-to-back launches trips
            # "App create rate limit exceeded" and the later models fail for a
            # reason that has nothing to do with them.
            print("\n... waiting {}s before the next model "
                  "(Modal app-create rate limit)".format(args.delay), flush=True)
            time.sleep(args.delay)
        ok, secs, why = run_one(model, MODELS[model], args.stage,
                                args.experiments, args.seed, args.extra)
        results.append((model, ok, secs, why))

    print("\n" + "=" * 74)
    print("SUMMARY -- stage: {}".format(args.stage))
    print("=" * 74)
    print("{:<14}{:>8}{:>7}   {}".format("model", "result", "secs", "why"))
    for model, ok, secs, why in results:
        print("{:<14}{:>8}{:>7.0f}   {}".format(
            model, "ok" if ok else "FAIL", secs, "" if ok else why[:90]))

    failed = [(m, w) for m, ok, _, w in results if not ok]
    if failed:
        print("\n{} model(s) failed: {}".format(
            len(failed), ", ".join(m for m, _ in failed)))
        reasons = {w[:90] for _, w in failed}
        if len(reasons) == 1 and len(failed) == len(results):
            print("\nEvery model failed with the same message, so this is one")
            print("shared cause, not five separate ones:")
            print("  {}".format(next(iter(reasons))))
        print("\nFix that before spending GPU time on a full sweep.")
        return 1
    print("\nall {} model(s) passed".format(len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
