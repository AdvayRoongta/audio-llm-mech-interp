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


# Lines that carry no diagnostic value: Modal draws its errors in a box, and
# apt/pip chatter dominates the tail of a failed build.
_NOISE = ("╭", "╰", "│ ", "─", "Unpacking ", "Setting up ", "Selecting ",
          "Preparing to unpack", "Downloading ", "Collecting ", "Reading database",
          "Get:", "Fetched ", "Installing collected", "Successfully installed",
          "[notice]", "Processing triggers")


def summarise_failure(out):
    """Pull the actual reason out of a failed run.

    The last line of Modal output is usually the bottom edge of its error box,
    which tells you nothing. This looks for the message inside the box, then
    for common exception text, before giving up and using the last real line.
    """
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]

    # Modal renders errors as "│ <message> │"; take the longest such body.
    boxed = [l.strip("│ ").strip() for l in lines if l.startswith("│")]
    boxed = [b for b in boxed if b and not set(b) <= set("─ ")]
    if boxed:
        return " ".join(boxed)[:300]

    for line in reversed(lines):
        if any(line.lstrip().startswith(k) for k in _NOISE):
            continue
        if any(tok in line for tok in ("Error", "error:", "Exception", "Traceback",
                                       "FileNotFoundError", "Usage:", "not exist")):
            return line.strip()[:300]

    for line in reversed(lines):
        if not any(line.lstrip().startswith(k) for k in _NOISE):
            return line.strip()[:300]
    return "(no output)"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["audit", "smoke", "full"])
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--experiments", default="all",
                    help='"all" or a space-separated subset, quoted')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=int, default=20,
                    help="seconds between models; Modal rate-limits app creation")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="anything after this is passed through to modal run")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

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
