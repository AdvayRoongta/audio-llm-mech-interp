# Pre-SALMONN-refactor originals

Verbatim copies of the three `eval/` files the SALMONN port modified, taken
from commit **`cb0e4cf`** — the last state of `main` before the backend split.
These are the versions that produced every committed Qwen result in
`results/exp*.json`.

| file | what changed after this snapshot |
|---|---|
| `_common.py` | split into a `MECH_BACKEND` dispatcher; Qwen bodies moved to `eval/_backend_qwen.py` |
| `exp2_encoder_probe.py` | two Qwen-specific assumptions fixed (the projector, and the dtype cast) |
| `exp3_circuit_localization.py` | two direct `import run_qwen2audio_eval` calls routed through the dispatcher |
| `gitignore.txt` | the `.gitignore` as it stood (now also ignores `*.pt`, `data/raw/**/audio/`) |

**These are a convenience copy, not the backup of record.** Git history is the
real one and it is complete — nothing here is unrecoverable without this
folder. To see or restore any original:

```bash
git show cb0e4cf:eval/_common.py                 # view
git checkout cb0e4cf -- eval/_common.py          # restore into the working tree
git diff cb0e4cf -- eval/                        # what the port changed
```

**Do not import from this directory.** It exists so the pre-refactor code can
be read side by side with the current version without a git checkout. The
equivalence test `eval/test_refactor_equivalence.py` deliberately pulls the old
`_common.py` out of git rather than from here, so the comparison cannot drift
if someone edits these files.

Why the refactor happened, what it changed, and the two bugs it surfaced:
`notes/SALMONN_PORT.md`. Chronological account: `notes/SALMONN_PORT_LOG.md`.
