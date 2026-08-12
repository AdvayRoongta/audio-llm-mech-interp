# third_party/

Vendored upstream code, kept as-is. Not part of this repo's build/data/eval pipeline —
don't wire it into the new scripts, and don't edit anything under here.

## MPP_Code/

Upstream MUStARD++ model + training code from the original
[cfiltnlp/MUStARD_Plus_Plus](https://github.com/cfiltnlp/MUStARD_Plus_Plus) repo, moved
here unmodified from the repo root.

- `README.md` — the upstream code's own README (originally `MPP_Code/README.md`).
- `README_MUStARD.md` — the original root `README.md` of the cloned repo, describing the
  MUStARD++ dataset itself. Preserved unmodified; moved here because it documents this
  vendored code rather than this project.
- `models/`, `training/` — upstream classification/regression/sarcasm training scripts.

No `LICENSE` file was present anywhere in the original clone, so none could be carried
over — check the upstream repo directly if you need licensing terms.

**Known pre-existing issue (not caused by this reorg):** the `training/*.py` scripts
reference data paths that were never included in this repo, e.g.
`path+'mustard_PP_utterance.csv'`, `'final_datasets/augmented_sarcastic_utterances.csv'`,
`'MPP_Code/data/splits_final_mustard++/...'`. These were broken in the original clone
before anything was moved. Left untouched.
