# AGENTS.md

## Cursor Cloud specific instructions

This repo contains two products: (1) the **Music Source Separation (MSS)** training/inference toolkit
(`inference.py`, `train.py`, `valid.py`, `ensemble.py`, models in `models/`), and (2) the
**`paper_digest`** arXiv emailer (`scripts/paper_digest/`, see `.cursor/skills/vocal-separation-paper-digest`).

### Environment
- CPU-only, headless Linux, no GPU. Python 3.12.
- Python deps are installed system-wide via `pip --break-system-packages` (see the startup update
  script). Just use `python3` directly; there is no virtualenv to activate.
- Run all MSS scripts from the repo root; they `sys.path`-append the root themselves.
- No lint/type tooling is configured (no ruff/flake8/black/mypy).

### Running the MSS app (core product)
- Inference works on CPU and does **not** require a checkpoint — with no `--start_check_point`
  the model runs with random weights, which is enough to smoke-test the pipeline end-to-end:
  ```
  python3 inference.py --model_type scnet --config_path configs/config_vocals_scnet.yaml \
      --input_folder <in_dir> --store_dir <out_dir> --force_cpu
  ```
  Always pass `--force_cpu` here. Output stems are written under `store_dir`.
- `train.py` needs a real dataset and is GPU-oriented; `*_ddp.py` need multiple GPUs. Not runnable here in practice.

### Tests
- `python3 tests/admin_test.py` synthesizes dummy audio and runs valid+inference for ~22 model
  configs with no checkpoints/data. It imports `train.py`, so the `[train]` extras (e.g. `wandb`)
  must be installed (they are, via the update script).
- Caveat: the **validation/metrics path is currently broken under `numpy>=2`** (the repo's declared
  floor): `utils/metrics.py` does `float()` on a 1-element ndarray, which raises
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`. This is a pre-existing
  code issue, not an environment problem. The **inference path is unaffected** and passes for
  apollo, htdemucs, mdx23c, scnet, segm_models, bs_roformer, mel_band_roformer.
- `tests/test.py` (user mode) requires real `--start_check_point` + `--data_path`/`--valid_path`.

### Things that cannot run in this environment
- `bandit` / `bandit_v2` models need `asteroid`, which fails to build (`pesq` needs `python3-dev`)
  and risks downgrading torch/numpy — intentionally not installed.
- `bs_mamba2` / `scnet_unofficial` need `mamba-ssm` (CUDA, x86_64 only).
- The wxPython GUI (`gui/gui-wx.py`) needs `wxPython` + a display — intentionally not installed.

### torchvision gotcha
`torch`/`torchaudio`/`torchvision` must all be the matching `+cpu` builds installed from
`https://download.pytorch.org/whl/cpu`. If `torchvision` gets pulled from PyPI (default CUDA build)
as a transitive dep of `timm`/`segmentation-models-pytorch`, imports fail with
`RuntimeError: operator torchvision::nms does not exist`. The update script installs the torch stack
from the CPU index first so the correct build is already satisfied.

### paper_digest
- `python3 -m scripts.paper_digest.main --dry-run` fetches from `export.arxiv.org` (needs outbound
  network) and prints an HTML preview without sending email — good for validation.
- Actually sending needs SMTP or Outlook Graph credentials via `PAPER_DIGEST_*` env vars
  (see the skill doc). `msal` is required only for `graph` mode.
