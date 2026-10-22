# Data generation

This folder contains scripts to generate **synthetic EEG and source data** for training and evaluating ViT-ESI and baseline models.

## Supported pipelines

| Pipeline | Folder | Description |
|----------|--------|--------------|
| **SEREEGA** | [sereega/](sereega/) | Extended-source simulations (patches, ERP-like signals). Python-only. |
| **NMM** | [nmm/simu_source_nmm/](nmm/simu_source_nmm/) | Neural mass model–based simulations (MATLAB + Python). |

For **ViT-ESI**, SEREEGA is the main data source; use the 994-region source space and the leadfield from `../anatomy/` (e.g. `leadfield_75_20k.mat`).

## Prerequisites

- **Anatomy:** The project `anatomy/` folder (or a custom path) must contain:
  - A leadfield matrix (e.g. `leadfield_75_20k.mat`, 75 sensors × 994 regions).
  - For SEREEGA extended sources: `sources_fsav_994.mat` (regional source positions).
- **Python:** Same environment as the rest of the project (`pip install -r requirements.txt` from repo root).

## Quick start (SEREEGA)

From the **project root** (`ViT-ESI/`), using `$PROJECT_ROOT` as the path to the repo:

```bash
python data_generation/sereega/simu_extended_source.py \
  -sin my_simulation \
  -ne 500 \
  -mk standard_1020 \
  -ss fsav_994 \
  -o constrained \
  -sn fsaverage \
  -rf "$PROJECT_ROOT" \
  --leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -fs 500 -d 1000 \
  -af "$PROJECT_ROOT/anatomy"
```

Output is written under `$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/my_simulation/` (or under `$PROJECT_ROOT/constrained/...` depending on layout). Use this path as `-simu_folder` when training.

See [sereega/README.md](sereega/README.md) for full SEREEGA options.

## NMM

The NMM pipeline uses MATLAB scripts and optional Python helpers. See [nmm/simu_source_nmm/readme.md](nmm/simu_source_nmm/readme.md) for setup and usage.
