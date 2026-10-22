# NMM-based source simulation

This folder contains scripts for **neural mass model (NMM)**–based synthetic EEG/source data generation. The pipeline uses **MATLAB** for the core NMM simulation and optional **Python** helpers for post-processing and integration.

## Contents

- **MATLAB:** `generate_synthetic_data.m`, `process_raw_nmm.m`, `find_spike_time.m`, and related `.m` files for running the NMM and producing spike/activity data.
- **Python:** `generate_tvb_data.py` and similar scripts for The Virtual Brain (TVB) or other pipelines, if used.
- **Shell:** `bash_tvb_simu.sh` for batch/scripted runs.

## Requirements

- **MATLAB** (with toolboxes required by the NMM scripts, if any).
- Optional: **Python** and project dependencies (`pip install -r requirements.txt` from repo root) for any Python-based steps.
- Anatomy and leadfield as in the main project (e.g. 994-region source space, 75-channel montage).

## Usage

1. Configure paths and parameters in the MATLAB scripts (e.g. leadfield path, output directory, number of trials).
2. Run the main MATLAB generation script (e.g. `generate_synthetic_data.m`) from MATLAB or via `matlab -batch "run('generate_synthetic_data.m')"`.
3. Output is typically written to a folder under the project (e.g. a “spikes” or “nmm” directory). That path is then used as **`-spikes_folder`** and the corresponding metadata path as **`-simu_folder`** (or equivalent) when training with **NMM** simu_type in `main_train.py`.

For exact steps and required MATLAB variables, refer to the comments and variable definitions inside the `.m` files in this directory.

## Integration with ViT-ESI training

In `main_train.py`, use `-simu_type NMM` and point to the NMM output folder and metadata. The main [model_training readme](../../../model_training/readme.md) describes training with NMM data; ensure the folder layout matches what `ModSpikeEEGBuild` and the loaders expect (see `model_training/loaders.py`).
