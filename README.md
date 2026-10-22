# ViT-ESI: Vision Transformer for Electrical Source Imaging

**ViT-ESI** is a Vision Transformer–based framework for EEG source imaging. It estimates the spatiotemporal distribution of intracranial neuronal sources from scalp-recorded EEG with attention over space and time.

This repository supports:
- **Synthetic data generation** (SEREEGA and NMM-based simulations)
- **Training** of ViT-ESI and baseline models (1D-CNN, LSTM, DeepSIF)
- **Evaluation** and comparison with linear methods (MNE, sLORETA) and neural networks

---

## Requirements

- **Python**: 3.10+ (tested with 3.12)
- **OS**: Linux, macOS, or Windows

See [requirements.txt](requirements.txt) for the full dependency list (PyTorch, PyTorch Lightning, MNE, scipy, etc.).

---

## Setup (any machine)

1. **Clone the repository** (or unpack the project):
   ```bash
   cd /path/to/your/workspace
   git clone <repo-url> ViT-ESI
   cd ViT-ESI
   ```
   Replace `/path/to/your/workspace` with your actual path (e.g. `~/projects/ViT-ESI` or `C:\Users\You\ViT-ESI` on Windows).

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Anatomy and leadfield** (for 994-region source space):
   - Ensure the `anatomy/` folder exists and contains at least:
     - `leadfield_75_20k.mat` (75 sensors × 994 regions)
     - `sources_fsav_994.mat` (for SEREEGA extended-source simulation)
   - If you use a different layout, point scripts to your leadfield and anatomy paths via the arguments below.

Use a single variable for the project root so you can rerun on any machine:

```bash
export PROJECT_ROOT=/path/to/ViT-ESI   # set this to your ViT-ESI folder
# Windows (PowerShell): $env:PROJECT_ROOT = "C:\path\to\ViT-ESI"
```

---

## Project structure

```
ViT-ESI/
├── anatomy/                    # Leadfield and source space (e.g. leadfield_75_20k.mat, sources_fsav_994.mat)
├── data_generation/             # Synthetic EEG/source data
│   ├── sereega/                 # SEREEGA-based simulation
│   └── nmm/                     # Neural mass model (NMM) based
├── model_training/              # Training and evaluation
│   ├── main_train.py            # Train ViT-ESI or baselines
│   ├── eval.py                  # Evaluate trained models
│   ├── loaders.py               # Dataset loaders
│   ├── models/                  # ViT, 1D-CNN, LSTM, DeepSIF
│   └── load_data/               # Head model, folder structure
├── simulation/                  # Output of data generation (created when you run scripts)
├── requirements.txt
└── README.md                    # This file
```

---

## How to run the code

### 1. Generate synthetic data (SEREEGA)

From the **project root** (`ViT-ESI/`):

```bash
python data_generation/sereega/simu_extended_source.py \
  -sin mes_debug_python \
  -ne 100 \
  -mk standard_1020 \
  -ss fsav_994 \
  -o constrained \
  -sn fsaverage \
  -rf "$PROJECT_ROOT" \
  --leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -fs 500 \
  -d 1000 \
  -af "$PROJECT_ROOT/anatomy"
```

- **`-sin`**: simulation name (folder created under `.../simu/`)
- **`-ne`**: number of examples (trials)
- **`-rf`**: root folder (your project root)
- **`-af`**: anatomy folder (for `sources_fsav_994.mat`)

Data is written under:
`$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python/`  
(or under `$PROJECT_ROOT/constrained/...` if not using the `simulation/<subject>/` layout).

See [data_generation/sereega/README.md](data_generation/sereega/README.md) for more options.

---

### 2. Train a model

From the **model_training/** directory (or run `python model_training/main_train.py` from project root with adjusted paths).

**ViT-ESI (recommended)**:
```bash
cd model_training
python main_train.py mes_debug_python \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -simu_type sereega \
  -source_space fsav_994 \
  -electrode_montage standard_1020 \
  -orientation constrained \
  -model VIT \
  -to_load 100 \
  -per_valid 0.2 \
  -n_times 500 \
  -eeg_snr 5 \
  -loss cosine \
  -scaler linear \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -vit_embed_dim 256 \
  -vit_depth 6 \
  -vit_heads 8 \
  -vit_mlp_dim 512 \
  -vit_dropout 0.1 \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  -n_epochs 25
```

You can omit **`-root_simu`** when **`-simu_folder`** is given. If you don’t use `-simu_folder`, add:
`-root_simu "$PROJECT_ROOT"`.

**Other models**: use the same call but set `-model` to `1DCNN`, `LSTM`, or `DEEPSIF` and the corresponding flags (see [model_training/readme.md](model_training/readme.md)).

Checkpoints and the best model are saved under:
`$PROJECT_ROOT/model_training/results/<simu_name><source_space>_/trainings/<run_subfolder>/`
- `pl_checkpoints/` — PyTorch Lightning checkpoints
- `trained_models/VIT_model.pt` (or `1DCNN_model.pt`, etc.) — best weights

---

### 3. Evaluate

From **model_training/**:

```bash
python eval.py mes_debug_python \
  -root_simu "$PROJECT_ROOT" \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -eval_simu_type sereega \
  -source_space fsav_994 \
  -electrode_montage standard_1020 \
  -orientation constrained \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  ...
```

Point the script to the same simulation and the run folder that contains `trained_models/VIT_model.pt` (or the model you trained). See [model_training/readme.md](model_training/readme.md) for full eval options.

---

## Citation

If you use this code, please cite the work as:

> In this work, we propose **ViT-ESI**, a Vision Transformer–based framework for EEG source imaging.

---

## License and data

See the repository license file. Synthetic data generated by the scripts is for research use; ensure compliance with any external data or anatomy (e.g. FreeSurfer) licenses.
