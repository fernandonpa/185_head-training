# Model training and evaluation

This folder contains training (`main_train.py`) and evaluation (`eval.py`) for **ViT-ESI** and baseline models (1D-CNN, LSTM, DeepSIF). Use the same data layout and paths as in the main [README](../README.md).

**Conventions:**
- `$PROJECT_ROOT` = path to the ViT-ESI repo (e.g. `/path/to/ViT-ESI`).
- Run commands from **`model_training/`** unless noted.

---

## Training: `main_train.py`

### Common arguments (all models)

| Argument | Description | Example |
|----------|-------------|---------|
| `simu_name` | Simulation name (positional) | `mes_debug_python` |
| `-results_path` | Where to save results and checkpoints | `$PROJECT_ROOT/model_training/results` |
| `-simu_type` | Data source: `sereega` or `NMM` | `sereega` |
| `-source_space` | Source space (e.g. `fsav_994`) | `fsav_994` |
| `-electrode_montage` | Montage name | `standard_1020` |
| `-orientation` | `constrained` or `unconstrained` | `constrained` |
| `-leadfield_mat` | Path to leadfield `.mat` | `$PROJECT_ROOT/anatomy/leadfield_75_20k.mat` |
| `-simu_folder` | **Optional.** Full path to simu folder; overrides `-root_simu` for loading data | `.../simu/mes_debug_python` |
| `-root_simu` | Root of simulation tree (required if `-simu_folder` not set) | `$PROJECT_ROOT` |
| `-to_load` | Number of samples (train+val) | `100` |
| `-per_valid` | Validation fraction (0–1) | `0.2` |
| `-n_times` | Time samples per trial | `500` |
| `-eeg_snr` | EEG SNR (dB) for noise | `5` |
| `-loss` | Loss: `cosine`, `mse`, `logmse` | `cosine` |
| `-scaler` | Normalization: `linear` or `max` | `linear` |
| `-n_epochs` | Max training epochs | `25` |
| `-batch_size` / `-bs` | Batch size | `8` |
| `-resume` | Path to a `.ckpt` file to resume training (restores model, optimizer, epoch) | `path/to/epoch=4-loss=0.32.ckpt` |

### Resuming training from a checkpoint

To continue training from a saved checkpoint (e.g. after interruption or to train for more epochs), pass the path to a PyTorch Lightning `.ckpt` file with **`-resume**:

```bash
python main_train.py mes_debug_python ... -resume "path/to/pl_checkpoints/epoch=4-train_loss=0.32.ckpt" -n_epochs 50
```

Use the same data/model arguments as the original run; the checkpoint restores model weights, optimizer state, and epoch counter. New checkpoints and the best model will be written to the **current run directory** (determined by `-results_path`, `simu_name`, and the run subfolder). To resume into the same run folder, use the same `-results_path` and other args so that `results_path` matches the directory that contains the checkpoint.

### Where outputs are saved

- **Checkpoints:**  
  `{results_path}/{simu_name}{source_space}_/trainings/{run_subfolder}/pl_checkpoints/`  
  Files: `{epoch}-{train_loss:.2f}.ckpt`

- **Best model (state dict):**  
  `.../trained_models/{MODEL}_model.pt`  
  e.g. `VIT_model.pt`, `1DCNN_model.pt`, `LSTM_model.pt`, `DEEPSIF_model.pt`

- **Logs:**  
  `.../logs/` (TensorBoard or CSV)

`run_subfolder` is built from simu_type, source_space, model name, dataset size, epochs, loss, and scaler (e.g. `simu_sereega_srcspace_fsav_994_model_VIT_trainset_80_epochs_25_loss_cosine_norm_linear`).

---

## Example commands (use your paths)

Replace `$PROJECT_ROOT` with your ViT-ESI directory (e.g. `/media/pasindu/DATA/fyp/ViT-ESI`).

### ViT-ESI (recommended)

```bash
python model_training/main_train.py mes_debug_python \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -simu_type sereega \
  -source_space fsav_994 \
  -electrode_montage standard_1020 \
  -orientation constrained \
  -model VIT \
  -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 \
  -loss cosine -scaler linear \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -vit_embed_dim 256 -vit_depth 6 -vit_heads 8 -vit_mlp_dim 512 -vit_dropout 0.1 \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  -n_epochs 25
```

### 1D-CNN

```bash
python main_train.py mes_debug_python \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained \
  -model 1DCNN -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 \
  -loss cosine -scaler linear \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -inter_layer 4096 -kernel_size 5 \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  -n_epochs 25
```

### LSTM

```bash
python main_train.py mes_debug_python \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained \
  -model LSTM -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 \
  -loss cosine -scaler linear \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  -n_epochs 25
```

### DeepSIF

```bash
python main_train.py mes_debug_python \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained \
  -model DEEPSIF -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 \
  -loss cosine -scaler linear \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -deepsif_temporal_input_size 500 \
  -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" \
  -n_epochs 25
```

---

## Evaluation: `eval.py`

Evaluates trained models and (optionally) linear methods (MNE, sLORETA) on the same simulation.

### Important arguments

- `simu_name`, `-root_simu`, `-results_path`, `-eval_simu_type`, `-source_space`, `-electrode_montage`, `-orientation`, `-leadfield_mat`: same as training.
- **`-train_run_dir`**: Path to the **run directory** that contains `trained_models/<MODEL>_model.pt`. If set, the script loads the NN from this folder instead of inferring from other training args.
- **`-simu_folder`**: Optional; full path to the simulation folder (same as in training).
- **`-mets` / `-methods`**: Methods to run, e.g. `eeg_vit` (ViT-ESI), `cnn_1d`, `lstm`, `deep_sif`, `MNE`, `sLORETA`.

### Example (ViT-ESI from a specific run)

After training, you get a run directory like:
`results/mes_debug_pythonfsav_994_/trainings/simu_sereega_srcspace_fsav_994_model_VIT_trainset_80_epochs_25_loss_cosine_norm_linear/`

```bash
python eval.py mes_debug_python \
  -root_simu "$PROJECT_ROOT" \
  -results_path "$PROJECT_ROOT/model_training/results" \
  -eval_simu_type sereega \
  -source_space fsav_994 \
  -electrode_montage standard_1020 \
  -orientation constrained \
  -leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -train_run_dir "$PROJECT_ROOT/model_training/results/mes_debug_pythonfsav_994_/trainings/simu_sereega_srcspace_fsav_994_model_VIT_trainset_80_epochs_25_loss_cosine_norm_linear" \
  -mets eeg_vit
```

Use `-mets MNE sLORETA eeg_vit` to compare linear methods and ViT-ESI.

---

## Running on another machine

1. Set `PROJECT_ROOT` to the ViT-ESI folder on that machine.
2. Install dependencies: `pip install -r requirements.txt` (from repo root).
3. Ensure `anatomy/` contains `leadfield_75_20k.mat` and (for SEREEGA) `sources_fsav_994.mat`.
4. Generate data (or copy existing `simulation/` and configs) so that `-simu_folder` or `-root_simu` points to the correct paths.
5. Run the same training/eval commands with `$PROJECT_ROOT` replaced by your path.
