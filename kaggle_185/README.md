# kaggle_185 — generate data, train, test and package the 185-electrode ViT-ESI (SEREEGA)

A small helper layer **around** the team's scripts: nothing in `data_generation/` or `model_training/` is modified.
Open `ViTESI_185_pipeline_kaggle.ipynb` on Kaggle (Accelerator **GPU T4 x2**, Internet **On**) and **Run All**.

| File | Role |
|------|------|
| `ViTESI_185_pipeline_kaggle.ipynb` | the notebook: clone → generate → train → test → analyse → zip (one setting cell at the top, `QUICK_TEST=True` = ~10-minute dry run) |
| `config_185.py` | all data-generation / training settings in one place |
| `generate_data_185.py` | runs `data_generation/sereega/simu_extended_source.py` for train and test (in parallel, different seeds), verifies the files, writes `data_manifest.json` and the 185-electrode `model/ch_fsav_994.mat` |
| `train_185.py` | runs the team's `main_train.py` unchanged and adds `callbacks_185.py` (recording, one model file overwritten per epoch) |
| `callbacks_185.py` | per-step/per-epoch records, validation proxies reduced over both GPUs, `last.ckpt`/`last_model.pt`/`best_val_model.pt` |
| `eval_185.py` | runs the team's `eval.py` unchanged and exports per-sample metrics **with sample ids**, summary and example predictions |
| `analyze_185.py` | figures and `SUMMARY.md` from the records |
| `package_results_185.py` | adds models/reference files and zips the results (never the data) |

## Settings (the previous team's 75-electrode setup, with 185 electrodes)

Evidence for every value: `document/validation_data_preprocessing/09_data_generation_split_training_parameters.md` (thesis, progress report, Kaggle notebook, and the 75-electrode dataset on disk).

| | Value |
|---|---|
| Leadfield | `anatomy/leadfield_185_20k.mat` (185 × 994, average-referenced) |
| Data | SEREEGA extended sources, 1000 Hz, 500 ms (500 samples), 1–4 patches of order 1–4, ERP centre 250 ms / width 50 ms (script defaults) |
| Split | `train_6000` (seed 0, 4800 train / 1200 validation by `-per_valid 0.2`) and a separately generated `test_2000` (seed 1) |
| Model / optimisation | ViT-ESI 256-6-8-512, dropout 0.1; Adam lr 1e-3; cosine loss; 100 epochs; `-scaler linear`; `-eeg_snr 5` (white noise) |
| Batch / precision | 32 per GPU × 2 T4 (the team's Kaggle notebook; thesis: effective 8), `16-mixed`, 2 workers per GPU |

## What the run writes (`/kaggle/working/results_185/`, zipped to `results_185.zip`)

* `training/training_epochs.csv` — per epoch: train/validation loss (reduced over both GPUs), cosine similarity, validation peak-hit rate and localisation proxies, learning rate, epoch time, GPU memory
* `training/training_steps.csv` — loss of every optimisation step
* **Models: exactly 3 files, updated in place** — `last_model.pt` and `last.ckpt` (rewritten after every epoch; `last.ckpt` allows resuming) and `best_val_model.pt` (rewritten only when the validation loss improves); the team's own `VIT_model.pt` (lowest training loss) is also kept
* `test/test_per_sample_metrics.csv` (LE, AUC, nMSE, PSNR, time error for each test sample, with id and number of patches), `test_summary.json`, `test_predictions_sample.npz`
* `logs/` (every step's console log, `data_manifest.json`, `nvidia-smi`), `run_manifest.json` (settings, versions, commit, timings), `figures/`, `SUMMARY.md`, `reference/` (leadfield, electrodes, source positions, noise model)

The generated data (≈ 11 GB) go to `/kaggle/temp`, which is not saved with the notebook output.

## Notes

* `main_train.py` hard-codes the montage folder name `standard_1020` for SEREEGA data, so the data are written into `<root>/simulation/fsaverage/constrained/standard_1020/...` under their own root (never mixed with the 75-electrode data).
* The team's `eval.py` uses the localisation error of the *seed* regions of each patch (see the code) and white noise at `-eeg_snr`; the validation proxies logged during training are cheaper and only for monitoring.
* Resume after an interruption in the same session: set `RESUME = True` in the setting cell.
* Tested locally on CPU (data generation → training incl. a 2-process DDP run → test → analysis → zip) with `QUICK_TEST`; the 2-GPU NCCL run itself can only be confirmed on Kaggle.
