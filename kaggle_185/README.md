# kaggle_185 — generate data, train, test and package the 185-electrode ViT-ESI (SEREEGA)

A small helper layer **around** the team's scripts: nothing in `data_generation/` or `model_training/` is modified.
Open `ViTESI_185_pipeline_kaggle.ipynb` on Kaggle (Accelerator **GPU T4 x2**, Internet **On**) and **Run All**.

| File | Role |
|------|------|
| `ViTESI_185_pipeline_kaggle.ipynb` | the ViT-ESI notebook: clone → generate → train → test → analyse → zip (one setting cell at the top, `QUICK_TEST=True` = ~10-minute dry run) |
| `DeepSIF_185_pipeline_kaggle.ipynb` | the same pipeline for the **DeepSIF** baseline, on the **same** generated data (see below) |
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

## Comparing ViT-ESI and DeepSIF (same data, same test)

* Both notebooks generate the data with the same code, leadfield, sizes and seeds (train 6000 / seed 0, test 2000 / seed 1). The generation is deterministic: `logs/data_manifest.json` contains a **fingerprint** of every split
  (`md_sha256`: all metadata files; `eeg_sha256`: 20 EEG samples) and two independent runs gave identical fingerprints (checked locally). In the DeepSIF notebook you can paste the ViT run's `md_sha256` values into
  `EXPECTED_FINGERPRINT` to make the notebook stop if the data differ.
* The test is the same `eval.py` on the same `test_2000`, evaluated at the same SNR (5 dB), so `test_per_sample_metrics.csv` of the two runs can be joined on `sample_id`.
* **Training settings.** The previous team's DeepSIF notebook uses a global batch of 8 and 32-bit precision, while their ViT-ESI notebook (and `ViTESI_185_pipeline_kaggle.ipynb`) uses a global batch of 64 and 16-bit mixed precision.
  `SETTINGS = "team_deepsif"` (default of the DeepSIF notebook) reproduces the DeepSIF notebook; `SETTINGS = "match_vit185"` trains DeepSIF with exactly the settings of the ViT-185 notebook (strictly identical setup).
  Everything else is identical for both models: 100 epochs, Adam lr 1e-3, cosine loss, 4800/1200 split, `-eeg_snr 5`, `-scaler linear`.
* DeepSIF with 185 electrodes has 22 528 396 trainable parameters (thesis: 22.38 M with 75 electrodes). Its zip omits `last.ckpt` (270 MB, only needed to resume) so that it stays around 250 MB.
