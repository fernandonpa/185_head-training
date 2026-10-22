# kaggle_75sub — the 75-electrode pipeline (electrodes selected from the ccepcoreg 185)

Two notebooks, identical in structure to the 185-electrode ones in `kaggle_185/` (they use the same helper scripts; the environment variable `ESI_PIPELINE=75sub`, set in the first cell, selects the 75-electrode leadfield/electrode files):

| Notebook | What it does (automatically, on 2 × T4) |
|----------|------------------------------------------|
| `ViTESI_75sub_pipeline_kaggle.ipynb` | generates 6000 train + 2000 test SEREEGA samples with `anatomy/leadfield_75sub_20k.mat`, trains ViT-ESI, tests on the generated test set, writes records/figures and `results_75sub.zip` |
| `DeepSIF_75sub_pipeline_kaggle.ipynb` | the same for DeepSIF on the **same** data (dataset fingerprint check), `results_75sub_deepsif.zip` |

Settings are those of the previous team, as in the 185-electrode notebooks (`kaggle_185/README.md`): fs 1000 Hz, 500 ms, 1–4 patches of order 1–4, ERP centre 250 ms / width 50 ms, seeds 0 (train) / 1 (test), 4800/1200 train/validation split, Adam 1e-3, cosine loss, 100 epochs, 5 dB white noise, linear scaling.
The only differences from the 185-electrode run are the electrode set (75 selected electrodes) and its leadfield. For DeepSIF the choice `SETTINGS = "team_deepsif"` / `"match_vit75sub"` (batch size and precision) is explained in the notebook.

Real-data validation (local): `validation_preprocessing/build_validation_75sub.py` builds `experiments/data_experiments/validation_ready/ccepcoreg_75sub/` (same runs, labels and levels as the 185-electrode dataset; 75 channels in the original-montage order, average-referenced over the 75).
Use `variant_C_1kHz/L2_good` as `-real_data_dir` with `-leadfield_mat anatomy/leadfield_75sub_20k.mat`.
