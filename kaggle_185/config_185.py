"""Single source of truth for the 185-electrode pipeline settings.

Every value below reproduces what the previous team used for the 75-electrode work (evidence in
document/validation_data_preprocessing/09_data_generation_split_training_parameters.md), except the electrode
count (185) and the leadfield file.  Nothing here modifies the team's scripts: they are called with these arguments.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data generation (SEREEGA)
LEADFIELD = REPO / "anatomy" / "leadfield_185_20k.mat"          # 185 x 994, average-referenced (keys: G, fwd, fwd_raw)
ELECTRODES = REPO / "anatomy" / "electrode_185.mat"
SOURCES_994 = REPO / "anatomy" / "sources_fsav_994.mat"
N_ELECTRODES = 185
N_SOURCES = 994

# The folder level called "montage" must be `standard_1020`: main_train.py hard-codes that name when it builds the
# SEREEGA dataset.  The data live in their own root, so they never mix with the 75-electrode data.
MONTAGE_DIR = "standard_1020"
SOURCE_SPACE = "fsav_994"
ORIENTATION = "constrained"
SUBJECT = "fsaverage"

SEREEGA = dict(
    fs=1000,            # Hz   (75-electrode data: fs 1000, 500 ms -> 500 samples)
    duree=500,          # ms
    n_patch_min=1, n_patch_max=5,       # upper bound exclusive -> 1..4 patches (as measured on the 75-electrode data)
    order_min=1, order_max=5,           # 1..4
    margin=2,
    sig_type="erp", amplitude=1, center=250, width=50,
)
N_TIMES = SEREEGA["fs"] * SEREEGA["duree"] // 1000   # 500

SPLITS = {
    "train": dict(name="train_6000", n=6000, seed=0),
    "test": dict(name="test_2000", n=2000, seed=1),      # a different seed: separate random draw
}

# ---------------------------------------------------------------- training (ViT-ESI, same hyper-parameters as the 75-electrode run)
TRAIN = dict(
    model="VIT", simu_type="sereega", loss="cosine", scaler="linear", eeg_snr=5, per_valid=0.2, n_epochs=100,
    vit_embed_dim=256, vit_depth=6, vit_heads=8, vit_mlp_dim=512, vit_dropout=0.1,
    # optimizer: Adam, lr 1e-3, set inside models/vit.py / main_train.py (not changed here)
    batch_per_gpu=32, num_workers=2, precision="16-mixed",   # the team's Kaggle notebook values (thesis: effective batch 8)
)
