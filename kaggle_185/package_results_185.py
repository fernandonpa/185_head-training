"""Collect everything needed for the local validation and further analysis into the results folder and zip it.

    python kaggle_185/package_results_185.py --results results_185 --zip results_185.zip

Adds to <results>:
  models/                best_val_model.pt, last_model.pt, last.ckpt (moved here from training/: the files overwritten each epoch), VIT_model.pt (the team's
                         own "lowest train loss" save, from main_train_output)
  reference/             leadfield_185_20k.mat, electrode_185.mat, sources_fsav_994.mat, region mapping, noise model (what the model was
                         trained with: needed to use/evaluate the model locally)
  README_RESULTS.md      what is inside and how to use it
  file_listing.csv       every file with size and sha256 (models)
The zip never contains the generated data.  The team's `pl_checkpoints/` (a 43 MB duplicate of a model) is left out of the zip.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

README = """# Results of the 185-electrode ViT-ESI run (SEREEGA data, trained on Kaggle)

| Folder / file | Content |
|---------------|---------|
| `SUMMARY.md`, `figures/` | summary tables and plots (learning curves, test metrics, example predictions) |
| `run_manifest.json` | settings, environment, git commit, timings of the whole run |
| `logs/` | console logs of every step (data generation, training, test, analysis), `data_manifest.json`, `nvidia-smi` snapshots |
| `training/` | `training_epochs.csv` (per-epoch train/val loss, cosine similarity, validation localisation proxies, time, GPU memory), `training_steps.csv` (per step), `best_val.json`, `run_summary.json`, `fit_info.json`, `pl_csv/` |
| `test/` | `test_per_sample_metrics.csv` (every test sample with id, number of patches, LE, AUC, nMSE, PSNR, time error), `test_summary.json`, `test_predictions_sample.npz`, `eval_original/` |
| `models/best_val_model.pt` | **weights with the lowest validation loss (used for the test; use this for validation on real data)** |
| `models/last_model.pt`, `models/last.ckpt` | weights / full checkpoint of the final epoch (the only copies kept; overwritten after every epoch) |
| `models/VIT_model.pt` | the team's own save (checkpoint with the lowest training loss) |
| `reference/` | leadfield (185 x 994), electrode list, source positions, region mapping and noise model used for this model |
| `main_train_output/` | the team's own output tree (TensorBoard logs, `trained_models/`) |

## Using the model for the real-data validation (local machine)

Model: `models.vit.EEGViTpl(num_sensor=185, num_source=994, n_times=500, embed_dim=256, depth=6, num_heads=8, mlp_dim=512, dropout=0.1, optimizer=None, lr=1e-3, criterion=None)`,
weights = `models/best_val_model.pt` (a Lightning-module `state_dict`).
Input: `(batch, 185, 500)`, channel order = `reference/electrode_185.mat` (= `electrodes_185.csv` of the validation dataset), average-referenced,
500 samples at **1000 Hz**, scaled by max-abs; output `(batch, 994, 500)` source activity for the 994 regions.
With the team's script (from `model_training/`):

    python predict/eval_real_vit.py <name> -real_data_dir <validation data folder> -leadfield_mat <results>/reference/leadfield_185_20k.mat \\
        -train_run_dir <results>/main_train_output/<...>/<run folder> -weights_path <results>/models/best_val_model.pt -n_times 500
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    (res / "models").mkdir(exist_ok=True)
    (res / "reference").mkdir(exist_ok=True)

    tr = res / "training"
    for name in ("best_val_model.pt", "last_model.pt", "last.ckpt"):          # moved (not copied): one copy only, in models/
        if (tr / name).exists():
            shutil.move(str(tr / name), str(res / "models" / name))
    cand = sorted((res / "main_train_output").glob("**/trained_models/VIT_model.pt"))
    if cand:
        shutil.copy2(cand[-1], res / "models" / "VIT_model.pt")
    for name in ("leadfield_185_20k.mat", "electrode_185.mat", "sources_fsav_994.mat", "fs_cortex_20k_region_mapping.mat", "noise_model_185.mat"):
        if (REPO / "anatomy" / name).exists():
            shutil.copy2(REPO / "anatomy" / name, res / "reference" / name)
    (res / "README_RESULTS.md").write_text(README)

    zpath = Path(a.zip)
    rows = ["path,size_bytes,sha256"]
    t0 = time.time()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(res.rglob("*")):
            if not p.is_file() or p.suffix == ".tmp" or "pl_checkpoints" in p.parts or p.resolve() == zpath.resolve():
                continue
            rel = p.relative_to(res)
            z.write(p, Path(res.name) / rel)
            rows.append(f"{rel.as_posix()},{p.stat().st_size},{sha256(p) if rel.parts[0] == 'models' else ''}")
        listing = res / "file_listing.csv"
        listing.write_text("\n".join(rows))
        z.write(listing, Path(res.name) / "file_listing.csv")
    print(f"zip: {zpath} ({zpath.stat().st_size / 2**20:.1f} MB, {len(rows) - 1} files, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
