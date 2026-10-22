"""Test with the team's `model_training/eval.py` (unchanged) and export analysis-ready records.

Use it exactly like eval.py (same arguments), after setting ESI185_TEST_RECORDS_DIR:

    ESI185_TEST_RECORDS_DIR=results_185/test python kaggle_185/eval_185.py test_2000 -root_simu <root> ... -mets eeg_vit

eval.py is executed with runpy; when it finishes, its in-memory results are read and written to the records directory:
  test_per_sample_metrics.csv   one row per test sample WITH its id and ground-truth description (eval.py itself drops the ids:
                                its order comes from a random split)  [localisation error mm, time error ms, nMSE, AUC, PSNR]
  test_summary.json             mean/std/median/quartiles overall and by number of patches
  test_predictions_sample.npz   input EEG, ground-truth sources and predicted sources of a few test samples (for plots)
  eval_original/                the CSV files eval.py wrote itself
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MT = REPO / "model_training"

sys.argv[0] = os.path.abspath(sys.argv[0])
sys.path.insert(0, str(MT))
os.chdir(MT)

METRIC_NAMES = {"loc_error": ("loc_error_mm", 1e3), "time_error": ("time_error_ms", 1e3), "nmse": ("nmse", 1.0), "auc": ("auc", 1.0), "psnr": ("psnr_db", 1.0)}


def export(g: dict, out: Path, n_pred: int = 12) -> None:
    import numpy as np
    import pandas as pd
    import torch

    out.mkdir(parents=True, exist_ok=True)
    val_ds = g["val_ds"]
    ds = val_ds.dataset
    idxs = list(val_ds.indices)
    md_keys = list(ds.md_dict.keys())
    methods = [m for m in g["methods"] if m != "gt"]
    dicts = {"loc_error": g["loc_error_dict"], "time_error": g["time_error_dict"], "nmse": g["nmse_dict"], "auc": g["auc_dict"], "psnr": g["psnr_dict"]}

    rows = []
    for pos, k in enumerate(idxs):
        md = ds.md_dict[md_keys[k]]
        patches = md["act_src"]
        rows.append(dict(position=pos, dataset_index=k, sample_key=md_keys[k], sample_id=md["id"], n_patch=md["n_patch"], seeds=json.dumps(md["seeds"]),
                         orders=json.dumps(md["orders"]), n_active_regions=sum(len(v) for v in patches.values()),
                         smallest_patch=min(len(v) for v in patches.values()), largest_patch=max(len(v) for v in patches.values())))
    df = pd.DataFrame(rows)
    for m in methods:
        for key, (col, scale) in METRIC_NAMES.items():
            v = np.asarray(dicts[key][m]).reshape(-1)
            if len(v) != len(df):
                raise RuntimeError(f"metric array length {len(v)} != {len(df)} samples ({m}, {key})")
            df[f"{m}_{col}"] = v * scale
    df.to_csv(out / "test_per_sample_metrics.csv", index=False)

    summary = dict(n_samples=len(df), methods=methods, noise_snr_db=g["args"].eeg_snr, simu_name=g["args"].simu_name, weights=str(g.get("vit_model_path", "") or g.get("deep_sif_model_path", "")))
    for m in methods:
        cols = [f"{m}_{c}" for c, _ in METRIC_NAMES.values()]
        desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        summary[m] = {c: {k: float(desc.loc[c, k]) for k in ("mean", "std", "min", "25%", "50%", "75%", "max")} for c in cols}
        summary[f"{m}_by_n_patch"] = {str(p): {c: float(sub[c].mean()) for c in cols} | {"n": int(len(sub))} for p, sub in df.groupby("n_patch")}
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2))

    eval_dir = Path(g["eval_results_path"])
    if eval_dir.is_dir():
        shutil.copytree(eval_dir, out / "eval_original", dirs_exist_ok=True)

    nn_methods = {"eeg_vit": ("eeg_vit", "vit_device"), "deep_sif": ("deep_sif", "sif_device")}      # method -> (model variable, device variable) in eval.py
    m_nn = next((m for m in methods if m in nn_methods and nn_methods[m][0] in g), None)
    if m_nn is not None:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(idxs), size=min(n_pred, len(idxs)), replace=False).tolist())
        model, dev, utl, fwd = g[nn_methods[m_nn][0]], g[nn_methods[m_nn][1]], g["utl"], torch.from_numpy(g["fwd"])
        eeg_l, gt_l, pr_l, ids_l = [], [], [], []
        for p in pick:
            k = idxs[p]
            M, j = ds[k]
            M, j = M.float(), j.float()
            M_un, j_un = M * ds.max_eeg[k], j * ds.max_src[k]
            with torch.no_grad():
                j_hat = model(M.unsqueeze(0).to(dev)).squeeze().float().cpu()
            j_hat = utl.gfp_scaling(M_un, j_hat, fwd)
            eeg_l.append(M.numpy().astype("float16")); gt_l.append(j_un.numpy().astype("float16")); pr_l.append(j_hat.numpy().astype("float16")); ids_l.append(int(df.sample_id[p]))
        np.savez_compressed(out / "test_predictions_sample.npz", method=m_nn, eeg=np.stack(eeg_l), gt=np.stack(gt_l), pred=np.stack(pr_l), sample_id=np.array(ids_l), position=np.array(pick))


if __name__ == "__main__":
    g = runpy.run_path(str(MT / "eval.py"), run_name="__main__")
    rec = os.environ.get("ESI185_TEST_RECORDS_DIR")
    if rec:
        try:
            export(g, Path(rec))
            print(f"[eval_185] records written to {rec}", flush=True)
        except Exception:
            Path(rec).mkdir(parents=True, exist_ok=True)
            (Path(rec) / "export_error.txt").write_text(traceback.format_exc())
            print("[eval_185] exporting the records FAILED:\n" + traceback.format_exc(), flush=True)
            sys.exit(1)
    else:
        print("[eval_185] ESI185_TEST_RECORDS_DIR not set: only eval.py's own output was written", flush=True)
