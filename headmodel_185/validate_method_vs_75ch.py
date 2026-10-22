"""Method validation: rebuild the team's 75-electrode leadfield with our code on the 72 electrodes that exist in MNE's montage and compare.

    python -I headmodel_185/validate_method_vs_75ch.py

Compares `anatomy/leadfield_75_20k.mat` (read only) with our MNE-based leadfield on the shared electrodes (all but Nz, TP11, TP12).
Writes headmodel_185/outputs/validation_vs_75ch.json and a cached vertex leadfield.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import headmodel_185 as H  # noqa: E402

out = HERE / "outputs"
out.mkdir(exist_ok=True)
t0 = time.time()

e75 = loadmat(str(REPO / "anatomy" / "electrode_75.mat"), simplify_cells=True)["eloc75"]
labels75 = [e["labels"] for e in e75]
LF75 = loadmat(str(REPO / "anatomy" / "leadfield_75_20k.mat"), simplify_cells=True)["fwd"]           # (75, 994)
import mne  # noqa: E402

montage = mne.channels.make_standard_montage("standard_1005")
common = [l for l in labels75 if l in montage.ch_names]
missing = [l for l in labels75 if l not in montage.ch_names]
idx75 = [labels75.index(l) for l in common]
print(f"shared electrodes: {len(common)} ; not in MNE montage: {missing}")

info = H.make_info(common, "standard_1005")
src, bem, trans = H.make_bem_and_src()
Lv = H.vertex_leadfield(info, src, bem, trans)                                                    # (72, 20484)
np.save(out / "vertex_leadfield_72ch_standard1005.npy", Lv.astype(np.float32))
rm = H.region_mapping(REPO)
mine_sum = H.sum_by_region(Lv, rm)
mine_mean = mine_sum / np.bincount(rm, minlength=994)[None, :]
team = LF75[idx75]

A = H.average_reference_matrix(len(common))


def topo_corr(a, b):
    a, b = A @ a, A @ b
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return num / den


res = {"n_shared": len(common), "missing": missing, "seconds": round(time.time() - t0, 1)}
for name, M in [("sum", mine_sum), ("mean", mine_mean)]:
    c = topo_corr(M, team)
    k = float((M * team).sum() / (M * M).sum())                                                 # least-squares scale team ~ k * mine (raw, no re-reference)
    res[name] = dict(topo_corr_median=float(np.median(c)), topo_corr_p05=float(np.percentile(c, 5)), topo_corr_min=float(c.min()),
                     scale_k_overall=k, rel_residual_after_scale=float(np.linalg.norm(team - k * M) / np.linalg.norm(team)))
# column-wise scale consistency (is team = k * mine with ONE k?)
kc = (mine_sum * team).sum(axis=0) / (mine_sum ** 2).sum(axis=0)
res["column_scale_k"] = dict(median=float(np.median(kc)), p05=float(np.percentile(kc, 5)), p95=float(np.percentile(kc, 95)), cv=float(np.std(kc) / abs(np.mean(kc))))
# reference convention of both leadfields (mean over electrodes of each column relative to its magnitude)
res["team_mean_over_electrodes_rel"] = float(np.abs(team.mean(axis=0)).mean() / np.abs(team).mean())
res["mine_mean_over_electrodes_rel"] = float(np.abs(mine_sum.mean(axis=0)).mean() / np.abs(mine_sum).mean())
(out / "validation_vs_75ch.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
