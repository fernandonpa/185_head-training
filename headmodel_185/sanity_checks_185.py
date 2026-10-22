"""Sanity checks of anatomy/leadfield_185_20k.mat (read only).

    python -I headmodel_185/sanity_checks_185.py

1 shape / order / reference   2 electrodes vs the scalp   3 projection invariance (electrodes shifted +-10 %)
4 single-source topographies  5 physical plausibility for all 994 regions   6 singular values
Results: headmodel_185/outputs/sanity_checks.json and figures/ (PNG).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import headmodel_185 as H  # noqa: E402

mne.set_log_level("ERROR")
out = HERE / "outputs"
fig_dir = out / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
res = {}

# ---- 1. shape, order, reference -----------------------------------------------------------------------------------
lf = loadmat(str(REPO / "anatomy" / "leadfield_185_20k.mat"), simplify_cells=True)
fwd, raw, names_lf = lf["fwd"], lf["fwd_raw"], [str(n) for n in lf["ch_names"]]
el = pd.read_csv(REPO / "experiments/data_experiments/validation_ready/ccepcoreg_185ch/electrodes_185.csv")
eloc = loadmat(str(REPO / "anatomy" / "electrode_185.mat"), simplify_cells=True)["eloc185"]
res["shape"] = list(fwd.shape)
res["order_equals_electrodes_185_csv"] = names_lf == list(el.name)
res["electrode_mat_order_equals_csv"] = [e["labels"] for e in eloc] == list(el.name)
res["G_equals_fwd"] = bool(np.array_equal(lf["G"], fwd))
res["avg_ref_max_column_mean"] = float(np.abs(fwd.mean(axis=0)).max())
res["raw_mean_over_electrodes_rel"] = float(np.abs(raw.mean(axis=0)).mean() / np.abs(raw).mean())
res["finite"] = bool(np.isfinite(fwd).all())
res["zero_columns"] = int((np.abs(fwd).sum(axis=0) == 0).sum())
print("1:", {k: res[k] for k in ["shape", "order_equals_electrodes_185_csv", "electrode_mat_order_equals_csv", "avg_ref_max_column_mean", "zero_columns"]})

# ---- 2. electrodes vs scalp (head frame -> MRI frame with fsaverage-trans) ----------------------------------------
fs_dir, _ = H.fsaverage_dirs()
trans = mne.read_trans(str(fs_dir / "bem" / "fsaverage-trans.fif"))                # head -> MRI
info = H.make_info(list(el.name), "GSN-HydroCel-256")
pos_head = np.array([c["loc"][:3] for c in info["chs"]])
pos_mri = mne.transforms.apply_trans(trans, pos_head)
scalp = [s for s in mne.read_bem_surfaces(str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem.fif")) if s["id"] == 4][0]
d, i = cKDTree(scalp["rr"]).query(pos_mri)
signed = ((pos_mri - scalp["rr"][i]) * scalp["nn"][i]).sum(1)
res["electrode_to_scalp_mm"] = dict(median=float(np.median(d) * 1e3), p95=float(np.percentile(d, 95) * 1e3), max=float(d.max() * 1e3),
                                     signed_median=float(np.median(signed) * 1e3), n_outside=int((signed > 0).sum()))
print("2:", res["electrode_to_scalp_mm"])

# ---- 3. projection invariance: shrink / expand the electrode cloud about its centroid and recompute the forward -----
src, bem, trans_path = H.make_bem_and_src()
base = np.load(out / "vertex_leadfield_185ch.npy").astype(float)
rm = H.region_mapping(REPO)
base_reg = H.sum_by_region(base, rm)
inv = {}
for factor in (0.9, 1.1):
    info2 = H.make_info(list(el.name), "GSN-HydroCel-256")
    c = pos_head.mean(axis=0)
    for ch, p in zip(info2["chs"], pos_head):
        ch["loc"][:3] = c + factor * (p - c)
    L2 = H.sum_by_region(H.vertex_leadfield(info2, src, bem, trans_path), rm)
    A = H.average_reference_matrix(185)
    a, b = A @ base_reg, A @ L2
    corr = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    inv[str(factor)] = dict(topo_corr_median=float(np.median(corr)), topo_corr_min=float(corr.min()), rel_diff=float(np.linalg.norm(a - b) / np.linalg.norm(a)))
res["electrode_cloud_scaling_sensitivity"] = inv
print("3:", inv)

# ---- 4/5. single-source topographies and plausibility -------------------------------------------------------------
ctx = loadmat(str(REPO / "anatomy" / "fs_cortex_20k.mat"), simplify_cells=True)
V = ctx["pos"]
cent_mri = np.array([V[rm == r].mean(0) for r in range(994)]) / 1000.0                  # m, MRI frame
cent_head = mne.transforms.apply_trans(np.linalg.inv(trans["trans"]), cent_mri)
ref_regions = [644, 243, 405, 640, 714]                                                    # ground-truth regions of five ccepcoreg runs
fig, axs = plt.subplots(1, len(ref_regions), figsize=(4 * len(ref_regions), 4))
for ax, r in zip(axs, ref_regions):
    im, _ = mne.viz.plot_topomap(fwd[:, r], info, axes=ax, show=False, sphere=None, cmap="RdBu_r")
    ax.set_title(f"region {r}: single source (average ref.)")
fig.tight_layout()
fig.savefig(fig_dir / "fig40_single_source_topographies.png", dpi=130)
plt.close(fig)

# strongest electrode vs the scalp point above the source (the electrode nearest to the radial projection of the centroid)
d_peak, d_over = [], []
tree_e = cKDTree(pos_head)
centre = pos_head.mean(axis=0)
for r in range(994):
    col = fwd[:, r]
    pk = np.argmax(np.abs(col))
    over = np.argmin(np.linalg.norm(pos_head - (centre + (cent_head[r] - centre) / np.linalg.norm(cent_head[r] - centre) * np.linalg.norm(pos_head - centre, axis=1).mean()), axis=1))
    d_peak.append(np.linalg.norm(pos_head[pk] - pos_head[over]) * 1e3)
d_peak = np.array(d_peak)
res["strongest_electrode_vs_overlying_electrode_mm"] = dict(median=float(np.median(d_peak)), p75=float(np.percentile(d_peak, 75)), p95=float(np.percentile(d_peak, 95)))
print("5:", res["strongest_electrode_vs_overlying_electrode_mm"])

# dipole polarity: a constrained source gives both a positive and negative pole -> amplitude ratio of the two extremes
pos_neg = np.array([abs(fwd[:, r].max() / fwd[:, r].min()) for r in range(994)])
res["pole_ratio_median"] = float(np.median(pos_neg))

# ---- 6. singular values -------------------------------------------------------------------------------------------
s = np.linalg.svd(fwd, compute_uv=False)
res["singular_values"] = dict(rank_numeric=int((s > s[0] * 1e-10).sum()), s1=float(s[0]), s_ratio_184_over_1=float(s[183] / s[0]), condition_first_184=float(s[0] / s[183]))
print("6:", res["singular_values"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(s / s[0]); ax[0].set_title("normalised singular values of the 185 x 994 leadfield"); ax[0].set_xlabel("index")
ax[1].hist(d_peak, bins=40, color="steelblue"); ax[1].set_title("distance: strongest electrode vs electrode above the source (mm)")
fig.tight_layout(); fig.savefig(fig_dir / "fig41_singular_values_and_peaks.png", dpi=130); plt.close(fig)

# 3-D view: scalp, electrodes, cortex centroids (head frame)
scalp_head = mne.transforms.apply_trans(np.linalg.inv(trans["trans"]), scalp["rr"])
fig = plt.figure(figsize=(14, 5))
for k, (e, a) in enumerate([(15, -90), (15, 0), (85, -90)]):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.scatter(*scalp_head[::6].T * 1e3, s=1, c="lightgray")
    ax.scatter(*(cent_head * 1e3).T, s=3, c="tab:red", alpha=0.5)
    ax.scatter(*(pos_head * 1e3).T, s=10, c="tab:blue")
    ax.view_init(e, a); ax.set_title(["left", "front", "top"][k])
fig.suptitle("fsaverage scalp (gray), 185 template electrodes (blue), 994 region centroids (red); MNE head frame, mm")
fig.tight_layout(); fig.savefig(fig_dir / "fig42_electrodes_scalp_cortex_3d.png", dpi=130); plt.close(fig)

(out / "sanity_checks.json").write_text(json.dumps(res, indent=2))
print("saved", out / "sanity_checks.json")
