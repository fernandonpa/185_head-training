"""Build the leadfield (75 x 994) of the 75 electrodes selected from the ccepcoreg 185 (see electrode_selection_75sub.ipynb).

    python -I headmodel_75sub/build_headmodel_75sub.py

Same recipe as headmodel_185 (fsaverage ico5 pial, 3-layer BEM 0.33/0.004125/0.33 S/m, constrained orientation, region = sum of its vertices); the electrodes are the
selected GSN-HydroCel-256 template positions, in the order of the original 75-electrode montage (`anatomy/electrode_75.mat`).
Writes: anatomy/leadfield_75sub_20k.mat (fwd, G = average-referenced over the 75 channels; fwd_raw), headmodel_75sub/outputs/build_metadata.json,
        outputs/sanity_checks_75sub.json and figures/fig73_*.png.   (anatomy/electrode_75sub.mat is written by the selection notebook.)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat, savemat  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "headmodel_185"))
import headmodel_185 as H  # noqa: E402

mne.set_log_level("ERROR")
out = HERE / "outputs"
(out / "figures").mkdir(parents=True, exist_ok=True)
t0 = time.time()

e75 = loadmat(str(REPO / "anatomy" / "electrode_75sub.mat"), simplify_cells=True)["eloc75sub"]
names = [str(e["labels"]) for e in e75]
orig = [str(e["original_label"]) for e in e75]
idx185 = np.array([int(e["idx185"]) for e in e75])
assert len(names) == 75 and len(set(names)) == 75
info = H.make_info(names, "GSN-HydroCel-256")
assert info.ch_names == names
src, bem, trans = H.make_bem_and_src()
Lv = H.vertex_leadfield(info, src, bem, trans)
assert Lv.shape == (75, 20484) and np.isfinite(Lv).all()
rm = H.region_mapping(REPO)
raw = H.sum_by_region(Lv, rm)
avg = H.average_reference_matrix(75) @ raw
savemat(REPO / "anatomy" / "leadfield_75sub_20k.mat", dict(
    fwd=avg, G=avg, fwd_raw=raw, ch_names=np.array(names, dtype=object), original_labels=np.array(orig, dtype=object), conductivity=np.array(H.CONDUCTIVITY),
    note="fwd/G: average-referenced over the 75 channels; fwd_raw: as returned by MNE. Units V/(A*m). Region = sum over the region's vertices. Rows follow the order of the original 75-electrode montage."))

# ---- checks -------------------------------------------------------------------------------------------------------------------------------
chk = {}
chk["shape"] = list(avg.shape)
chk["avg_ref_max_column_mean"] = float(np.abs(avg.mean(axis=0)).max())
chk["numerical_rank"] = int((np.linalg.svd(avg, compute_uv=False) > 1e-10 * np.linalg.svd(avg, compute_uv=False)[0]).sum())
cache185 = REPO / "headmodel_185" / "outputs" / "vertex_leadfield_185ch.npy"
if cache185.exists():                       # the forward of an electrode does not depend on the other electrodes: rows of the 185 forward must be identical
    L185 = np.load(cache185).astype(float)[idx185]
    chk["max_rel_diff_vs_rows_of_185_forward"] = float(np.abs(Lv - L185).max() / np.abs(Lv).max())
fs_dir, _ = H.fsaverage_dirs()
tr = mne.read_trans(str(fs_dir / "bem" / "fsaverage-trans.fif"))
pos_head = np.array([c["loc"][:3] for c in info["chs"]])
scalp = [s for s in mne.read_bem_surfaces(str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem.fif")) if s["id"] == 4][0]
from scipy.spatial import cKDTree  # noqa: E402

d, _ = cKDTree(scalp["rr"]).query(mne.transforms.apply_trans(tr, pos_head))
chk["electrode_to_scalp_vertex_mm"] = dict(median=float(np.median(d) * 1e3), max=float(d.max() * 1e3))
(out / "sanity_checks_75sub.json").write_text(json.dumps(chk, indent=2))
(out / "build_metadata_75sub.json").write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), mne=mne.__version__, n_electrodes=75,
                                                              conductivity=H.CONDUCTIVITY, leadfield_shape=list(avg.shape), abs_median_raw=float(np.median(np.abs(raw))),
                                                              electrodes=names, original_labels=orig), indent=2))
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, r in zip(axs, [644, 243, 405, 640]):
    mne.viz.plot_topomap(avg[:, r], info, axes=ax, show=False, sphere=None, cmap="RdBu_r"); ax.set_title(f"region {r}: single source")
fig.tight_layout(); fig.savefig(out / "figures" / "fig73_single_source_topographies_75sub.png", dpi=130)
print(json.dumps(chk, indent=2))
