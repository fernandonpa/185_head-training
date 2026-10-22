"""Select the 75 of the 185 ccepcoreg electrodes that best reproduce the montage of the original 75-electrode work.

Definitions (standard techniques, no tuning on any model result):
* target electrodes: the 75 electrodes of the original montage (`anatomy/electrode_75.mat`, 10-10 names). 72 of them exist in MNE's
  `standard_1005` template (head frame); the nasion electrode `Nz` is the montage's nasion fiducial; `TP11`/`TP12` are not in any MNE template and
  are placed with the similarity transform that maps the original (Curry-space) coordinates onto the MNE head frame, fitted on the 72 shared electrodes.
* candidates: the 185 electrodes released with ccepcoreg = positions of the EGI GSN-HydroCel-256 template (MNE head frame).
* both sets are projected onto the fsaverage scalp surface (`fsaverage-5120-5120-5120-bem.fif`), exactly what MNE does with EEG electrodes when it
  computes a forward solution; distances are measured between the projected points.
* assignment: one-to-one (every candidate used at most once) by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`); variants: minimum sum of
  distances, minimum sum of squared distances, minimum bottleneck (smallest possible worst distance, then minimum sum), and a naive greedy baseline.
* quality of a subset = fidelity of its forward model to the forward model of the original montage (same MNE recipe): per-region topography correlation and
  relative error of the average-referenced region leadfields.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "headmodel_185"))
import headmodel_185 as H  # noqa: E402

from mne.surface import _project_onto_surface  # noqa: E402  (the function MNE's forward code uses for EEG electrodes)


# ------------------------------------------------------------------------------------------------ geometry
def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity transform dst ~ c R src + t (rotation only, no reflection)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt
    c = (sig * np.diag(Dm)).sum() / ((S ** 2).sum() / len(src))
    return c, R, mu_d - c * R @ mu_s


def original_montage():
    """labels, positions in the MNE head frame (m), source of each position, fit statistics of the transform."""
    e75 = loadmat(str(REPO / "anatomy" / "electrode_75.mat"), simplify_cells=True)["eloc75"]
    labels = [e["labels"] for e in e75]
    curry = np.array([[e["X"], e["Y"], e["Z"]] for e in e75], float)                       # mm, Curry space
    montage = mne.channels.make_standard_montage("standard_1005")
    common = [l for l in labels if l in montage.ch_names]
    info = H.make_info(common, "standard_1005")
    pos_common = np.array([c["loc"][:3] for c in info["chs"]])                             # m, head frame
    idx_common = [labels.index(l) for l in common]
    c, R, t = umeyama(curry[idx_common], pos_common * 1000.0)                              # mm
    fitted = (c * (R @ curry.T)).T + t
    resid = np.linalg.norm(fitted[idx_common] - pos_common * 1000.0, axis=1)
    pos = np.zeros((len(labels), 3))
    source = []
    for i, l in enumerate(labels):
        if l in common:
            pos[i] = pos_common[common.index(l)]
            source.append("MNE standard_1005")
        else:
            pos[i] = fitted[i] / 1000.0
            source.append("similarity transform of the original coordinates")
    # Nz: the nasion fiducial of the montage (the electrode sits on the nasion)
    nas = [d["r"] for d in info["dig"] if d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL and d["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION][0]
    i_nz = labels.index("Nz")
    nz_transform_vs_nasion = float(np.linalg.norm(fitted[i_nz] - nas * 1000.0))
    pos[i_nz] = nas
    source[i_nz] = "montage nasion fiducial"
    stats = dict(n_shared=len(common), fit_scale=float(c), fit_residual_median_mm=float(np.median(resid)), fit_residual_max_mm=float(resid.max()),
                 nz_transform_vs_nasion_mm=nz_transform_vs_nasion, missing=[l for l in labels if l not in common])
    return labels, pos, source, stats


def candidate_montage(names):
    info = H.make_info(list(names), "GSN-HydroCel-256")
    return np.array([c["loc"][:3] for c in info["chs"]])                                    # m, head frame


def scalp_surface():
    fs_dir, _ = H.fsaverage_dirs()
    surf = [s for s in mne.read_bem_surfaces(str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem.fif"), verbose=False) if s["id"] == 4][0]
    trans = mne.read_trans(str(fs_dir / "bem" / "fsaverage-trans.fif"))                    # head -> MRI
    return surf, trans


def project_to_scalp(pos_head_m: np.ndarray, surf, trans) -> np.ndarray:
    """Project head-frame positions (m) onto the scalp; returns head-frame positions (m) of the projected points."""
    pos_mri = mne.transforms.apply_trans(trans, pos_head_m)
    proj_mri = _project_onto_surface(pos_mri, surf, project_rrs=True)[2]
    return mne.transforms.apply_trans(np.linalg.inv(trans["trans"]), proj_mri)


# ------------------------------------------------------------------------------------------------ assignment methods
def hungarian(cost: np.ndarray) -> np.ndarray:
    r, c = linear_sum_assignment(cost)
    out = np.empty(cost.shape[0], int)
    out[r] = c
    return out


def bottleneck(D: np.ndarray):
    """Smallest possible worst-case distance of a one-to-one assignment, then the minimum total distance among those."""
    vals = np.unique(D)
    lo, hi = 0, len(vals) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        match = maximum_bipartite_matching(csr_matrix(D <= vals[mid]), perm_type="column")
        if (match >= 0).sum() == D.shape[0]:
            hi = mid
        else:
            lo = mid + 1
    thr = vals[lo]
    cost = np.where(D <= thr, D, 1e6)
    return hungarian(cost), float(thr)


def greedy(D: np.ndarray) -> np.ndarray:
    """Naive baseline: repeatedly fix the globally closest (target, candidate) pair."""
    D = D.copy()
    out = -np.ones(D.shape[0], int)
    for _ in range(D.shape[0]):
        i, j = np.unravel_index(np.argmin(D), D.shape)
        out[i] = j
        D[i, :] = np.inf
        D[:, j] = np.inf
    return out


def candidate_assignments(D: np.ndarray) -> dict:
    out = {"hungarian_sum": hungarian(D), "hungarian_sq": hungarian(D ** 2), "greedy": greedy(D)}
    out["bottleneck"], _ = bottleneck(D)
    return out


# ------------------------------------------------------------------------------------------------ forward-model fidelity
def region_topography_fidelity(L_target: np.ndarray, L_sub: np.ndarray) -> dict:
    """Both (75, 994): compare the average-referenced region topographies."""
    A = H.average_reference_matrix(L_target.shape[0])
    a, b = A @ L_target, A @ L_sub
    corr = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    rel = np.linalg.norm(a - b, axis=0) / np.linalg.norm(a, axis=0)
    return dict(corr_median=float(np.median(corr)), corr_p05=float(np.percentile(corr, 5)), corr_min=float(corr.min()), rel_err_median=float(np.median(rel)),
                rel_err_p95=float(np.percentile(rel, 95)), rel_err_global=float(np.linalg.norm(a - b) / np.linalg.norm(a))), corr, rel


def target_leadfield(labels, pos_head_m, rm, cache: Path | None = None) -> np.ndarray:
    """Region leadfield (75, 994, MNE units, not re-referenced) of the original montage on the MNE template (same recipe as the 185-electrode build)."""
    if cache is not None and cache.exists():
        Lv = np.load(cache).astype(float)
    else:
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(labels, pos_head_m)), coord_frame="head")
        info = mne.create_info(list(labels), 1000.0, ch_types="eeg")
        info.set_montage(montage)
        src, bem, trans = H.make_bem_and_src()
        Lv = H.vertex_leadfield(info, src, bem, trans)
        if cache is not None:
            np.save(cache, Lv.astype(np.float32))
    return H.sum_by_region(Lv, rm)
