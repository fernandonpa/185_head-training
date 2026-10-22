"""185-electrode head model / leadfield for the ccepcoreg montage (GSN-HydroCel-256 subset).

Follows the recipe of `model_training/utils/utils_headmodel.py::create_deepsif_based_headmodel` (read, NOT imported or modified):
fsaverage, ico5 *pial* source space (20484 vertices), 3-layer BEM (ico=4) with conductivities (0.33, 0.004125, 0.33) S/m,
the standard `fsaverage-trans.fif`, constrained (surface-normal) orientation, and the leadfield of a region = sum of the columns
of its vertices (region_mapping = anatomy/fs_cortex_20k_region_mapping.mat, 994 regions).

Only the electrode set differs: the names/order come from the validation dataset's `electrodes_185.csv`.
"""
from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

CONDUCTIVITY = (0.33, 0.004125, 0.33)          # brain, skull, scalp -- value used by create_deepsif_based_headmodel
BEM_ICO = 4                                    # same as the team's make_bem_model call
SRC_SPACING = "ico5"
SRC_SURFACE = "pial"
SUBJECT = "fsaverage"


def fsaverage_dirs():
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=0))
    return fs_dir, str(fs_dir.parent)


def make_info(ch_names, montage_name: str) -> mne.Info:
    """Info with the template electrode positions in the MNE *head* frame (montage fiducials define the frame)."""
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(list(ch_names), 1000.0, ch_types=["eeg"] * len(ch_names))
    info.set_montage(montage, on_missing="raise")
    return info


def make_bem_and_src(conductivity=CONDUCTIVITY):
    fs_dir, subjects_dir = fsaverage_dirs()
    src = mne.setup_source_space(SUBJECT, spacing=SRC_SPACING, surface=SRC_SURFACE, subjects_dir=subjects_dir, add_dist=False, verbose=0)
    model = mne.make_bem_model(subject=SUBJECT, ico=BEM_ICO, conductivity=conductivity, subjects_dir=subjects_dir, verbose=0)
    bem = mne.make_bem_solution(model, verbose=0)
    trans = str(fs_dir / "bem" / "fsaverage-trans.fif")
    return src, bem, trans


def vertex_leadfield(info: mne.Info, src, bem, trans) -> np.ndarray:
    """(n_channels, 20484) constrained-orientation leadfield (MNE units: V per A*m)."""
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, n_jobs=1, verbose=0)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0)
    return np.asarray(fwd["sol"]["data"], dtype=np.float64)


def region_mapping(repo: Path) -> np.ndarray:
    rm = loadmat(str(repo / "anatomy" / "fs_cortex_20k_region_mapping.mat"), simplify_cells=True)["rm"].astype(int)
    assert rm.shape == (20484,) and rm.min() == 0 and rm.max() == 993 and len(np.unique(rm)) == 994, (rm.shape, rm.min(), rm.max())
    return rm


def sum_by_region(L: np.ndarray, rm: np.ndarray) -> np.ndarray:
    out = np.zeros((L.shape[0], 994))
    for r in range(994):
        out[:, r] = L[:, rm == r].sum(axis=1)
    return out


def average_reference_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n
