"""75-channel noise model for the selected 75 electrodes, derived from the measured 185-channel noise model (anatomy/noise_model_185.mat).

    python -I headmodel_75sub/noise_75sub.py

Standard re-referencing identity: the 185-channel noise is average-referenced over the 185 channels. The selected channels, re-referenced to their own average
(the reference used for the 75-electrode data and leadfield), have the covariance  C75 = H C_sub H^T  with  H = I - 11^T/75  and  C_sub = the 75 x 75 block of C185.
The temporal colouring (PSD) is unchanged.  Writes anatomy/noise_model_75sub.mat and anatomy/realistic_noise_75sub.mat (20 surrogate segments, same layout as the
team's realistic_noise.mat: `data` (n, 500, 75), `npower` (n, 75) = sum of squares over time) and outputs/noise_75sub_validation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "headmodel_185"))
import noise_185 as N185  # noqa: E402

m = loadmat(str(REPO / "anatomy" / "noise_model_185.mat"), simplify_cells=True)
idx = np.array([int(e["idx185"]) for e in loadmat(str(REPO / "anatomy" / "electrode_75sub.mat"), simplify_cells=True)["eloc75sub"]])
C185 = m["rel_std"][:, None] * m["corr"] * m["rel_std"][None]
Hm = np.eye(75) - np.ones((75, 75)) / 75
C75 = Hm @ C185[np.ix_(idx, idx)] @ Hm.T
sd = np.sqrt(np.diag(C75) / np.diag(C75).mean())
corr = C75 / np.sqrt(np.outer(np.diag(C75), np.diag(C75)))
model = dict(corr=corr, rel_std=sd, psd_f=m["psd_f"], psd=m["psd"])
savemat(REPO / "anatomy" / "noise_model_75sub.mat", dict(**model, note="Spatial correlation/relative std of the 75 selected channels re-referenced to their own average (H C_sub H^T of the pooled 185-channel baseline noise), PSD as in noise_model_185.mat."))
segs = N185.generate(model, n_times=500, n_samples=20, rng=0).astype(np.float32)
savemat(REPO / "anatomy" / "realistic_noise_75sub.mat", dict(data=segs, npower=(segs ** 2).sum(axis=1),
        note="SURROGATE realistic noise (20 segments x 500 samples x 75 channels, float32) generated from measured statistics; same layout as realistic_noise.mat."))

gen = N185.generate(model, 500, 200, rng=1)
Rg = np.mean([np.corrcoef(g.T) for g in gen], axis=0)
w = lambda M: np.sort(np.linalg.eigvalsh(M))[::-1]  # noqa: E731
n90 = lambda e: int(np.searchsorted(np.cumsum(e) / e.sum(), 0.9) + 1)  # noqa: E731
res = dict(corr_cosine_generated_vs_target=float((Rg * corr).sum() / np.linalg.norm(Rg) / np.linalg.norm(corr)), n90_target=n90(w(corr)), n90_generated=n90(w(Rg)),
           generated_avg_ref_residual=float(np.abs(gen.mean(axis=2)).max()), target_rank=int((np.linalg.eigvalsh(C75) > 1e-10 * np.linalg.eigvalsh(C75).max()).sum()))
(HERE / "outputs" / "noise_75sub_validation.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
