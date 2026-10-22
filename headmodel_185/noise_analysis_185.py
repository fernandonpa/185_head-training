"""Properties of the real noise in the validation data (pre-stimulus baseline of the *averaged* runs) and the training-SNR they imply.

    python -I headmodel_185/noise_analysis_185.py

Reads (read only): validation_ready/ccepcoreg_185ch (variant A, level L2) and anatomy/realistic_noise.mat, electrode_75.mat, electrode_185.csv.
Writes: headmodel_185/outputs/noise_analysis.json, noise_run_stats.csv, figures/fig50_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.signal import welch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DS = REPO / "experiments/data_experiments/validation_ready/ccepcoreg_185ch"
out = HERE / "outputs"
fig_dir = out / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
FS = 500
B0, B1 = 5, 125            # baseline samples: t = -290 .. -52 ms (first 5 dropped: FIR edge of the decimation)

man = pd.read_csv(DS / "manifest.csv")
man = man[man.L2_good].reset_index(drop=True)
el = pd.read_csv(DS / "electrodes_185.csv")
pos = el[["x_mm", "y_mm", "z_mm"]].to_numpy()
D = np.linalg.norm(pos[:, None] - pos[None], axis=2)

segs, stats, covs, subj = [], [], [], []
for _, r in man.iterrows():
    m = loadmat(DS / r.file_variant_A, simplify_cells=True)
    x = m["data"].astype(float) * float(m["scale_uV"])                             # (185, 500) uV, average-referenced, baseline-corrected
    b = x[:, B0:B1]
    p_ch = (b ** 2).mean(axis=1)
    e_win = (x ** 2).sum()                                                       # whole-window energy (training definition uses the whole matrix)
    n_win = (b ** 2).mean() * x.shape[1] * x.shape[0]                             # noise energy over the window assuming stationarity
    snr_matrix = 10 * np.log10(max(e_win - n_win, 1e-12) / n_win)
    sig_p = ((x ** 2).mean(axis=1) - p_ch).clip(1e-12)
    snr_elec = 10 * np.log10(sig_p / p_ch)                                       # per-electrode (DeepSIF's definition)
    stats.append(dict(subj=r.subj, run=r.run, noise_rms_uV=float(np.sqrt(p_ch.mean())), snr_matrix_db=float(snr_matrix), snr_elec_median_db=float(np.median(snr_elec)),
                      snr_evoked_db=float(r.snr_evoked_db)))
    segs.append(b)
    c = b @ b.T / b.shape[1]
    covs.append(c / np.trace(c) * c.shape[0])                                    # unit mean variance (removes run-to-run amplitude)
    subj.append(r.subj)
st = pd.DataFrame(stats)
st.to_csv(out / "noise_run_stats.csv", index=False)
Cpool = np.mean(covs, axis=0)
sd = np.sqrt(np.diag(Cpool))
Rpool = Cpool / sd[:, None] / sd[None]

res = {"n_runs": len(st), "baseline_ms": [-300 + 2 * B0, -300 + 2 * B1]}
res["noise_rms_uV"] = {k: float(v) for k, v in st.noise_rms_uV.describe()[["min", "25%", "50%", "75%", "max"]].items()}
res["snr_matrix_db"] = {k: float(v) for k, v in st.snr_matrix_db.describe(percentiles=[.05, .25, .5, .75, .95])[["min", "5%", "25%", "50%", "75%", "95%", "max"]].items()}
res["snr_elec_median_db"] = {k: float(v) for k, v in st.snr_elec_median_db.describe(percentiles=[.05, .25, .5, .75, .95])[["min", "5%", "25%", "50%", "75%", "95%", "max"]].items()}

# spatial structure: correlation vs distance (pooled), eigen-spectrum
iu = np.triu_indices(185, 1)
bins = np.arange(0, 241, 20)
idx = np.digitize(D[iu], bins)
corr_bin = [float(np.mean(Rpool[iu][idx == k])) for k in range(1, len(bins))]
res["corr_vs_distance_mm"] = dict(zip([f"{bins[k - 1]}-{bins[k]}" for k in range(1, len(bins))], corr_bin))
w = np.sort(np.linalg.eigvalsh(Rpool))[::-1]
res["eig_participation_ratio"] = float(w.sum() ** 2 / (w ** 2).sum())
res["eig_n_for_90pct"] = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.9) + 1)
res["eig_n_for_99pct"] = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.99) + 1)

# spectrum
f, P = welch(np.array(segs), fs=FS, nperseg=120, axis=2)
Pm = P.mean(axis=(0, 1))
sel = (f >= 4) & (f <= 40)
res["psd_loglog_slope_4_40Hz"] = float(np.polyfit(np.log10(f[sel]), np.log10(Pm[sel]), 1)[0])
res["psd_rel_to_10Hz"] = {str(int(round(x))): float(Pm[np.argmin(abs(f - x))] / Pm[np.argmin(abs(f - 10))]) for x in (4, 10, 20, 40, 80, 150, 240)}

# cross-subject generalisation: covariance from half of the subjects vs the other half
subs = np.array(subj); us = np.unique(subs); rng = np.random.default_rng(0)
cos = []
for _ in range(50):
    pick = rng.permutation(us); A_ = np.isin(subs, pick[:len(us) // 2])
    Ca, Cb = np.mean([c for c, a in zip(covs, A_) if a], axis=0), np.mean([c for c, a in zip(covs, A_) if not a], axis=0)
    cos.append(float((Ca * Cb).sum() / np.linalg.norm(Ca) / np.linalg.norm(Cb)))
res["cov_split_half_subjects_cosine"] = dict(mean=float(np.mean(cos)), min=float(np.min(cos)))
one = np.mean([float((c * Cpool).sum() / np.linalg.norm(c) / np.linalg.norm(Cpool)) for c in covs])
res["single_run_cov_vs_pooled_cosine_mean"] = float(one)

# the team's 75-electrode realistic noise for comparison
rn = loadmat(str(REPO / "anatomy/realistic_noise.mat"), simplify_cells=True)["data"]            # (4, 500, 75)
e75 = loadmat(str(REPO / "anatomy/electrode_75.mat"), simplify_cells=True)["eloc75"]
P75 = np.array([[e["X"], e["Y"], e["Z"]] for e in e75]); D75 = np.linalg.norm(P75[:, None] - P75[None], axis=2)
R75 = np.mean([np.corrcoef(s.T) for s in rn], axis=0)
iu75 = np.triu_indices(75, 1); idx75 = np.digitize(D75[iu75], bins)
res["team75_corr_vs_distance_mm"] = {f"{bins[k - 1]}-{bins[k]}": float(np.mean(R75[iu75][idx75 == k])) if (idx75 == k).any() else None for k in range(1, len(bins))}
f75, P75s = welch(rn, fs=FS, nperseg=250, axis=1)
P75m = P75s.mean(axis=(0, 2)); s75 = (f75 >= 4) & (f75 <= 40)
res["team75_psd_loglog_slope_4_40Hz"] = float(np.polyfit(np.log10(f75[s75]), np.log10(P75m[s75]), 1)[0])
w75 = np.sort(np.linalg.eigvalsh(R75))[::-1]
res["team75_eig_n_for_90pct"] = int(np.searchsorted(np.cumsum(w75) / w75.sum(), 0.9) + 1)
(out / "noise_analysis.json").write_text(json.dumps(res, indent=2))
np.savez_compressed(out / "noise_pooled_covariance_185.npz", cov=Cpool, corr=Rpool, psd_f=f, psd=Pm)

fig, ax = plt.subplots(2, 3, figsize=(17, 8))
ax[0, 0].hist(st.snr_matrix_db, bins=30, color="steelblue"); ax[0, 0].axvline(-5, color="r", ls="--"); ax[0, 0].axvline(10, color="r", ls="--", label="current SEREEGA range -5..10 dB"); ax[0, 0].legend(fontsize=7)
ax[0, 0].set_title("equivalent training SNR of each L2 run (whole 500x185 matrix, dB)")
ax[0, 1].hist(st.snr_elec_median_db, bins=30, color="steelblue"); ax[0, 1].set_title("median per-electrode SNR of each run (DeepSIF definition, dB)")
ax[0, 2].hist(st.noise_rms_uV, bins=30, color="steelblue"); ax[0, 2].set_title("rms of the baseline noise in the averaged data (uV)")
ax[1, 0].plot(np.arange(2, 242, 20) - 10 + 10, corr_bin, "o-", label="ccepcoreg 185 (pooled L2 baselines)")
c75 = [res["team75_corr_vs_distance_mm"][k] for k in res["team75_corr_vs_distance_mm"]]
ax[1, 0].plot(np.arange(2, 242, 20), [np.nan if v is None else v for v in c75], "s-", label="team 75-ch realistic noise"); ax[1, 0].axhline(0, color="k", lw=0.5)
ax[1, 0].set_xlabel("electrode distance (mm, bin start +10)"); ax[1, 0].set_title("noise spatial correlation vs distance"); ax[1, 0].legend(fontsize=7)
ax[1, 1].loglog(f[1:], Pm[1:], label=f"ccepcoreg (slope {res['psd_loglog_slope_4_40Hz']:.2f})"); ax[1, 1].loglog(f75[1:], P75m[1:] / P75m[np.argmin(abs(f75 - 10))] * Pm[np.argmin(abs(f - 10))], label=f"team 75-ch (slope {res['team75_psd_loglog_slope_4_40Hz']:.2f})")
ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_title("noise PSD (arbitrary scale, matched at 10 Hz)"); ax[1, 1].legend(fontsize=7)
ax[1, 2].semilogy(np.cumsum(w) / w.sum()); ax[1, 2].set_title("cumulative eigenvalue share of the pooled noise correlation (185 ch)"); ax[1, 2].set_xlabel("component")
fig.tight_layout(); fig.savefig(fig_dir / "fig50_noise_analysis.png", dpi=130)
print(json.dumps(res, indent=2))
