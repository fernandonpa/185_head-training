"""185-channel noise model for training/evaluation, estimated from the real baselines of the ccepcoreg validation data.

Model: Gaussian noise  n(t) = A z(t)  with
  * spatial factor A from the pooled, unit-variance noise covariance of the 185 channels (average-referenced, so rank 184),
  * temporal colouring of every latent series by the measured noise PSD (1/f^~1.75, band-limited like the data),
  * zero time-mean per channel and zero channel-mean per sample (the same normalisations as the team's NMM path).
This is a *documented substitute* for a measured 185-channel realistic-noise file: the released epochs only contain 250 ms of baseline,
too short to cut 1-s noise segments, so segments are generated from the measured second-order statistics.

    python -I headmodel_185/noise_185.py build       # writes anatomy/noise_model_185.mat and anatomy/realistic_noise_185.mat
    python -I headmodel_185/noise_185.py validate    # compares generated noise with the real statistics (fig51)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FS = 500


def load_model(path: Path | None = None) -> dict:
    m = loadmat(str(path or REPO / "anatomy" / "noise_model_185.mat"), simplify_cells=True)
    return dict(corr=m["corr"], rel_std=m["rel_std"], psd_f=m["psd_f"], psd=m["psd"])


def spatial_factor(model: dict) -> np.ndarray:
    C = model["rel_std"][:, None] * model["corr"] * model["rel_std"][None]
    w, U = np.linalg.eigh(C)
    return U * np.sqrt(np.clip(w, 0, None))[None]


def generate(model: dict, n_times: int = 500, n_samples: int = 1, fs: float = FS, rng=None) -> np.ndarray:
    """(n_samples, n_times, 185) noise with unit mean variance across channels."""
    rng = np.random.default_rng(rng)
    A = spatial_factor(model)
    nf = A.shape[0]
    freqs = np.fft.rfftfreq(n_times, 1 / fs)
    amp = np.sqrt(np.interp(np.log10(np.maximum(freqs, 1e-3)), np.log10(np.maximum(model["psd_f"], 1e-3)), model["psd"], left=model["psd"][1], right=0.0))
    amp[0] = 0.0
    out = np.empty((n_samples, n_times, nf))
    for s in range(n_samples):
        z = rng.standard_normal((nf, n_times))
        z = np.fft.irfft(np.fft.rfft(z, axis=1) * amp[None], n=n_times, axis=1)           # temporal colouring
        x = A @ z                                                                             # spatial mixing
        x -= x.mean(axis=1, keepdims=True)                                                    # zero time mean
        x -= x.mean(axis=0, keepdims=True)                                                    # average reference
        out[s] = (x / np.sqrt((x ** 2).mean())).T
    return out


def build():
    npz = np.load(HERE / "outputs" / "noise_pooled_covariance_185.npz")
    C, R = npz["cov"], npz["corr"]
    rel_std = np.sqrt(np.diag(C) / np.diag(C).mean())
    f, P = npz["psd_f"], npz["psd"]
    P = P / P[np.argmin(abs(f - 10))]
    savemat(REPO / "anatomy" / "noise_model_185.mat", dict(corr=R, rel_std=rel_std, psd_f=f, psd=P,
            note="Pooled spatial correlation, relative channel std and PSD (relative to 10 Hz) of the averaged pre-stimulus baselines of the 242 L2 runs of ccepcoreg, 185 channels, 500 Hz."))
    segs = generate(dict(corr=R, rel_std=rel_std, psd_f=f, psd=P), n_times=500, n_samples=20, rng=0).astype(np.float32)     # uV scale is irrelevant: add_white_noise rescales per electrode
    npower = (segs ** 2).sum(axis=1)                                                                          # sum of squares over time, as in realistic_noise.mat
    savemat(REPO / "anatomy" / "realistic_noise_185.mat", dict(data=segs, npower=npower,
            note="SURROGATE realistic noise (20 segments x 500 samples x 185 channels, float32) generated from measured statistics, same layout as realistic_noise.mat; npower = sum of squares over time."))
    print("written noise_model_185.mat and realistic_noise_185.mat", segs.shape, npower.shape)


def validate():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import welch

    real = np.load(HERE / "outputs" / "noise_pooled_covariance_185.npz")
    model = load_model()
    gen = generate(model, 500, 200, rng=1)                                                    # (200, 500, 185)
    el = pd.read_csv(REPO / "experiments/data_experiments/validation_ready/ccepcoreg_185ch/electrodes_185.csv")
    pos = el[["x_mm", "y_mm", "z_mm"]].to_numpy()
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    iu = np.triu_indices(185, 1)
    Rg = np.mean([np.corrcoef(g.T) for g in gen], axis=0)
    bins = np.arange(0, 221, 20); idx = np.digitize(D[iu], bins)
    cr = [float(np.mean(real["corr"][iu][idx == k])) for k in range(1, len(bins))]
    cg = [float(np.mean(Rg[iu][idx == k])) for k in range(1, len(bins))]
    f, Pg = welch(gen, fs=FS, nperseg=120, axis=1); Pg = Pg.mean(axis=(0, 2))
    sel = (f >= 4) & (f <= 40)
    slope_g = float(np.polyfit(np.log10(f[sel]), np.log10(Pg[sel]), 1)[0])
    slope_r = float(np.polyfit(np.log10(real["psd_f"][sel]), np.log10(real["psd"][sel]), 1)[0])
    wg = np.sort(np.linalg.eigvalsh(Rg))[::-1]; wr = np.sort(np.linalg.eigvalsh(real["corr"]))[::-1]
    n90 = lambda w: int(np.searchsorted(np.cumsum(w) / w.sum(), 0.9) + 1)  # noqa: E731
    res = dict(corr_distance_max_abs_diff=float(np.max(np.abs(np.array(cr) - np.array(cg)))), corr_matrix_cosine=float((Rg * real["corr"]).sum() / np.linalg.norm(Rg) / np.linalg.norm(real["corr"])),
               psd_slope_real=slope_r, psd_slope_generated=slope_g, n90_real=n90(wr), n90_generated=n90(wg),
               generated_avg_ref_residual=float(np.abs(gen.mean(axis=2)).max()), generated_time_mean_max=float(np.abs(gen.mean(axis=1)).max()))
    (HERE / "outputs" / "noise_generator_validation.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(bins[:-1] + 10, cr, "o-", label="real (pooled baselines)"); ax[0].plot(bins[:-1] + 10, cg, "s--", label="generated"); ax[0].set_title("spatial correlation vs distance (mm)"); ax[0].legend()
    ax[1].loglog(real["psd_f"][1:], real["psd"][1:] / real["psd"][np.argmin(abs(real["psd_f"] - 10))], label="real"); ax[1].loglog(f[1:], Pg[1:] / Pg[np.argmin(abs(f - 10))], "--", label="generated"); ax[1].set_title("PSD (relative to 10 Hz)"); ax[1].legend()
    ax[2].plot(gen[0, :, :6]); ax[2].set_title("generated noise, 6 channels (one segment)")
    fig.tight_layout(); fig.savefig(HERE / "outputs" / "figures" / "fig51_noise_generator_validation.png", dpi=130)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"build": build, "validate": validate}[sys.argv[1]]()
