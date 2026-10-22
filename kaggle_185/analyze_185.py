"""Figures and a summary report from the training / test records.

    python kaggle_185/analyze_185.py --results results_185

Reads  <results>/training/{training_epochs.csv, training_steps.csv, best_val.json, run_summary.json, fit_info.json}
       <results>/test/{test_per_sample_metrics.csv, test_summary.json, test_predictions_sample.npz}
Writes <results>/figures/*.png and <results>/SUMMARY.md   (every part is optional: missing inputs are skipped)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def training_figures(tr: Path, fig: Path, md: list):
    ep_f, st_f = tr / "training_epochs.csv", tr / "training_steps.csv"
    if not ep_f.exists():
        return
    ep = pd.read_csv(ep_f)
    best = json.loads((tr / "best_val.json").read_text()) if (tr / "best_val.json").exists() else {}
    fig_, ax = plt.subplots(2, 3, figsize=(17, 8))
    ax[0, 0].plot(ep.epoch, ep.train_loss, label="train"); ax[0, 0].plot(ep.epoch, ep.val_loss, label="validation"); ax[0, 0].set_title("loss (negative mean cosine similarity)"); ax[0, 0].legend()
    ax[0, 1].plot(ep.epoch, ep.train_cos_sim, label="train"); ax[0, 1].plot(ep.epoch, ep.val_cos_sim, label="validation"); ax[0, 1].set_title("cosine similarity (prediction vs ground truth)"); ax[0, 1].legend()
    ax[0, 2].plot(ep.epoch, ep.val_peak_hit_rate); ax[0, 2].set_title("validation: predicted peak region inside the active patch (rate)")
    ax[1, 0].plot(ep.epoch, ep.val_le_nearest_mm, label="to nearest active region"); ax[1, 0].plot(ep.epoch, ep.val_le_peak_mm, label="to ground-truth peak region")
    ax[1, 0].set_title("validation: distance of the predicted peak region (mm)"); ax[1, 0].legend()
    ax[1, 1].plot(ep.epoch, ep.epoch_time_s); ax[1, 1].set_title("seconds per epoch"); ax[1, 1].set_xlabel("epoch")
    if st_f.exists():
        st = pd.read_csv(st_f)
        w = max(1, len(st) // 200)
        ax[1, 2].plot(st.global_step, st.loss_rank0, lw=0.4, alpha=0.5); ax[1, 2].plot(st.global_step, st.loss_rank0.rolling(w, min_periods=1).mean(), lw=1.5)
        ax[1, 2].set_title("training loss per optimisation step (rank 0)"); ax[1, 2].set_xlabel("step")
    if best:
        for a in ax.ravel()[:5]:
            a.axvline(best["epoch"], color="r", ls=":", lw=1)
    fig_.tight_layout(); fig_.savefig(fig / "fig_training_curves.png"); plt.close(fig_)
    last = ep.iloc[-1]
    md += ["## Training", "",
           f"* epochs run: {len(ep)}; total time {ep.elapsed_s.iloc[-1] / 3600:.2f} h; median epoch time {ep.epoch_time_s.median():.0f} s",
           f"* best validation loss {best.get('val_loss', float('nan')):.4f} at epoch {best.get('epoch', '?')} (cosine similarity {-best.get('val_loss', float('nan')):.4f})",
           f"* last epoch: train loss {last.train_loss:.4f}, validation loss {last.val_loss:.4f}, validation peak-hit rate {last.val_peak_hit_rate:.3f}, distance to nearest active region {last.val_le_nearest_mm:.1f} mm",
           f"* the saved test model is the best-validation one (`models/best_val_model.pt`); `models/last_model.pt` is the final epoch", ""]


def test_figures(te: Path, fig: Path, md: list):
    f = te / "test_per_sample_metrics.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    summ = json.loads((te / "test_summary.json").read_text()) if (te / "test_summary.json").exists() else {}
    m = next((x for x in summ.get("methods", []) if x != "gt"), "eeg_vit")
    cols = [("loc_error_mm", "localisation error (mm)"), ("auc", "AUC"), ("nmse", "nMSE"), ("psnr_db", "PSNR (dB)"), ("time_error_ms", "time error (ms)")]
    fig_, ax = plt.subplots(2, 5, figsize=(20, 7))
    for i, (c, t) in enumerate(cols):
        v = df[f"{m}_{c}"]
        ax[0, i].hist(v.dropna(), bins=40, color="steelblue"); ax[0, i].axvline(v.mean(), color="r", ls="--"); ax[0, i].set_title(f"{t}\nmean {v.mean():.3g}, median {v.median():.3g}")
        data = [df.loc[df.n_patch == p, f"{m}_{c}"].dropna() for p in sorted(df.n_patch.unique())]
        ax[1, i].boxplot(data, tick_labels=[str(p) for p in sorted(df.n_patch.unique())]); ax[1, i].set_xlabel("number of patches"); ax[1, i].set_title(f"{t} by number of patches")
    fig_.tight_layout(); fig_.savefig(fig / "fig_test_metrics.png"); plt.close(fig_)
    s = summ
    md += ["## Test set (simulated, SEREEGA)", "", f"* method: {m}; {len(df)} samples, evaluation SNR {s.get('noise_snr_db', '?')} dB (white noise)", "",
           "| metric | mean | std | median | 25 % | 75 % |", "|---|---|---|---|---|---|"]
    for c, t in cols:
        d = s.get(m, {}).get(f"{m}_{c}")
        if d:
            md.append(f"| {t} | {d['mean']:.4g} | {d['std']:.4g} | {d['50%']:.4g} | {d['25%']:.4g} | {d['75%']:.4g} |")
    md += ["", "By number of patches (mean):", "", "| patches | n | LE (mm) | AUC | nMSE | PSNR (dB) | time error (ms) |", "|---|---|---|---|---|---|---|"]
    for p, sub in df.groupby("n_patch"):
        md.append(f"| {p} | {len(sub)} | " + " | ".join(f"{sub[f'{m}_{c}'].mean():.4g}" for c, _ in cols) + " |")
    md.append("")


def prediction_figure(te: Path, fig: Path):
    f = te / "test_predictions_sample.npz"
    if not f.exists():
        return
    from scipy.io import loadmat

    d = np.load(f)
    pos = np.asarray(loadmat(str(REPO / "anatomy" / "sources_fsav_994.mat"))["positions"], float) * 1000
    n = min(6, d["gt"].shape[0])
    fig_, ax = plt.subplots(n, 5, figsize=(21, 3.4 * n))
    ax = np.atleast_2d(ax)
    for r in range(n):
        gt, pr, eeg = d["gt"][r].astype(float), d["pred"][r].astype(float), d["eeg"][r].astype(float)
        tg = int(np.abs(gt).max(axis=0).argmax()); tp = int(np.abs(pr).max(axis=0).argmax())
        ax[r, 0].plot(eeg.T, lw=0.3); ax[r, 0].set_title(f"test sample {int(d['sample_id'][r])}: input EEG ({eeg.shape[0]} ch)")
        for c, (vec, ttl) in enumerate([(np.abs(gt[:, tg]), f"ground truth at {tg} samples"), (np.abs(pr[:, tp]), f"prediction at {tp} samples")]):
            vec = vec / (vec.max() + 1e-12)
            ax[r, 1 + 2 * c].scatter(pos[:, 1], pos[:, 0], c="lightgray", s=2); ax[r, 1 + 2 * c].scatter(pos[:, 1], pos[:, 0], c=vec, s=6 + 30 * vec, cmap="Reds", vmin=0, vmax=1)
            ax[r, 2 + 2 * c].scatter(pos[:, 1], pos[:, 2], c="lightgray", s=2); ax[r, 2 + 2 * c].scatter(pos[:, 1], pos[:, 2], c=vec, s=6 + 30 * vec, cmap="Reds", vmin=0, vmax=1)
            ax[r, 1 + 2 * c].set_title(ttl + " (top view)"); ax[r, 2 + 2 * c].set_title("side view")
        for a in ax[r, 1:]:
            a.set_aspect("equal"); a.axis("off")
    fig_.tight_layout(); fig_.savefig(fig / "fig_test_predictions.png"); plt.close(fig_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--label", default="ViT-ESI", help="model name used in the title")
    a = ap.parse_args()
    root = Path(a.results)
    fig = root / "figures"
    fig.mkdir(exist_ok=True)
    md = [f"# Results summary (185-electrode {a.label}, SEREEGA)", ""]
    for name in ("run_manifest.json",):
        f = root / name
        if f.exists():
            m = json.loads(f.read_text())
            md += ["## Run", "", "```json", json.dumps({k: m[k] for k in m if k in ("created", "git_commit", "settings", "environment")}, indent=1)[:3000], "```", ""]
    training_figures(root / "training", fig, md)
    test_figures(root / "test", fig, md)
    prediction_figure(root / "test", fig)
    (root / "SUMMARY.md").write_text("\n".join(md))
    print("figures:", sorted(p.name for p in fig.glob("*.png")))
    print("summary:", root / "SUMMARY.md")


if __name__ == "__main__":
    main()
