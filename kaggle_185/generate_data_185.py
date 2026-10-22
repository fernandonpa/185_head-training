"""Generate the 185-electrode SEREEGA train/test sets with the team's own generator (not modified) and verify them.

    python kaggle_185/generate_data_185.py --root /kaggle/temp/data_185 --logs_dir results_185/logs [--n_train 6000 --n_test 2000]

Runs `data_generation/sereega/simu_extended_source.py` once per split (the two runs in parallel), with the settings of
config_185.py, then checks the files and writes `<root>/data_manifest.json` (parameters, seeds, counts, statistics).
Output layout (what main_train.py / eval.py expect):
    <root>/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/{train_6000,test_2000}/...
    <root>/simulation/fsaverage/constrained/standard_1020/fsav_994/model/ch_fsav_994.mat   (185-electrode info)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_185 as C  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def generation_cmd(name: str, n: int, seed: int, root: Path) -> list[str]:
    s = C.SEREEGA
    return [
        sys.executable, str(C.REPO / "data_generation" / "sereega" / "simu_extended_source.py"),
        "-sin", name, "-ne", str(n), "--seed", str(seed),
        "-mk", C.MONTAGE_DIR, "-ss", C.SOURCE_SPACE, "-o", C.ORIENTATION, "-sn", C.SUBJECT,
        "-rf", str(root), "-af", str(C.REPO / "anatomy"), "--leadfield_mat", str(C.LEADFIELD),
        "-fs", str(s["fs"]), "-d", str(s["duree"]),
        "-np_min", str(s["n_patch_min"]), "-np_max", str(s["n_patch_max"]),
        "-o_min", str(s["order_min"]), "-o_max", str(s["order_max"]), "-m", str(s["margin"]),
        "-s_type", s["sig_type"], "-amp", str(s["amplitude"]), "-c", str(s["center"]), "-w", str(s["width"]),
    ]


def simu_dir(root: Path, name: str) -> Path:
    return root / "simulation" / C.SUBJECT / C.ORIENTATION / C.MONTAGE_DIR / C.SOURCE_SPACE / "simu" / name


def write_model_folder(root: Path) -> Path:
    """185-electrode info in the file format HeadModel.ElectrodeSpace reads (otherwise it silently falls back to the 75-electrode file)."""
    e = loadmat(str(C.ELECTRODES), simplify_cells=True)[C.ELOC_KEY]
    names = np.array([str(x["labels"]) for x in e], dtype=object)
    pos = np.array([[x["X"], x["Y"], x["Z"]] for x in e], float)
    d = root / "simulation" / C.SUBJECT / C.ORIENTATION / C.MONTAGE_DIR / C.SOURCE_SPACE / "model"
    d.mkdir(parents=True, exist_ok=True)
    savemat(d / f"ch_{C.SOURCE_SPACE}.mat", dict(nb_channels=len(names), positions=pos, names=names))
    return d


def fingerprint(d: Path, n: int) -> dict:
    """Identifies the generated data: two runs with the same code, leadfield, seed and size must give the same fingerprint (so that two
    notebooks, e.g. ViT-ESI and DeepSIF, can be shown to use the same dataset).  md_sha256: exact text of all metadata files (seeds,
    orders, patches); eeg_sha256: 20 EEG samples rounded to 5 decimals after scaling (robust against last-digit floating-point differences)."""
    h = hashlib.sha256()
    for i in range(1, n + 1):
        h.update((d / "md" / f"{i}_md_json_flie.json").read_bytes())
    he = hashlib.sha256()
    for i in np.unique(np.linspace(1, n, 20).astype(int)):
        x = loadmat(d / "eeg" / "infdb" / f"{i}_eeg.mat", simplify_cells=True)["eeg_data"]["EEG"]
        he.update(np.round(x / np.abs(x).max(), 5).tobytes())
    return dict(md_sha256=h.hexdigest(), eeg_sha256=he.hexdigest())


def verify_split(root: Path, name: str, n: int, seed: int) -> dict:
    d = simu_dir(root, name)
    cfg = json.load(open(d / f"{name}{C.SOURCE_SPACE}_config.json"))
    match = json.load(open(d / f"{name}{C.SOURCE_SPACE}_match_json_file.json"))
    n_eeg = len(list((d / "eeg" / "infdb").glob("*_eeg.mat")))
    n_src = len(list((d / "sources" / "Jact").glob("*_src_act.mat")))
    n_md = len(list((d / "md").glob("*_md_json_flie.json")))
    problems = []
    if cfg["electrode_space"]["n_electrodes"] != C.N_ELECTRODES:
        problems.append(f"n_electrodes={cfg['electrode_space']['n_electrodes']}")
    if cfg["rec_info"]["fs"] != C.SEREEGA["fs"] or cfg["rec_info"]["n_times"] != C.N_TIMES:
        problems.append(f"rec_info={cfg['rec_info']}")
    if not (n_eeg == n_src == n_md == len(match) == n):
        problems.append(f"file counts eeg={n_eeg} src={n_src} md={n_md} match={len(match)} expected={n}")
    rng = np.random.default_rng(0)
    ids = rng.choice(np.arange(1, n + 1), size=min(30, n), replace=False)
    amp, col_mean_rel, n_active = [], [], []
    for i in ids:
        x = loadmat(d / "eeg" / "infdb" / f"{i}_eeg.mat", simplify_cells=True)["eeg_data"]["EEG"]
        if x.shape != (C.N_ELECTRODES, C.N_TIMES) or not np.isfinite(x).all() or np.abs(x).max() == 0:
            problems.append(f"bad EEG sample {i}: shape {x.shape}")
            continue
        amp.append(float(np.abs(x).max()))
        col_mean_rel.append(float(np.abs(x.mean(axis=0)).max() / np.abs(x).max()))
        j = loadmat(d / "sources" / "Jact" / f"{i}_src_act.mat", simplify_cells=True)["Jact"]["Jact"]
        n_active.append(int(np.atleast_2d(j).shape[0]))
    n_patch, orders, patch_sizes = [], [], []
    for f in sorted((d / "md").glob("*_md_json_flie.json")):
        m = json.load(open(f))
        n_patch.append(m["n_patch"])
        orders += m["orders"]
        patch_sizes += [len(v) for v in m["act_src"].values()]
    def hist(v):
        u, c = np.unique(v, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}
    return dict(name=name, n=n, seed=seed, path=str(d), config=cfg, n_eeg_files=n_eeg, problems=problems,
                eeg_max_abs_checked_samples=dict(min=min(amp, default=None), median=float(np.median(amp)) if amp else None, max=max(amp, default=None)),
                eeg_avg_reference_residual_rel_max=max(col_mean_rel, default=None), n_active_regions_checked=dict(min=min(n_active, default=None), max=max(n_active, default=None)),
                fingerprint=fingerprint(d, n), n_patch_hist=hist(n_patch), order_hist=hist(orders),
                patch_size=dict(min=int(min(patch_sizes)), median=float(np.median(patch_sizes)), max=int(max(patch_sizes))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root; data go to <root>/simulation/fsaverage/...")
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--n_train", type=int, default=C.SPLITS["train"]["n"])
    ap.add_argument("--n_test", type=int, default=C.SPLITS["test"]["n"])
    ap.add_argument("--train_name", default=C.SPLITS["train"]["name"])
    ap.add_argument("--test_name", default=C.SPLITS["test"]["name"])
    ap.add_argument("--seed_train", type=int, default=C.SPLITS["train"]["seed"])
    ap.add_argument("--seed_test", type=int, default=C.SPLITS["test"]["seed"])
    ap.add_argument("--sequential", action="store_true", help="run the two generations one after the other")
    a = ap.parse_args()
    if a.seed_train == a.seed_test:
        sys.exit("train and test need different seeds, otherwise they start from the same random state")
    root, logs = Path(a.root), Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    lf = loadmat(str(C.LEADFIELD))
    G = lf["G"]
    assert G.shape == (C.N_ELECTRODES, C.N_SOURCES), G.shape
    print(f"leadfield {C.LEADFIELD.name}: {G.shape}, sha256 {sha256(C.LEADFIELD)[:16]}...", flush=True)

    jobs = [(a.train_name, a.n_train, a.seed_train), (a.test_name, a.n_test, a.seed_test)]
    t0 = time.time()
    procs, times = [], {}
    for name, n, seed in jobs:
        cmd = generation_cmd(name, n, seed, root)
        (logs / f"generate_{name}.cmd.txt").write_text(" ".join(cmd) + "\n")
        log = open(logs / f"generate_{name}.log", "w")
        print(f"[{time.strftime('%H:%M:%S')}] generating {name}: {n} samples, seed {seed}", flush=True)
        p = subprocess.Popen(cmd, cwd=str(C.REPO), stdout=log, stderr=subprocess.STDOUT)
        if a.sequential:
            p.wait()
            times[name] = time.time() - t0
            if p.returncode != 0:
                sys.exit(f"generation of {name} failed, see {logs / f'generate_{name}.log'}")
        procs.append((name, p, log))
    for name, p, log in procs:
        p.wait()
        log.close()
        times.setdefault(name, time.time() - t0)
        if p.returncode != 0:
            sys.exit(f"generation of {name} failed (exit {p.returncode}), see {logs / f'generate_{name}.log'}")
    print(f"generation finished in {time.time() - t0:.0f}s", flush=True)

    model_dir = write_model_folder(root)
    splits = {}
    for name, n, seed in jobs:
        v = verify_split(root, name, n, seed)
        v["generation_seconds_wallclock"] = round(times[name], 1)
        splits[name] = v
        print(f"{name}: files {v['n_eeg_files']}/{n}, problems: {v['problems'] or 'none'}", flush=True)
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), root=str(root), model_folder=str(model_dir),
                    leadfield=dict(path=str(C.LEADFIELD), shape=list(G.shape), sha256=sha256(C.LEADFIELD)),
                    sereega=C.SEREEGA, montage_folder_name=C.MONTAGE_DIR, n_electrodes=C.N_ELECTRODES, splits=splits)
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    (logs / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    bad = {k: v["problems"] for k, v in splits.items() if v["problems"]}
    if bad:
        sys.exit(f"verification failed: {bad}")
    print("data verified OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
