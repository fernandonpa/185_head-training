"""Build the 185-electrode leadfield (185 x 994) for the ccepcoreg montage.

    python -I headmodel_185/build_headmodel_185.py [--electrodes CSV]

Writes (new files only, nothing existing is modified):
    anatomy/leadfield_185_20k.mat   fwd, G (average-referenced, MNE units V/(A*m)), fwd_raw, ch_names, provenance
    anatomy/electrode_185.mat       eloc185: labels, X, Y, Z (mm, MNE head frame), same field style as eloc75
    headmodel_185/outputs/          vertex_leadfield_185ch.npy (cache), build_metadata.json, forward.log
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import savemat

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import headmodel_185 as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--electrodes", default=str(REPO / "experiments/data_experiments/validation_ready/ccepcoreg_185ch/electrodes_185.csv"))
args = ap.parse_args()
out = HERE / "outputs"
out.mkdir(exist_ok=True)
t0 = time.time()

el = pd.read_csv(args.electrodes)
names = list(el.name)
assert len(names) == 185 and len(set(names)) == 185
info = H.make_info(names, "GSN-HydroCel-256")
assert info.ch_names == names                                                       # order preserved
pos_head_mm = np.array([c["loc"][:3] for c in info["chs"]]) * 1000.0
src, bem, trans = H.make_bem_and_src()
print(f"source space: {src[0]['nuse']} + {src[1]['nuse']} vertices ({time.time() - t0:.0f}s); BEM conductivities {H.CONDUCTIVITY}")

# capture MNE's forward log (documents e.g. the projection of the electrodes onto the scalp)
buf = io.StringIO()
handler = logging.StreamHandler(buf)
mne.utils.logger.addHandler(handler)
mne.set_log_level("INFO")
Lv = H.vertex_leadfield(info, src, bem, trans)
mne.set_log_level("ERROR")
mne.utils.logger.removeHandler(handler)
(out / "forward.log").write_text(buf.getvalue())
proj_lines = [l for l in buf.getvalue().splitlines() if "project" in l.lower() or "electrode" in l.lower() or "EEG" in l]
print("forward log (relevant lines):", *proj_lines[:10], sep="\n  ")
assert Lv.shape == (185, 20484) and np.isfinite(Lv).all()
np.save(out / "vertex_leadfield_185ch.npy", Lv.astype(np.float32))

rm = H.region_mapping(REPO)
raw = H.sum_by_region(Lv, rm)                                                       # (185, 994), as in create_deepsif_based_headmodel
avg = H.average_reference_matrix(185) @ raw                                         # average-referenced leadfield
assert raw.shape == (185, 994) and np.abs(avg.mean(axis=0)).max() < 1e-9 * np.abs(avg).max() + 1e-9

savemat(REPO / "anatomy" / "leadfield_185_20k.mat", dict(
    fwd=avg, G=avg, fwd_raw=raw, ch_names=np.array(names, dtype=object), conductivity=np.array(H.CONDUCTIVITY),
    note="fwd/G: average-referenced over the 185 channels; fwd_raw: as returned by MNE (no re-reference). Units V/(A*m). Region = sum over the region's vertices."))
eloc = [dict(labels=n, X=float(p[0]), Y=float(p[1]), Z=float(p[2])) for n, p in zip(names, pos_head_mm)]
savemat(REPO / "anatomy" / "electrode_185.mat", dict(eloc185=np.array(eloc, dtype=object)))
meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), mne=mne.__version__, montage="GSN-HydroCel-256", n_electrodes=185,
            source_space=f"{H.SUBJECT} {H.SRC_SPACING} {H.SRC_SURFACE}", n_vertices=20484, n_regions=994, bem_ico=H.BEM_ICO, conductivity=H.CONDUCTIVITY,
            leadfield_shape=list(avg.shape), abs_median_raw=float(np.median(np.abs(raw))), abs_max_raw=float(np.abs(raw).max()))
(out / "build_metadata.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
