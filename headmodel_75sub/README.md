# headmodel_75sub — the 75 of the 185 ccepcoreg electrodes that best match the original 75-electrode montage

| File | Role |
|------|------|
| `electrode_selection_75sub.py` | library: original montage in the MNE head frame, projection onto the fsaverage scalp, assignment methods (Hungarian sum / squared / bottleneck / greedy), forward-model fidelity |
| `electrode_selection_75sub.ipynb` | **the selection notebook** (executed): method, figures, tables, the saved result |
| `build_headmodel_75sub.py` | builds `anatomy/leadfield_75sub_20k.mat` (75 × 994, average-referenced over the 75; keys `fwd`, `G`, `fwd_raw`) with the same recipe as `headmodel_185/`, plus checks |
| `noise_75sub.py` | builds `anatomy/noise_model_75sub.mat` and `anatomy/realistic_noise_75sub.mat` from the measured 185-channel noise (re-referenced to the 75-channel average) |
| `outputs/` | `electrode_selection_75sub.csv` (original electrode → selected electrode, distance), summaries (`*.json`), `figures/` |

New files in `anatomy/`: `electrode_75sub.mat` (`eloc75sub`: selected electrode name, original label, template position, distance), `leadfield_75sub_20k.mat`, `noise_model_75sub.mat`, `realistic_noise_75sub.mat`.
Channel order everywhere = the order of the original 75-electrode montage (`anatomy/electrode_75.mat`).

## Result (details: `document/validation_data_preprocessing/12_75sub_electrode_selection_and_pipeline.md`)
* Method: both montages projected onto the fsaverage scalp; one-to-one Hungarian assignment; the subset is chosen by how well its forward model reproduces the forward model of the original montage (rule fixed in advance).
* Selected: minimum sum of squared distances (identical to the bottleneck assignment). Median distance to the original electrode 8.7 mm (43 of 75 within 10 mm, 90th percentile 21 mm, max 31.9 mm).
  Median region-topography correlation with the original montage's forward model **0.990** (5th percentile 0.977, minimum 0.932); a random assignment gives ≈ 0.
* The electrodes of the lowest row (TP9–TP12, F9/F10, FT9/FT10, T9/T10) have no close counterpart in the released 185 (the neck/cheek electrodes were removed): 19–32 mm.
* Checks: the leadfield is average-referenced (column mean 8e-13), numerical rank 74, and **identical to the corresponding rows of the 185-electrode forward model** (max relative difference 0).

Reproduce: run the selection notebook, then `python -I headmodel_75sub/build_headmodel_75sub.py` and `python -I headmodel_75sub/noise_75sub.py` (needs the MNE fsaverage template and `headmodel_185/outputs/vertex_leadfield_185ch.npy`).
