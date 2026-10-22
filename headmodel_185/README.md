# headmodel_185

Builds the 185-electrode leadfield (`anatomy/leadfield_185_20k.mat`, 185 × 994) for the `ccepcoreg` montage (GSN-HydroCel-256 subset), following the recipe of
`model_training/utils/utils_headmodel.py::create_deepsif_based_headmodel` without importing or modifying it.

| Script | Purpose |
|--------|---------|
| `headmodel_185.py` | shared functions |
| `validate_method_vs_75ch.py` | compares our method with the team's 75-electrode leadfield on 72 shared electrodes |
| `build_headmodel_185.py` | builds `anatomy/leadfield_185_20k.mat` and `anatomy/electrode_185.mat` |
| `sanity_checks_185.py` | shape/order/reference, electrodes vs scalp, projection invariance, topographies, singular values |

Full description: `document/validation_data_preprocessing/07_head_model_185_build.md`. Needs the MNE fsaverage template (`mne.datasets.fetch_fsaverage`).
`leadfield_185_20k.mat` keys: `fwd` and `G` (average-referenced), `fwd_raw`, `ch_names`.

## Noise (step 2)

| Script | Purpose |
|--------|---------|
| `noise_analysis_185.py` | measures the noise of the validation data (baseline of the averaged runs) and the training SNR it implies |
| `noise_185.py build / validate` | writes `anatomy/noise_model_185.mat` and `anatomy/realistic_noise_185.mat` (surrogate, 20 segments); checks the generator against the real statistics |

Description and numbers: `document/validation_data_preprocessing/08_noise_185_channels.md`.
