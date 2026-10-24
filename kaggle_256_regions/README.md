# ViT-ESI 256 electrodes, with the validation regions in the training data

Same pipeline as `kaggle_256/`, with two differences:
1. The 8 ground-truth regions of the coreg-spikes validation sessions (`target_regions.json`: 748, 418, 907, 908, 904, 480, 917, 975, 0-based) are forced into the simulated data, train and test: in 25 % of the samples (`FORCE_FRACTION`) the first patch is centred on one of them, the 8 regions in turn (equal numbers). `make_generator.py` writes a patched copy of the team's generator (`data_generation/sereega/simu_extended_source_regions.py`); the original is not modified. `check_regions.py` counts the samples per region (saved as `results_256_regions/regions_in_data.csv`).
2. Training noise 15 dB (best setting so far), 25 epochs, 6000 train / 2000 test.

Caveat: after this the real-data check on these 8 regions no longer tests locations the model has never seen; it tests whether seeing them helps on real data.
`make_notebook_regions.py` regenerates the notebook from `kaggle_256/`.
