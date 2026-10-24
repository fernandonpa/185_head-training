"""Derives kaggle_256_regions/ViTESI_256_regions_pipeline_kaggle.ipynb from the 256-electrode notebook: same pipeline, but the simulated data (train and test)
contain the 8 ground-truth regions of the coreg-spikes validation sessions (generator copy with forced regions), training noise 15 dB."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
nb = json.load(open(R / "kaggle_256" / "ViTESI_256_pipeline_kaggle.ipynb", encoding="utf8"))

COMMON = [
    ("ViTESI_256_pipeline_kaggle.ipynb", "ViTESI_256_regions_pipeline_kaggle.ipynb"),
    ("results_256", "results_256_regions"), ("data_256", "data_256_regions"),
    ("ViT-ESI-256", "ViT-ESI-256-regions"), ("ViTESI-256", "ViTESI-256-regions"), ("ViT-256", "ViT-256-regions"), ("vit256", "vit256regions"),
    ("N_EPOCHS = 2 if QUICK_TEST else 100", "N_EPOCHS = 2 if QUICK_TEST else 25"),
    ("BATCH_PER_GPU = 4 if QUICK_TEST else 32", "BATCH_PER_GPU = 4 if QUICK_TEST else 64"),
    ("EEG_SNR = 5 ", "EEG_SNR = 15"),
]
ENV_OLD = 'os.environ["ESI_PIPELINE"] = "256"'
ENV_NEW = ('FORCE_FRACTION = 0.25                      # share of the samples whose first patch is centred on one of the 8 validation regions (the 8 regions are used in turn)\n'
           'REGIONS = [748, 418, 907, 908, 904, 480, 917, 975]   # ground-truth regions (0-based) of the 8 coreg-spikes sessions, kaggle_256_regions/target_regions.json\n'
           'os.environ["ESI_FORCE_REGIONS"] = ",".join(map(str, REGIONS)); os.environ["ESI_FORCE_FRACTION"] = str(FORCE_FRACTION)\n' + ENV_OLD)

GEN_OLD = "t0 = time.time()\nrun([sys.executable, REPO_DIR / \"kaggle_185\" / \"generate_data_185.py\""
GEN_NEW = ("run([sys.executable, REPO_DIR / \"kaggle_256_regions\" / \"make_generator.py\"], \"make_generator.log\")      # patched copy of the team's generator (original untouched)\n"
           "os.environ[\"ESI_GENERATOR\"] = str(REPO_DIR / \"data_generation\" / \"sereega\" / \"simu_extended_source_regions.py\")\n"
           + GEN_OLD)
CHECK = ("\nrun([sys.executable, REPO_DIR / \"kaggle_256_regions\" / \"check_regions.py\", \"--root\", DATA_ROOT, \"--splits\", TRAIN_NAME, TEST_NAME, \"--out\", RESULTS / \"regions_in_data.csv\"], \"regions_in_data.log\")\n")

for i, c in enumerate(nb["cells"]):
    t = "".join(c["source"])
    for a, b in COMMON:
        t = t.replace(a, b)
    t = t.replace("generate_data_256_regions.py", "generate_data_185.py").replace("package_results_256_regions.py", "package_results_185.py")
    if c["cell_type"] == "code" and ENV_OLD in t:
        t = t.replace(ENV_OLD, ENV_NEW, 1)
    if c["cell_type"] == "code" and GEN_OLD in t:
        t = t.replace(GEN_OLD, GEN_NEW, 1) + CHECK
    if i == 0:
        t = t.replace("\n", "\n", 1) + ("\n\n**Variant with the validation regions.** The 8 ground-truth regions of the coreg-spikes validation sessions (748, 418, 907, 908, 904, 480, 917, 975) are forced into the simulated "
                                      "data: in 25 % of the samples (train and test) the first patch is centred on one of them (used in turn, so each gets the same number of samples). "
                                      "Training noise 15 dB, 25 epochs. Note that the real-data check on these 8 regions is then no longer a test on locations the model has never seen.")
    c["source"] = t.splitlines(True)
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
out = R / "kaggle_256_regions" / "ViTESI_256_regions_pipeline_kaggle.ipynb"
json.dump(nb, open(out, "w", encoding="utf8"), indent=1)
txt = "\n".join("".join(c["source"]) for c in nb["cells"])
for k in ("ESI_FORCE", "ESI_GENERATOR", "EEG_SNR =", "N_EPOCHS =", "BATCH_PER_GPU =", "generate_data", "package_results", "results_256_regions\""):
    print(k, "->", [l.strip()[:150] for l in txt.splitlines() if k in l][:3])
