"""Derives kaggle_256/ViTESI_256_pipeline_kaggle.ipynb from the 185-electrode notebook (same pipeline, 256-electrode profile of the shared helper scripts)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
COMMON = [
    ("ViTESI_185_pipeline_kaggle.ipynb", "ViTESI_256_pipeline_kaggle.ipynb"),
    ("results_185", "results_256"), ("data_185", "data_256"),
    ("leadfield_185_20k.mat", "leadfield_256_20k.mat"),
    ("with 185 electrodes on Kaggle", "with 256 electrodes (coreg-spikes net) on Kaggle"),
    ("185-electrode", "256-electrode"), ("185 electrodes", "256 electrodes"), ("(185 electrodes", "(256 electrodes"),
    ("generate_data_256.py", "generate_data_185.py"), ("package_results_256.py", "package_results_185.py"),   # shared helper scripts keep their 185 names
    ("ViT-ESI-185", "ViT-ESI-256"), ("ViTESI-185", "ViTESI-256"), ("ViT-185", "ViT-256"), ("vit185", "vit256"),
]
ENV = ('from pathlib import Path\n\nWORK = Path("/kaggle/working")',
       'from pathlib import Path\n\nos.environ["ESI_PIPELINE"] = "256"        # selects the 256-electrode profile of the shared helper scripts (kaggle_185/config_185.py)\n\nWORK = Path("/kaggle/working")')
nb = json.load(open(R / "kaggle_185/ViTESI_185_pipeline_kaggle.ipynb", encoding="utf8"))
for c in nb["cells"]:
    t = "".join(c["source"])
    for a, b in COMMON: t = t.replace(a, b)
    if c["cell_type"] == "code" and ENV[0] in t: t = t.replace(*ENV)
    c["source"] = t.splitlines(True)
    if c["cell_type"] == "code": c["outputs"] = []; c["execution_count"] = None
json.dump(nb, open(R / "kaggle_256/ViTESI_256_pipeline_kaggle.ipynb", "w", encoding="utf8"), indent=1)
for i, c in enumerate(nb["cells"]):
    for l in "".join(c["source"]).splitlines():
        if "185" in l: print(i, l.strip()[:160])
