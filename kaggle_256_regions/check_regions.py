"""Counts, in the generated splits, how many samples have each target region as a patch seed (centre) and how many contain it in any patch.

    python kaggle_256_regions/check_regions.py --root DATA_ROOT --splits train_6000 test_2000 [--out regions_in_data.csv]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--splits", nargs="+", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
T = json.load(open(HERE / "target_regions.json"))
rows = []
for sp in a.splits:
    d = Path(a.root) / "simulation/fsaverage/constrained/standard_1020/fsav_994/simu" / sp / "md"
    files = sorted(d.glob("*_md_json_flie.json"))
    seeds, inside = {r: 0 for r in T["regions"]}, {r: 0 for r in T["regions"]}
    for f in files:
        m = json.load(open(f))
        s_set = set(m["seeds"]); all_src = {x for v in m["act_src"].values() for x in v}
        for r in T["regions"]:
            seeds[r] += r in s_set
            inside[r] += r in all_src
    for sub, r in zip(T["sessions"], T["regions"]):
        rows.append(dict(split=sp, n_samples=len(files), session=sub, region=r, samples_as_patch_centre=seeds[r], samples_with_region_active=inside[r]))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
if a.out:
    df.to_csv(a.out, index=False)
