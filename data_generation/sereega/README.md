# SEREEGA extended-source simulation

Generates synthetic EEG and source activity using **extended (patch) sources** and a leadfield matrix. Used for training and evaluating **ViT-ESI** and other models.

## Requirements

- Python environment with dependencies from the repo root: `pip install -r requirements.txt`
- **Anatomy folder** with:
  - `sources_fsav_994.mat` — regional source positions (required when using 994-region leadfield)
  - Optional: leadfield in a head-model folder, or use `--leadfield_mat` to point to a `.mat` file
- **Leadfield:** Either `--leadfield_mat <path>` (e.g. `anatomy/leadfield_75_20k.mat`) or a head-model folder with `LF_<source_space>.mat`

## Usage

Run from the **project root** (ViT-ESI/):

```bash
python data_generation/sereega/simu_extended_source.py \
  -sin SIMU_NAME \
  -ne N_EXAMPLES \
  -mk MONTAGE \
  -ss SOURCE_SPACE \
  -o ORIENTATION \
  -sn SUBJECT \
  -rf ROOT_FOLDER \
  [--leadfield_mat PATH] \
  [-af ANATOMY_FOLDER] \
  -fs FS -d DURATION_MS
```

### Main arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--simu_name` | `-sin` | Simulation name (output subfolder) | `mes_debug_python` |
| `--n_examples` | `-ne` | Number of trials/samples to generate | `100` |
| `--montage_kind` | `-mk` | Electrode montage | `standard_1020` |
| `--source_sampling` | `-ss` | Source space | `fsav_994` |
| `--orientation` | `-o` | `constrained` or `unconstrained` | `constrained` |
| `--subject_name` | `-sn` | Subject (folder layout) | `fsaverage` |
| `--root_folder` | `-rf` | Root path for output (project root) | `$PROJECT_ROOT` |
| `--leadfield_mat` | — | Path to leadfield `.mat` (75×994, etc.) | `$PROJECT_ROOT/anatomy/leadfield_75_20k.mat` |
| `--anatomy_folder` | `-af` | Folder for `sources_fsav_994.mat` | `$PROJECT_ROOT/anatomy` |
| `--fs` | `-fs` | Sampling frequency (Hz) | `500` |
| `--duree` | `-d` | Trial duration (ms) | `1000` |

### Optional (spatial/temporal)

- **`-np_min` / `-np_max`**: Min/max number of patches per trial (default 1–5).
- **`-o_min` / `-o_max`**: Min/max patch “order” (spatial extent).
- **`-s_type`**: Signal type (default `erp`).
- **`-amp`**, **`-c`**, **`-w`**: ERP amplitude, center (ms), width (ms).
- **`-ds`**: Don’t save (dry run).

## Output layout

With `-rf $PROJECT_ROOT`, `-sn fsaverage`, `-o constrained`, `-mk standard_1020`, `-ss fsav_994`, `-sin my_sim`:

- **Directory:**  
  `$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/my_sim/`
- **Contents:** Config JSON, source and EEG data (format depends on script). This path is what you pass as **`-simu_folder`** in `main_train.py` and `eval.py`.

## Example (copy-paste, any machine)

Set your project root once:

```bash
export PROJECT_ROOT=/path/to/ViT-ESI
```

Then:

```bash
python data_generation/sereega/simu_extended_source.py \
  -sin mes_debug_python \
  -ne 200 \
  -mk standard_1020 \
  -ss fsav_994 \
  -o constrained \
  -sn fsaverage \
  -rf "$PROJECT_ROOT" \
  --leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" \
  -fs 500 \
  -d 1000 \
  -af "$PROJECT_ROOT/anatomy"
```

After this, train with:

```bash
cd model_training
python main_train.py mes_debug_python ... -simu_folder "$PROJECT_ROOT/simulation/fsaverage/constrained/standard_1020/fsav_994/simu/mes_debug_python" ...
```

## Notes

- For **ViT-ESI** we use the 994-region source space (`fsav_994`) and a 75-sensor leadfield; `sources_fsav_994.mat` must be in the anatomy folder when using `--leadfield_mat` with a 994-column leadfield.
- If `sources_fsav_994.mat` is missing, the script will error; ensure the anatomy folder is provided with `-af` and that the file exists there or in the path you use.
