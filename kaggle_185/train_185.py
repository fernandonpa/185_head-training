"""Train with the team's `model_training/main_train.py` (unchanged), adding the recording callbacks of callbacks_185.py.

Use it exactly like main_train.py (same arguments), after setting ESI185_RECORDS_DIR:

    ESI185_RECORDS_DIR=results_185/training python kaggle_185/train_185.py train_6000 -results_path ... -simu_type sereega ...

How it works: it patches `pytorch_lightning.Trainer` in this process (and, because PyTorch Lightning's DDP launcher re-runs this
very script in the other GPU processes, in those too) and then executes main_train.py.  main_train.py itself is not modified.

Note: main_train.py is executed with exec() in a fresh namespace and NOT with runpy.run_path: run_path temporarily sets sys.argv[0] to
main_train.py, and Lightning would then start the other GPU processes with main_train.py directly -- without this wrapper (no recorder there,
and the cross-GPU reductions of the recorder would wait forever).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MT = REPO / "model_training"

# PyTorch Lightning's DDP launcher re-executes `abspath(sys.argv[0])` after the working directory has changed below
sys.argv[0] = os.path.abspath(sys.argv[0])
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MT))
os.chdir(MT)
os.environ.setdefault("ESI185_SOURCES_MAT", str(REPO / "anatomy" / "sources_fsav_994.mat"))

import callbacks_185  # noqa: E402

if callbacks_185.install_trainer_patch():
    print(f"[train_185] recording to {os.environ[callbacks_185.ENV_RECORDS]}", flush=True)
else:
    print("[train_185] ESI185_RECORDS_DIR not set: running main_train.py without the extra recording", flush=True)

_script = MT / "main_train.py"
_ns = {"__name__": "__main__", "__file__": str(_script), "__builtins__": __builtins__}
exec(compile(_script.read_text(encoding="utf-8"), str(_script), "exec"), _ns)
