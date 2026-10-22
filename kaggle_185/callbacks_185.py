"""Recording callbacks injected into the team's `main_train.py` without editing it (see train_185.py).

Per run (records directory = env ESI185_RECORDS_DIR):
  training_steps.csv   one row per optimisation step on rank 0: epoch, step, loss, lr, time
  training_epochs.csv  one row per epoch: global train/val loss (reduced over all GPUs), cosine similarity ("accuracy" of the loss),
                       validation peak-localisation proxies, learning rate, epoch time, GPU memory, flags
  best_val.json        epoch/value of the best validation loss so far (rewritten when it improves)
  last_model.pt        state_dict after the latest epoch (ONE file, overwritten every epoch)
  last.ckpt            full Lightning checkpoint after the latest epoch (ONE file, overwritten) -> training can be resumed
  best_val_model.pt    state_dict of the epoch with the lowest validation loss (overwritten only when it improves)
  run_summary.json     written when training ends (also on an exception)
  pl_csv/              Lightning CSVLogger output (metrics.csv, hparams.yaml)
So the number of model files is constant (3), however many epochs are run.

Validation proxies (computed on the full validation set every epoch, summed over both GPUs):
  peak region   = argmax over regions of max_t |predicted source|
  peak_hit      = the predicted peak region is one of the active (non-zero) regions of the ground truth
  le_peak_mm    = distance between the predicted peak region and the ground-truth peak region
  le_nearest_mm = distance from the predicted peak region to the nearest active ground-truth region (0 if it is inside)
They are quick proxies for monitoring; the official test metrics (LE, AUC, nMSE, PSNR, time error) come from eval.py.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback

ENV_RECORDS = "ESI185_RECORDS_DIR"
ENV_SOURCES = "ESI185_SOURCES_MAT"          # anatomy/sources_fsav_994.mat (region positions, metres)


def _loss_value(outputs):
    if isinstance(outputs, dict):
        outputs = outputs.get("loss")
    if outputs is None:
        return None
    return float(outputs.detach().float().item())


class Recorder185(Callback):
    def __init__(self, records_dir: str, sources_mat: str | None = None):
        super().__init__()
        self.dir = Path(records_dir)
        self.sources_mat = sources_mat
        self.t_fit = time.time()
        self._step_f = None
        self._step_w = None
        self.best_val = float("inf")
        self.best_epoch = -1
        self.last_row = {}
        self.D = None
        self._epochs_done = 0

    # ------------------------------------------------------------------ helpers
    def _rank0(self, trainer) -> bool:
        return trainer.is_global_zero

    def _reduce_sum(self, trainer, t: torch.Tensor) -> torch.Tensor:
        if trainer.world_size > 1:
            return trainer.strategy.reduce(t, reduce_op="sum")
        return t

    def _append_csv(self, name: str, row: dict):
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------ setup
    def setup(self, trainer, pl_module, stage=None):
        if self._rank0(trainer):
            self.dir.mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        dev = pl_module.device
        if self.sources_mat and os.path.isfile(self.sources_mat):
            from scipy.io import loadmat

            pos = np.asarray(loadmat(self.sources_mat)["positions"], dtype=np.float64) * 1000.0     # metres -> mm
            pos = torch.tensor(pos, dtype=torch.float32, device=dev)
            self.D = torch.cdist(pos, pos)                                                               # (994, 994) mm
        if self._rank0(trainer):
            info = dict(
                started=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, world_size=trainer.world_size, max_epochs=trainer.max_epochs,
                python=sys.version.split()[0], torch=torch.__version__, pytorch_lightning=pl.__version__, platform=platform.platform(),
                cuda=torch.cuda.is_available(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                n_parameters=int(sum(p.numel() for p in pl_module.parameters())),
                precision=str(trainer.precision),
            )
            (self.dir / "fit_info.json").write_text(json.dumps(info, indent=2))
            self.t_fit = time.time()

    # ------------------------------------------------------------------ training
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0 and self._rank0(trainer):
            info_path = self.dir / "fit_info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            info["n_train_batches_per_epoch_per_gpu"] = int(trainer.num_training_batches)
            info["n_val_batches_per_gpu"] = int(sum(trainer.num_val_batches)) if trainer.num_val_batches else None
            info_path.write_text(json.dumps(info, indent=2))
        self._tr = torch.zeros(2, device=pl_module.device)          # [sum(loss * n), n]
        self._ep_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = _loss_value(outputs)
        if loss is None:
            return
        n = int(batch[0].shape[0])
        self._tr[0] += loss * n
        self._tr[1] += n
        if self._rank0(trainer):
            lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
            self._append_csv("training_steps.csv", dict(epoch=trainer.current_epoch, global_step=trainer.global_step, batch_idx=batch_idx,
                                                        loss_rank0=round(loss, 6), lr=lr, time_s=round(time.time() - self.t_fit, 2)))

    # ------------------------------------------------------------------ validation
    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._va = torch.zeros(5, device=pl_module.device)          # [loss*n, hit, le_peak, le_nearest, n]

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        eeg, src = batch
        n = int(eeg.shape[0])
        loss = _loss_value(outputs)
        if loss is not None:
            self._va[0] += loss * n
        self._va[4] += n
        if self.D is None:
            return
        pred = pl_module(eeg.float().to(pl_module.device))                        # (B, S, T), eval mode, fp32
        src = src.float().to(pl_module.device)
        pred_peak = pred.abs().amax(dim=2).argmax(dim=1)                          # (B,)
        gt_amp = src.abs().amax(dim=2)                                            # (B, S)
        gt_peak = gt_amp.argmax(dim=1)
        active = gt_amp > 0
        ar = torch.arange(n, device=pred.device)
        dist_to_pred = self.D[pred_peak]                                          # (B, S)
        self._va[1] += active[ar, pred_peak].float().sum()
        self._va[2] += self.D[pred_peak, gt_peak].sum()
        self._va[3] += dist_to_pred.masked_fill(~active, float("inf")).amin(dim=1).sum()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._va_global = self._reduce_sum(trainer, self._va).detach().cpu().numpy()

    # ------------------------------------------------------------------ end of epoch (validation of this epoch is finished)
    def on_train_epoch_end(self, trainer, pl_module):
        tr = self._reduce_sum(trainer, self._tr).detach().cpu().numpy()
        va = getattr(self, "_va_global", None)
        epoch = trainer.current_epoch
        train_loss = float(tr[0] / max(tr[1], 1))
        row = dict(epoch=epoch, train_loss=round(train_loss, 6), train_cos_sim=round(-train_loss, 6))
        improved = False
        if va is not None and va[4] > 0:
            val_loss = float(va[0] / va[4])
            row.update(val_loss=round(val_loss, 6), val_cos_sim=round(-val_loss, 6), val_peak_hit_rate=round(float(va[1] / va[4]), 4),
                       val_le_peak_mm=round(float(va[2] / va[4]), 3), val_le_nearest_mm=round(float(va[3] / va[4]), 3), n_val=int(va[4]))
            if val_loss < self.best_val:
                self.best_val, self.best_epoch, improved = val_loss, epoch, True
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        row.update(lr=lr, epoch_time_s=round(time.time() - self._ep_t0, 2), elapsed_s=round(time.time() - self.t_fit, 1), is_best_val=int(improved),
                   gpu_mem_max_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0)
        self.last_row = row

        # model files: ONE last checkpoint, overwritten after every epoch (collective call: every rank takes part)
        trainer.save_checkpoint(str(self.dir / "last.ckpt"))
        if self._rank0(trainer):
            tmp = self.dir / "last_model.pt.tmp"
            torch.save(pl_module.state_dict(), tmp)
            os.replace(tmp, self.dir / "last_model.pt")
            if improved:
                tmp = self.dir / "best_val_model.pt.tmp"
                torch.save(pl_module.state_dict(), tmp)
                os.replace(tmp, self.dir / "best_val_model.pt")
                (self.dir / "best_val.json").write_text(json.dumps({"epoch": epoch, **{k: row[k] for k in row if k.startswith("val_")}}, indent=2))
            row["last_model_saved"] = 1
            self._append_csv("training_epochs.csv", row)
        self._epochs_done += 1

    # ------------------------------------------------------------------ end
    def _write_summary(self, trainer, status: str):
        if not self._rank0(trainer):
            return
        s = dict(status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"), epochs_completed=self._epochs_done,
                 total_time_s=round(time.time() - self.t_fit, 1), best_val_loss=self.best_val if self.best_epoch >= 0 else None,
                 best_epoch=self.best_epoch, last_epoch_row=self.last_row)
        (self.dir / "run_summary.json").write_text(json.dumps(s, indent=2))

    def on_fit_end(self, trainer, pl_module):
        self._write_summary(trainer, "completed")

    def on_exception(self, trainer, pl_module, exception):
        self._write_summary(trainer, f"exception: {type(exception).__name__}: {exception}")


def install_trainer_patch():
    """Make every `pl.Trainer(...)` created afterwards (by main_train.py) carry the recorder and a CSV logger."""
    records = os.environ.get(ENV_RECORDS)
    if not records:
        return False
    from pytorch_lightning.loggers import CSVLogger

    base = pl.Trainer

    class Trainer185(base):          # noqa: N801
        def __init__(self, *args, **kwargs):
            cbs = list(kwargs.get("callbacks") or [])
            cbs.append(Recorder185(records, os.environ.get(ENV_SOURCES)))
            kwargs["callbacks"] = cbs
            lg = kwargs.get("logger")
            extra = CSVLogger(save_dir=records, name="pl_csv", version=0)
            kwargs["logger"] = ([lg] if lg not in (None, True, False) and not isinstance(lg, (list, tuple)) else list(lg or [])) + [extra]
            super().__init__(*args, **kwargs)

    pl.Trainer = Trainer185
    return True
