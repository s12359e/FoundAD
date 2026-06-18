
from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.utils.synthesis import CutPasteUnion
from src.foundad import VisionModule

_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
            weights_path=mcfg.get("weights_path"), repo_dir=mcfg.get("repo_dir"), arch=mcfg.get("arch", "dinov3_vitb16"),
            config_file=mcfg.get("config_file"),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")

        # --- masked-neighbor context config (additive; default = dropout = unchanged) ---
        self.context_mode = mcfg.get("context_mode", "dropout")  # "dropout" | "masked"
        self.mask_ratio = mcfg.get("mask_ratio", 0.5)
        self.mask_type = mcfg.get("mask_type", "block")
        self.mask_block = tuple(mcfg.get("mask_block", [2, 2]))
        logger.info(f"Context mode: {self.context_mode}"
                    + (f" (ratio={self.mask_ratio}, type={self.mask_type}, block={self.mask_block})"
                       if self.context_mode == "masked" else ""))

        # --- defect-aware loss config (additive; default off = unchanged) ---
        # Weights the reconstruction loss higher at patches where the synthetic
        # defect was pasted, so the (tiny) defect signal isn't diluted by the
        # ~90% of patches that are identical between clean and abnormal images.
        self.defect_aware_loss = mcfg.get("defect_aware_loss", False)
        self.w_defect = mcfg.get("w_defect", 5.0)
        if self.defect_aware_loss:
            logger.info(f"Defect-aware loss ON (w_defect={self.w_defect})")

        # ---------- data ----------
        dcfg = args["data"]
        if dcfg["dataset"] in ("mvtec", "visa"):
            # For the benchmark datasets, sanity-check the few-shot folder name matches.
            assert dcfg["dataset"] in dcfg["data_name"], (
                f"data.dataset='{dcfg['dataset']}' should appear in data.data_name='{dcfg['data_name']}'"
            )
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip",False),
            use_vflip=dcfg.get("use_vflip",False),
            use_rotate90=dcfg.get("use_rotate90",False),
            use_color_jitter=dcfg.get("use_color_jitter",False),
            use_gray=dcfg.get("use_gray",False),
            use_blur=dcfg.get("use_blur",False),
        )
        self.cutpaste = CutPasteUnion(colorJitter=0.5)
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],                         # for cosine_warmup
            min_lr=ocfg.get("min_lr", 1e-6),                  # for cosine_warmup
            warmup_epoch=ocfg.get("warmup_epoch", 5),         # for cosine_warmup
            step_size=ocfg.get("step_size", 300),             # for step
            gamma=ocfg.get("gamma", 0.1),                     # for step
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        # log_dir.mkdir(parents=True, exist_ok=True)     
        self.ckpt_dir = log_dir

        self.tag = lcfg.get("write_tag", "train")      
        
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _masked_loss_fn(self, h, p, mask) -> torch.Tensor:
        """Loss computed only at masked positions (masked-neighbor context path).

        Args:
            h: [B, N, C] target features from frozen encoder.
            p: [B, N, C] predictor output.
            mask: [B, N] bool, True = masked (reconstruct these).
        """
        h_m = h[mask]  # [M, C]
        p_m = p[mask]  # [M, C]
        if self.loss_mode == 'l2':
            return F.mse_loss(h_m, p_m, reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h_m, p_m, reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _defect_aware_loss_fn(self, h, p, pixel_mask) -> torch.Tensor:
        """Per-patch reconstruction loss, up-weighted at pasted-defect patches.

        Args:
            h: [B, N, C] target features from frozen encoder.
            p: [B, N, C] predictor output.
            pixel_mask: [B, 1, H_img, W_img] binary mask of the pasted defect.
        """
        B, N, C = h.shape
        g = int(round(N ** 0.5))
        assert g * g == N, f"Patch count {N} is not a perfect square"

        # Pixel mask -> patch grid: a patch is "defect" if ANY of its pixels are.
        pm = F.adaptive_max_pool2d(pixel_mask.float(), output_size=(g, g))  # [B, 1, g, g]
        pm = (pm > 0).reshape(B, N)                                          # [B, N] bool

        # Per-patch error (mean over channels), respecting loss_mode.
        if self.loss_mode == 'l2':
            err = F.mse_loss(h, p, reduction="none").mean(dim=2)             # [B, N]
        elif self.loss_mode == 'smooth_l1':
            err = F.smooth_l1_loss(h, p, reduction="none").mean(dim=2)
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

        w = torch.where(pm, float(self.w_defect), 1.0).to(err.dtype)        # [B, N]
        return (err * w).sum() / w.sum()

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save({"predictor": self.model.predictor.state_dict(),
                    "projector": self.model.projector.state_dict() if self.model.projector else None,
                    "epoch": ep, "lr": self.optimizer.param_groups[0]["lr"]}, self.ckpt_dir/name)

    def train(self):
        mp.set_start_method("spawn", force=True); gstep = 0
        for ep in range(self.epochs):
            logger.info("Epoch %d", ep+1); self.sampler.set_epoch(ep); loss_m, time_m = AverageMeter(), AverageMeter()
            for itr, (imgs, labels, paths) in enumerate(self.loader):
                imgs = imgs.to(self.device, non_blocking=True)
                if self.defect_aware_loss:
                    _, imgs_abn, paste_mask = self.cutpaste(imgs, labels, return_mask=True) # anomaly synthesis (+ paste mask)
                else:
                    _, imgs_abn = self.cutpaste(imgs, labels) # anomaly synthesis
                def _step():
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        if self.context_mode == "masked":
                            # --- masked-neighbor context path ---
                            ctx_imgs = imgs if np.random.rand() < 0.5 else imgs_abn
                            h = self.model.target_features(ctx_imgs, paths, n_layer=self.n_layer)
                            _, p, mask = self.model.context_features_masked(
                                ctx_imgs, paths, n_layer=self.n_layer,
                                mask_ratio=self.mask_ratio,
                                mask_type=self.mask_type,
                                mask_block=self.mask_block,
                            )
                            return self._masked_loss_fn(h, p, mask)
                        else:
                            # --- existing dropout context path (unchanged) ---
                            if np.random.rand() < 0.5:
                                h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs, paths, n_layer=self.n_layer); used_abn = False
                            else:
                                h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs_abn, paths, n_layer=self.n_layer); used_abn = True
                            # Defect-aware weighting only applies to the abnormal branch
                            # (the clean->clean branch has no pasted defect to up-weight).
                            if self.defect_aware_loss and used_abn:
                                return self._defect_aware_loss_fn(h, p, paste_mask)
                            return self._loss_fn(h, p,)
                (loss,), t = gpu_timer(lambda: [_step()])
                if self.use_bf16: self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                else: loss.backward(); self.optimizer.step()
                grad_stats = grad_logger(self.model.predictor.named_parameters()); self.optimizer.zero_grad()
                loss_m.update(loss.item()); time_m.update(t); gstep += 1
                if gstep % 100 == 0: self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep+1, itr, loss.item(), t)
                if itr % 100 == 0:
                    logger.info("[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB (%.1fms)", ep+1, itr, loss.item(), loss_m.avg, torch.cuda.max_memory_allocated()/1024**2, time_m.avg)
                    if grad_stats:
                        logger.info("    grad: [%.2e %.2e] (%.2e %.2e)", grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max)
            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]['lr']
            )
            if self.scheduler is not None:
                self.scheduler.step()

def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml");
        if not cfg_path.exists(): raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f: args = yaml.safe_load(f)
    Trainer(args).train()

if __name__ == "__main__":
    main()