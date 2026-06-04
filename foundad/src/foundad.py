
import multiprocessing as mp
from typing import Any, Dict, Tuple, Optional, List
import importlib   
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from transformers import AutoProcessor, SiglipVisionModel, CLIPVisionModel



class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(self, model_name: str, pred_depth: int, pred_emb_dim: int, use_cuda: bool = True, if_pe: bool = True, feat_normed: bool = False,
                 weights_path: Optional[str] = None, repo_dir: Optional[str] = None, arch: str = "dinov3_vitb16"):
        super().__init__()
        # Used only by the "dinov3_local" branch (a user-trained DINOv3 backbone).
        self.weights_path = weights_path
        self.repo_dir = repo_dir
        self.arch = arch
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](num_patches=self.num_patches, embed_dim=self.embed_dim,
                                                         predictor_embed_dim=pred_emb_dim, depth=pred_depth, if_pe=if_pe, feat_normed=feat_normed)
        self._init_predictor(self.predictor)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()
        self.feat_normed = self.predictor.feat_normed # it depends on the predictor
        print(f"Normed features: {self.feat_normed}")

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)
    
    def target_features(self, images, paths, n_layer=3):
        with torch.no_grad():
            return self._extract(images, paths, n_layer=n_layer)

    def context_features(self, images, paths, n_layer=3):
        z = self._extract(images, paths, n_layer=n_layer)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):

        projector = processor = None
        if model == "dinov2":
            enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval(); num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3":
            enc = torch.hub.load("facebookresearch/dinov3", 'dinov3_vitb16', source="github").eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3_local":
            # A DINOv3 ViT backbone you trained yourself (SSL). We build the architecture
            # and load your local checkpoint instead of Meta's official weights.
            enc = self._load_local_dinov3(self.arch, self.weights_path, self.repo_dir).eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval(); num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval(); processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512"); num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval(); processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16"); num_patches, embed_dim = 196, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config      
            
            config = Config()
            enc = init_vit_backbone(config)

            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")
        if model != 'dinosiglip':
            for p in enc.parameters():
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    def _load_local_dinov3(self, arch: str, weights_path: Optional[str], repo_dir: Optional[str]):
        """Build a DINOv3 ViT and load user-trained SSL weights from a local file.

        - ``arch``: hub entrypoint name, e.g. ``dinov3_vitb16``.
        - ``weights_path``: path to your checkpoint (SSL training ckpt or a plain backbone state_dict).
        - ``repo_dir``: local clone of facebookresearch/dinov3. If ``None`` the code is fetched from GitHub.
        """
        if not weights_path:
            raise ValueError(
                "model='dinov3_local' requires meta.weights_path pointing to your trained backbone checkpoint."
            )
        # 1) Instantiate the architecture WITHOUT downloading Meta's official weights.
        src = "local" if repo_dir else "github"
        repo = repo_dir if repo_dir else "facebookresearch/dinov3"
        try:
            enc = torch.hub.load(repo, arch, source=src, pretrained=False)
        except TypeError:
            # Some hub entrypoints expose `weights=` rather than `pretrained=`.
            enc = torch.hub.load(repo, arch, source=src, weights=None)
        # 2) Load your checkpoint, unwrapping common SSL containers/prefixes.
        ckpt = torch.load(weights_path, map_location="cpu")
        state = self._extract_backbone_state_dict(ckpt)
        missing, unexpected = enc.load_state_dict(state, strict=False)
        matched = len(state) - len(unexpected)
        print(f"[dinov3_local] Loaded backbone weights from: {weights_path}")
        print(f"[dinov3_local] matched={matched} | missing={len(missing)} | unexpected={len(unexpected)}")
        if matched == 0:
            raise RuntimeError(
                "[dinov3_local] No weights matched the architecture. Check that meta.arch matches the "
                "checkpoint you trained, and that the file is a DINOv3 backbone/SSL checkpoint."
            )
        if missing:
            print(f"[dinov3_local] e.g. missing keys: {list(missing)[:5]}")
        if unexpected:
            print(f"[dinov3_local] e.g. unexpected keys: {list(unexpected)[:5]}")
        return enc

    @staticmethod
    def _extract_backbone_state_dict(ckpt):
        """Return a flat backbone state_dict from an arbitrary DINOv3/DINOv2 checkpoint.

        Handles training checkpoints wrapped as {"teacher"/"student"/"model"/"state_dict": ...},
        strips common prefixes (module./backbone./teacher./...), and drops SSL head params.
        """
        if isinstance(ckpt, dict):
            for key in ("teacher", "model", "state_dict", "student", "backbone"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        prefixes = ("module.", "backbone.", "teacher.", "student.", "encoder.")
        drop_substrings = ("dino_head", "ibot_head", "head.", "mask_token", "criterion")
        new_sd = {}
        for k, v in ckpt.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for pref in prefixes:
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                        changed = True
            if any(s in nk for s in drop_substrings):
                continue
            new_sd[nk] = v
        return new_sd

    def _extract(self, imgs: torch.Tensor, paths: List[str], n_layer: int = 3):
        if self.model_name == "dinov2":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] # the thrid last block
        elif self.model_name in ("dinov3", "dinov3_local"):
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0]
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer)[0][:,1:,:]
        elif self.model_name == "siglip":
            pil_list = [Image.open(p).convert("RGB") for p in paths]
            proc = self.processor(images=pil_list, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(imgs.device)

            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states  # tuple: [embeddings, block1, ..., blockL]; len = L+1

            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, :, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, 1:, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0,2,1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            h = F.normalize(h, dim=-1)

        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
