# model_merlin_aux.py
# Merlin-assisted multi-task model
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121

from .config import ROI_SIZE


def patch_merlin_offline(local_model_dir: str):
    """
    Offline Merlin loader patch.
    Keep Merlin as a frozen auxiliary medical foundation representation branch.
    """
    import torchvision.models as tv_models
    from merlin import Merlin
    import merlin.utils.huggingface_download as hf_dl
    import merlin.models.load as merlin_load
    import merlin.models.build as merlin_build

    local_model_dir = str(Path(local_model_dir).expanduser().resolve())

    def _download_file_local_only(repo_id: str, filename: str, local_dir: str = None):
        candidate_dirs = [Path(local_model_dir)]
        if local_dir:
            candidate_dirs.append(Path(local_dir).expanduser().resolve())
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidate_dirs.append(Path(hf_home).expanduser().resolve() / "Merlin")

        checked = []
        for d in candidate_dirs:
            f = d / filename
            checked.append(str(f))
            if f.exists():
                print(f"[Merlin local] using checkpoint: {f}")
                return str(f)

        raise FileNotFoundError(
            "Merlin checkpoint not found locally: "
            + filename
            + "\nChecked:\n- "
            + "\n- ".join(checked)
        )

    hf_dl.download_file = _download_file_local_only
    merlin_load.download_file = _download_file_local_only

    original_resnet152 = tv_models.resnet152

    def resnet152_no_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        print("[Merlin local] patched torchvision.models.resnet152 -> weights=None")
        return original_resnet152(*args, **kwargs)

    tv_models.resnet152 = resnet152_no_download

    class DummyTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, text_labels):
            bs = len(text_labels) if isinstance(text_labels, (list, tuple)) else 1
            return torch.zeros(bs, 512)

    merlin_build.TextEncoder = DummyTextEncoder
    print("[Merlin local] patched TextEncoder -> DummyTextEncoder")

    original_load_state_dict = merlin_build.MerlinArchitecture.load_state_dict

    def relaxed_load_state_dict(self, state_dict, strict=True, assign=False):
        print("[Merlin local] patched MerlinArchitecture.load_state_dict -> strict=False")
        return original_load_state_dict(self, state_dict, strict=False, assign=assign)

    merlin_build.MerlinArchitecture.load_state_dict = relaxed_load_state_dict

    return Merlin


class GatedFusion(nn.Module):
    """
    Small and safe Merlin fusion.
    DenseNet remains the main branch. Merlin only provides a constrained auxiliary prior.
    """
    def __init__(self, densenet_dim: int, merlin_dim: int, init_scale: float = 0.05, dropout: float = 0.10):
        super().__init__()
        self.merlin_proj = nn.Sequential(
            nn.LayerNorm(merlin_dim),
            nn.Linear(merlin_dim, densenet_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(densenet_dim + merlin_dim),
            nn.Linear(densenet_dim + merlin_dim, densenet_dim),
            nn.Sigmoid(),
        )
        # Learnable but initialized very small, so Merlin cannot destroy run1 performance at the start.
        self.fusion_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, dense_feat, merlin_feat):
        merlin_z = self.merlin_proj(merlin_feat)
        gate = self.gate(torch.cat([dense_feat, merlin_feat], dim=1))
        scale = torch.clamp(self.fusion_scale, 0.0, 0.20)
        return dense_feat + scale * gate * merlin_z


class MerlinAuxMultiTaskClassifier(nn.Module):
    """
    Main branch: run1-style 3D DenseNet121 using CT + BQ mask.
    Auxiliary branch: frozen Merlin using CT channel only.
    Fusion: gated residual fusion with small learnable scale.
    Heads: four task heads.
    """
    def __init__(
        self,
        in_channels=2,
        tasks_config=None,
        merlin_local_dir=None,
        fusion_init_scale=0.05,
        fusion_dropout=0.10,
        freeze_merlin=True,
    ):
        super().__init__()
        self.tasks = tasks_config or {}

        # run1 main backbone
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
        )
        self.feature_extractor = self.backbone.features

        with torch.no_grad():
            dummy = torch.randn(1, in_channels, *ROI_SIZE)
            feat = self.feature_extractor(dummy)
            feat = F.relu(feat, inplace=False)
            feat = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
            dense_dim = feat.shape[1]

        self.dense_dim = dense_dim

        # Merlin auxiliary backbone
        Merlin = patch_merlin_offline(merlin_local_dir)
        self.merlin = Merlin(ImageEmbedding=True)

        if freeze_merlin:
            for p in self.merlin.parameters():
                p.requires_grad = False

        self.freeze_merlin = freeze_merlin
        self.fusion = None
        self.merlin_dim = None
        self.fusion_init_scale = fusion_init_scale
        self.fusion_dropout = fusion_dropout

        self.heads = nn.ModuleDict()
        for task, num_classes in self.tasks.items():
            if num_classes == 2:
                self.heads[task] = nn.Linear(dense_dim, 1)
            else:
                self.heads[task] = nn.Linear(dense_dim, num_classes)

    def extract_dense_feat(self, x):
        feat = self.feature_extractor(x)
        feat = F.relu(feat, inplace=False)
        feat = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
        return feat

    def _extract_single_merlin(self, x_single):
        # x_single: [1,1,H,W,D], CT channel only
        if self.freeze_merlin:
            with torch.no_grad():
                out = self.merlin(x_single)
        else:
            out = self.merlin(x_single)

        if isinstance(out, torch.Tensor):
            feat = out
        elif isinstance(out, dict):
            for key in ["image_embedding", "embeddings", "features", "image_features"]:
                if key in out:
                    feat = out[key]
                    break
            else:
                raise ValueError(f"Merlin output dict keys not recognized: {list(out.keys())}")
        elif isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            raise ValueError(f"Unsupported Merlin output type: {type(out)}")

        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        return feat

    def extract_merlin_feat(self, x):
        # Use only CT image channel; mask remains for DenseNet main branch and ROI crop.
        x_ct = x[:, 0:1]
        feats = []
        for i in range(x_ct.shape[0]):
            feats.append(self._extract_single_merlin(x_ct[i:i + 1]))
        return torch.cat(feats, dim=0)

    def maybe_build_fusion(self, merlin_feat):
        if self.fusion is None:
            self.merlin_dim = int(merlin_feat.shape[1])
            self.fusion = GatedFusion(
                densenet_dim=self.dense_dim,
                merlin_dim=self.merlin_dim,
                init_scale=self.fusion_init_scale,
                dropout=self.fusion_dropout,
            ).to(merlin_feat.device)
            print(f"[Fusion] built: dense_dim={self.dense_dim}, merlin_dim={self.merlin_dim}, init_scale={self.fusion_init_scale}")

    def forward(self, x):
        dense_feat = self.extract_dense_feat(x)
        merlin_feat = self.extract_merlin_feat(x)
        self.maybe_build_fusion(merlin_feat)
        feat = self.fusion(dense_feat, merlin_feat)

        outputs = {}
        for task, head in self.heads.items():
            out = head(feat)
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.squeeze(1)
            outputs[task] = out

        return outputs
