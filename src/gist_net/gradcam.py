#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate pre-ablation multi-view Grad-CAM figures for four task heads:
1) is_gist (Stage1)
2) risk (Stage2)
3) mutation (Stage2)
4) prognosis (Stage2)

Outputs per task:
- axial base / overlay
- coronal base / overlay
- sagittal base / overlay
- 3D semi-transparent ROI
- 3D semi-transparent Grad-CAM on ROI surface
- NIfTI heatmap
- task prediction CSV

This script keeps the Grad-CAM generation logic aligned with the current
pre-ablation workflow: crop by ROI bbox -> resize to ROI_SIZE -> inference ->
Grad-CAM -> mask-guided CAM -> restore to full-size volume.
"""

import os
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_dilation
from monai.networks.nets import DenseNet121
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .model import GatedFusion, MerlinAuxMultiTaskClassifier

warnings.filterwarnings("ignore")

ROI_SIZE = (128, 128, 96)
HU_MIN = -200.0
HU_MAX = 300.0

TASK_CONFIGS = [
    {
        "key": "is_gist",
        "display": "GIST vs. non-GIST",
        "stage": "stage1",
        "task_type": "binary",
        "folder": "01_is_gist"
    },
    {
        "key": "risk",
        "display": "Risk grade",
        "stage": "stage2",
        "task_type": "multiclass",
        "folder": "02_risk"
    },
    {
        "key": "mutation",
        "display": "Molecular subtype",
        "stage": "stage2",
        "task_type": "multiclass",
        "folder": "03_mutation"
    },
    {
        "key": "prognosis",
        "display": "Recurrence (within 24 months)",
        "stage": "stage2",
        "task_type": "binary",
        "folder": "04_prognosis"
    },
]


# =========================
# basic utilities
# =========================
def read_csv_auto(csv_path):
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"[CSV] loaded with encoding: {enc}, shape={df.shape}")
            return df
        except Exception as e:
            last_err = e
    raise last_err



def normalize_ct(ct):
    ct = np.asarray(ct, dtype=np.float32)
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN + 1e-8)
    return ct.astype(np.float32)



def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)



def get_bbox(mask, margin=20):
    pos = np.where(mask > 0)
    if len(pos[0]) == 0:
        raise ValueError("mask is empty")

    x0, x1 = int(pos[0].min()), int(pos[0].max()) + 1
    y0, y1 = int(pos[1].min()), int(pos[1].max()) + 1
    z0, z1 = int(pos[2].min()), int(pos[2].max()) + 1

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    z0 = max(0, z0 - margin)
    x1 = min(mask.shape[0], x1 + margin)
    y1 = min(mask.shape[1], y1 + margin)
    z1 = min(mask.shape[2], z1 + margin)
    return x0, x1, y0, y1, z0, z1



def crop_array(arr, bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    return arr[x0:x1, y0:y1, z0:z1]



def resize_3d_np(arr, out_size, mode="trilinear"):
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if mode == "nearest":
            y = F.interpolate(t, size=out_size, mode=mode)
        else:
            y = F.interpolate(t, size=out_size, mode=mode, align_corners=False)
    return y[0, 0].cpu().numpy().astype(np.float32)



def resize_mask_np(arr, out_size):
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        y = F.interpolate(t, size=out_size, mode="nearest")
    return y[0, 0].cpu().numpy().astype(np.float32)



def restore_to_full(cam_small, bbox, full_shape):
    x0, x1, y0, y1, z0, z1 = bbox
    roi_shape = (x1 - x0, y1 - y0, z1 - z0)
    cam_roi = zoom(cam_small, np.array(roi_shape) / np.array(cam_small.shape), order=1)
    out = np.zeros(full_shape, dtype=np.float32)
    out[x0:x1, y0:y1, z0:z1] = cam_roi
    return out



def prepare_model_input(ct_full, mask_full, bbox):
    ct_crop = crop_array(ct_full, bbox)
    mask_crop = crop_array(mask_full, bbox).astype(np.float32)

    ct_crop_norm = normalize_ct(ct_crop)
    ct_in = resize_3d_np(ct_crop_norm, ROI_SIZE, mode="trilinear")
    mask_in = resize_mask_np(mask_crop, ROI_SIZE)
    mask_in = (mask_in > 0.5).astype(np.float32)

    x_np = np.stack([ct_in, mask_in], axis=0).astype(np.float32)
    return x_np, ct_in, mask_in



def build_guided_cam_full(cam_resized, mask_in, bbox, full_shape, dilate_iter=3):
    mask_guided = binary_dilation(mask_in > 0.5, iterations=max(1, int(dilate_iter))).astype(np.float32)
    cam_guided_small = normalize_map(cam_resized * mask_guided)

    x0, x1, y0, y1, z0, z1 = bbox
    crop_shape = (x1 - x0, y1 - y0, z1 - z0)

    cam_crop = resize_3d_np(cam_guided_small, crop_shape, mode="trilinear")
    cam_full = restore_to_full(cam_crop, bbox, full_shape)
    cam_full = normalize_map(cam_full)
    return cam_full



def select_axial_slice(mask):
    area = mask.sum(axis=(0, 1))
    if np.max(area) <= 0:
        return mask.shape[2] // 2
    return int(np.argmax(area))



def select_coronal_slice(mask):
    area = mask.sum(axis=(0, 2))
    if np.max(area) <= 0:
        return mask.shape[1] // 2
    return int(np.argmax(area))



def select_sagittal_slice(mask):
    area = mask.sum(axis=(1, 2))
    if np.max(area) <= 0:
        return mask.shape[0] // 2
    return int(np.argmax(area))


# =========================
# model loading
# =========================
def strip_prefix_state_dict(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("model."):
            nk = nk[6:]
        new_state[nk] = v
    return new_state



def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError("checkpoint format not supported")



def load_stage1_model(checkpoint_path, device):
    model = DenseNet121(spatial_dims=3, in_channels=2, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(ckpt)
    state = strip_prefix_state_dict(state)
    msg = model.load_state_dict(state, strict=False)
    print("[STAGE1 CHECKPOINT]")
    print("missing keys:", len(msg.missing_keys))
    print("unexpected keys:", len(msg.unexpected_keys))
    model.eval()
    return model



def build_stage2_model(checkpoint_path, merlin_local_dir, device):
    kwargs = {
        "in_channels": 2,
        "tasks_config": {
            "risk": 3,
            "mutation": 4,
            "prognosis": 2,
        },
    }
    kwargs.update({"merlin_local_dir": merlin_local_dir, "fusion_init_scale": 0.05,
                   "fusion_dropout": 0.10, "freeze_merlin": True})
    model = MerlinAuxMultiTaskClassifier(**kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(ckpt)
    # Fusion is lazily constructed in the training model. Recreate it before
    # loading so that the trained fusion parameters are not silently skipped.
    merlin_dim = int(state["fusion.merlin_proj.1.weight"].shape[1])
    model.merlin_dim = merlin_dim
    model.fusion = GatedFusion(
        densenet_dim=model.dense_dim,
        merlin_dim=merlin_dim,
        init_scale=kwargs.get("fusion_init_scale", 0.03),
        dropout=kwargs.get("fusion_dropout", 0.10),
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("[STAGE2 CHECKPOINT]")
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    model = model.to(device)
    model.eval()
    return model


# =========================
# generic Grad-CAM engine
# =========================
class GenericGradCAMEngine:
    def __init__(self, model, target_layer, stage, task_name=None, task_type="binary"):
        self.model = model
        self.target_layer = target_layer
        self.stage = stage
        self.task_name = task_name
        self.task_type = task_type
        self.feature = None
        self.gradient = None
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.feature = out

        def _save_grad(g):
            self.gradient = g

        if torch.is_tensor(out) and out.requires_grad:
            out.register_hook(_save_grad)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_logits(self, x):
        if self.stage == "stage1":
            logits = self.model(x)
            return logits, None

        outputs = self.model(x)
        if not isinstance(outputs, dict):
            raise RuntimeError("Stage2 model output is not dict")
        if self.task_name not in outputs:
            raise RuntimeError(f"Task head not found: {self.task_name}")
        return outputs[self.task_name], outputs

    def generate(self, x, target_class=None):
        self.feature = None
        self.gradient = None

        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        logits, outputs = self.get_logits(x)

        if self.task_type == "binary":
            logits_flat = logits.reshape(-1)
            score = logits_flat[0]
            prob = float(torch.sigmoid(score).item())
            pred_cls = int(prob >= 0.5)
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1)[0].item())
            score = logits[0, target_class]
            prob = float(torch.softmax(logits, dim=1)[0, target_class].item())
            pred_cls = int(torch.argmax(logits, dim=1)[0].item())

        score.backward(retain_graph=True)

        if self.feature is None:
            raise RuntimeError("GradCAM feature not captured")
        if self.gradient is None:
            raise RuntimeError("GradCAM gradient not captured")

        weights = self.gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.feature).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam = normalize_map(cam)
        return cam, prob, pred_cls, outputs


# =========================
# plotting helpers
# =========================
def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def draw_mask_contour(ax, mask2d, color="lime", lw=0.5, transpose=False):
    if mask2d is None or np.max(mask2d) <= 0:
        return
    if transpose:
        ax.contour(mask2d.T, levels=[0.5], colors=[color], linewidths=lw, origin="lower")
    else:
        ax.contour(mask2d, levels=[0.5], colors=[color], linewidths=lw)



def apply_heat_overlay(ax, base_img, heat_img, alpha=0.55, low_cut=0.20, transpose=False):
    if transpose:
        ax.imshow(base_img.T, cmap="gray", vmin=0, vmax=1, origin="lower")
        show = np.array(heat_img, dtype=np.float32).copy().T
        show[show < low_cut] = np.nan
        ax.imshow(show, cmap="jet", alpha=alpha, vmin=0, vmax=1, origin="lower")
    else:
        ax.imshow(base_img, cmap="gray", vmin=0, vmax=1)
        show = np.array(heat_img, dtype=np.float32).copy()
        show[show < low_cut] = np.nan
        ax.imshow(show, cmap="jet", alpha=alpha, vmin=0, vmax=1)



def save_axial_base(ct, mask, z, out_png, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ct[:, :, z], cmap="gray", vmin=0, vmax=1)
    draw_mask_contour(ax, mask[:, :, z], color="lime", lw=0.5)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def save_axial_overlay(ct, cam, mask, z, out_png, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_heat_overlay(ax, ct[:, :, z], cam[:, :, z], alpha=0.55, low_cut=0.20, transpose=False)
    draw_mask_contour(ax, mask[:, :, z], color="lime", lw=0.5)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def save_coronal_base(ct, mask, y, out_png, title=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(ct[:, y, :].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    draw_mask_contour(ax, mask[:, y, :], color="lime", lw=0.5, transpose=True)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def save_coronal_overlay(ct, cam, mask, y, out_png, title=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    apply_heat_overlay(ax, ct[:, y, :], cam[:, y, :], alpha=0.55, low_cut=0.20, transpose=True)
    draw_mask_contour(ax, mask[:, y, :], color="lime", lw=0.5, transpose=True)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def save_sagittal_base(ct, mask, x, out_png, title=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(ct[x, :, :].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    draw_mask_contour(ax, mask[x, :, :], color="lime", lw=0.5, transpose=True)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def save_sagittal_overlay(ct, cam, mask, x, out_png, title=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    apply_heat_overlay(ax, ct[x, :, :], cam[x, :, :], alpha=0.55, low_cut=0.20, transpose=True)
    draw_mask_contour(ax, mask[x, :, :], color="lime", lw=0.5, transpose=True)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    save_fig(fig, out_png)



def set_axes_equal_3d(ax, limits):
    xlim, ylim, zlim = limits
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    max_range = max(xr, yr, zr)
    xmid = np.mean(xlim)
    ymid = np.mean(ylim)
    zmid = np.mean(zlim)
    ax.set_xlim(xmid - max_range / 2.0, xmid + max_range / 2.0)
    ax.set_ylim(ymid - max_range / 2.0, ymid + max_range / 2.0)
    ax.set_zlim(zmid - max_range / 2.0, zmid + max_range / 2.0)



def sample_volume_nn(vol, verts):
    idx = np.round(verts).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, vol.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, vol.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, vol.shape[2] - 1)
    return vol[idx[:, 0], idx[:, 1], idx[:, 2]]



def _crop_for_3d(mask, cam=None, margin=5):
    """
    Crop 3D rendering area around ROI to avoid the tumor appearing tiny
    in the whole CT coordinate space.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Empty mask for 3D rendering")

    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    z0 = max(0, z0 - margin)

    x1 = min(mask.shape[0], x1 + margin)
    y1 = min(mask.shape[1], y1 + margin)
    z1 = min(mask.shape[2], z1 + margin)

    mask_crop = mask[x0:x1, y0:y1, z0:z1]

    if cam is not None:
        cam_crop = cam[x0:x1, y0:y1, z0:z1]
        return mask_crop, cam_crop

    return mask_crop


def save_3d_roi(mask, out_png, title=None, elev=25, azim=45):
    """
    Zoomed 3D ROI rendering.
    Only render local tumor region.
    """
    mask_crop = _crop_for_3d(mask, margin=5)

    verts, faces, _, _ = marching_cubes(
        (mask_crop > 0).astype(np.float32),
        level=0.5
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        verts[faces],
        alpha=0.35
    )
    mesh.set_facecolor((0.35, 0.85, 0.35, 0.35))
    mesh.set_edgecolor((0.10, 0.55, 0.10, 0.15))
    ax.add_collection3d(mesh)

    set_axes_equal_3d(
        ax,
        ((0, mask_crop.shape[0]),
         (0, mask_crop.shape[1]),
         (0, mask_crop.shape[2]))
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(mask_crop.shape)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=10)

    save_fig(fig, out_png)


def save_3d_cam(mask, cam, out_png, title=None, elev=25, azim=45):
    """
    Zoomed 3D Grad-CAM rendering.
    Render heatmap on ROI surface.
    """
    mask_crop, cam_crop = _crop_for_3d(mask, cam, margin=5)

    verts, faces, _, _ = marching_cubes(
        (mask_crop > 0).astype(np.float32),
        level=0.5
    )

    vert_vals = sample_volume_nn(cam_crop, verts)
    vert_vals = normalize_map(vert_vals)

    face_vals = vert_vals[faces].mean(axis=1)

    cmap = plt.get_cmap("jet")
    face_colors = cmap(face_vals)
    face_colors[:, 3] = 0.85

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        verts[faces],
        alpha=0.85
    )
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor((0.1,0.1,0.1,0.05))
    ax.add_collection3d(mesh)

    set_axes_equal_3d(
        ax,
        ((0, mask_crop.shape[0]),
         (0, mask_crop.shape[1]),
         (0, mask_crop.shape[2]))
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(mask_crop.shape)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=10)

    save_fig(fig, out_png)



# =========================
# task processing
# =========================
def process_task(row, task_cfg, stage1_model, stage2_model, out_root, device,
                 margin=20, dilate_iter=3):
    case_id = str(row["id"])
    task_key = task_cfg["key"]
    task_display = task_cfg["display"]
    stage = task_cfg["stage"]
    task_type = task_cfg["task_type"]
    out_dir = os.path.join(out_root, task_cfg["folder"])
    os.makedirs(out_dir, exist_ok=True)

    print(f"[TASK] {task_key} | case={case_id}")

    ct_nii = nib.load(row["image"])
    ct = ct_nii.get_fdata().astype(np.float32)
    mask_nii = nib.load(row["mask"])
    mask = (mask_nii.get_fdata() > 0).astype(np.uint8)

    bbox = get_bbox(mask, margin=margin)
    x_np, _, _ = prepare_model_input(ct, mask, bbox)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)

    if stage == "stage1":
        model = stage1_model
        target_layer = model.features.denseblock4
        engine = GenericGradCAMEngine(model, target_layer, stage="stage1", task_name=None, task_type=task_type)
    else:
        model = stage2_model
        target_layer = model.feature_extractor.denseblock4
        engine = GenericGradCAMEngine(model, target_layer, stage="stage2", task_name=task_key, task_type=task_type)

    try:
        cam_resized, prob, pred_cls, _ = engine.generate(x)
    finally:
        engine.remove()

    cam_full = build_guided_cam_full(
        cam_resized,
        x[0, 1].detach().cpu().numpy(),
        bbox,
        ct.shape,
        dilate_iter=dilate_iter
    )

    ct_norm = normalize_ct(ct)
    axial_idx = select_axial_slice(mask)
    coronal_idx = select_coronal_slice(mask)
    sagittal_idx = select_sagittal_slice(mask)

    nib.save(
        nib.Nifti1Image(cam_full.astype(np.float32), ct_nii.affine),
        os.path.join(out_dir, f"{case_id}_{task_key}_before_heatmap.nii.gz")
    )

    save_axial_base(
        ct_norm, mask, axial_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_01_axial_base.png"),
        title=f"{task_display} | axial CT"
    )
    save_axial_overlay(
        ct_norm, cam_full, mask, axial_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_02_axial_gradcam.png"),
        title=f"{task_display} | axial Grad-CAM"
    )

    save_coronal_base(
        ct_norm, mask, coronal_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_03_coronal_base.png"),
        title=f"{task_display} | coronal CT"
    )
    save_coronal_overlay(
        ct_norm, cam_full, mask, coronal_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_04_coronal_gradcam.png"),
        title=f"{task_display} | coronal Grad-CAM"
    )

    save_sagittal_base(
        ct_norm, mask, sagittal_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_05_sagittal_base.png"),
        title=f"{task_display} | sagittal CT"
    )
    save_sagittal_overlay(
        ct_norm, cam_full, mask, sagittal_idx,
        os.path.join(out_dir, f"{case_id}_{task_key}_06_sagittal_gradcam.png"),
        title=f"{task_display} | sagittal Grad-CAM"
    )

    save_3d_roi(
        mask,
        os.path.join(out_dir, f"{case_id}_{task_key}_07_3d_roi.png"),
        title=f"{task_display} | 3D ROI"
    )
    save_3d_cam(
        mask, cam_full,
        os.path.join(out_dir, f"{case_id}_{task_key}_08_3d_gradcam.png"),
        title=f"{task_display} | 3D Grad-CAM"
    )

    pred_df = pd.DataFrame([{
        "id": case_id,
        "task": task_key,
        "task_display": task_display,
        "task_type": task_type,
        "pred_class": pred_cls,
        "pred_prob": prob,
        "axial_idx": axial_idx,
        "coronal_idx": coronal_idx,
        "sagittal_idx": sagittal_idx,
    }])
    pred_df.to_csv(os.path.join(out_dir, f"{case_id}_{task_key}_prediction.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] {task_key} | pred_cls={pred_cls} | pred_prob={prob:.4f}")
    return {
        "id": case_id,
        "task": task_key,
        "pred_class": pred_cls,
        "pred_prob": prob,
        "axial_idx": axial_idx,
        "coronal_idx": coronal_idx,
        "sagittal_idx": sagittal_idx,
        "status": "OK",
    }


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Generate original multi-view Grad-CAM figures for four tasks"
    )
    parser.add_argument("--csv", required=True, help="cases_filtered.csv path")
    parser.add_argument("--case_id", required=True, help="case id, e.g. 59.nii")
    parser.add_argument("--stage1_ckpt", required=True, help="stage1 main_best_model.pt")
    parser.add_argument("--stage2_ckpt", required=True, help="stage2 best_model_merlin_aux_subtasks.pt")
    parser.add_argument("--merlin_local_dir", required=True, help="local Merlin folder")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--dilate_iter", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = read_csv_auto(args.csv)
    row_df = df[df["id"].astype(str) == str(args.case_id)]
    if len(row_df) == 0:
        raise ValueError(f"case not found: {args.case_id}")
    row = row_df.iloc[0]

    print("[INFO] device:", args.device)
    stage1_model = load_stage1_model(args.stage1_ckpt, args.device)
    stage2_model = build_stage2_model(args.stage2_ckpt, args.merlin_local_dir, args.device)

    records = []
    for task_cfg in TASK_CONFIGS:
        try:
            rec = process_task(
                row=row,
                task_cfg=task_cfg,
                stage1_model=stage1_model,
                stage2_model=stage2_model,
                out_root=args.out_dir,
                device=args.device,
                margin=args.margin,
                dilate_iter=args.dilate_iter,
            )
            records.append(rec)
        except Exception as e:
            print(f"[FAILED] {task_cfg['key']}: {repr(e)}")
            traceback.print_exc()
            records.append({
                "id": str(args.case_id),
                "task": task_cfg["key"],
                "status": "FAILED",
                "error": repr(e),
            })

    pd.DataFrame(records).to_csv(
        os.path.join(args.out_dir, f"{args.case_id}_four_tasks_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print("[DONE]", os.path.join(args.out_dir, f"{args.case_id}_four_tasks_summary.csv"))


if __name__ == "__main__":
    main()
