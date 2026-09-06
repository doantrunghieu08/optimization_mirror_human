"""
overlay_debug.py
================
Chiếu SMPL (HMR4D) xuống 2D và so với ViTPose trên vài frame.

Dùng để chốt convention real / mirror trước khi tin reprojection loss.

Usage:
    python visualizations/overlay_debug.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml

# Cho phép chạy file trực tiếp từ thư mục visualizations/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloaders.dataset import MirrorFusionDataset
from utils.camera_utils import project_3d_to_2d
from utils.smpl_utils import SMPLForwardPass, mirror_axis_angle, remirror_body_pose
from visualizations.vis_utils import overlay_2d_skeleton


def _read_frame(video_path: str, index: int, width: int, height: int) -> np.ndarray:
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
    return np.zeros((height, width, 3), dtype=np.uint8)


def _canvas_size(K: torch.Tensor, default_w: int) -> tuple[int, int]:
    cx = float(K[0, 2].item())
    cy = float(K[1, 2].item())
    width = int(round(default_w)) if default_w else int(round(cx * 2))
    height = int(round(cy * 2)) if cy > 1 else 1080
    return max(width, 64), max(height, 64)


def main(config: dict) -> None:
    data_cfg = config["data"]
    vis_cfg = config.get("visualization", {})
    out_dir = os.path.join(vis_cfg.get("output_dir", "output"), "overlay_debug")
    os.makedirs(out_dir, exist_ok=True)

    ds = MirrorFusionDataset(
        real_dir=data_cfg["real_dir"],
        mirror_dir=data_cfg["mirror_dir"],
        is_train=True,
        train_ratio=1.0,
        gt_dir=data_cfg.get("gt_dir"),
        image_width=data_cfg.get("image_width", 1920),
        mirror_intrinsics_mode=data_cfg.get("mirror_intrinsics_mode", "raw"),
        beta_mode=data_cfg.get("beta_mode", "median"),
        allow_missing_vitpose=True,
    )
    n = len(ds)
    if n == 0:
        raise ValueError("Dataset is empty")

    frames = sorted({0, n // 2, n - 1})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPLForwardPass(model_path=config["model"]["smpl_model_path"], device=device)
    video_path = vis_cfg.get("input_video", "")

    for item in frames:
        sample = ds[item]
        idx = ds.indices[item]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        joints_real = smpl(
            global_orient=batch["real_global_orient"],
            body_pose=batch["real_body_pose"],
            betas=batch["real_betas"],
            transl=batch["real_transl"],
        )
        proj_real, _ = project_3d_to_2d(joints_real[:, :17], batch["K_real"])

        mirrored_bp = remirror_body_pose(batch["mirror_body_pose"].view(1, 21, 3)).reshape(1, 63)
        raw_go = mirror_axis_angle(batch["mirror_global_orient"])
        raw_tr = batch["mirror_transl"].clone()
        raw_tr[..., 0] = -raw_tr[..., 0]
        joints_mirror = smpl(
            global_orient=raw_go,
            body_pose=mirrored_bp,
            betas=batch["real_betas"],
            transl=raw_tr,
        )
        proj_mirror, _ = project_3d_to_2d(joints_mirror[:, :17], batch["K_mirror"])

        w, h = _canvas_size(sample["K_real"], data_cfg.get("image_width", 1920))
        frame = _read_frame(video_path, idx, w, h)

        real_smpl = overlay_2d_skeleton(frame.copy(), proj_real[0].detach().cpu(), color=(0, 255, 0))
        real_vp = overlay_2d_skeleton(frame.copy(), sample["kp2d_real"], color=(255, 0, 0))
        real_both = overlay_2d_skeleton(real_smpl, sample["kp2d_real"], color=(255, 0, 0))

        mw, mh = _canvas_size(sample["K_mirror"], data_cfg.get("image_width", 1920))
        mirror_canvas = np.zeros((mh, mw, 3), dtype=np.uint8)
        mir_smpl = overlay_2d_skeleton(mirror_canvas.copy(), proj_mirror[0].detach().cpu(), color=(0, 255, 0))
        mir_both = overlay_2d_skeleton(mir_smpl, sample["kp2d_mirror"], color=(255, 0, 0))

        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}_real_smpl.png"), real_smpl)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}_real_vitpose.png"), real_vp)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}_real_overlay.png"), real_both)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}_mirror_overlay.png"), mir_both)
        print(f"[overlay] wrote frame {idx} → {out_dir}")

    print("Green = SMPL projection, Red = ViTPose. Check left/right shoulders and wrists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay SMPL vs ViTPose")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        main(yaml.safe_load(f))
