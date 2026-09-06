"""
gt_utils.py — Tiện ích đọc Ground Truth từ thư mục inputs/GT/*.json

Mỗi file JSON chứa danh sách người trong khung hình.
Format mỗi keypoint: [x, y, z, confidence]
Quy ước: OpenPose-25 / MoCapStudio.
Chiều cao ước tính từ chuỗi xương (không phụ thuộc trục đứng).
"""

from __future__ import annotations

import os
import json
import glob
import numpy as np
import torch


# Index trong GT keypoints
GT_NOSE_IDX   = 0
GT_RANKLE_IDX = 11
GT_LANKLE_IDX = 14

# Hệ số bù mũi → đỉnh đầu (~13-15% chiều cao toàn thân)
HEIGHT_SCALE  = 1.15

# Ngưỡng chiều cao hợp lệ (m) — loại bỏ frame bị khuất/missing
HEIGHT_MIN    = 1.0
HEIGHT_MAX    = 2.5

# Fallback khi frame không hợp lệ (dùng giá trị median của toàn video)
FALLBACK_HEIGHT_M = 1.65

# OpenPose-25 indices (cùng quy ước evaluate.py)
_OP25_NOSE = 0
_OP25_L_SHOULDER = 5
_OP25_R_SHOULDER = 2
_OP25_L_HIP = 12
_OP25_R_HIP = 9
_OP25_L_KNEE = 13
_OP25_R_KNEE = 10
_OP25_L_ANKLE = 14
_OP25_R_ANKLE = 11


def normalize_optional_dir(value) -> str | None:
    """YAML `None` / `null` thường thành chuỗi 'None' — chuẩn hóa về Python None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def estimate_height_op25(kp: np.ndarray) -> float:
    """
    Chiều cao từ chuỗi xương (không phụ thuộc trục đứng Y/Z).
    kp: (25, 3+) OpenPose-25, đơn vị mét.
    """
    xyz = kp[:, :3]
    mid_sh = 0.5 * (xyz[_OP25_L_SHOULDER] + xyz[_OP25_R_SHOULDER])
    mid_hip = 0.5 * (xyz[_OP25_L_HIP] + xyz[_OP25_R_HIP])
    head = np.linalg.norm(xyz[_OP25_NOSE] - mid_sh)
    torso = np.linalg.norm(mid_sh - mid_hip)
    l_leg = (
        np.linalg.norm(xyz[_OP25_L_HIP] - xyz[_OP25_L_KNEE])
        + np.linalg.norm(xyz[_OP25_L_KNEE] - xyz[_OP25_L_ANKLE])
    )
    r_leg = (
        np.linalg.norm(xyz[_OP25_R_HIP] - xyz[_OP25_R_KNEE])
        + np.linalg.norm(xyz[_OP25_R_KNEE] - xyz[_OP25_R_ANKLE])
    )
    return float((head + torso + 0.5 * (l_leg + r_leg)) * HEIGHT_SCALE)


def load_gt_heights_from_dir(gt_dir: str, num_frames: int, person_id: int = 0) -> torch.Tensor:
    """
    Đọc chiều cao GT per-frame từ thư mục chứa các file JSON.

    Mỗi file XXXXXX.json tương ứng với 1 frame (sort theo tên file).
    Chỉ lấy người đầu tiên (data[0]) trong mỗi frame.

    Chiều cao ước tính từ chuỗi xương OpenPose-25 (mét).

    Args:
        gt_dir    : Đường dẫn đến thư mục chứa XXXXXX.json
        num_frames: Số frame của video (để kiểm tra khớp với dataset)

    Returns:
        gt_heights: Tensor (num_frames,) — chiều cao GT per-frame (mét)
                    Frame không hợp lệ → được thay bằng median của các frame hợp lệ.
    """
    json_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))

    if len(json_files) == 0:
        print(f"[gt_utils] WARNING: Không tìm thấy JSON trong {gt_dir}. "
              f"Dùng fallback height={FALLBACK_HEIGHT_M}m.")
        return torch.full((num_frames,), FALLBACK_HEIGHT_M, dtype=torch.float32)

    if len(json_files) != num_frames:
        print(f"[gt_utils] WARNING: Số file GT ({len(json_files)}) "
              f"!= num_frames ({num_frames}). Sẽ dùng min(len, num_frames) frames.")

    n = min(len(json_files), num_frames)
    raw_heights = np.zeros(n, dtype=np.float32)

    for i, fpath in enumerate(json_files[:n]):
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            # Lấy người đầu tiên trong frame
            person = next((item for item in data if item.get("id") == person_id), None)
            if person is None:
                raise ValueError(f"person_id={person_id} not found")
            kp = np.array(person["keypoints3d"], dtype=np.float32)  # (25, 4)
            raw_heights[i] = estimate_height_op25(kp)

        except Exception as e:
            print(f"[gt_utils] WARNING: Lỗi đọc {fpath}: {e}. Gán 0 tạm thời.")
            raw_heights[i] = 0.0

    # ── Tính median của các frame hợp lệ để dùng làm fallback ─────────
    valid_mask = (raw_heights >= HEIGHT_MIN) & (raw_heights <= HEIGHT_MAX)
    valid_heights = raw_heights[valid_mask]

    if len(valid_heights) > 0:
        median_h = float(np.median(valid_heights))
    else:
        print(f"[gt_utils] WARNING: Không có frame hợp lệ. "
              f"Dùng fallback={FALLBACK_HEIGHT_M}m.")
        median_h = FALLBACK_HEIGHT_M

    # ── Thay thế các frame không hợp lệ bằng median ────────────────────
    raw_heights[~valid_mask] = median_h

    # ── Nếu num_frames > n (thiếu GT), pad bằng median ─────────────────
    if num_frames > n:
        pad = np.full(num_frames - n, median_h, dtype=np.float32)
        raw_heights = np.concatenate([raw_heights, pad])

    print(f"[gt_utils] Loaded {n} GT height frames | "
          f"valid={valid_mask.sum()}/{n} | "
          f"median={median_h:.3f}m | fallback frames={int((~valid_mask).sum())}")

    return torch.from_numpy(raw_heights)  # (num_frames,)
